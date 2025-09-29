"""
Main integration bridge: QLabs QCar2 + SimLingo adapters + model wrapper.

This module orchestrates:
- QLabs connection and QCar2 actor lifecycle
- Camera acquisition -> data adapter -> model inference -> control adapter
- Real-time control loop for basic autonomous driving demo
"""
from __future__ import annotations

import sys
import time
import logging
import math

import io
from contextlib import redirect_stdout, redirect_stderr
import threading
import select
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Paths
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]  # .../Qcar2
QVL_PATH = REPO_ROOT / "0_libraries" / "python"

# Local imports (do not import QVL yet; delay until connect())
from adapters.data_adapter import Qcar2DataAdapter
from adapters.control_adapter import Qcar2ControlAdapter
from models.simlingo_wrapper import SimLingoModel # ModelInferenceError

logger = logging.getLogger(__name__)


@dataclass
class DriverConfig:
    host: str = "localhost"
    camera: Optional[int] = None  # resolved after importing QVL
    hz: float = 10.0
    duration_sec: float = 30.0
    try_load_weights: bool = False  # heavy; keep False by default
    # Spawn/initial placement config (tweak to avoid spawning on obstacles)
    spawn_location: tuple[float, float, float] = (0.0, 0.0, 0.1)
    spawn_yaw_deg: float = 90.0
    spawn_autosearch: bool = True  # try a few candidate spots if initial spot collides


class SimLingoQcar2Driver:
    def __init__(self, cfg: DriverConfig) -> None:
        self.cfg = cfg
        self.qlabs = None  # type: ignore
        self.car = None  # type: ignore
        self.QLabsQCar2 = None  # resolved at runtime

        # Adapters and model
        # Use the exact CSI camera resolution 820x410 from QVL
        self.data_adapter = Qcar2DataAdapter(target_size=(820, 410))
        self.control_adapter = Qcar2ControlAdapter(max_forward_speed=2.0, max_turn_angle=0.6)
        self.model = SimLingoModel(
            model_root=REPO_ROOT / "simlingo_qcar2_integration" / "models" / "simlingo",
        )

        # Runtime state for speed estimation
        self._last_location: Optional[tuple[float, float, float]] = None
        self._last_time: Optional[float] = None
        self._last_speed_mps: float = 0.0

        # QVL CSI front camera calibration (820x410, 160° FOV approximated as pinhole)
        fx = 820.0 / (2.0 * math.tan(math.radians(160.0 / 2.0)))
        self._K = [[fx, 0.0, 820.0/2.0],
                   [0.0, fx, 410.0/2.0],
                   [0.0, 0.0, 1.0]]
        self._E = [
            [1.0, 0.0, 0.0, 1.83],
            [0.0, 1.0, 0.0, 0.00],
            [0.0, 0.0, 1.0, 1.10],
            [0.0, 0.0, 0.0, 1.0],
        ]



        # Instruction input thread controls
        self._stop_event = threading.Event()
        self._instr_thread: threading.Thread | None = None

    def connect(self) -> bool:
        # Add QVL path and import lazily to avoid requiring Quanser package at import-time
        if str(QVL_PATH) not in sys.path:
            sys.path.insert(0, str(QVL_PATH))
        try:
            from qvl.qlabs import QuanserInteractiveLabs
            from qvl.qcar2 import QLabsQCar2
        except Exception as e:
            logger.error(f"Failed to import QVL modules. Ensure Quanser Python SDK is available: {e}")
            return False

        # Save class for later use
        self.QLabsQCar2 = QLabsQCar2

        # Resolve camera if not provided
        if self.cfg.camera is None:
            try:
                self.cfg.camera = QLabsQCar2.CAMERA_CSI_FRONT
            except Exception:
                self.cfg.camera = 3  # fallback to typical CSI front camera id


        self.qlabs = QuanserInteractiveLabs()
        if not self.qlabs.open(self.cfg.host):
            logger.error("Unable to connect to QLabs. Ensure QLabs is running and a layout is open.")
            return False
        logger.info("Connected to QLabs")
        return True

    def spawn_vehicle(self) -> bool:
        assert self.qlabs is not None and self.QLabsQCar2 is not None

        self.car = self.QLabsQCar2(self.qlabs)
        status, actor_num = self.car.spawn(location=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1])
        if status != 0:
            logger.error(f"Failed to spawn QCar2 (status={status})")
            return False


        def try_set(loc_xyz, yaw_deg):
            try:
                ok, loc, rot_deg, fv, uv, front_hit, rear_hit = self.car.set_transform_and_request_state_degrees(
                    location=[float(loc_xyz[0]), float(loc_xyz[1]), float(loc_xyz[2])],
                    rotation=[0.0, 0.0, float(yaw_deg)],
                    enableDynamics=True,
                    headlights=False,
                    leftTurnSignal=False,
                    rightTurnSignal=False,
                    brakeSignal=False,
                    reverseSignal=False,
                    waitForConfirmation=True,
                )
                return ok, loc, yaw_deg, front_hit, rear_hit
            except Exception as e:

                return False, None, yaw_deg, True, True

        # First, try the configured location
        chosen = None
        x, y, z = self.cfg.spawn_location
        yaw_deg = self.cfg.spawn_yaw_deg
        ok, loc, yaw_used, front_hit, rear_hit = try_set((x, y, z), yaw_deg)
        if ok and not front_hit and not rear_hit:
            chosen = (loc, yaw_used)

        elif self.cfg.spawn_autosearch:
            # Search a few candidate spots near origin and along axes

            candidates = [
                ((0.0, 0.0, 0.12), 0.0),
                ((0.0, 2.0, 0.12), 0.0),
                ((0.0, -2.0, 0.12), 180.0),
                ((2.0, 0.0, 0.12), 90.0),
                ((-2.0, 0.0, 0.12), -90.0),
                ((4.0, 0.0, 0.12), 90.0),
                ((0.0, 4.0, 0.12), 0.0),
                ((-4.0, 0.0, 0.12), -90.0),
                ((0.0, -4.0, 0.12), 180.0),
            ]
            for (cx, cy, cz), cyaw in candidates:
                ok, loc, yaw_used, front_hit, rear_hit = try_set((cx, cy, cz), cyaw)
                if ok and not front_hit and not rear_hit:
                    chosen = (loc, yaw_used)

                    break

        if chosen is None:
            if ok:
                pass
            else:
                pass
        return True

    def possess_camera(self) -> None:
        try:
            assert self.car is not None and self.QLabsQCar2 is not None
            self.car.possess(camera=self.QLabsQCar2.CAMERA_TRAILING)
        except Exception as e:
            pass

    def _instruction_input_loop(self) -> None:
        """Background thread: read user instructions from stdin and update model prompt."""
        logger.info("Instruction input: type a new driving instruction and press Enter (e.g., 'keep right and slow down').")
        logger.info("Type '/show' to print the current instruction. Ctrl+D to stop input thread.")
        while not self._stop_event.is_set():
            try:
                # poll stdin so we can exit when stop_event is set
                rlist, _, _ = select.select([sys.stdin], [], [], 0.5)
                if not rlist:
                    continue
                line = sys.stdin.readline()
                if line == '':  # EOF
                    break
                line = line.strip()
                if not line:
                    continue
                if line == '/show':
                    try:
                        cur = self.model.get_instruction()
                        logger.info(f"Current instruction: {cur}")
                    except Exception:
                        pass
                    continue
                # Update model instruction
                try:
                    self.model.set_instruction(line)
                except Exception:
                    pass
            except Exception:
                # Keep the thread alive; don't spam logs
                time.sleep(0.5)

    def _start_instruction_thread(self) -> None:
        if self._instr_thread is None or not self._instr_thread.is_alive():
            self._stop_event.clear()
            t = threading.Thread(target=self._instruction_input_loop, name="instruction-input", daemon=True)
            t.start()
            self._instr_thread = t

    def _stop_instruction_thread(self) -> None:
        try:
            self._stop_event.set()
            if self._instr_thread is not None:
                self._instr_thread.join(timeout=1.0)
        except Exception:
            pass

            assert self.car is not None and self.QLabsQCar2 is not None
            self.car.possess(camera=self.QLabsQCar2.CAMERA_TRAILING)
        except Exception as e:
            pass

    def initialize_model(self) -> bool:

        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                ok = self.model.load(try_load_weights=self.cfg.try_load_weights)
        finally:
            captured_out = buf_out.getvalue()
            captured_err = buf_err.getvalue()
        if ok:
            logger.info("SimLingo model ready (real VLA inference; no fallback)")
        else:
            # silently ignore captured init logs to avoid noise
            if captured_out or captured_err:
                pass
        return ok

    def control_loop(self) -> None:
        assert self.car is not None
        period = 1.0 / max(1e-3, self.cfg.hz)
        end_time = time.time() + self.cfg.duration_sec


        while time.time() < end_time:
            t0 = time.time()
            try:
                # 1) Acquire camera
                img = self.data_adapter.get_qcar2_camera_data(self.car, int(self.cfg.camera))
                if img is None:
                    # If no image, stop car briefly
                    self.control_adapter.send_control_command(self.car, 0.0, 0.0)
                    time.sleep(period)
                    continue

                # 2) Prepare model input
                model_input = self.data_adapter.process_camera_image(img)

                # 3) Model inference with real context (CSI front camera 820x410)
                H, W = model_input.shape[:2]
                camera_ctx = {
                    "width": W,
                    "height": H,
                    "intrinsics": self._K,
                    "extrinsics": self._E,
                }
                vehicle_ctx = {"speed_mps": float(self._last_speed_mps)}
                out = self.model.inference(model_input, camera_info=camera_ctx, vehicle_info=vehicle_ctx)

                # Log only the commentary (language) output
                try:
                    lang_txt = out.get("language_text") if isinstance(out, dict) else None
                    if isinstance(lang_txt, str) and lang_txt.strip():
                        t = lang_txt.strip()
                        logger.info(f"lang: {t}")
                except Exception:
                    pass

                # 4) Convert and send to QCar2 (no logging)
                fwd, turn = self.control_adapter.process_simlingo_output(out)
                ok, info = self.control_adapter.send_control_command(self.car, fwd, turn)

                # 4b) Update speed estimate from location delta
                try:
                    now = time.time()
                    loc = info.get("location")
                    rot = info.get("rotation")
                    if loc is not None and rot is not None:
                        if self._last_location is not None and self._last_time is not None:
                            dt = max(1e-3, now - self._last_time)
                            dx = float(loc[0]) - float(self._last_location[0])
                            dy = float(loc[1]) - float(self._last_location[1])
                            yaw = float(rot[2])  # radians
                            fx = math.cos(yaw); fy = math.sin(yaw)
                            self._last_speed_mps = (dx * fx + dy * fy) / dt
                        self._last_location = (float(loc[0]), float(loc[1]), float(loc[2]))
                        self._last_time = now
                except Exception:
                    pass






            except KeyboardInterrupt:
                break
            except ModelInferenceError as e:
                logger.error(f"Model inference error: {e}. Stopping vehicle and exiting loop.")
                try:
                    self.control_adapter.send_control_command(self.car, 0.0, 0.0)
                except Exception:
                    pass
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                # Safe stop on error, then continue
                try:
                    self.control_adapter.send_control_command(self.car, 0.0, 0.0)
                except Exception:
                    pass

            # Maintain loop rate
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

        # Stop the vehicle at end
        try:
            self.control_adapter.send_control_command(self.car, 0.0, 0.0)
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self.qlabs is not None:
                self.qlabs.close()
        except Exception:
            pass

    def run(self) -> bool:
        if not self.connect():
            return False
        try:
            if not self.spawn_vehicle():
                return False
            self.possess_camera()
            if not self.initialize_model():
                logger.error("Model failed to initialize; aborting (no fallback driving).")
                return False
            # Start instruction input thread (keyboard)
            try:
                self._start_instruction_thread()
                _ = self.model.get_instruction()

            except Exception as e:
                logger.error(f"Could not start instruction input thread: {e}")
            self.control_loop()
            return True
        finally:
            # Stop input thread and close QLabs
            try:
                self._stop_instruction_thread()
            except Exception:
                pass
            self.close()


def run_cli(
    hz: float = 10.0,
    duration: float = 30.0,
    try_load_weights: bool = False,
    spawn_x: float | None = None,
    spawn_y: float | None = None,
    spawn_z: float | None = None,
    spawn_yaw: float | None = None,
    spawn_autosearch: bool = True,
) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    # Reduce third-party library verbosity during model initialization
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers_modules").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)
    # Defaults mirror DriverConfig dataclass defaults
    def_loc = (0.0, 0.0, 0.1)
    def_yaw = 90.0
    cfg = DriverConfig(
        hz=hz,
        duration_sec=duration,
        try_load_weights=try_load_weights,
        spawn_location=(
            float(spawn_x) if spawn_x is not None else def_loc[0],
            float(spawn_y) if spawn_y is not None else def_loc[1],
            float(spawn_z) if spawn_z is not None else def_loc[2],
        ),
        spawn_yaw_deg=float(spawn_yaw) if spawn_yaw is not None else def_yaw,
        spawn_autosearch=spawn_autosearch,
    )
    driver = SimLingoQcar2Driver(cfg)
    ok = driver.run()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run_cli())

