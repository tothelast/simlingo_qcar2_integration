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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

# Paths
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # .../simlingo_qcar2_integration


# Local imports (do not import QVL yet; delay until connect())
from adapters.data_adapter import Qcar2DataAdapter
from adapters.control_adapter import Qcar2ControlAdapter
from models.simlingo_wrapper import SimLingoModel

logger = logging.getLogger(__name__)


@dataclass
class DriverConfig:
    host: str = "localhost"
    hz: float = 5.0  # SimLingo runs at 5 Hz (0.2s intervals between waypoints)
    duration_sec: float = 30.0
    spawn_location: tuple[float, float, float] = (0.0, 0.0, 0.1)
    spawn_yaw_deg: float = 90.0


class SimLingoQcar2Driver:
    def __init__(self, cfg: DriverConfig) -> None:
        self.cfg = cfg
        self.qlabs = None  # type: ignore
        self.car = None  # type: ignore
        self.QLabsQCar2 = None  # resolved at runtime
        self.camera_id = None  # CSI Front camera ID (resolved after QVL import)

        # Adapters and model
        # Use the exact CSI camera resolution 820x410 from QVL
        self.data_adapter = Qcar2DataAdapter(target_size=(820, 410))
        self.control_adapter = Qcar2ControlAdapter(max_forward_speed=2.0, max_turn_angle=0.6)
        self.model = SimLingoModel(
            model_root=REPO_ROOT / "models" / "simlingo",
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
            [1.0, 0.0, 0.0, 0.193],
            [0.0, 1.0, 0.0, 0.00],
            [0.0, 0.0, 1.0, 0.085],
            [0.0, 0.0, 0.0, 1.0],
        ]


    def connect(self) -> bool:
        # Save class for later use
        self.QLabsQCar2 = QLabsQCar2
        # Set camera ID to CSI Front
        self.camera_id = QLabsQCar2.CAMERA_CSI_FRONT

        self.qlabs = QuanserInteractiveLabs()
        if not self.qlabs.open(self.cfg.host):
            logger.error("Unable to connect to QLabs. Ensure QLabs is running and a layout is open.")
            return False
        logger.info("Connected to QLabs")
        return True

    def spawn_vehicle(self) -> bool:
        assert self.qlabs is not None and self.QLabsQCar2 is not None

        self.car = self.QLabsQCar2(self.qlabs)
        status, _ = self.car.spawn(location=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1])
        if status != 0:
            logger.error(f"Failed to spawn QCar2 (status={status})")
            return False

        # Set initial position
        x, y, z = self.cfg.spawn_location
        yaw_deg = self.cfg.spawn_yaw_deg
        self.car.set_transform_and_request_state_degrees(
            location=[float(x), float(y), float(z)],
            rotation=[0.0, 0.0, float(yaw_deg)],
            enableDynamics=True,
            headlights=False,
            leftTurnSignal=False,
            rightTurnSignal=False,
            brakeSignal=False,
            reverseSignal=False,
            waitForConfirmation=True,
        )
        return True

    def possess_camera(self) -> None:
        assert self.car is not None and self.QLabsQCar2 is not None
        self.car.possess(camera=self.QLabsQCar2.CAMERA_TRAILING)

    def initialize_model(self) -> bool:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            ok = self.model.load()
        if ok:
            logger.info("SimLingo model ready")
        return ok

    def control_loop(self) -> None:
        assert self.car is not None
        period = 1.0 / max(1e-3, self.cfg.hz)
        end_time = time.time() + self.cfg.duration_sec


        while time.time() < end_time:
            t0 = time.time()
            try:
                # 1) Acquire camera
                img = self.data_adapter.get_qcar2_camera_data(self.car, self.camera_id)
                if img is None:
                    # If no image, stop car briefly
                    self.control_adapter.send_control_command(self.car, 0.0, 0.0)
                    time.sleep(period)
                    continue

                # 2) Prepare model input
                model_input = self.data_adapter.process_camera_image(img)

                # 3) Model inference
                H, W = model_input.shape[:2]
                camera_ctx = {
                    "width": W,
                    "height": H,
                    "intrinsics": self._K,
                    "extrinsics": self._E,
                }
                vehicle_ctx = {"speed_mps": float(self._last_speed_mps)}
                out = self.model.inference(model_input, camera_info=camera_ctx, vehicle_info=vehicle_ctx)

                # 4) Convert and send to QCar2
                fwd, turn = self.control_adapter.process_simlingo_output(out, current_speed=self._last_speed_mps)
                _, info = self.control_adapter.send_control_command(self.car, fwd, turn)

                # 5) Update speed estimate
                now = time.time()
                loc = info.get("location")
                rot = info.get("rotation")
                if loc is not None and rot is not None:
                    if self._last_location is not None and self._last_time is not None:
                        dt = max(1e-3, now - self._last_time)
                        dx = float(loc[0]) - float(self._last_location[0])
                        dy = float(loc[1]) - float(self._last_location[1])
                        yaw = float(rot[2])
                        fx = math.cos(yaw); fy = math.sin(yaw)
                        self._last_speed_mps = (dx * fx + dy * fy) / dt
                    self._last_location = (float(loc[0]), float(loc[1]), float(loc[2]))
                    self._last_time = now






            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                self.control_adapter.send_control_command(self.car, 0.0, 0.0)

            # Maintain loop rate
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

        # Stop the vehicle
        self.control_adapter.send_control_command(self.car, 0.0, 0.0)

    def close(self) -> None:
        if self.qlabs is not None:
            self.qlabs.close()

    def run(self) -> bool:
        if not self.connect():
            return False
        try:
            if not self.spawn_vehicle():
                return False
            self.possess_camera()
            if not self.initialize_model():
                logger.error("Model failed to initialize")
                return False
            self.control_loop()
            return True
        finally:
            self.close()


def run_cli(
    hz: float = 5.0,
    duration: float = 30.0,
) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers_modules").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)

    cfg = DriverConfig(hz=hz, duration_sec=duration)
    driver = SimLingoQcar2Driver(cfg)
    ok = driver.run()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run_cli())

