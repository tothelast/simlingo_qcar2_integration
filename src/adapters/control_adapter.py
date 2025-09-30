"""
Control Adapter for converting SimLingo outputs to Qcar2 control commands.

Uses the original SimLingo finite-difference speed calculation and braking logic.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Qcar2ControlAdapter:
    """Adapter for converting SimLingo outputs to Qcar2 control commands."""

    def __init__(self,
                 max_forward_speed: float = 5.0,
                 max_turn_angle: float = 0.6,
                 brake_speed: float = 0.4,
                 brake_ratio: float = 1.1):
        """
        Initialize the control adapter.

        Args:
            max_forward_speed: Maximum forward speed in m/s
            max_turn_angle: Maximum turn angle in radians
            brake_speed: Speed threshold below which brake is triggered (m/s)
            brake_ratio: Ratio of current/desired speed above which brake is triggered
        """
        self.max_forward_speed = max_forward_speed
        self.max_turn_angle = max_turn_angle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio

    def process_simlingo_output(self, model_output: Dict[str, Any], current_speed: float = 0.0) -> tuple[float, float]:
        """
        Convert SimLingo predictions to (forward_speed, turn_angle) using original SimLingo logic.

        Args:
            model_output: Dict containing 'pred_speed_wps' [N,2] waypoints in vehicle frame
            current_speed: Current vehicle speed in m/s

        Returns:
            (forward_speed, turn_angle) tuple
        """
        try:
            # SimLingo outputs pred_speed_wps: [N,2] waypoints at 5 Hz (0.2s intervals)
            if 'pred_speed_wps' not in model_output:
                logger.warning("No pred_speed_wps in model output")
                return 0.0, 0.0

            wps = np.asarray(model_output['pred_speed_wps'], dtype=float)

            if wps.ndim != 2 or wps.shape[1] != 2 or wps.shape[0] < 3:
                logger.warning(f"Invalid waypoint shape: {wps.shape}, expected [N>=3, 2]")
                return 0.0, 0.0

            # Original SimLingo speed calculation (finite-difference)
            # Waypoints are at 5 Hz: wp[0]=t0, wp[1]=t0.2s, wp[2]=t0.4s
            # desired_speed = ||wp[0] - wp[2]|| * 2.0
            desired_speed = float(np.linalg.norm(wps[2] - wps[0]) * 2.0)

            # Original SimLingo braking logic
            should_brake = (desired_speed < self.brake_speed) or \
                          (current_speed > 0.01 and (current_speed / max(desired_speed, 0.01)) > self.brake_ratio)

            if should_brake:
                forward_speed = 0.0
            else:
                forward_speed = np.clip(desired_speed, 0.0, self.max_forward_speed)

            # Steering: angle to waypoint at index 2 (0.4s ahead)
            dx, dy = float(wps[2, 0]), float(wps[2, 1])
            angle = float(np.arctan2(dy, dx))
            turn_angle = np.clip(angle, -self.max_turn_angle, self.max_turn_angle)

            return float(forward_speed), float(turn_angle)

        except Exception as e:
            logger.error(f"Error processing SimLingo output: {e}")
            return 0.0, 0.0

    def send_control_command(self,
                           qcar2_vehicle,
                           forward_speed: float,
                           turn_angle: float) -> tuple[bool, dict]:
        """
        Send control command to Qcar2 vehicle.

        Args:
            qcar2_vehicle: QLabsQCar2 instance
            forward_speed: Forward speed in m/s
            turn_angle: Turn angle in radians

        Returns:
            (success, info) where info includes location, rotation, front_hit, rear_hit
        """
        try:
            # Send command to vehicle
            success, location, rotation, front_hit, rear_hit = qcar2_vehicle.set_velocity_and_request_state(
                forward=forward_speed,
                turn=turn_angle,
                headlights=False,
                leftTurnSignal=False,
                rightTurnSignal=False,
                brakeSignal=False,
                reverseSignal=False
            )

            info = {
                "location": location,
                "rotation": rotation,
                "front_hit": front_hit,
                "rear_hit": rear_hit,
            }

            return success, info

        except Exception as e:
            logger.error(f"Error sending control command: {e}")
            return False, {"location": [0,0,0], "rotation": [0,0,0], "front_hit": False, "rear_hit": False}
