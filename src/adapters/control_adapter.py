"""
Control Adapter for converting SimLingo outputs to Qcar2 control commands.

This module handles the conversion of SimLingo model outputs to Qcar2
vehicle control commands.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Qcar2ControlAdapter:
    """Adapter for converting SimLingo outputs to Qcar2 control commands."""
    
    def __init__(self, 
                 max_forward_speed: float = 5.0,
                 max_turn_angle: float = 0.6):
        """
        Initialize the control adapter.
        
        Args:
            max_forward_speed: Maximum forward speed in m/s
            max_turn_angle: Maximum turn angle in radians
        """
        self.max_forward_speed = max_forward_speed
        self.max_turn_angle = max_turn_angle
        
    def process_simlingo_output(self, model_output: Dict[str, Any]) -> tuple[float, float]:
        """
        Convert SimLingo predictions to (forward_speed, turn_angle).
        Supports:
        - pred_speed_wps: [N,2] (dx,dy) waypoints in vehicle frame OR [N,1] speed sequence
        - pred_route: [M,2] route waypoints in vehicle frame
        - actions: [throttle, steering]
        - direct keys: forward_speed, turn_angle
        """
        try:
            # Preferred: use predicted speed waypoints
            if 'pred_speed_wps' in model_output and isinstance(model_output['pred_speed_wps'], (list, np.ndarray)):
                wps = np.asarray(model_output['pred_speed_wps'], dtype=float)
                if wps.ndim == 2 and wps.shape[0] >= 1:
                    # Use a short-horizon waypoint to reduce jitter
                    idx = min(2, wps.shape[0]-1)
                    if wps.shape[1] == 2:
                        dx, dy = float(wps[idx, 0]), float(wps[idx, 1])
                        dist = float(np.hypot(dx, dy))
                        # Heading to waypoint
                        angle = float(np.arctan2(dy, dx))
                        # Simple proportional mapping
                        turn_angle = np.clip(angle, -self.max_turn_angle, self.max_turn_angle)
                        forward_speed = np.clip(1.5 * dist, -self.max_forward_speed, self.max_forward_speed)
                        return float(forward_speed), float(turn_angle)
                    elif wps.shape[1] == 1:
                        # Pure speed sequence, try to use route for steering
                        speed = float(wps[idx, 0])
                        if 'pred_route' in model_output:
                            route = np.asarray(model_output['pred_route'], dtype=float)
                            if route.ndim == 2 and route.shape[0] >= 1 and route.shape[1] == 2:
                                rdx, rdy = float(route[min(2, route.shape[0]-1), 0]), float(route[min(2, route.shape[0]-1), 1])
                                angle = float(np.arctan2(rdy, rdx))
                                turn_angle = np.clip(angle, -self.max_turn_angle, self.max_turn_angle)
                            else:
                                turn_angle = 0.0
                        else:
                            turn_angle = 0.0
                        forward_speed = np.clip(speed, -self.max_forward_speed, self.max_forward_speed)
                        return float(forward_speed), float(turn_angle)

            # Next: use route only
            if 'pred_route' in model_output and isinstance(model_output['pred_route'], (list, np.ndarray)):
                route = np.asarray(model_output['pred_route'], dtype=float)
                if route.ndim == 2 and route.shape[0] >= 1 and route.shape[1] == 2:
                    idx = min(2, route.shape[0]-1)
                    dx, dy = float(route[idx, 0]), float(route[idx, 1])
                    angle = float(np.arctan2(dy, dx))
                    turn_angle = np.clip(angle, -self.max_turn_angle, self.max_turn_angle)
                    dist = float(np.hypot(dx, dy))
                    forward_speed = np.clip(1.5 * dist, -self.max_forward_speed, self.max_forward_speed)
                    return float(forward_speed), float(turn_angle)

            # Actions interface
            if 'actions' in model_output:
                actions = model_output['actions']
                if isinstance(actions, (list, np.ndarray)) and len(actions) >= 2:
                    throttle = float(actions[0])
                    steering = float(actions[1])
                    forward_speed = self._convert_throttle_to_speed(throttle)
                    turn_angle = self._convert_steering_to_angle(steering)
                    return forward_speed, turn_angle

            # Direct keys
            forward_speed = float(model_output.get('forward_speed', 0.0))
            turn_angle = float(model_output.get('turn_angle', 0.0))
            forward_speed = np.clip(forward_speed, -self.max_forward_speed, self.max_forward_speed)
            turn_angle = np.clip(turn_angle, -self.max_turn_angle, self.max_turn_angle)
            return float(forward_speed), float(turn_angle)

        except Exception as e:
            logger.error(f"Error processing SimLingo output: {e}")
            return 0.0, 0.0  # Safe fallback

    def _convert_throttle_to_speed(self, throttle: float) -> float:
        """Convert normalized throttle [-1, 1] to speed in m/s."""
        # Clamp throttle to valid range
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # Convert to speed (simple linear mapping)
        speed = throttle * self.max_forward_speed
        
        return speed
    
    def _convert_steering_to_angle(self, steering: float) -> float:
        """Convert normalized steering [-1, 1] to angle in radians."""
        # Clamp steering to valid range
        steering = np.clip(steering, -1.0, 1.0)
        
        # Convert to angle (simple linear mapping)
        angle = steering * self.max_turn_angle
        
        return angle
    
    def send_control_command(self,
                           qcar2_vehicle,
                           forward_speed: float,
                           turn_angle: float,
                           headlights: bool = False,
                           left_turn_signal: bool = False,
                           right_turn_signal: bool = False,
                           brake_signal: bool = False,
                           reverse_signal: bool = False) -> tuple[bool, dict]:
        """
        Send control command to Qcar2 vehicle.

        Args:
            qcar2_vehicle: QLabsQCar2 instance
            forward_speed: Forward speed in m/s
            turn_angle: Turn angle in radians
            headlights: Headlights state
            left_turn_signal: Left turn signal state
            right_turn_signal: Right turn signal state
            brake_signal: Brake signal state
            reverse_signal: Reverse signal state

        Returns:
            (success, info) where info includes location, rotation, front_hit, rear_hit
        """
        try:
            # Determine brake signal based on speed
            if forward_speed < -0.1:  # Reverse
                reverse_signal = True
                brake_signal = False
            elif abs(forward_speed) < 0.1:  # Stopped
                brake_signal = True
                reverse_signal = False
            else:  # Forward
                brake_signal = False
                reverse_signal = False

            # Determine turn signals based on steering angle
            if turn_angle > 0.1:  # Right turn
                right_turn_signal = True
                left_turn_signal = False
            elif turn_angle < -0.1:  # Left turn
                left_turn_signal = True
                right_turn_signal = False
            else:  # Straight
                left_turn_signal = False
                right_turn_signal = False

            # Send command to vehicle
            success, location, rotation, front_hit, rear_hit = qcar2_vehicle.set_velocity_and_request_state(
                forward=forward_speed,
                turn=turn_angle,
                headlights=headlights,
                leftTurnSignal=left_turn_signal,
                rightTurnSignal=right_turn_signal,
                brakeSignal=brake_signal,
                reverseSignal=reverse_signal
            )

            info = {
                "location": location,
                "rotation": rotation,
                "front_hit": front_hit,
                "rear_hit": rear_hit,
            }

            if success:
                return True, info
            else:
                logger.error("Failed to send control command to Qcar2")
                return False, info

        except Exception as e:
            logger.error(f"Error sending control command: {e}")
            return False, {"location": [0,0,0], "rotation": [0,0,0], "front_hit": False, "rear_hit": False}
