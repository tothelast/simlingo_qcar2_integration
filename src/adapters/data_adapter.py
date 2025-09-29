"""
Data Adapter for converting Qcar2 sensor data to SimLingo input format.
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class Qcar2DataAdapter:
    
    def __init__(self, target_size: Tuple[int, int] = (820, 410)):
        """
        Initialize the data adapter.

        Args:
            target_size: Target image size (width, height) for SimLingo input. Default is CSI front cam 820x410.
        """
        self.target_size = target_size
        self.target_width, self.target_height = target_size
        
    def process_camera_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process Qcar2 camera image to SimLingo format.
        
        Args:
            image: Raw camera image from Qcar2 (BGR format from OpenCV)
            
        Returns:
            Processed image in SimLingo format (RGB, resized), dtype=uint8
        """
        try:
            # Convert BGR to RGB (OpenCV uses BGR by default)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            # Resize to target dimensions
            w,h = image_rgb.shape[1], image_rgb.shape[0]
            if (w, h) != self.target_size:
                resized_image = cv2.resize(image_rgb, self.target_size, cv2.INTER_LINEAR)
            else:
                resized_image = image_rgb
            
            return resized_image
            
        except Exception as e:
            logger.error(f"Error processing camera image: {e}")
            raise
    
    def get_qcar2_camera_data(self, qcar2_vehicle, camera_type: int = 3) -> np.ndarray:
        """
        Get camera data from Qcar2 vehicle.
        
        Args:
            qcar2_vehicle: QLabsQCar2 instance
            camera_type: Camera type (3 = CAMERA_CSI_FRONT by default)
            
        Returns:
            Camera image as numpy array
        """
        _, image_data = qcar2_vehicle.get_image(camera_type)
        return image_data
        
    
    def prepare_model_input(self, qcar2_vehicle, camera_type: int = 3) -> np.ndarray:
        """
        Complete pipeline: get Qcar2 camera data and prepare for SimLingo.
        
        Args:
            qcar2_vehicle: QLabsQCar2 instance
            camera_type: Camera type to use
            
        Returns:
            Processed image ready for SimLingo model input
        """
        # Get raw camera data
        raw_image = self.get_qcar2_camera_data(qcar2_vehicle, camera_type)
            
        # Process for SimLingo
        processed_image = self.process_camera_image(raw_image)
        
        # Add batch dimension for model input
        model_input = np.expand_dims(processed_image, axis=0)
        
        return model_input
