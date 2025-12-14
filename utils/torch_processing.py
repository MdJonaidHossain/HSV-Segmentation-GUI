"""
PyTorch-accelerated image processing utilities for HSV segmentation.
"""
import torch
import numpy as np
import cv2
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class TorchImageProcessor:
    """Image processor using PyTorch for GPU acceleration."""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize PyTorch image processor.
        
        Args:
            use_gpu: Whether to use GPU if available
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"TorchImageProcessor initialized with device: {self.device}")
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to PyTorch tensor.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            PyTorch tensor on configured device
        """
        try:
            if image is None or image.size == 0:
                return torch.zeros((1, 1, 3), device=self.device, dtype=torch.uint8)
            
            tensor = torch.from_numpy(image).to(self.device)
            return tensor
        except Exception as e:
            logger.error(f"Error converting to tensor: {e}")
            return torch.zeros((1, 1, 3), device=self.device, dtype=torch.uint8)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            Numpy array
        """
        try:
            return tensor.cpu().numpy()
        except Exception as e:
            logger.error(f"Error converting to numpy: {e}")
            return np.zeros((1, 1, 3), dtype=np.uint8)
    
    def bgr_to_hsv_torch(self, bgr_image: np.ndarray) -> torch.Tensor:
        """
        Convert BGR image to HSV using PyTorch operations.
        
        Args:
            bgr_image: BGR image as numpy array
            
        Returns:
            HSV tensor
        """
        try:
            # Use OpenCV for color conversion (optimized)
            hsv_np = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            return self.to_tensor(hsv_np)
        except Exception as e:
            logger.error(f"Error in BGR to HSV conversion: {e}")
            return torch.zeros_like(self.to_tensor(bgr_image))
    
    def threshold_torch(self, image: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """
        Apply HSV threshold using PyTorch for potential GPU acceleration.
        
        Args:
            image: Input BGR image
            lower: Lower HSV threshold [H, S, V]
            upper: Upper HSV threshold [H, S, V]
            
        Returns:
            Binary mask as numpy array
        """
        try:
            if image is None or image.size == 0:
                return np.zeros((100, 100), dtype=np.uint8)
            
            # Convert to HSV (use OpenCV for accurate color conversion)
            hsv_np = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_tensor = self.to_tensor(hsv_np)
            
            # Convert thresholds to tensors
            lower_t = torch.tensor(lower, device=self.device, dtype=torch.uint8)
            upper_t = torch.tensor(upper, device=self.device, dtype=torch.uint8)
            
            # Vectorized thresholding - all channels at once
            in_range = torch.ones(hsv_tensor.shape[:2], device=self.device, dtype=torch.bool)
            
            for i in range(3):
                in_range = in_range & (hsv_tensor[:, :, i] >= lower_t[i]) & (hsv_tensor[:, :, i] <= upper_t[i])
            
            mask = (in_range.to(torch.uint8) * 255)
            
            return self.to_numpy(mask)
        except Exception as e:
            logger.error(f"Error in torch threshold: {e}")
            # Fallback to OpenCV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            return cv2.inRange(hsv, lower, upper)
    
    def calculate_metrics_torch(self, mask: np.ndarray) -> dict:
        """
        Calculate mask metrics using PyTorch.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary with metrics
        """
        try:
            if mask is None or mask.size == 0:
                return {
                    'pixel_count': 0,
                    'percentage': 0.0,
                    'total_pixels': 0
                }
            
            mask_tensor = self.to_tensor(mask)
            
            total_pixels = mask_tensor.numel()
            white_pixels = torch.sum(mask_tensor > 0).item()
            percentage = (white_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
            
            return {
                'pixel_count': int(white_pixels),
                'percentage': float(percentage),
                'total_pixels': int(total_pixels)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics with torch: {e}")
            return {
                'pixel_count': 0,
                'percentage': 0.0,
                'total_pixels': 0
            }
    
    def batch_process_masks(self, images: list, lower: np.ndarray, upper: np.ndarray) -> list:
        """
        Process multiple images in batch for improved GPU utilization.
        
        Args:
            images: List of BGR images
            lower: Lower HSV threshold
            upper: Upper HSV threshold
            
        Returns:
            List of binary masks
        """
        try:
            if not images:
                return []
            
            # For actual batch processing, images need same dimensions
            # Here we process individually but could be extended for true batching
            masks = []
            
            # Convert all images to HSV tensors
            hsv_tensors = []
            for img in images:
                hsv_np = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv_tensors.append(self.to_tensor(hsv_np))
            
            # Process all on GPU
            lower_t = torch.tensor(lower, device=self.device, dtype=torch.uint8)
            upper_t = torch.tensor(upper, device=self.device, dtype=torch.uint8)
            
            for hsv_tensor in hsv_tensors:
                in_range = torch.ones(hsv_tensor.shape[:2], device=self.device, dtype=torch.bool)
                for i in range(3):
                    in_range = in_range & (hsv_tensor[:, :, i] >= lower_t[i]) & (hsv_tensor[:, :, i] <= upper_t[i])
                
                mask = (in_range.to(torch.uint8) * 255)
                masks.append(self.to_numpy(mask))
            
            return masks
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Fallback to sequential OpenCV processing
            return [cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower, upper) for img in images]


def get_processor(use_gpu: bool = None) -> TorchImageProcessor:
    """
    Get image processor instance.
    
    Args:
        use_gpu: Whether to use GPU (None = auto-detect)
        
    Returns:
        TorchImageProcessor instance
    """
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    return TorchImageProcessor(use_gpu=use_gpu)
