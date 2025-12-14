"""
Utility functions for HSV segmentation and image processing.
"""
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


def gray_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for a grayscale image.
    
    Args:
        image: Grayscale image as numpy array
        
    Returns:
        Dictionary containing mean, std, min, max values
    """
    try:
        if image is None or image.size == 0:
            logger.warning("Empty image provided to gray_stats")
            return {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0}
        
        return {
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': int(np.min(image)),
            'max': int(np.max(image))
        }
    except Exception as e:
        logger.error(f"Error calculating gray stats: {e}")
        return {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0}


def threshold(image: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Apply HSV threshold to create binary mask.
    
    Args:
        image: Input image in BGR format
        lower: Lower HSV threshold array [H, S, V]
        upper: Upper HSV threshold array [H, S, V]
        
    Returns:
        Binary mask as numpy array
    """
    try:
        if image is None or image.size == 0:
            logger.warning("Empty image provided to threshold")
            return np.zeros((100, 100), dtype=np.uint8)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    except Exception as e:
        logger.error(f"Error in threshold: {e}")
        return np.zeros(image.shape[:2], dtype=np.uint8) if image is not None else np.zeros((100, 100), dtype=np.uint8)


def calculate_metrics(mask: np.ndarray, image: np.ndarray) -> Dict[str, any]:
    """
    Calculate segmentation metrics from mask.
    
    Args:
        mask: Binary mask
        image: Original image
        
    Returns:
        Dictionary with metrics: pixel_count, percentage, stats
    """
    try:
        if mask is None or mask.size == 0:
            logger.warning("Empty mask provided to calculate_metrics")
            return {
                'pixel_count': 0,
                'percentage': 0.0,
                'stats': {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0}
            }
        
        total_pixels = mask.size
        white_pixels = np.sum(mask > 0)
        percentage = (white_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
        
        # Calculate statistics for masked region
        stats = gray_stats(mask)
        
        return {
            'pixel_count': int(white_pixels),
            'percentage': float(percentage),
            'total_pixels': int(total_pixels),
            'stats': stats
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            'pixel_count': 0,
            'percentage': 0.0,
            'stats': {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0}
        }


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image to extract segmented region.
    
    Args:
        image: Original image
        mask: Binary mask
        
    Returns:
        Masked image
    """
    try:
        if image is None or mask is None:
            logger.warning("Null image or mask provided to apply_mask_to_image")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        return cv2.bitwise_and(image, image, mask=mask)
    except Exception as e:
        logger.error(f"Error applying mask: {e}")
        return np.zeros_like(image) if image is not None else np.zeros((100, 100, 3), dtype=np.uint8)


def resize_image(image: np.ndarray, max_dimension: int = 1920) -> np.ndarray:
    """
    Resize image if it exceeds maximum dimension while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_dimension: Maximum width or height
        
    Returns:
        Resized image
    """
    try:
        if image is None or image.size == 0:
            logger.warning("Empty image provided to resize_image")
            return image
        
        h, w = image.shape[:2]
        
        if max(h, w) <= max_dimension:
            return image
        
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image


def load_image(filepath: str, resize_large: bool = True, max_dimension: int = 1920) -> Optional[np.ndarray]:
    """
    Load image from file with error handling.
    
    Args:
        filepath: Path to image file
        resize_large: Whether to resize large images
        max_dimension: Maximum dimension for resizing
        
    Returns:
        Loaded image or None if failed
    """
    try:
        image = cv2.imread(filepath)
        
        if image is None:
            logger.error(f"Failed to load image: {filepath}")
            return None
        
        logger.info(f"Successfully loaded image: {filepath} with shape {image.shape}")
        
        if resize_large:
            image = resize_image(image, max_dimension)
        
        return image
    except Exception as e:
        logger.error(f"Error loading image {filepath}: {e}")
        return None


def save_image(filepath: str, image: np.ndarray) -> bool:
    """
    Save image to file with error handling.
    
    Args:
        filepath: Output path
        image: Image to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if image is None or image.size == 0:
            logger.error("Cannot save empty image")
            return False
        
        success = cv2.imwrite(filepath, image)
        
        if success:
            logger.info(f"Successfully saved image: {filepath}")
        else:
            logger.error(f"Failed to save image: {filepath}")
        
        return success
    except Exception as e:
        logger.error(f"Error saving image {filepath}: {e}")
        return False
