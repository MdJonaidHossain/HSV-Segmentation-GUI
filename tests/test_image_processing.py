"""
Unit tests for image processing utilities.
"""
import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_processing import (
    gray_stats, threshold, calculate_metrics,
    apply_mask_to_image, resize_image, load_image, save_image
)


class TestImageProcessing(unittest.TestCase):
    """Test cases for image processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = [0, 0, 255]  # Red square
        
        # Create grayscale image
        self.gray_image = np.ones((100, 100), dtype=np.uint8) * 128
    
    def test_gray_stats_normal(self):
        """Test gray_stats with normal image."""
        stats = gray_stats(self.gray_image)
        
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        
        self.assertAlmostEqual(stats['mean'], 128.0, delta=1.0)
        self.assertEqual(stats['min'], 128)
        self.assertEqual(stats['max'], 128)
    
    def test_gray_stats_empty(self):
        """Test gray_stats with empty image."""
        empty = np.array([])
        stats = gray_stats(empty)
        
        self.assertEqual(stats['mean'], 0.0)
        self.assertEqual(stats['std'], 0.0)
    
    def test_gray_stats_none(self):
        """Test gray_stats with None."""
        stats = gray_stats(None)
        
        self.assertEqual(stats['mean'], 0.0)
        self.assertEqual(stats['std'], 0.0)
    
    def test_threshold_normal(self):
        """Test threshold with normal parameters."""
        lower = np.array([0, 0, 200])
        upper = np.array([10, 255, 255])
        
        mask = threshold(self.test_image, lower, upper)
        
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (100, 100))
        self.assertTrue(np.any(mask > 0))  # Should have some white pixels
    
    def test_threshold_empty(self):
        """Test threshold with empty image."""
        empty = np.array([])
        lower = np.array([0, 0, 0])
        upper = np.array([179, 255, 255])
        
        mask = threshold(empty, lower, upper)
        
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (100, 100))  # Should return default size
    
    def test_threshold_none(self):
        """Test threshold with None image."""
        lower = np.array([0, 0, 0])
        upper = np.array([179, 255, 255])
        
        mask = threshold(None, lower, upper)
        
        self.assertIsNotNone(mask)
    
    def test_calculate_metrics_normal(self):
        """Test calculate_metrics with normal mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # 50x50 white square
        
        metrics = calculate_metrics(mask, self.test_image)
        
        self.assertIn('pixel_count', metrics)
        self.assertIn('percentage', metrics)
        self.assertIn('total_pixels', metrics)
        
        self.assertEqual(metrics['pixel_count'], 2500)  # 50*50
        self.assertEqual(metrics['total_pixels'], 10000)  # 100*100
        self.assertAlmostEqual(metrics['percentage'], 25.0, delta=0.1)
    
    def test_calculate_metrics_empty(self):
        """Test calculate_metrics with empty mask."""
        empty = np.array([])
        
        metrics = calculate_metrics(empty, self.test_image)
        
        self.assertEqual(metrics['pixel_count'], 0)
        self.assertEqual(metrics['percentage'], 0.0)
    
    def test_calculate_metrics_all_white(self):
        """Test calculate_metrics with all white mask."""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        metrics = calculate_metrics(mask, self.test_image)
        
        self.assertEqual(metrics['pixel_count'], 10000)
        self.assertAlmostEqual(metrics['percentage'], 100.0, delta=0.1)
    
    def test_calculate_metrics_all_black(self):
        """Test calculate_metrics with all black mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        metrics = calculate_metrics(mask, self.test_image)
        
        self.assertEqual(metrics['pixel_count'], 0)
        self.assertEqual(metrics['percentage'], 0.0)
    
    def test_apply_mask_to_image(self):
        """Test apply_mask_to_image."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        result = apply_mask_to_image(self.test_image, mask)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Check that masked region has color
        self.assertTrue(np.any(result[40, 40] > 0))
        # Check that unmasked region is black
        self.assertTrue(np.all(result[10, 10] == 0))
    
    def test_apply_mask_none_image(self):
        """Test apply_mask_to_image with None image."""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        result = apply_mask_to_image(None, mask)
        
        self.assertIsNotNone(result)
    
    def test_apply_mask_none_mask(self):
        """Test apply_mask_to_image with None mask."""
        result = apply_mask_to_image(self.test_image, None)
        
        self.assertIsNotNone(result)
    
    def test_resize_image_no_resize_needed(self):
        """Test resize_image when image is already small."""
        small_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = resize_image(small_image, max_dimension=1920)
        
        self.assertEqual(result.shape, small_image.shape)
    
    def test_resize_image_resize_needed(self):
        """Test resize_image when image is too large."""
        large_image = np.zeros((2000, 3000, 3), dtype=np.uint8)
        
        result = resize_image(large_image, max_dimension=1920)
        
        self.assertLessEqual(max(result.shape[:2]), 1920)
        # Check aspect ratio maintained
        aspect_ratio_orig = large_image.shape[1] / large_image.shape[0]
        aspect_ratio_new = result.shape[1] / result.shape[0]
        self.assertAlmostEqual(aspect_ratio_orig, aspect_ratio_new, delta=0.01)
    
    def test_resize_image_empty(self):
        """Test resize_image with empty image."""
        empty = np.array([])
        
        result = resize_image(empty, max_dimension=1920)
        
        # Should return input unchanged
        self.assertEqual(result.size, 0)
    
    def test_resize_image_none(self):
        """Test resize_image with None."""
        result = resize_image(None, max_dimension=1920)
        
        self.assertIsNone(result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_threshold_with_inverted_range(self):
        """Test threshold when min > max."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        lower = np.array([100, 100, 100])
        upper = np.array([50, 50, 50])  # Inverted
        
        mask = threshold(image, lower, upper)
        
        self.assertIsNotNone(mask)
        # Should return all black mask
        self.assertFalse(np.any(mask > 0))
    
    def test_gray_stats_with_varied_values(self):
        """Test gray_stats with varied pixel values."""
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        stats = gray_stats(image)
        
        self.assertGreater(stats['std'], 0)  # Should have variation
        self.assertGreaterEqual(stats['min'], 0)
        self.assertLessEqual(stats['max'], 255)


if __name__ == '__main__':
    unittest.main()
