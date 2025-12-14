"""
Generate sample images for demonstration.
"""
import cv2
import numpy as np
from pathlib import Path


def create_sample_images():
    """Create sample images in various colors for HSV segmentation demonstration."""
    
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a colorful test image with various HSV regions
    height, width = 480, 640
    
    # Sample 1: Multi-colored blocks
    img1 = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Red region (top-left)
    img1[0:height//2, 0:width//3] = [0, 0, 255]
    
    # Green region (top-center)
    img1[0:height//2, width//3:2*width//3] = [0, 255, 0]
    
    # Blue region (top-right)
    img1[0:height//2, 2*width//3:width] = [255, 0, 0]
    
    # Yellow region (bottom-left)
    img1[height//2:height, 0:width//3] = [0, 255, 255]
    
    # Magenta region (bottom-center)
    img1[height//2:height, width//3:2*width//3] = [255, 0, 255]
    
    # Cyan region (bottom-right)
    img1[height//2:height, 2*width//3:width] = [255, 255, 0]
    
    cv2.imwrite(str(sample_dir / "color_blocks.png"), img1)
    
    # Sample 2: Gradient image
    img2 = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        hue = int(179 * i / height)
        img2[i, :] = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    
    cv2.imwrite(str(sample_dir / "hsv_gradient.png"), img2)
    
    # Sample 3: Circles with different colors
    img3 = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    colors = [
        (0, 0, 255),      # Red
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (0, 165, 255),    # Orange
    ]
    
    positions = [
        (160, 120), (480, 120),
        (160, 360), (480, 360),
        (320, 120), (320, 360)
    ]
    
    for pos, color in zip(positions, colors):
        cv2.circle(img3, pos, 80, color, -1)
    
    cv2.imwrite(str(sample_dir / "colored_circles.png"), img3)
    
    print(f"Created 3 sample images in {sample_dir}/")


if __name__ == "__main__":
    create_sample_images()
