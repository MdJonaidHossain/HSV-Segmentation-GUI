"""
Simple command-line demonstration of HSV segmentation.
This script demonstrates the core functionality without requiring a GUI.
"""
import cv2
import numpy as np
from pathlib import Path
from utils import load_image, threshold, calculate_metrics, apply_mask_to_image, save_image


def demo_segmentation():
    """Demonstrate HSV segmentation on sample images."""
    
    print("=" * 60)
    print("HSV Segmentation Demonstration")
    print("=" * 60)
    
    sample_dir = Path("sample_images")
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Demo 1: Red color segmentation
    print("\n1. Segmenting red colors from color blocks...")
    img = load_image(str(sample_dir / "color_blocks.png"))
    
    if img is not None:
        # Red range in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        
        mask = threshold(img, lower_red, upper_red)
        result = apply_mask_to_image(img, mask)
        metrics = calculate_metrics(mask, img)
        
        print(f"   - Red pixels detected: {metrics['pixel_count']:,}")
        print(f"   - Coverage: {metrics['percentage']:.2f}%")
        
        save_image(str(output_dir / "red_mask.png"), mask)
        save_image(str(output_dir / "red_result.png"), result)
        print(f"   ✓ Saved to {output_dir}/red_*.png")
    
    # Demo 2: Green color segmentation
    print("\n2. Segmenting green colors from color blocks...")
    if img is not None:
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        mask = threshold(img, lower_green, upper_green)
        result = apply_mask_to_image(img, mask)
        metrics = calculate_metrics(mask, img)
        
        print(f"   - Green pixels detected: {metrics['pixel_count']:,}")
        print(f"   - Coverage: {metrics['percentage']:.2f}%")
        
        save_image(str(output_dir / "green_mask.png"), mask)
        save_image(str(output_dir / "green_result.png"), result)
        print(f"   ✓ Saved to {output_dir}/green_*.png")
    
    # Demo 3: Blue color segmentation
    print("\n3. Segmenting blue colors from color blocks...")
    if img is not None:
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask = threshold(img, lower_blue, upper_blue)
        result = apply_mask_to_image(img, mask)
        metrics = calculate_metrics(mask, img)
        
        print(f"   - Blue pixels detected: {metrics['pixel_count']:,}")
        print(f"   - Coverage: {metrics['percentage']:.2f}%")
        
        save_image(str(output_dir / "blue_mask.png"), mask)
        save_image(str(output_dir / "blue_result.png"), result)
        print(f"   ✓ Saved to {output_dir}/blue_*.png")
    
    # Demo 4: Process colored circles
    print("\n4. Segmenting from colored circles image...")
    circles_img = load_image(str(sample_dir / "colored_circles.png"))
    
    if circles_img is not None:
        # Segment all non-white colors
        lower = np.array([0, 50, 50])
        upper = np.array([179, 255, 255])
        
        mask = threshold(circles_img, lower, upper)
        result = apply_mask_to_image(circles_img, mask)
        metrics = calculate_metrics(mask, circles_img)
        
        print(f"   - Colored pixels detected: {metrics['pixel_count']:,}")
        print(f"   - Coverage: {metrics['percentage']:.2f}%")
        
        save_image(str(output_dir / "circles_mask.png"), mask)
        save_image(str(output_dir / "circles_result.png"), result)
        print(f"   ✓ Saved to {output_dir}/circles_*.png")
    
    print("\n" + "=" * 60)
    print(f"✅ Demo complete! Check the '{output_dir}' folder for results.")
    print("=" * 60)


if __name__ == "__main__":
    demo_segmentation()
