"""
Main GUI application for HSV Segmentation Tool.
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging
from pathlib import Path

from utils import (
    load_image, save_image, threshold, calculate_metrics,
    apply_mask_to_image, Config, PresetManager
)
from utils.torch_processing import get_processor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hsv_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HSVVisualizerFrame(ttk.Frame):
    """Frame for displaying HSV color space visualization."""
    
    def __init__(self, parent, width=200, height=200):
        """
        Initialize HSV visualizer.
        
        Args:
            parent: Parent widget
            width: Canvas width
            height: Canvas height
        """
        super().__init__(parent)
        self.width = width
        self.height = height
        
        self.canvas = tk.Canvas(self, width=width, height=height, bg='black')
        self.canvas.pack()
        
        self.update_visualization(0, 179, 0, 255, 0, 255)
    
    def update_visualization(self, h_min, h_max, s_min, s_max, v_min, v_max):
        """
        Update HSV visualization.
        
        Args:
            h_min, h_max: Hue range
            s_min, s_max: Saturation range
            v_min, v_max: Value range
        """
        try:
            # Create HSV color visualization
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            for y in range(self.height):
                for x in range(self.width):
                    h = int((x / self.width) * 179)
                    s = 255 - int((y / self.height) * 255)
                    v = 200
                    
                    # Check if in range
                    if h_min <= h <= h_max and s_min <= s <= s_max:
                        img[y, x] = [h, s, v]
                    else:
                        img[y, x] = [0, 0, 50]  # Dark gray for out of range
            
            bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(rgb)
            self.photo = ImageTk.PhotoImage(pil_img)
            
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        except Exception as e:
            logger.error(f"Error updating HSV visualization: {e}")


class HSVSegmentationGUI:
    """Main GUI application for HSV segmentation."""
    
    def __init__(self):
        """Initialize the GUI application."""
        # Load configuration
        self.config = Config()
        self.preset_manager = PresetManager()
        
        # Initialize PyTorch processor
        use_gpu = self.config.get('performance', 'use_gpu', default=False)
        self.torch_processor = get_processor(use_gpu=use_gpu)
        
        # Initialize main window
        theme = self.config.get('window', 'theme', default='darkly')
        self.root = ttk.Window(themename=theme)
        self.root.title(self.config.get('window', 'title', default='HSV Segmentation Tool'))
        self.root.geometry(self.config.get('window', 'geometry', default='1400x900'))
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.mask = None
        self.result_image = None
        
        # HSV slider variables
        self.h_min = tk.IntVar(value=self.config.get('hsv_defaults', 'h_min', default=0))
        self.h_max = tk.IntVar(value=self.config.get('hsv_defaults', 'h_max', default=179))
        self.s_min = tk.IntVar(value=self.config.get('hsv_defaults', 's_min', default=0))
        self.s_max = tk.IntVar(value=self.config.get('hsv_defaults', 's_max', default=255))
        self.v_min = tk.IntVar(value=self.config.get('hsv_defaults', 'v_min', default=0))
        self.v_max = tk.IntVar(value=self.config.get('hsv_defaults', 'v_max', default=255))
        
        # Create GUI
        self.create_widgets()
        
        logger.info("HSV Segmentation GUI initialized")
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_container = ttk.Frame(self.root, padding=10)
        main_container.pack(fill=BOTH, expand=YES)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_container, padding=10)
        left_panel.pack(side=LEFT, fill=Y)
        
        # Right panel - Images
        right_panel = ttk.Frame(main_container, padding=10)
        right_panel.pack(side=RIGHT, fill=BOTH, expand=YES)
        
        self.create_control_panel(left_panel)
        self.create_image_panel(right_panel)
    
    def create_control_panel(self, parent):
        """Create control panel with sliders and buttons."""
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding=10)
        file_frame.pack(fill=X, pady=5)
        
        load_btn = ttk.Button(file_frame, text="Load Image", command=self.load_image, bootstyle=PRIMARY)
        load_btn.pack(fill=X, pady=2)
        ToolTip(load_btn, text="Load an image for HSV segmentation")
        
        sample_btn = ttk.Button(file_frame, text="Load Sample", command=self.load_sample, bootstyle=INFO)
        sample_btn.pack(fill=X, pady=2)
        ToolTip(sample_btn, text="Load a sample demonstration image")
        
        save_btn = ttk.Button(file_frame, text="Save Mask", command=self.save_mask, bootstyle=SUCCESS)
        save_btn.pack(fill=X, pady=2)
        ToolTip(save_btn, text="Save the current segmentation mask")
        
        save_result_btn = ttk.Button(file_frame, text="Save Result", command=self.save_result, bootstyle=SUCCESS)
        save_result_btn.pack(fill=X, pady=2)
        ToolTip(save_result_btn, text="Save the segmented result image")
        
        # Presets
        preset_frame = ttk.LabelFrame(parent, text="Presets", padding=10)
        preset_frame.pack(fill=X, pady=5)
        
        self.preset_combo = ttk.Combobox(preset_frame, values=self.preset_manager.get_preset_names(), state='readonly')
        self.preset_combo.pack(fill=X, pady=2)
        self.preset_combo.bind('<<ComboboxSelected>>', self.load_preset)
        ToolTip(self.preset_combo, text="Select a color preset")
        
        save_preset_btn = ttk.Button(preset_frame, text="Save Current as Preset", command=self.save_preset, bootstyle=WARNING)
        save_preset_btn.pack(fill=X, pady=2)
        
        # HSV Sliders
        hsv_frame = ttk.LabelFrame(parent, text="HSV Adjustments", padding=10)
        hsv_frame.pack(fill=BOTH, expand=YES, pady=5)
        
        self.create_slider(hsv_frame, "H Min", self.h_min, 0, 179, "Minimum Hue value (0-179)")
        self.create_slider(hsv_frame, "H Max", self.h_max, 0, 179, "Maximum Hue value (0-179)")
        self.create_slider(hsv_frame, "S Min", self.s_min, 0, 255, "Minimum Saturation value (0-255)")
        self.create_slider(hsv_frame, "S Max", self.s_max, 0, 255, "Maximum Saturation value (0-255)")
        self.create_slider(hsv_frame, "V Min", self.v_min, 0, 255, "Minimum Value/Brightness (0-255)")
        self.create_slider(hsv_frame, "V Max", self.v_max, 0, 255, "Maximum Value/Brightness (0-255)")
        
        # Reset button
        reset_btn = ttk.Button(hsv_frame, text="Reset to Default", command=self.reset_sliders, bootstyle=DANGER)
        reset_btn.pack(fill=X, pady=5)
        
        # HSV Visualizer
        viz_frame = ttk.LabelFrame(parent, text="HSV Visualization", padding=10)
        viz_frame.pack(fill=X, pady=5)
        
        self.hsv_visualizer = HSVVisualizerFrame(viz_frame, width=200, height=200)
        self.hsv_visualizer.pack()
        
        # Metrics
        self.metrics_frame = ttk.LabelFrame(parent, text="Metrics", padding=10)
        self.metrics_frame.pack(fill=X, pady=5)
        
        self.metrics_label = ttk.Label(self.metrics_frame, text="Load an image to see metrics", font=('Arial', 9))
        self.metrics_label.pack()
    
    def create_slider(self, parent, label, variable, from_, to, tooltip):
        """Create a slider with label and tooltip."""
        frame = ttk.Frame(parent)
        frame.pack(fill=X, pady=5)
        
        lbl = ttk.Label(frame, text=f"{label}:", width=8)
        lbl.pack(side=LEFT)
        ToolTip(lbl, text=tooltip)
        
        slider = ttk.Scale(frame, from_=from_, to=to, variable=variable, orient=HORIZONTAL, 
                          command=lambda x: self.on_slider_change(), bootstyle=INFO)
        slider.pack(side=LEFT, fill=X, expand=YES, padx=5)
        
        value_lbl = ttk.Label(frame, textvariable=variable, width=5)
        value_lbl.pack(side=RIGHT)
    
    def create_image_panel(self, parent):
        """Create image display panel."""
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=BOTH, expand=YES)
        
        # Original image tab
        original_frame = ttk.Frame(notebook)
        notebook.add(original_frame, text="Original Image")
        self.original_canvas = tk.Canvas(original_frame, bg='gray20')
        self.original_canvas.pack(fill=BOTH, expand=YES)
        
        # Mask tab
        mask_frame = ttk.Frame(notebook)
        notebook.add(mask_frame, text="Mask")
        self.mask_canvas = tk.Canvas(mask_frame, bg='gray20')
        self.mask_canvas.pack(fill=BOTH, expand=YES)
        
        # Result tab
        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="Result")
        self.result_canvas = tk.Canvas(result_frame, bg='gray20')
        self.result_canvas.pack(fill=BOTH, expand=YES)
    
    def load_image(self):
        """Load image from file."""
        try:
            supported_formats = self.config.get('image', 'supported_formats', 
                                               default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
            filetypes = [('Image files', ' '.join(f'*{fmt}' for fmt in supported_formats))]
            
            filepath = filedialog.askopenfilename(
                title="Select Image",
                filetypes=filetypes
            )
            
            if filepath:
                resize_large = self.config.get('performance', 'resize_large_images', default=True)
                max_dim = self.config.get('performance', 'max_dimension', default=1920)
                
                self.current_image = load_image(filepath, resize_large=resize_large, max_dimension=max_dim)
                
                if self.current_image is not None:
                    self.current_image_path = filepath
                    self.update_display()
                    logger.info(f"Loaded image: {filepath}")
                    messagebox.showinfo("Success", "Image loaded successfully!")
                else:
                    messagebox.showerror("Error", "Failed to load image. Check the file format.")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def load_sample(self):
        """Load a sample image."""
        try:
            sample_dir = Path("sample_images")
            if not sample_dir.exists():
                messagebox.showwarning("Warning", "Sample images not found. Please create samples first.")
                return
            
            samples = list(sample_dir.glob("*.png"))
            if not samples:
                messagebox.showwarning("Warning", "No sample images found.")
                return
            
            # Load first sample
            filepath = str(samples[0])
            self.current_image = load_image(filepath)
            
            if self.current_image is not None:
                self.current_image_path = filepath
                self.update_display()
                logger.info(f"Loaded sample: {filepath}")
            else:
                messagebox.showerror("Error", "Failed to load sample image.")
        except Exception as e:
            logger.error(f"Error loading sample: {e}")
            messagebox.showerror("Error", f"Error loading sample: {str(e)}")
    
    def save_mask(self):
        """Save the current mask."""
        try:
            if self.mask is None:
                messagebox.showwarning("Warning", "No mask to save. Please process an image first.")
                return
            
            filepath = filedialog.asksaveasfilename(
                title="Save Mask",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if filepath:
                if save_image(filepath, self.mask):
                    messagebox.showinfo("Success", f"Mask saved to {filepath}")
                else:
                    messagebox.showerror("Error", "Failed to save mask.")
        except Exception as e:
            logger.error(f"Error saving mask: {e}")
            messagebox.showerror("Error", f"Error saving mask: {str(e)}")
    
    def save_result(self):
        """Save the result image."""
        try:
            if self.result_image is None:
                messagebox.showwarning("Warning", "No result to save. Please process an image first.")
                return
            
            filepath = filedialog.asksaveasfilename(
                title="Save Result",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            
            if filepath:
                if save_image(filepath, self.result_image):
                    messagebox.showinfo("Success", f"Result saved to {filepath}")
                else:
                    messagebox.showerror("Error", "Failed to save result.")
        except Exception as e:
            logger.error(f"Error saving result: {e}")
            messagebox.showerror("Error", f"Error saving result: {str(e)}")
    
    def load_preset(self, event=None):
        """Load selected preset."""
        try:
            preset_name = self.preset_combo.get()
            if preset_name:
                preset = self.preset_manager.get_preset(preset_name)
                if preset:
                    self.h_min.set(preset.get('h_min', 0))
                    self.h_max.set(preset.get('h_max', 179))
                    self.s_min.set(preset.get('s_min', 0))
                    self.s_max.set(preset.get('s_max', 255))
                    self.v_min.set(preset.get('v_min', 0))
                    self.v_max.set(preset.get('v_max', 255))
                    self.on_slider_change()
                    logger.info(f"Loaded preset: {preset_name}")
        except Exception as e:
            logger.error(f"Error loading preset: {e}")
            messagebox.showerror("Error", f"Error loading preset: {str(e)}")
    
    def save_preset(self):
        """Save current settings as a preset."""
        try:
            name = tk.simpledialog.askstring("Save Preset", "Enter preset name:")
            if name:
                preset_values = {
                    'h_min': self.h_min.get(),
                    'h_max': self.h_max.get(),
                    's_min': self.s_min.get(),
                    's_max': self.s_max.get(),
                    'v_min': self.v_min.get(),
                    'v_max': self.v_max.get()
                }
                self.preset_manager.add_preset(name, preset_values)
                self.preset_manager.save()
                self.preset_combo['values'] = self.preset_manager.get_preset_names()
                messagebox.showinfo("Success", f"Preset '{name}' saved successfully!")
                logger.info(f"Saved preset: {name}")
        except Exception as e:
            logger.error(f"Error saving preset: {e}")
            messagebox.showerror("Error", f"Error saving preset: {str(e)}")
    
    def reset_sliders(self):
        """Reset sliders to default values."""
        self.h_min.set(0)
        self.h_max.set(179)
        self.s_min.set(0)
        self.s_max.set(255)
        self.v_min.set(0)
        self.v_max.set(255)
        self.on_slider_change()
    
    def on_slider_change(self):
        """Handle slider value changes."""
        if self.current_image is not None:
            self.update_segmentation()
        self.update_hsv_visualizer()
    
    def update_hsv_visualizer(self):
        """Update HSV color space visualizer."""
        try:
            self.hsv_visualizer.update_visualization(
                self.h_min.get(), self.h_max.get(),
                self.s_min.get(), self.s_max.get(),
                self.v_min.get(), self.v_max.get()
            )
        except Exception as e:
            logger.error(f"Error updating visualizer: {e}")
    
    def update_segmentation(self):
        """Update segmentation with current HSV values."""
        try:
            if self.current_image is None:
                return
            
            lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()])
            upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
            
            # Use PyTorch processor if available
            if self.torch_processor.device.type == 'cuda':
                self.mask = self.torch_processor.threshold_torch(self.current_image, lower, upper)
            else:
                self.mask = threshold(self.current_image, lower, upper)
            
            self.result_image = apply_mask_to_image(self.current_image, self.mask)
            
            self.update_display()
            self.update_metrics()
        except Exception as e:
            logger.error(f"Error updating segmentation: {e}")
            messagebox.showerror("Error", f"Error updating segmentation: {str(e)}")
    
    def update_display(self):
        """Update all image displays."""
        try:
            if self.current_image is not None:
                self.display_image(self.current_image, self.original_canvas)
            
            if self.mask is not None:
                self.display_image(self.mask, self.mask_canvas)
            
            if self.result_image is not None:
                self.display_image(self.result_image, self.result_canvas)
        except Exception as e:
            logger.error(f"Error updating display: {e}")
    
    def display_image(self, image, canvas):
        """Display image on canvas with automatic resizing."""
        try:
            if image is None:
                return
            
            # Convert to RGB for PIL
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to fit canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w = int(w * scale * 0.95)
                new_h = int(h * scale * 0.95)
                
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            pil_img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            canvas.delete("all")
            canvas.create_image(canvas_width // 2, canvas_height // 2, anchor='center', image=photo)
            canvas.image = photo  # Keep reference
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
    
    def update_metrics(self):
        """Update metrics display."""
        try:
            if self.mask is None:
                return
            
            metrics = calculate_metrics(self.mask, self.current_image)
            
            metrics_text = (
                f"Pixels Selected: {metrics['pixel_count']:,}\n"
                f"Total Pixels: {metrics.get('total_pixels', 0):,}\n"
                f"Percentage: {metrics['percentage']:.2f}%"
            )
            
            self.metrics_label.config(text=metrics_text)
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def run(self):
        """Start the GUI application."""
        try:
            logger.info("Starting HSV Segmentation GUI")
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error running GUI: {e}")
            raise


def main():
    """Main entry point."""
    try:
        app = HSVSegmentationGUI()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
