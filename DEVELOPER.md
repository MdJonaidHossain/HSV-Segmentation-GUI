# Developer Guide

## Architecture Overview

The HSV Segmentation GUI follows a modular architecture with clear separation of concerns:

```
HSV-Segmentation-GUI/
├── hsv_segmentation_gui.py    # Main GUI application entry point
├── utils/                      # Core utility modules
│   ├── image_processing.py    # Image processing algorithms
│   ├── config.py              # Configuration and preset management
│   └── torch_processing.py    # PyTorch GPU acceleration
├── tests/                      # Unit test suite
└── sample_images/              # Demo images
```

## Module Responsibilities

### 1. Image Processing (`utils/image_processing.py`)

**Purpose**: Core image processing algorithms using OpenCV and NumPy

**Key Functions**:
- `gray_stats()`: Calculate statistics for grayscale images
- `threshold()`: HSV-based color segmentation
- `calculate_metrics()`: Compute segmentation metrics
- `apply_mask_to_image()`: Apply binary mask to image
- `resize_image()`: Intelligent image resizing
- `load_image()` / `save_image()`: File I/O with error handling

**Design Patterns**:
- Error handling with graceful degradation
- Comprehensive logging for debugging
- Edge case handling (None, empty arrays)

### 2. Configuration (`utils/config.py`)

**Purpose**: Manage application settings and user presets

**Classes**:

#### `Config`
- Loads from `config.json` with fallback to defaults
- Supports nested key access with `get()` method
- Thread-safe configuration updates
- Automatic persistence with `save()` method

#### `PresetManager`
- Manages HSV range presets
- Default presets for common colors (Red, Green, Blue, Yellow, Orange)
- User preset creation and deletion
- JSON-based persistence

**Usage Example**:
```python
from utils.config import Config, PresetManager

# Load configuration
config = Config()
theme = config.get('window', 'theme', default='darkly')

# Manage presets
presets = PresetManager()
red_preset = presets.get_preset('Red')
presets.add_preset('Custom', {...})
presets.save()
```

### 3. PyTorch Processing (`utils/torch_processing.py`)

**Purpose**: GPU-accelerated image processing using PyTorch

**Class**: `TorchImageProcessor`

**Features**:
- Automatic GPU/CPU device selection
- PyTorch tensor conversion utilities
- GPU-accelerated HSV thresholding
- Batch processing support

**Usage Example**:
```python
from utils.torch_processing import get_processor

processor = get_processor(use_gpu=True)
mask = processor.threshold_torch(image, lower, upper)
metrics = processor.calculate_metrics_torch(mask)
```

**Performance Notes**:
- GPU acceleration most beneficial for large images (>2MP)
- Batch processing reduces GPU transfer overhead
- Falls back to OpenCV if PyTorch/CUDA unavailable

### 4. GUI Application (`hsv_segmentation_gui.py`)

**Purpose**: User interface built with ttkbootstrap

**Classes**:

#### `HSVVisualizerFrame`
- Real-time HSV color space visualization
- Shows selected HSV range as color gradient
- Updates on slider changes

#### `HSVSegmentationGUI`
- Main application window
- Control panel with sliders and buttons
- Tabbed image display (Original, Mask, Result)
- Real-time metrics display

**Design Patterns**:
- MVC-like separation (GUI / Logic / Data)
- Event-driven slider updates
- Lazy loading of images
- Responsive layout with resizing

## Adding New Features

### Adding a New Utility Function

1. **Define the function** in appropriate module:
```python
# utils/image_processing.py
def new_processing_function(image: np.ndarray, param: int) -> np.ndarray:
    """
    Description of function.
    
    Args:
        image: Input image
        param: Parameter description
        
    Returns:
        Processed image
    """
    try:
        # Implementation
        result = cv2.someOperation(image, param)
        logger.info(f"Processed image with param={param}")
        return result
    except Exception as e:
        logger.error(f"Error in new_processing_function: {e}")
        return image
```

2. **Export in `__init__.py`**:
```python
# utils/__init__.py
from .image_processing import new_processing_function

__all__ = [
    # ... existing exports
    'new_processing_function'
]
```

3. **Add unit tests**:
```python
# tests/test_image_processing.py
def test_new_processing_function(self):
    """Test new processing function."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = new_processing_function(image, 5)
    
    self.assertIsNotNone(result)
    self.assertEqual(result.shape, image.shape)
```

### Adding a New Preset

Programmatically:
```python
from utils.config import PresetManager

presets = PresetManager()
presets.add_preset('Purple', {
    'h_min': 130,
    'h_max': 160,
    's_min': 50,
    's_max': 255,
    'v_min': 50,
    'v_max': 255
})
presets.save()
```

Or edit `user_presets.json` directly:
```json
{
  "Purple": {
    "h_min": 130,
    "h_max": 160,
    "s_min": 50,
    "s_max": 255,
    "v_min": 50,
    "v_max": 255
  }
}
```

### Extending the GUI

1. **Add new control**:
```python
# In create_control_panel()
new_btn = ttk.Button(
    parent, 
    text="New Feature", 
    command=self.new_feature_handler,
    bootstyle=PRIMARY
)
new_btn.pack(fill=X, pady=2)
ToolTip(new_btn, text="Description of new feature")
```

2. **Implement handler**:
```python
def new_feature_handler(self):
    """Handle new feature button click."""
    try:
        # Implementation
        logger.info("New feature activated")
        messagebox.showinfo("Success", "Feature completed!")
    except Exception as e:
        logger.error(f"Error in new feature: {e}")
        messagebox.showerror("Error", str(e))
```

## Testing Guidelines

### Running Tests

```bash
# All tests
python -m unittest discover tests -v

# Specific test file
python -m unittest tests/test_image_processing.py

# Specific test case
python -m unittest tests.test_image_processing.TestImageProcessing.test_threshold_normal
```

### Writing Tests

Follow the existing pattern:

```python
class TestNewFeature(unittest.TestCase):
    """Test cases for new feature."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = create_test_data()
    
    def test_normal_case(self):
        """Test normal operation."""
        result = new_function(self.test_data)
        self.assertIsNotNone(result)
    
    def test_edge_case_empty(self):
        """Test with empty input."""
        result = new_function(np.array([]))
        # Assert expected behavior
    
    def test_edge_case_none(self):
        """Test with None input."""
        result = new_function(None)
        # Assert expected behavior
```

### Test Coverage Goals

- **Utility functions**: 100% coverage
- **Edge cases**: Empty, None, invalid inputs
- **Configuration**: Load, save, defaults
- **GUI components**: Not required (use manual testing)

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile function
cProfile.run('threshold(image, lower, upper)', 'stats.prof')

# View results
p = pstats.Stats('stats.prof')
p.sort_stats('cumulative').print_stats(10)
```

### GPU Acceleration

Enable GPU in `config.json`:
```json
{
  "performance": {
    "use_gpu": true
  }
}
```

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Memory Optimization

Large images are automatically resized based on config:
```json
{
  "performance": {
    "resize_large_images": true,
    "max_dimension": 1920
  }
}
```

## Logging

Configure logging in `config.json`:
```json
{
  "logging": {
    "level": "INFO",
    "file": "hsv_segmentation.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

Logging levels:
- `DEBUG`: Detailed information for diagnosis
- `INFO`: General informational messages
- `WARNING`: Warning messages (non-critical)
- `ERROR`: Error messages

View logs:
```bash
tail -f hsv_segmentation.log
```

## Code Style

Follow PEP 8 guidelines:
- 4 spaces for indentation
- Max line length: 100 characters
- Docstrings for all functions/classes
- Type hints where applicable

Example:
```python
def process_image(
    image: np.ndarray, 
    lower: np.ndarray, 
    upper: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Process image with HSV thresholding.
    
    Args:
        image: Input BGR image
        lower: Lower HSV bounds
        upper: Upper HSV bounds
        
    Returns:
        Tuple of (mask, metrics)
    """
    mask = threshold(image, lower, upper)
    metrics = calculate_metrics(mask, image)
    return mask, metrics
```

## Debugging Tips

1. **Enable debug logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check intermediate results**:
```python
# Save intermediate images
cv2.imwrite('debug_mask.png', mask)
cv2.imwrite('debug_result.png', result)
```

3. **Print array info**:
```python
print(f"Shape: {image.shape}, dtype: {image.dtype}")
print(f"Min: {np.min(image)}, Max: {np.max(image)}")
print(f"Mean: {np.mean(image):.2f}")
```

4. **Use try-except with logging**:
```python
try:
    result = risky_operation()
except Exception as e:
    logger.exception("Detailed error with traceback")
    raise
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes with tests
4. Ensure all tests pass: `python -m unittest discover tests`
5. Commit: `git commit -m "Add feature"`
6. Push: `git push origin feature-name`
7. Create Pull Request

## Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Version bumped in setup.py
- [ ] Sample images included
- [ ] README screenshots updated
- [ ] Code reviewed
- [ ] Security check completed
