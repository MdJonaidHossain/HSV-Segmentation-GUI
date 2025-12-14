# HSV Segmentation GUI

An interactive, modern GUI tool for HSV-based image segmentation with real-time preview, preset management, and optional GPU acceleration using PyTorch.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Features

### Core Functionality
- **Interactive HSV Adjustment**: Real-time sliders for Hue, Saturation, and Value ranges
- **Multi-view Display**: Tabbed interface showing original image, mask, and segmented result
- **HSV Visualizer**: Live visualization of selected HSV color space
- **Metrics Display**: Real-time statistics on selected pixels and coverage

### User Experience
- **Modern UI**: Built with ttkbootstrap for a contemporary look and feel
- **Preset Management**: Save and load custom HSV range presets
- **Sample Images**: Built-in sample images for demonstration
- **Tooltips**: Helpful hints on all controls
- **Error Handling**: Graceful handling of corrupted or unsupported files

### Performance
- **PyTorch Acceleration**: Optional GPU acceleration for faster processing
- **Smart Resizing**: Automatic resizing of large images for better performance
- **Efficient Processing**: Optimized algorithms for real-time updates

### Developer Features
- **Modular Architecture**: Well-organized code structure with separate utilities
- **Comprehensive Testing**: Unit tests for all core functions
- **Detailed Logging**: Built-in logging for debugging and monitoring
- **Configuration Management**: JSON-based configuration system
- **Edge Case Handling**: Robust error handling for various scenarios

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/MdJonaidHossain/HSV-Segmentation-GUI.git
cd HSV-Segmentation-GUI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate sample images (optional):
```bash
python create_samples.py
```

4. Run the application:
```bash
python hsv_segmentation_gui.py
```

## Usage

### Basic Workflow

1. **Load Image**: Click "Load Image" to select an image file, or "Load Sample" to use demonstration images
2. **Adjust HSV Ranges**: Use the sliders to adjust Hue, Saturation, and Value ranges
3. **View Results**: Switch between tabs to see the original, mask, and segmented result
4. **Save Output**: Click "Save Mask" or "Save Result" to export your work

### Using Presets

1. **Load Preset**: Select a color preset from the dropdown (e.g., "Red", "Green", "Blue")
2. **Adjust**: Fine-tune the preset using sliders
3. **Save Custom Preset**: Click "Save Current as Preset" and enter a name

### Understanding HSV

- **Hue (H)**: Color type (0-179 in OpenCV)
  - 0-10: Red
  - 40-80: Green
  - 100-130: Blue
  - 20-40: Yellow

- **Saturation (S)**: Color intensity (0-255)
  - 0: Grayscale
  - 255: Fully saturated color

- **Value (V)**: Brightness (0-255)
  - 0: Black
  - 255: Bright

## Configuration

Edit `config.json` to customize:

```json
{
  "window": {
    "title": "HSV Segmentation Tool",
    "geometry": "1400x900",
    "theme": "darkly"
  },
  "performance": {
    "use_gpu": false,
    "resize_large_images": true,
    "max_dimension": 1920
  }
}
```

### Available Themes
- darkly (default)
- solar
- superhero
- cyborg
- vapor
- cosmo
- flatly
- journal
- litera
- lumen
- minty
- pulse
- sandstone
- united
- yeti

## Development

### Project Structure

```
HSV-Segmentation-GUI/
├── hsv_segmentation_gui.py    # Main GUI application
├── create_samples.py           # Sample image generator
├── config.json                 # Application configuration
├── requirements.txt            # Python dependencies
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── image_processing.py    # Core image processing functions
│   ├── config.py              # Configuration management
│   └── torch_processing.py    # PyTorch acceleration
├── tests/                      # Unit tests
│   ├── test_image_processing.py
│   └── test_config.py
└── sample_images/              # Sample demonstration images
```

### Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests/test_image_processing.py

# Run with verbose output
python -m unittest discover tests -v
```

### Adding Custom Utility Functions

1. Add function to appropriate module in `utils/`
2. Export in `utils/__init__.py`
3. Add unit tests in `tests/`
4. Update documentation

### GPU Acceleration

To enable GPU acceleration:

1. Ensure CUDA-compatible GPU is available
2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. Set `use_gpu: true` in `config.json`

## API Reference

### Core Functions

#### `load_image(filepath, resize_large=True, max_dimension=1920)`
Load and optionally resize an image.

**Parameters:**
- `filepath` (str): Path to image file
- `resize_large` (bool): Whether to resize large images
- `max_dimension` (int): Maximum width or height

**Returns:** numpy.ndarray or None

#### `threshold(image, lower, upper)`
Apply HSV threshold to create binary mask.

**Parameters:**
- `image` (np.ndarray): Input BGR image
- `lower` (np.ndarray): Lower HSV threshold [H, S, V]
- `upper` (np.ndarray): Upper HSV threshold [H, S, V]

**Returns:** Binary mask as numpy.ndarray

#### `calculate_metrics(mask, image)`
Calculate segmentation metrics.

**Parameters:**
- `mask` (np.ndarray): Binary mask
- `image` (np.ndarray): Original image

**Returns:** Dictionary with pixel_count, percentage, and stats

#### `gray_stats(image)`
Calculate statistics for grayscale image.

**Parameters:**
- `image` (np.ndarray): Grayscale image

**Returns:** Dictionary with mean, std, min, max

## Troubleshooting

### Common Issues

**Issue**: Application won't start
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: GPU not detected
- **Solution**: Install PyTorch with CUDA support or set `use_gpu: false` in config

**Issue**: Image won't load
- **Solution**: Check file format is supported (.jpg, .png, .bmp, .tiff)

**Issue**: Application is slow
- **Solution**: Enable image resizing in config or reduce `max_dimension`

### Logging

Check `hsv_segmentation.log` for detailed error messages and debugging information.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV for computer vision operations
- ttkbootstrap for modern UI components
- PyTorch for GPU acceleration capabilities

## Future Enhancements

- [ ] Advanced morphological operations
- [ ] Color space conversions (LAB, YCrCb)
- [ ] Batch processing mode
- [ ] Video segmentation support
- [ ] Export configuration as code
- [ ] Histogram equalization
- [ ] Undo/redo functionality

## Support

For issues, questions, or suggestions, please open an issue on GitHub.