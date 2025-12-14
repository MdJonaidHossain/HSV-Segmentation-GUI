"""
Configuration management for HSV Segmentation GUI.
"""
import json
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for application settings."""
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return config
            else:
                logger.warning(f"Config file {self.config_file} not found, using defaults")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "window": {
                "title": "HSV Segmentation Tool",
                "geometry": "1400x900",
                "theme": "darkly"
            },
            "hsv_defaults": {
                "h_min": 0,
                "h_max": 179,
                "s_min": 0,
                "s_max": 255,
                "v_min": 0,
                "v_max": 255
            },
            "image": {
                "preview_width": 640,
                "preview_height": 480,
                "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            },
            "performance": {
                "use_gpu": False,
                "resize_large_images": True,
                "max_dimension": 1920
            },
            "logging": {
                "level": "INFO",
                "file": "hsv_segmentation.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            *keys: Configuration keys (e.g., 'window', 'title')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, value: Any, *keys) -> None:
        """
        Set configuration value.
        
        Args:
            value: Value to set
            *keys: Configuration keys
        """
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def save(self) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False


class PresetManager:
    """Manager for HSV range presets."""
    
    def __init__(self, preset_file: str = "user_presets.json"):
        """
        Initialize preset manager.
        
        Args:
            preset_file: Path to preset file
        """
        self.preset_file = Path(preset_file)
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, Dict[str, int]]:
        """Load presets from file."""
        try:
            if self.preset_file.exists():
                with open(self.preset_file, 'r') as f:
                    presets = json.load(f)
                logger.info(f"Loaded {len(presets)} presets from {self.preset_file}")
                return presets
            else:
                return self._default_presets()
        except Exception as e:
            logger.error(f"Error loading presets: {e}, using defaults")
            return self._default_presets()
    
    def _default_presets(self) -> Dict[str, Dict[str, int]]:
        """Return default presets."""
        return {
            "Red": {
                "h_min": 0, "h_max": 10,
                "s_min": 100, "s_max": 255,
                "v_min": 100, "v_max": 255
            },
            "Green": {
                "h_min": 40, "h_max": 80,
                "s_min": 50, "s_max": 255,
                "v_min": 50, "v_max": 255
            },
            "Blue": {
                "h_min": 100, "h_max": 130,
                "s_min": 50, "s_max": 255,
                "v_min": 50, "v_max": 255
            },
            "Yellow": {
                "h_min": 20, "h_max": 40,
                "s_min": 100, "s_max": 255,
                "v_min": 100, "v_max": 255
            },
            "Orange": {
                "h_min": 10, "h_max": 25,
                "s_min": 100, "s_max": 255,
                "v_min": 100, "v_max": 255
            }
        }
    
    def get_preset(self, name: str) -> Dict[str, int]:
        """
        Get preset by name.
        
        Args:
            name: Preset name
            
        Returns:
            Preset values or empty dict
        """
        return self.presets.get(name, {})
    
    def add_preset(self, name: str, values: Dict[str, int]) -> None:
        """
        Add or update preset.
        
        Args:
            name: Preset name
            values: HSV values
        """
        self.presets[name] = values
        logger.info(f"Added/updated preset: {name}")
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete preset.
        
        Args:
            name: Preset name
            
        Returns:
            True if deleted, False if not found
        """
        if name in self.presets:
            del self.presets[name]
            logger.info(f"Deleted preset: {name}")
            return True
        return False
    
    def get_preset_names(self) -> list:
        """Get list of preset names."""
        return list(self.presets.keys())
    
    def save(self) -> bool:
        """Save presets to file."""
        try:
            with open(self.preset_file, 'w') as f:
                json.dump(self.presets, f, indent=2)
            logger.info(f"Saved {len(self.presets)} presets to {self.preset_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving presets: {e}")
            return False
