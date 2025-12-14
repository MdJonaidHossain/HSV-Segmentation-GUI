"""
Unit tests for configuration management.
"""
import unittest
import json
import sys
from pathlib import Path
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import Config, PresetManager


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
        os.rmdir(self.temp_dir)
    
    def test_config_load_default(self):
        """Test loading default config when file doesn't exist."""
        config = Config(self.config_file)
        
        self.assertIsNotNone(config.config)
        self.assertIn('window', config.config)
        self.assertIn('hsv_defaults', config.config)
    
    def test_config_get_nested(self):
        """Test getting nested configuration values."""
        config = Config(self.config_file)
        
        title = config.get('window', 'title')
        self.assertIsNotNone(title)
        
        h_min = config.get('hsv_defaults', 'h_min')
        self.assertEqual(h_min, 0)
    
    def test_config_get_with_default(self):
        """Test getting config with default value."""
        config = Config(self.config_file)
        
        value = config.get('nonexistent', 'key', default='default_value')
        self.assertEqual(value, 'default_value')
    
    def test_config_set(self):
        """Test setting configuration values."""
        config = Config(self.config_file)
        
        config.set(100, 'hsv_defaults', 'h_min')
        value = config.get('hsv_defaults', 'h_min')
        
        self.assertEqual(value, 100)
    
    def test_config_save_and_load(self):
        """Test saving and loading configuration."""
        config1 = Config(self.config_file)
        config1.set(150, 'hsv_defaults', 'h_max')
        config1.save()
        
        config2 = Config(self.config_file)
        value = config2.get('hsv_defaults', 'h_max')
        
        self.assertEqual(value, 150)
    
    def test_config_load_corrupted(self):
        """Test loading corrupted config file."""
        # Create corrupted JSON file
        with open(self.config_file, 'w') as f:
            f.write("{ invalid json }")
        
        config = Config(self.config_file)
        
        # Should fall back to defaults
        self.assertIsNotNone(config.config)
        self.assertIn('window', config.config)


class TestPresetManager(unittest.TestCase):
    """Test cases for PresetManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.preset_file = os.path.join(self.temp_dir, 'test_presets.json')
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.preset_file):
            os.remove(self.preset_file)
        os.rmdir(self.temp_dir)
    
    def test_preset_load_default(self):
        """Test loading default presets."""
        manager = PresetManager(self.preset_file)
        
        self.assertIsNotNone(manager.presets)
        self.assertGreater(len(manager.presets), 0)
    
    def test_preset_get_preset(self):
        """Test getting a preset."""
        manager = PresetManager(self.preset_file)
        
        red_preset = manager.get_preset('Red')
        
        self.assertIsNotNone(red_preset)
        self.assertIn('h_min', red_preset)
        self.assertIn('h_max', red_preset)
    
    def test_preset_get_nonexistent(self):
        """Test getting nonexistent preset."""
        manager = PresetManager(self.preset_file)
        
        preset = manager.get_preset('NonExistent')
        
        self.assertEqual(preset, {})
    
    def test_preset_add_preset(self):
        """Test adding a new preset."""
        manager = PresetManager(self.preset_file)
        
        new_preset = {
            'h_min': 50,
            'h_max': 100,
            's_min': 100,
            's_max': 255,
            'v_min': 100,
            'v_max': 255
        }
        
        manager.add_preset('Custom', new_preset)
        retrieved = manager.get_preset('Custom')
        
        self.assertEqual(retrieved, new_preset)
    
    def test_preset_update_existing(self):
        """Test updating existing preset."""
        manager = PresetManager(self.preset_file)
        
        new_values = {
            'h_min': 5,
            'h_max': 15,
            's_min': 150,
            's_max': 255,
            'v_min': 150,
            'v_max': 255
        }
        
        manager.add_preset('Red', new_values)
        retrieved = manager.get_preset('Red')
        
        self.assertEqual(retrieved, new_values)
    
    def test_preset_delete(self):
        """Test deleting a preset."""
        manager = PresetManager(self.preset_file)
        
        result = manager.delete_preset('Red')
        
        self.assertTrue(result)
        self.assertEqual(manager.get_preset('Red'), {})
    
    def test_preset_delete_nonexistent(self):
        """Test deleting nonexistent preset."""
        manager = PresetManager(self.preset_file)
        
        result = manager.delete_preset('NonExistent')
        
        self.assertFalse(result)
    
    def test_preset_get_names(self):
        """Test getting preset names."""
        manager = PresetManager(self.preset_file)
        
        names = manager.get_preset_names()
        
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)
        self.assertIn('Red', names)
        self.assertIn('Green', names)
    
    def test_preset_save_and_load(self):
        """Test saving and loading presets."""
        manager1 = PresetManager(self.preset_file)
        
        custom_preset = {
            'h_min': 75,
            'h_max': 125,
            's_min': 50,
            's_max': 200,
            'v_min': 50,
            'v_max': 200
        }
        
        manager1.add_preset('TestPreset', custom_preset)
        manager1.save()
        
        manager2 = PresetManager(self.preset_file)
        retrieved = manager2.get_preset('TestPreset')
        
        self.assertEqual(retrieved, custom_preset)


if __name__ == '__main__':
    unittest.main()
