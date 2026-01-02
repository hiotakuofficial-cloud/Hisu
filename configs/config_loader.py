"""
Configuration loader and manager
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigLoader:
    """
    Configuration loader for ML experiments
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader

        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.config = None

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        path = Path(config_path)

        if path.suffix == '.json':
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif path.suffix == '.py':
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.config = module.config
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

        return self.config

    def save(self, config: Dict[str, Any], filepath: str):
        """
        Save configuration to file

        Args:
            config: Configuration dictionary
            filepath: Path to save config
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to {filepath}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key

        Args:
            key: Configuration key (supports nested keys with dots)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if self.config is None:
            return default

        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value

        Args:
            key: Configuration key (supports nested keys with dots)
            value: Value to set
        """
        if self.config is None:
            self.config = {}

        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def merge(self, other_config: Dict[str, Any]):
        """
        Merge another configuration

        Args:
            other_config: Configuration to merge
        """
        if self.config is None:
            self.config = {}

        self.config = self._deep_merge(self.config, other_config)

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Deep merge two dictionaries

        Args:
            base: Base dictionary
            update: Dictionary to merge

        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if valid, False otherwise
        """
        if self.config is None:
            return False

        required_keys = ['model', 'training', 'data']

        for key in required_keys:
            if key not in self.config:
                print(f"Missing required configuration key: {key}")
                return False

        return True

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return self.config

    def reset(self):
        """Reset configuration to empty"""
        self.config = None
