import yaml
from pathlib import Path
from typing import Any


class ConfigManager:
    """
    Loads and manages configuration from a YAML file.
    """
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of ConfigManager exists."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path (str | Path | None): Path to the config.yaml file. If None, uses default path at project root.
        """
        if self._config is None:  # Only load once
            if config_path is None:
                # Default path: project root / config.yaml
                project_root = Path(__file__).resolve().parents[1]
                config_path = project_root / "config.yaml"
            
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            
            # Store project root for path resolution
            self._project_root = Path(__file__).resolve().parents[1]
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter by key.
        Supports nested keys using dot notation (e.g., 'paths.dataset').
        
        Args:
            key (str): Configuration key (supports dot notation for nested keys)
            default (Any): Default value if key is not found
            
        Returns:
            Any: Configuration value
            
        Example:
            >>> config = ConfigManager()
            >>> config.get_param('paths.dataset')
            'datasets/neo.csv'
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """
        Get a path from configuration and resolve it relative to project root.
        
        Args:
            key (str): Configuration key for the path
            
        Returns:
            Path: Resolved absolute path
            
        Example:
            >>> config = ConfigManager()
            >>> config.get_path('paths.dataset')
            Path('C:/Users/.../IFT712_Project/datasets/neo.csv')
        """
        path_str = self.get_param(key)
        if path_str is None:
            raise KeyError(f"Path key '{key}' not found in configuration")
        
        path = Path(path_str)
        if not path.is_absolute():
            path = self._project_root / path
        
        return path
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root
    
    @property
    def config(self) -> dict:
        """Get the full configuration dictionary."""
        return self._config


# Convenience function to get a global config instance
def get_config() -> ConfigManager:
    """Get the global ConfigManager instance."""
    return ConfigManager()