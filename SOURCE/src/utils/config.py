"""Configuration loader and validator."""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for StreamGuard."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    @property
    def raw_data_path(self) -> Path:
        """Get raw data directory path."""
        return Path(self.get('data.raw_path'))
    
    @property
    def processed_data_path(self) -> Path:
        """Get processed data directory path."""
        return Path(self.get('data.processed_path'))
    
    @property
    def features_path(self) -> Path:
        """Get features directory path."""
        return Path(self.get('data.features_path'))
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.raw_data_path,
            self.processed_data_path,
            self.features_path,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Directories created/verified")


# Global config instance
config = Config()