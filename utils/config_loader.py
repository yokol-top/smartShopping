import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    def __init__(self, config_path: str = "./config/settings.yaml"):
        self.config_path = config_path
        self.config_dir = os.path.dirname(os.path.abspath(config_path))
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def resolve_path(self, path: str) -> str:
        """
        Resolve a path relative to the config file directory
        
        Args:
            path: Path from config (can be relative or absolute)
            
        Returns:
            Absolute path
        """
        if not path or os.path.isabs(path):
            return path
        
        # Resolve relative to config file directory
        return os.path.abspath(os.path.join(self.config_dir, path))
    
    def reload(self):
        self.config = self._load_config()
