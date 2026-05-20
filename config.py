import yaml
from pathlib import Path

class Config:
    """Configuration management for the brain tumor classifier."""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_defaults()
    
    def _get_defaults(self):
        """Return default configuration."""
        return {
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'splits_dir': 'data/splits',
                'patch_size': 64,  # 64x64x64 or 96x96x96
                'stride': 32,  # Patch stride
                'modalities': ['t1', 't1ce', 't2', 'flair'],
                'normalize': True,
                'denoise': True,
            },
            'training': {
                'batch_size': 8,
                'epochs': 100,
                'learning_rate': 1e-3,
                'validation_split': 0.15,
                'test_split': 0.15,
                'shuffle': True,
                'augment': True,
            },
            'augmentation': {
                'rotation_angles': [10, 20, 30],
                'flip_axes': [0, 1, 2],
                'zoom_range': [0.8, 1.2],
            },
            'model': {
                'architecture': 'unet3d',
                'input_channels': 4,
                'output_classes': 4,  # Background + 3 tumor classes
                'depth': 4,
                'filters_base': 32,
            },
            'training_device': {
                'backend': 'tensorflow',
                'use_metal': True,  # Mac M2 acceleration
                'mixed_precision': False,
            },
            'paths': {
                'models_dir': 'models',
                'logs_dir': 'logs',
                'results_dir': 'results',
            }
        }
    
    def save(self):
        """Save configuration to YAML file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key, default=None):
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def __repr__(self):
        return f"Config({self.config})"

# Global config instance
config = Config()
