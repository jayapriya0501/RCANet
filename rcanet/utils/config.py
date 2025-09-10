"""Configuration utilities for RCANet.

Provides configuration classes and utilities for managing
RCANet model and training configurations.
"""

import json
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings


@dataclass
class RCANetConfig:
    """Configuration class for RCANet model.
    
    Contains all hyperparameters and settings for the RCANet architecture,
    training process, and data handling.
    """
    
    # Model architecture
    input_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = 'gelu'
    
    # Attention configuration
    row_attention_dim: int = 256
    col_attention_dim: int = 256
    attention_dropout: float = 0.1
    use_positional_encoding: bool = True
    max_sequence_length: int = 1000
    
    # Fusion configuration
    fusion_strategy: str = 'cross_attention'  # 'cross_attention', 'gated', 'adaptive'
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.1
    
    # Output configuration
    output_dim: int = 1
    task_type: str = 'regression'  # 'classification', 'regression'
    num_classes: Optional[int] = None
    
    # Contrastive learning
    use_contrastive: bool = True
    contrastive_dim: int = 128
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.1
    
    # Regularization
    weight_decay: float = 1e-4
    layer_norm_eps: float = 1e-12
    gradient_clip_norm: float = 1.0
    
    # Training configuration
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 10
    
    # Optimizer configuration
    optimizer_type: str = 'adamw'
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduler configuration
    scheduler_type: str = 'cosine'
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle_data: bool = True
    random_seed: int = 42
    
    # Preprocessing configuration
    numerical_strategy: str = 'standard'
    categorical_strategy: str = 'label'
    missing_value_strategy: str = 'median'
    feature_selection: Optional[str] = None
    dimensionality_reduction: Optional[str] = None
    
    # Augmentation configuration
    use_augmentation: bool = False
    augmentation_types: List[str] = field(default_factory=lambda: ['noise', 'swap'])
    noise_level: float = 0.01
    swap_probability: float = 0.1
    
    # Training utilities
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    save_best_only: bool = True
    
    # Logging and checkpointing
    log_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    save_frequency: int = 10
    verbose: int = 1
    
    # Device configuration
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0', etc.
    mixed_precision: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate dimensions
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
            
        # Validate attention dimensions
        if self.hidden_dim % self.num_heads != 0:
            warnings.warn(
                f"hidden_dim ({self.hidden_dim}) should be divisible by num_heads ({self.num_heads})"
            )
            
        # Validate dropout rates
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if not 0 <= self.attention_dropout <= 1:
            raise ValueError("attention_dropout must be between 0 and 1")
        if not 0 <= self.fusion_dropout <= 1:
            raise ValueError("fusion_dropout must be between 0 and 1")
            
        # Validate task configuration
        if self.task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")
            
        if self.task_type == 'classification':
            if self.num_classes is None:
                raise ValueError("num_classes must be specified for classification tasks")
            if self.num_classes <= 1:
                raise ValueError("num_classes must be greater than 1 for classification")
            if self.output_dim != self.num_classes:
                warnings.warn(
                    f"Setting output_dim to num_classes ({self.num_classes}) for classification"
                )
                self.output_dim = self.num_classes
                
        # Validate data splits
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
            
        # Validate learning parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
            
        # Validate strategies
        valid_numerical_strategies = ['standard', 'minmax', 'robust', 'quantile']
        if self.numerical_strategy not in valid_numerical_strategies:
            raise ValueError(f"numerical_strategy must be one of {valid_numerical_strategies}")
            
        valid_categorical_strategies = ['label', 'onehot', 'ordinal']
        if self.categorical_strategy not in valid_categorical_strategies:
            raise ValueError(f"categorical_strategy must be one of {valid_categorical_strategies}")
            
        valid_missing_strategies = ['mean', 'median', 'mode', 'knn', 'drop']
        if self.missing_value_strategy not in valid_missing_strategies:
            raise ValueError(f"missing_value_strategy must be one of {valid_missing_strategies}")
            
        valid_fusion_strategies = ['cross_attention', 'gated', 'adaptive']
        if self.fusion_strategy not in valid_fusion_strategies:
            raise ValueError(f"fusion_strategy must be one of {valid_fusion_strategies}")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
        
    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """Convert configuration to JSON string or save to file."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=indent)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
                
        return json_str
        
    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Convert configuration to YAML string or save to file."""
        config_dict = self.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(yaml_str)
                
        return yaml_str
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RCANetConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
        
    @classmethod
    def from_json(cls, filepath_or_str: str) -> 'RCANetConfig':
        """Create configuration from JSON file or string."""
        if Path(filepath_or_str).exists():
            # Load from file
            with open(filepath_or_str, 'r') as f:
                config_dict = json.load(f)
        else:
            # Parse as JSON string
            config_dict = json.loads(filepath_or_str)
            
        return cls.from_dict(config_dict)
        
    @classmethod
    def from_yaml(cls, filepath_or_str: str) -> 'RCANetConfig':
        """Create configuration from YAML file or string."""
        if Path(filepath_or_str).exists():
            # Load from file
            with open(filepath_or_str, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            # Parse as YAML string
            config_dict = yaml.safe_load(filepath_or_str)
            
        return cls.from_dict(config_dict)
        
    def update(self, **kwargs) -> 'RCANetConfig':
        """Update configuration with new values."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.from_dict(config_dict)
        
    def copy(self) -> 'RCANetConfig':
        """Create a copy of the configuration."""
        return self.from_dict(self.to_dict())
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration parameters."""
        model_params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'activation': self.activation,
            'row_attention_dim': self.row_attention_dim,
            'col_attention_dim': self.col_attention_dim,
            'attention_dropout': self.attention_dropout,
            'use_positional_encoding': self.use_positional_encoding,
            'max_sequence_length': self.max_sequence_length,
            'fusion_strategy': self.fusion_strategy,
            'fusion_hidden_dim': self.fusion_hidden_dim,
            'fusion_dropout': self.fusion_dropout,
            'output_dim': self.output_dim,
            'task_type': self.task_type,
            'num_classes': self.num_classes,
            'use_contrastive': self.use_contrastive,
            'contrastive_dim': self.contrastive_dim,
            'contrastive_temperature': self.contrastive_temperature,
            'layer_norm_eps': self.layer_norm_eps
        }
        return model_params
        
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration parameters."""
        training_params = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'warmup_epochs': self.warmup_epochs,
            'optimizer_type': self.optimizer_type,
            'optimizer_params': self.optimizer_params,
            'scheduler_type': self.scheduler_type,
            'scheduler_params': self.scheduler_params,
            'weight_decay': self.weight_decay,
            'gradient_clip_norm': self.gradient_clip_norm,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'save_best_only': self.save_best_only,
            'device': self.device,
            'mixed_precision': self.mixed_precision
        }
        return training_params
        
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration parameters."""
        data_params = {
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'shuffle_data': self.shuffle_data,
            'random_seed': self.random_seed,
            'numerical_strategy': self.numerical_strategy,
            'categorical_strategy': self.categorical_strategy,
            'missing_value_strategy': self.missing_value_strategy,
            'feature_selection': self.feature_selection,
            'dimensionality_reduction': self.dimensionality_reduction,
            'use_augmentation': self.use_augmentation,
            'augmentation_types': self.augmentation_types,
            'noise_level': self.noise_level,
            'swap_probability': self.swap_probability
        }
        return data_params


def create_config_for_dataset(
    dataset_info: Dict[str, Any],
    task_type: str = 'auto',
    **kwargs
) -> RCANetConfig:
    """Create configuration optimized for a specific dataset.
    
    Args:
        dataset_info: Dictionary containing dataset information
            - 'n_samples': Number of samples
            - 'n_features': Number of features
            - 'n_classes': Number of classes (for classification)
            - 'feature_types': List of feature types
        task_type: Task type ('classification', 'regression', 'auto')
        **kwargs: Additional configuration parameters
        
    Returns:
        Optimized RCANetConfig
    """
    n_samples = dataset_info.get('n_samples', 1000)
    n_features = dataset_info.get('n_features', 10)
    n_classes = dataset_info.get('n_classes')
    
    # Auto-detect task type
    if task_type == 'auto':
        if n_classes is not None and n_classes > 1:
            task_type = 'classification'
        else:
            task_type = 'regression'
            
    # Base configuration
    config_params = {
        'input_dim': n_features,
        'task_type': task_type
    }
    
    # Task-specific configuration
    if task_type == 'classification':
        config_params.update({
            'output_dim': n_classes,
            'num_classes': n_classes
        })
    else:
        config_params.update({
            'output_dim': 1,
            'num_classes': None
        })
        
    # Scale model size based on dataset size
    if n_samples < 1000:
        # Small dataset
        config_params.update({
            'hidden_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'batch_size': 16
        })
    elif n_samples < 10000:
        # Medium dataset
        config_params.update({
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'batch_size': 32
        })
    else:
        # Large dataset
        config_params.update({
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 16,
            'batch_size': 64
        })
        
    # Scale attention dimensions
    config_params.update({
        'row_attention_dim': config_params['hidden_dim'],
        'col_attention_dim': config_params['hidden_dim'],
        'fusion_hidden_dim': config_params['hidden_dim']
    })
    
    # Feature-based adjustments
    if n_features > 100:
        # High-dimensional data
        config_params.update({
            'feature_selection': 'kbest',
            'dimensionality_reduction': 'pca'
        })
        
    # Update with user-provided parameters
    config_params.update(kwargs)
    
    return RCANetConfig(**config_params)


def get_preset_config(preset_name: str, **kwargs) -> RCANetConfig:
    """Get a preset configuration.
    
    Args:
        preset_name: Name of the preset ('small', 'medium', 'large', 'xlarge')
        **kwargs: Additional configuration parameters
        
    Returns:
        Preset RCANetConfig
    """
    presets = {
        'small': {
            'hidden_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'row_attention_dim': 128,
            'col_attention_dim': 128,
            'fusion_hidden_dim': 128,
            'batch_size': 16,
            'learning_rate': 1e-3
        },
        'medium': {
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'row_attention_dim': 256,
            'col_attention_dim': 256,
            'fusion_hidden_dim': 256,
            'batch_size': 32,
            'learning_rate': 1e-3
        },
        'large': {
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 16,
            'row_attention_dim': 512,
            'col_attention_dim': 512,
            'fusion_hidden_dim': 512,
            'batch_size': 64,
            'learning_rate': 5e-4
        },
        'xlarge': {
            'hidden_dim': 768,
            'num_layers': 8,
            'num_heads': 24,
            'row_attention_dim': 768,
            'col_attention_dim': 768,
            'fusion_hidden_dim': 768,
            'batch_size': 128,
            'learning_rate': 2e-4
        }
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
    preset_params = presets[preset_name]
    preset_params.update(kwargs)
    
    return RCANetConfig(**preset_params)


def load_config(filepath: str) -> RCANetConfig:
    """Load configuration from file.
    
    Args:
        filepath: Path to configuration file (.json or .yaml)
        
    Returns:
        Loaded RCANetConfig
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
    if filepath.suffix.lower() == '.json':
        return RCANetConfig.from_json(str(filepath))
    elif filepath.suffix.lower() in ['.yaml', '.yml']:
        return RCANetConfig.from_yaml(str(filepath))
    else:
        raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")


def save_config(config: RCANetConfig, filepath: str):
    """Save configuration to file.
    
    Args:
        config: Configuration to save
        filepath: Path to save configuration file (.json or .yaml)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix.lower() == '.json':
        config.to_json(str(filepath))
    elif filepath.suffix.lower() in ['.yaml', '.yml']:
        config.to_yaml(str(filepath))
    else:
        raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")