"""Dataset classes for RCANet tabular data loading.

Provides dataset classes for loading and preprocessing tabular data
for training and evaluation with RCANet models.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings


class TabularDataset(Dataset):
    """Dataset class for tabular data.
    
    Handles loading, preprocessing, and batching of tabular data for RCANet.
    Supports both numerical and categorical features with automatic preprocessing.
    
    Args:
        data: Input data as DataFrame, numpy array, or file path
        target_column: Name or index of target column
        categorical_columns: List of categorical column names/indices
        numerical_columns: List of numerical column names/indices
        feature_types: Optional mapping of features to types
        transform: Whether to apply preprocessing transforms
        scaler: Scaler for numerical features
        label_encoder: Encoder for categorical features
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray, str],
        target_column: Union[str, int],
        categorical_columns: Optional[List[Union[str, int]]] = None,
        numerical_columns: Optional[List[Union[str, int]]] = None,
        feature_types: Optional[Dict[str, str]] = None,
        transform: bool = True,
        scaler: Optional[Any] = None,
        label_encoders: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        sample_strategy: str = 'random'  # 'random', 'stratified', 'sequential'
    ):
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.feature_types = feature_types or {}
        self.transform = transform
        self.max_samples = max_samples
        self.sample_strategy = sample_strategy
        
        # Load data
        self.data = self._load_data(data)
        
        # Separate features and targets
        self.features, self.targets = self._separate_features_targets()
        
        # Auto-detect column types if not provided
        if not self.categorical_columns and not self.numerical_columns:
            self._auto_detect_column_types()
            
        # Apply sampling if specified
        if max_samples and len(self.data) > max_samples:
            self._apply_sampling()
            
        # Initialize preprocessors
        self.scaler = scaler or StandardScaler()
        self.label_encoders = label_encoders or {}
        
        # Preprocess data
        if transform:
            self._preprocess_data()
            
        # Convert to tensors
        self._convert_to_tensors()
        
    def _load_data(self, data: Union[pd.DataFrame, np.ndarray, str]) -> pd.DataFrame:
        """Load data from various sources."""
        if isinstance(data, str):
            # Load from file
            if data.endswith('.csv'):
                return pd.read_csv(data)
            elif data.endswith('.parquet'):
                return pd.read_parquet(data)
            elif data.endswith(('.xlsx', '.xls')):
                return pd.read_excel(data)
            else:
                raise ValueError(f"Unsupported file format: {data}")
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
    def _separate_features_targets(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and targets."""
        if isinstance(self.target_column, str):
            targets = self.data[self.target_column]
            features = self.data.drop(columns=[self.target_column])
        else:
            targets = self.data.iloc[:, self.target_column]
            features = self.data.drop(self.data.columns[self.target_column], axis=1)
            
        return features, targets
        
    def _auto_detect_column_types(self):
        """Automatically detect categorical and numerical columns."""
        for col in self.features.columns:
            if self.features[col].dtype in ['object', 'category']:
                self.categorical_columns.append(col)
            elif self.features[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self.numerical_columns.append(col)
                
    def _apply_sampling(self):
        """Apply sampling strategy to reduce dataset size."""
        if self.sample_strategy == 'random':
            sampled_indices = np.random.choice(
                len(self.data), self.max_samples, replace=False
            )
        elif self.sample_strategy == 'stratified':
            _, sampled_indices = train_test_split(
                range(len(self.data)),
                train_size=self.max_samples,
                stratify=self.targets,
                random_state=42
            )
        elif self.sample_strategy == 'sequential':
            sampled_indices = list(range(self.max_samples))
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sample_strategy}")
            
        self.features = self.features.iloc[sampled_indices].reset_index(drop=True)
        self.targets = self.targets.iloc[sampled_indices].reset_index(drop=True)
        
    def _preprocess_data(self):
        """Preprocess numerical and categorical features."""
        # Handle missing values
        self._handle_missing_values()
        
        # Encode categorical features
        self._encode_categorical_features()
        
        # Scale numerical features
        self._scale_numerical_features()
        
        # Create feature type mapping
        self._create_feature_type_mapping()
        
    def _handle_missing_values(self):
        """Handle missing values in the dataset."""
        # Fill numerical missing values with median
        for col in self.numerical_columns:
            if col in self.features.columns:
                self.features[col].fillna(self.features[col].median(), inplace=True)
                
        # Fill categorical missing values with mode
        for col in self.categorical_columns:
            if col in self.features.columns:
                mode_value = self.features[col].mode()
                if len(mode_value) > 0:
                    self.features[col].fillna(mode_value[0], inplace=True)
                else:
                    self.features[col].fillna('unknown', inplace=True)
                    
    def _encode_categorical_features(self):
        """Encode categorical features."""
        for col in self.categorical_columns:
            if col in self.features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    self.features[col] = self.label_encoders[col].fit_transform(
                        self.features[col].astype(str)
                    )
                else:
                    # Transform using existing encoder
                    try:
                        self.features[col] = self.label_encoders[col].transform(
                            self.features[col].astype(str)
                        )
                    except ValueError:
                        # Handle unseen categories
                        warnings.warn(f"Unseen categories in column {col}")
                        self.features[col] = self.label_encoders[col].transform(
                            self.features[col].astype(str).fillna('unknown')
                        )
                        
    def _scale_numerical_features(self):
        """Scale numerical features."""
        if self.numerical_columns:
            numerical_data = self.features[self.numerical_columns]
            
            if not hasattr(self.scaler, 'scale_'):
                # Fit scaler
                scaled_data = self.scaler.fit_transform(numerical_data)
            else:
                # Transform using existing scaler
                scaled_data = self.scaler.transform(numerical_data)
                
            self.features[self.numerical_columns] = scaled_data
            
    def _create_feature_type_mapping(self):
        """Create mapping of features to their types."""
        self.feature_type_indices = {}
        type_counter = 0
        
        for col in self.features.columns:
            if col in self.categorical_columns:
                self.feature_type_indices[col] = 0  # Categorical type
            elif col in self.numerical_columns:
                self.feature_type_indices[col] = 1  # Numerical type
            else:
                self.feature_type_indices[col] = 2  # Unknown type
                
    def _convert_to_tensors(self):
        """Convert data to PyTorch tensors."""
        # Convert features to tensor
        self.feature_tensor = torch.FloatTensor(self.features.values)
        
        # Convert targets to tensor
        if hasattr(self.targets, 'dtype') and self.targets.dtype in ['object', 'category']:
            # Categorical targets
            if not hasattr(self, 'target_encoder'):
                self.target_encoder = LabelEncoder()
                encoded_targets = self.target_encoder.fit_transform(self.targets)
            else:
                encoded_targets = self.target_encoder.transform(self.targets)
            self.target_tensor = torch.LongTensor(encoded_targets)
        else:
            # Numerical targets
            self.target_tensor = torch.FloatTensor(self.targets.values)
            
        # Create feature type tensor
        feature_types = [self.feature_type_indices[col] for col in self.features.columns]
        self.feature_type_tensor = torch.LongTensor(feature_types)
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.feature_tensor)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        return {
            'features': self.feature_tensor[idx],
            'targets': self.target_tensor[idx],
            'feature_types': self.feature_type_tensor,
            'index': torch.tensor(idx)
        }
        
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about features."""
        return {
            'n_features': len(self.features.columns),
            'feature_names': list(self.features.columns),
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_types': self.feature_type_indices,
            'n_classes': len(np.unique(self.target_tensor)) if self.target_tensor.dtype == torch.long else 1
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'n_samples': len(self),
            'n_features': self.feature_tensor.shape[1],
            'target_distribution': {}
        }
        
        if self.target_tensor.dtype == torch.long:
            # Classification statistics
            unique, counts = torch.unique(self.target_tensor, return_counts=True)
            stats['target_distribution'] = {
                f'class_{i}': count.item() for i, count in zip(unique, counts)
            }
        else:
            # Regression statistics
            stats['target_distribution'] = {
                'mean': self.target_tensor.mean().item(),
                'std': self.target_tensor.std().item(),
                'min': self.target_tensor.min().item(),
                'max': self.target_tensor.max().item()
            }
            
        return stats


class MultiTableDataset(Dataset):
    """Dataset for handling multiple related tables.
    
    Useful for scenarios where tabular data comes from multiple related sources
    that need to be processed together.
    """
    
    def __init__(
        self,
        tables: Dict[str, Union[pd.DataFrame, str]],
        target_table: str,
        target_column: str,
        join_keys: Dict[str, str],
        **kwargs
    ):
        self.tables = {}
        self.target_table = target_table
        self.target_column = target_column
        self.join_keys = join_keys
        
        # Load all tables
        for name, table in tables.items():
            if isinstance(table, str):
                self.tables[name] = pd.read_csv(table)
            else:
                self.tables[name] = table.copy()
                
        # Join tables
        self.merged_data = self._join_tables()
        
        # Create single table dataset
        self.dataset = TabularDataset(
            data=self.merged_data,
            target_column=target_column,
            **kwargs
        )
        
    def _join_tables(self) -> pd.DataFrame:
        """Join multiple tables based on join keys."""
        result = self.tables[self.target_table]
        
        for table_name, join_key in self.join_keys.items():
            if table_name != self.target_table:
                result = result.merge(
                    self.tables[table_name],
                    on=join_key,
                    how='left'
                )
                
        return result
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]
        
    def get_feature_info(self) -> Dict[str, Any]:
        return self.dataset.get_feature_info()
        
    def get_statistics(self) -> Dict[str, Any]:
        return self.dataset.get_statistics()


def create_data_loaders(
    dataset: TabularDataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    stratify: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """Create train/validation/test data loaders.
    
    Args:
        dataset: TabularDataset instance
        batch_size: Batch size for data loaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        shuffle: Whether to shuffle data
        stratify: Whether to stratify splits
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Get indices
    indices = list(range(len(dataset)))
    
    if stratify and dataset.target_tensor.dtype == torch.long:
        # Stratified split for classification
        targets = dataset.target_tensor.numpy()
        
        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_split,
            stratify=targets,
            random_state=42
        )
        
        # Second split: val vs test
        if test_split > 0:
            val_size = val_split / (val_split + test_split)
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_size,
                stratify=targets[temp_indices],
                random_state=42
            )
        else:
            val_indices = temp_indices
            test_indices = []
    else:
        # Random split
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_end = int(train_split * len(indices))
        val_end = int((train_split + val_split) * len(indices))
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:] if test_split > 0 else []
        
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    data_loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    if test_indices:
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        data_loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
    return data_loaders


def collate_tabular_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for tabular data batches.
    
    Handles variable-length sequences and different data types.
    """
    # Stack features and targets
    features = torch.stack([item['features'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    indices = torch.stack([item['index'] for item in batch])
    
    # Feature types should be the same for all items
    feature_types = batch[0]['feature_types']
    
    return {
        'features': features,
        'targets': targets,
        'feature_types': feature_types,
        'indices': indices,
        'batch_size': len(batch)
    }