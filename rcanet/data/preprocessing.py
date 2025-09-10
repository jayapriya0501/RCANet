"""Preprocessing utilities for tabular data.

Provides comprehensive preprocessing capabilities for tabular data
including feature engineering, normalization, and data augmentation.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import warnings


class TabularPreprocessor:
    """Comprehensive preprocessor for tabular data.
    
    Handles missing values, feature scaling, encoding, feature selection,
    and data augmentation for tabular datasets.
    
    Args:
        numerical_strategy: Strategy for numerical feature preprocessing
        categorical_strategy: Strategy for categorical feature preprocessing
        missing_value_strategy: Strategy for handling missing values
        feature_selection: Whether to apply feature selection
        dimensionality_reduction: Whether to apply dimensionality reduction
        augmentation: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        numerical_strategy: str = 'standard',  # 'standard', 'minmax', 'robust', 'quantile'
        categorical_strategy: str = 'label',   # 'label', 'onehot', 'ordinal'
        missing_value_strategy: str = 'median', # 'mean', 'median', 'mode', 'knn', 'drop'
        feature_selection: Optional[str] = None, # 'kbest', 'mutual_info', None
        dimensionality_reduction: Optional[str] = None, # 'pca', None
        augmentation: bool = False,
        random_state: int = 42
    ):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.missing_value_strategy = missing_value_strategy
        self.feature_selection = feature_selection
        self.dimensionality_reduction = dimensionality_reduction
        self.augmentation = augmentation
        self.random_state = random_state
        
        # Initialize preprocessors
        self.numerical_scaler = None
        self.categorical_encoders = {}
        self.missing_value_imputer = None
        self.feature_selector = None
        self.dim_reducer = None
        
        # Store column information
        self.numerical_columns = []
        self.categorical_columns = []
        self.feature_names = []
        self.is_fitted = False
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ) -> 'TabularPreprocessor':
        """Fit the preprocessor on training data.
        
        Args:
            X: Input features
            y: Target values (optional, needed for supervised feature selection)
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            
        Returns:
            Self for method chaining
        """
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
        self.feature_names = list(X.columns)
        
        # Auto-detect column types if not provided
        if categorical_columns is None or numerical_columns is None:
            cat_cols, num_cols = self._auto_detect_column_types(X)
            self.categorical_columns = categorical_columns or cat_cols
            self.numerical_columns = numerical_columns or num_cols
        else:
            self.categorical_columns = categorical_columns
            self.numerical_columns = numerical_columns
            
        # Fit missing value imputer
        self._fit_missing_value_imputer(X)
        
        # Apply missing value imputation
        X_imputed = self._apply_missing_value_imputation(X)
        
        # Fit categorical encoders
        self._fit_categorical_encoders(X_imputed)
        
        # Apply categorical encoding
        X_encoded = self._apply_categorical_encoding(X_imputed)
        
        # Fit numerical scaler
        self._fit_numerical_scaler(X_encoded)
        
        # Apply numerical scaling
        X_scaled = self._apply_numerical_scaling(X_encoded)
        
        # Fit feature selector
        if self.feature_selection:
            self._fit_feature_selector(X_scaled, y)
            X_selected = self._apply_feature_selection(X_scaled)
        else:
            X_selected = X_scaled
            
        # Fit dimensionality reducer
        if self.dimensionality_reduction:
            self._fit_dimensionality_reducer(X_selected)
            
        self.is_fitted = True
        return self
        
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data using fitted preprocessors.
        
        Args:
            X: Input features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        # Apply transformations in order
        X_transformed = self._apply_missing_value_imputation(X)
        X_transformed = self._apply_categorical_encoding(X_transformed)
        X_transformed = self._apply_numerical_scaling(X_transformed)
        
        if self.feature_selection:
            X_transformed = self._apply_feature_selection(X_transformed)
            
        if self.dimensionality_reduction:
            X_transformed = self._apply_dimensionality_reduction(X_transformed)
            
        return X_transformed
        
    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Fit preprocessor and transform data."""
        return self.fit(X, y, **kwargs).transform(X)
        
    def _auto_detect_column_types(
        self,
        X: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Automatically detect categorical and numerical columns."""
        categorical_columns = []
        numerical_columns = []
        
        for col in X.columns:
            if X[col].dtype in ['object', 'category', 'bool']:
                categorical_columns.append(col)
            elif X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Check if it's actually categorical (low cardinality)
                unique_values = X[col].nunique()
                if unique_values <= 10 and X[col].dtype in ['int64', 'int32']:
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
                    
        return categorical_columns, numerical_columns
        
    def _fit_missing_value_imputer(self, X: pd.DataFrame):
        """Fit missing value imputer."""
        if self.missing_value_strategy == 'drop':
            return  # No imputer needed
            
        if self.missing_value_strategy in ['mean', 'median', 'mode']:
            strategy = 'most_frequent' if self.missing_value_strategy == 'mode' else self.missing_value_strategy
            self.missing_value_imputer = SimpleImputer(strategy=strategy)
        elif self.missing_value_strategy == 'knn':
            self.missing_value_imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Unknown missing value strategy: {self.missing_value_strategy}")
            
        # Fit on numerical columns only for mean/median
        if self.missing_value_strategy in ['mean', 'median'] and self.numerical_columns:
            self.missing_value_imputer.fit(X[self.numerical_columns])
        elif self.missing_value_strategy == 'knn':
            # For KNN, we need to encode categorical variables first
            X_temp = X.copy()
            for col in self.categorical_columns:
                if col in X_temp.columns:
                    le = LabelEncoder()
                    X_temp[col] = le.fit_transform(X_temp[col].astype(str))
            self.missing_value_imputer.fit(X_temp)
            
    def _apply_missing_value_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value imputation."""
        if self.missing_value_strategy == 'drop':
            return X.dropna()
            
        X_imputed = X.copy()
        
        if self.missing_value_strategy in ['mean', 'median']:
            if self.numerical_columns:
                X_imputed[self.numerical_columns] = self.missing_value_imputer.transform(
                    X[self.numerical_columns]
                )
            # Handle categorical columns separately
            for col in self.categorical_columns:
                if col in X_imputed.columns:
                    mode_value = X_imputed[col].mode()
                    if len(mode_value) > 0:
                        X_imputed[col].fillna(mode_value[0], inplace=True)
                        
        elif self.missing_value_strategy == 'mode':
            for col in X.columns:
                mode_value = X[col].mode()
                if len(mode_value) > 0:
                    X_imputed[col].fillna(mode_value[0], inplace=True)
                    
        elif self.missing_value_strategy == 'knn':
            # Encode categorical variables temporarily
            temp_encoders = {}
            for col in self.categorical_columns:
                if col in X_imputed.columns:
                    temp_encoders[col] = LabelEncoder()
                    X_imputed[col] = temp_encoders[col].fit_transform(X_imputed[col].astype(str))
                    
            # Apply KNN imputation
            X_imputed_values = self.missing_value_imputer.transform(X_imputed)
            X_imputed = pd.DataFrame(X_imputed_values, columns=X.columns, index=X.index)
            
            # Decode categorical variables
            for col, encoder in temp_encoders.items():
                X_imputed[col] = encoder.inverse_transform(X_imputed[col].astype(int))
                
        return X_imputed
        
    def _fit_categorical_encoders(self, X: pd.DataFrame):
        """Fit categorical encoders."""
        for col in self.categorical_columns:
            if col in X.columns:
                if self.categorical_strategy == 'label':
                    encoder = LabelEncoder()
                elif self.categorical_strategy == 'onehot':
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                elif self.categorical_strategy == 'ordinal':
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                else:
                    raise ValueError(f"Unknown categorical strategy: {self.categorical_strategy}")
                    
                encoder.fit(X[col].astype(str).values.reshape(-1, 1))
                self.categorical_encoders[col] = encoder
                
    def _apply_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding."""
        X_encoded = X.copy()
        
        for col in self.categorical_columns:
            if col in X.columns and col in self.categorical_encoders:
                encoder = self.categorical_encoders[col]
                
                if self.categorical_strategy == 'label':
                    try:
                        X_encoded[col] = encoder.transform(X[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        warnings.warn(f"Unseen categories in column {col}")
                        # Fill unseen categories with most frequent class
                        most_frequent = encoder.classes_[0]
                        X_temp = X[col].astype(str).copy()
                        mask = ~X_temp.isin(encoder.classes_)
                        X_temp[mask] = most_frequent
                        X_encoded[col] = encoder.transform(X_temp)
                        
                elif self.categorical_strategy == 'onehot':
                    encoded_values = encoder.transform(X[col].astype(str).values.reshape(-1, 1))
                    feature_names = [f"{col}_{cls}" for cls in encoder.categories_[0]]
                    
                    # Add one-hot encoded columns
                    for i, feature_name in enumerate(feature_names):
                        X_encoded[feature_name] = encoded_values[:, i]
                        
                    # Remove original column
                    X_encoded.drop(columns=[col], inplace=True)
                    
                elif self.categorical_strategy == 'ordinal':
                    X_encoded[col] = encoder.transform(X[col].astype(str).values.reshape(-1, 1)).flatten()
                    
        return X_encoded
        
    def _fit_numerical_scaler(self, X: pd.DataFrame):
        """Fit numerical scaler."""
        # Get numerical columns (may have changed after encoding)
        current_numerical_cols = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                current_numerical_cols.append(col)
                
        if not current_numerical_cols:
            return
            
        if self.numerical_strategy == 'standard':
            self.numerical_scaler = StandardScaler()
        elif self.numerical_strategy == 'minmax':
            self.numerical_scaler = MinMaxScaler()
        elif self.numerical_strategy == 'robust':
            self.numerical_scaler = RobustScaler()
        elif self.numerical_strategy == 'quantile':
            self.numerical_scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown numerical strategy: {self.numerical_strategy}")
            
        self.numerical_scaler.fit(X[current_numerical_cols])
        self.current_numerical_columns = current_numerical_cols
        
    def _apply_numerical_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply numerical scaling."""
        if self.numerical_scaler is None:
            return X
            
        X_scaled = X.copy()
        if hasattr(self, 'current_numerical_columns') and self.current_numerical_columns:
            X_scaled[self.current_numerical_columns] = self.numerical_scaler.transform(
                X[self.current_numerical_columns]
            )
            
        return X_scaled
        
    def _fit_feature_selector(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]):
        """Fit feature selector."""
        if y is None:
            warnings.warn("No target provided for feature selection, skipping")
            return
            
        if self.feature_selection == 'kbest':
            # Determine score function based on target type
            if pd.api.types.is_numeric_dtype(y):
                score_func = f_regression
            else:
                score_func = f_classif
                
            k = min(50, X.shape[1])  # Select top 50 features or all if less
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            
        elif self.feature_selection == 'mutual_info':
            if pd.api.types.is_numeric_dtype(y):
                warnings.warn("Mutual info feature selection not implemented for regression")
                return
            else:
                k = min(50, X.shape[1])
                self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
                
        if self.feature_selector:
            self.feature_selector.fit(X, y)
            
    def _apply_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection."""
        if self.feature_selector is None:
            return X
            
        X_selected = self.feature_selector.transform(X)
        
        # Get selected feature names
        selected_features = self.feature_selector.get_support()
        selected_feature_names = [name for name, selected in zip(X.columns, selected_features) if selected]
        
        return pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
        
    def _fit_dimensionality_reducer(self, X: pd.DataFrame):
        """Fit dimensionality reducer."""
        if self.dimensionality_reduction == 'pca':
            # Keep 95% of variance or max 100 components
            n_components = min(100, X.shape[1], int(0.95 * X.shape[1]))
            self.dim_reducer = PCA(n_components=n_components, random_state=self.random_state)
            self.dim_reducer.fit(X)
            
    def _apply_dimensionality_reduction(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply dimensionality reduction."""
        if self.dim_reducer is None:
            return X
            
        X_reduced = self.dim_reducer.transform(X)
        feature_names = [f'PC_{i+1}' for i in range(X_reduced.shape[1])]
        
        return pd.DataFrame(X_reduced, columns=feature_names, index=X.index)
        
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available."""
        if self.feature_selector and hasattr(self.feature_selector, 'scores_'):
            feature_names = self.feature_names
            if hasattr(self, 'current_numerical_columns'):
                # Account for one-hot encoding changes
                pass  # This would need more complex logic
                
            scores = self.feature_selector.scores_
            return dict(zip(feature_names[:len(scores)], scores))
            
        return None
        
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about preprocessing steps applied."""
        info = {
            'numerical_strategy': self.numerical_strategy,
            'categorical_strategy': self.categorical_strategy,
            'missing_value_strategy': self.missing_value_strategy,
            'feature_selection': self.feature_selection,
            'dimensionality_reduction': self.dimensionality_reduction,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'is_fitted': self.is_fitted
        }
        
        if self.dim_reducer and hasattr(self.dim_reducer, 'explained_variance_ratio_'):
            info['explained_variance_ratio'] = self.dim_reducer.explained_variance_ratio_.tolist()
            info['cumulative_variance_ratio'] = np.cumsum(self.dim_reducer.explained_variance_ratio_).tolist()
            
        return info


class TabularAugmenter:
    """Data augmentation for tabular data.
    
    Provides various augmentation techniques specifically designed for tabular data.
    """
    
    def __init__(
        self,
        noise_level: float = 0.01,
        swap_probability: float = 0.1,
        mixup_alpha: float = 0.2,
        cutmix_probability: float = 0.1
    ):
        self.noise_level = noise_level
        self.swap_probability = swap_probability
        self.mixup_alpha = mixup_alpha
        self.cutmix_probability = cutmix_probability
        
    def add_noise(
        self,
        X: torch.Tensor,
        categorical_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Add Gaussian noise to numerical features."""
        X_aug = X.clone()
        
        if categorical_mask is not None:
            # Only add noise to numerical features
            numerical_mask = ~categorical_mask
            noise = torch.randn_like(X_aug) * self.noise_level
            X_aug = X_aug + noise * numerical_mask.float()
        else:
            # Add noise to all features
            noise = torch.randn_like(X_aug) * self.noise_level
            X_aug = X_aug + noise
            
        return X_aug
        
    def feature_swap(
        self,
        X: torch.Tensor,
        categorical_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Randomly swap feature values between samples."""
        X_aug = X.clone()
        batch_size, n_features = X.shape
        
        for i in range(n_features):
            if categorical_mask is None or not categorical_mask[i]:
                # Only swap numerical features
                if torch.rand(1) < self.swap_probability:
                    # Randomly permute this feature across the batch
                    perm_indices = torch.randperm(batch_size)
                    X_aug[:, i] = X_aug[perm_indices, i]
                    
        return X_aug
        
    def mixup(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        alpha = alpha or self.mixup_alpha
        batch_size = X.shape[0]
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix inputs and targets
        X_mixed = lam * X + (1 - lam) * X[index]
        y_mixed = lam * y + (1 - lam) * y[index]
        
        return X_mixed, y_mixed, lam
        
    def cutmix(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation adapted for tabular data."""
        batch_size, n_features = X.shape
        
        if torch.rand(1) > self.cutmix_probability:
            return X, y, 1.0
            
        # Generate random lambda
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        rand_index = torch.randperm(batch_size)
        
        # Determine which features to cut
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(n_features * cut_ratio)
        
        # Random feature indices to cut
        cx = np.random.randint(n_features)
        bbx1 = np.clip(cx - cut_w // 2, 0, n_features)
        bbx2 = np.clip(cx + cut_w // 2, 0, n_features)
        
        # Apply cutmix
        X_cutmix = X.clone()
        X_cutmix[:, bbx1:bbx2] = X[rand_index, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) / n_features)
        
        return X_cutmix, y, lam
        
    def augment_batch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        categorical_mask: Optional[torch.Tensor] = None,
        augmentation_types: List[str] = ['noise', 'swap']
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multiple augmentation techniques to a batch."""
        X_aug = X.clone()
        y_aug = y.clone()
        
        for aug_type in augmentation_types:
            if aug_type == 'noise':
                X_aug = self.add_noise(X_aug, categorical_mask)
            elif aug_type == 'swap':
                X_aug = self.feature_swap(X_aug, categorical_mask)
            elif aug_type == 'mixup':
                X_aug, y_aug, _ = self.mixup(X_aug, y_aug)
            elif aug_type == 'cutmix':
                X_aug, y_aug, _ = self.cutmix(X_aug, y_aug)
                
        return X_aug, y_aug