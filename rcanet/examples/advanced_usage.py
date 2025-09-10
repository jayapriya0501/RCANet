"""Advanced usage example for RCANet.

This script demonstrates advanced features including:
- Custom configurations and hyperparameter tuning
- Real dataset loading and preprocessing
- Advanced training techniques
- Model interpretation and analysis
- Comparison with baseline models
"""

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
from typing import Dict, List, Tuple, Any

# RCANet imports
from rcanet.models import RCANet
from rcanet.data import TabularDataset, TabularPreprocessor, create_data_loaders
from rcanet.training import RCANetTrainer, create_optimizer, create_scheduler
from rcanet.utils import RCANetConfig, compute_metrics, MetricsTracker


class RCANetExperiment:
    """Class for running comprehensive RCANet experiments."""
    
    def __init__(self, dataset_name: str, task_type: str):
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.results = {}
        self.models = {}
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and return dataset."""
        print(f"Loading {self.dataset_name} dataset...")
        
        if self.dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            
        elif self.dataset_name == 'wine':
            data = load_wine()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            
        elif self.dataset_name == 'california_housing':
            data = fetch_california_housing()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        print(f"Dataset shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Features: {len(feature_names)}")
        
        if self.task_type == 'classification':
            print(f"Classes: {len(np.unique(y))}")
            print(f"Class distribution: {np.bincount(y)}")
        else:
            print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
            
        return X, y, feature_names
    
    def create_advanced_config(self, input_dim: int, output_dim: int) -> RCANetConfig:
        """Create advanced configuration with optimized hyperparameters."""
        if self.task_type == 'classification':
            config = RCANetConfig(
                # Model architecture
                input_dim=input_dim,
                hidden_dim=256,
                num_heads=8,
                num_layers=4,
                output_dim=output_dim,
                
                # Task configuration
                task_type='classification',
                
                # Attention configuration
                attention_dropout=0.1,
                fusion_strategy='cross_attention',
                use_positional_encoding=True,
                
                # Training configuration
                learning_rate=0.0005,
                weight_decay=0.01,
                batch_size=64,
                num_epochs=200,
                
                # Regularization
                dropout=0.15,
                layer_norm=True,
                
                # Contrastive learning
                use_contrastive=True,
                contrastive_weight=0.2,
                contrastive_temperature=0.1,
                
                # Optimization
                optimizer='adamw',
                scheduler='cosine',
                warmup_steps=100,
                
                # Early stopping
                early_stopping=True,
                patience=20,
                min_delta=0.001
            )
        else:  # regression
            config = RCANetConfig(
                # Model architecture
                input_dim=input_dim,
                hidden_dim=128,
                num_heads=6,
                num_layers=3,
                output_dim=1,
                
                # Task configuration
                task_type='regression',
                
                # Attention configuration
                attention_dropout=0.05,
                fusion_strategy='gated',
                use_positional_encoding=False,
                
                # Training configuration
                learning_rate=0.001,
                weight_decay=0.005,
                batch_size=32,
                num_epochs=150,
                
                # Regularization
                dropout=0.1,
                layer_norm=True,
                
                # No contrastive learning for regression
                use_contrastive=False,
                
                # Optimization
                optimizer='adam',
                scheduler='plateau',
                
                # Early stopping
                early_stopping=True,
                patience=15,
                min_delta=0.0001
            )
            
        return config
    
    def train_baseline_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train baseline models for comparison."""
        print("\nTraining baseline models...")
        
        baselines = {}
        
        if self.task_type == 'classification':
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            start_time = time.time()
            rf.fit(X_train, y_train)
            rf_time = time.time() - start_time
            rf_pred = rf.predict(X_test)
            rf_score = rf.score(X_test, y_test)
            
            baselines['RandomForest'] = {
                'model': rf,
                'predictions': rf_pred,
                'accuracy': rf_score,
                'training_time': rf_time
            }
            
            # Logistic Regression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            start_time = time.time()
            lr.fit(X_train, y_train)
            lr_time = time.time() - start_time
            lr_pred = lr.predict(X_test)
            lr_score = lr.score(X_test, y_test)
            
            baselines['LogisticRegression'] = {
                'model': lr,
                'predictions': lr_pred,
                'accuracy': lr_score,
                'training_time': lr_time
            }
            
        else:  # regression
            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            start_time = time.time()
            rf.fit(X_train, y_train)
            rf_time = time.time() - start_time
            rf_pred = rf.predict(X_test)
            rf_score = rf.score(X_test, y_test)  # R²
            
            baselines['RandomForest'] = {
                'model': rf,
                'predictions': rf_pred,
                'r2_score': rf_score,
                'training_time': rf_time
            }
            
            # Linear Regression
            lr = LinearRegression()
            start_time = time.time()
            lr.fit(X_train, y_train)
            lr_time = time.time() - start_time
            lr_pred = lr.predict(X_test)
            lr_score = lr.score(X_test, y_test)  # R²
            
            baselines['LinearRegression'] = {
                'model': lr,
                'predictions': lr_pred,
                'r2_score': lr_score,
                'training_time': lr_time
            }
            
        return baselines
    
    def train_rcanet_with_cv(self, X: np.ndarray, y: np.ndarray, 
                           feature_names: List[str], n_folds: int = 5) -> Dict[str, Any]:
        """Train RCANet with cross-validation."""
        print(f"\nTraining RCANet with {n_folds}-fold cross-validation...")
        
        # Create cross-validation splits
        if self.task_type == 'classification':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
        cv_results = {
            'fold_scores': [],
            'fold_losses': [],
            'fold_times': [],
            'models': [],
            'histories': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create configuration
            output_dim = len(np.unique(y)) if self.task_type == 'classification' else 1
            config = self.create_advanced_config(X.shape[1], output_dim)
            
            # Reduce epochs for CV
            config.num_epochs = 100
            config.patience = 10
            
            # Preprocess data
            preprocessor = TabularPreprocessor(
                numerical_features=list(range(len(feature_names))),
                categorical_features=[],
                scale_features=True,
                handle_missing='mean'
            )
            
            X_train_processed = preprocessor.fit_transform(X_train_fold)
            X_val_processed = preprocessor.transform(X_val_fold)
            
            # Create datasets and loaders
            train_dataset = TabularDataset(
                X_train_processed, y_train_fold,
                feature_names=feature_names,
                task_type=self.task_type
            )
            
            val_dataset = TabularDataset(
                X_val_processed, y_val_fold,
                feature_names=feature_names,
                task_type=self.task_type
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=config.batch_size, shuffle=False
            )
            
            # Create and train model
            model = RCANet(config)
            trainer = RCANetTrainer(
                model=model,
                config=config,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            start_time = time.time()
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config.num_epochs
            )
            training_time = time.time() - start_time
            
            # Evaluate
            val_metrics = trainer.evaluate(val_loader)
            
            # Store results
            if self.task_type == 'classification':
                cv_results['fold_scores'].append(val_metrics['accuracy'])
            else:
                cv_results['fold_scores'].append(val_metrics['r2_score'])
                
            cv_results['fold_losses'].append(val_metrics['loss'])
            cv_results['fold_times'].append(training_time)
            cv_results['models'].append(model)
            cv_results['histories'].append(history)
            
            print(f"Fold {fold + 1} completed in {training_time:.2f}s")
            
        # Compute CV statistics
        cv_results['mean_score'] = np.mean(cv_results['fold_scores'])
        cv_results['std_score'] = np.std(cv_results['fold_scores'])
        cv_results['mean_loss'] = np.mean(cv_results['fold_losses'])
        cv_results['std_loss'] = np.std(cv_results['fold_losses'])
        cv_results['mean_time'] = np.mean(cv_results['fold_times'])
        
        score_name = 'Accuracy' if self.task_type == 'classification' else 'R² Score'
        print(f"\nCross-validation results:")
        print(f"{score_name}: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        print(f"Loss: {cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}")
        print(f"Training time: {cv_results['mean_time']:.2f}s ± {np.std(cv_results['fold_times']):.2f}s")
        
        return cv_results
    
    def analyze_feature_importance(self, model: RCANet, trainer: RCANetTrainer,
                                 data_loader: torch.utils.data.DataLoader,
                                 feature_names: List[str]) -> Dict[str, float]:
        """Analyze feature importance using attention weights."""
        print("\nAnalyzing feature importance...")
        
        # Get attention weights
        _, _, attention_weights = trainer.predict(data_loader, return_attention=True)
        
        if attention_weights is None:
            print("No attention weights available.")
            return {}
            
        # Extract attention weights
        if isinstance(attention_weights, dict):
            row_attention = attention_weights.get('row_attention')
            col_attention = attention_weights.get('col_attention')
        else:
            row_attention = attention_weights
            col_attention = None
            
        # Compute feature importance from attention
        feature_importance = {}
        
        if row_attention is not None:
            # Average attention across samples, heads, and sequence positions
            row_importance = torch.mean(row_attention, dim=(0, 1, 2)).cpu().numpy()
            
            for i, feature_name in enumerate(feature_names):
                if i < len(row_importance):
                    feature_importance[feature_name] = float(row_importance[i])
                    
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        # Print top features
        print("Top 10 most important features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            print(f"{i+1:2d}. {feature}: {importance:.4f}")
            
        return feature_importance
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run complete experiment."""
        print(f"\n{'='*60}")
        print(f"Running RCANet Advanced Experiment")
        print(f"Dataset: {self.dataset_name}")
        print(f"Task: {self.task_type}")
        print(f"{'='*60}")
        
        # Load dataset
        X, y, feature_names = self.load_dataset()
        
        # Train-test split
        test_size = 0.2
        if self.task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
        # Train baseline models
        baselines = self.train_baseline_models(X_train, X_test, y_train, y_test)
        
        # Train RCANet with cross-validation
        cv_results = self.train_rcanet_with_cv(X_train, y_train, feature_names)
        
        # Train final RCANet model on full training set
        print("\nTraining final RCANet model...")
        output_dim = len(np.unique(y)) if self.task_type == 'classification' else 1
        config = self.create_advanced_config(X.shape[1], output_dim)
        
        # Preprocess data
        preprocessor = TabularPreprocessor(
            numerical_features=list(range(len(feature_names))),
            categorical_features=[],
            scale_features=True,
            handle_missing='mean'
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Create datasets and loaders
        train_dataset = TabularDataset(
            X_train_processed, y_train,
            feature_names=feature_names,
            task_type=self.task_type
        )
        
        test_dataset = TabularDataset(
            X_test_processed, y_test,
            feature_names=feature_names,
            task_type=self.task_type
        )
        
        train_loader, val_loader = create_data_loaders(
            train_dataset,
            batch_size=config.batch_size,
            validation_split=0.2,
            shuffle=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False
        )
        
        # Create and train final model
        final_model = RCANet(config)
        final_trainer = RCANetTrainer(
            model=final_model,
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        start_time = time.time()
        final_history = final_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs
        )
        final_training_time = time.time() - start_time
        
        # Evaluate final model
        test_metrics = final_trainer.evaluate(test_loader)
        predictions, targets, attention_weights = final_trainer.predict(
            test_loader, return_attention=True
        )
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance(
            final_model, final_trainer, test_loader, feature_names
        )
        
        # Compile results
        results = {
            'dataset': self.dataset_name,
            'task_type': self.task_type,
            'baselines': baselines,
            'cv_results': cv_results,
            'final_model': {
                'model': final_model,
                'trainer': final_trainer,
                'config': config,
                'history': final_history,
                'test_metrics': test_metrics,
                'training_time': final_training_time,
                'predictions': predictions,
                'targets': targets,
                'attention_weights': attention_weights
            },
            'feature_importance': feature_importance,
            'preprocessor': preprocessor
        }
        
        # Print final comparison
        self.print_final_comparison(results)
        
        return results
    
    def print_final_comparison(self, results: Dict[str, Any]):
        """Print final model comparison."""
        print(f"\n{'='*60}")
        print("FINAL RESULTS COMPARISON")
        print(f"{'='*60}")
        
        if self.task_type == 'classification':
            metric_name = 'Accuracy'
            rcanet_score = results['final_model']['test_metrics']['accuracy']
            cv_score = results['cv_results']['mean_score']
        else:
            metric_name = 'R² Score'
            rcanet_score = results['final_model']['test_metrics']['r2_score']
            cv_score = results['cv_results']['mean_score']
            
        print(f"\nRCANet Results:")
        print(f"Cross-validation {metric_name}: {cv_score:.4f} ± {results['cv_results']['std_score']:.4f}")
        print(f"Test {metric_name}: {rcanet_score:.4f}")
        print(f"Training time: {results['final_model']['training_time']:.2f}s")
        
        print(f"\nBaseline Results:")
        for name, baseline in results['baselines'].items():
            if self.task_type == 'classification':
                score = baseline['accuracy']
            else:
                score = baseline['r2_score']
            print(f"{name} {metric_name}: {score:.4f} (time: {baseline['training_time']:.2f}s)")
            
        # Determine best model
        best_baseline_score = max(
            baseline['accuracy' if self.task_type == 'classification' else 'r2_score']
            for baseline in results['baselines'].values()
        )
        
        improvement = rcanet_score - best_baseline_score
        print(f"\nRCANet improvement over best baseline: {improvement:+.4f}")
        
        if improvement > 0:
            print("✅ RCANet outperforms baseline models!")
        else:
            print("❌ RCANet underperforms compared to baselines.")


def run_classification_experiment():
    """Run classification experiments."""
    datasets = ['breast_cancer', 'wine']
    
    for dataset in datasets:
        try:
            experiment = RCANetExperiment(dataset, 'classification')
            results = experiment.run_experiment()
            
            # Save results (optional)
            # torch.save(results, f'rcanet_{dataset}_results.pt')
            
        except Exception as e:
            print(f"Error in {dataset} experiment: {e}")
            import traceback
            traceback.print_exc()


def run_regression_experiment():
    """Run regression experiments."""
    datasets = ['california_housing']
    
    for dataset in datasets:
        try:
            experiment = RCANetExperiment(dataset, 'regression')
            results = experiment.run_experiment()
            
            # Save results (optional)
            # torch.save(results, f'rcanet_{dataset}_results.pt')
            
        except Exception as e:
            print(f"Error in {dataset} experiment: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run advanced usage examples."""
    print("RCANet Advanced Usage Examples")
    print("==============================\n")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Running classification experiments...")
    run_classification_experiment()
    
    print("\nRunning regression experiments...")
    run_regression_experiment()
    
    print("\n=== Advanced Examples Complete ===")
    print("Results include cross-validation, baseline comparisons, and feature importance analysis.")


if __name__ == "__main__":
    main()