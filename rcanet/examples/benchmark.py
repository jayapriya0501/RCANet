"""Benchmarking script for RCANet.

This script provides comprehensive benchmarking capabilities to compare
RCANet against other state-of-the-art tabular deep learning methods.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, fetch_california_housing,
    make_classification, make_regression
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# RCANet imports
from rcanet.models import RCANet
from rcanet.data import TabularDataset, TabularPreprocessor, create_data_loaders
from rcanet.training import RCANetTrainer
from rcanet.utils import RCANetConfig

warnings.filterwarnings('ignore')


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    dataset_name: str
    task_type: str
    train_time: float
    test_score: float
    test_loss: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None
    model_params: Optional[int] = None
    

class TabularBenchmark:
    """Comprehensive benchmarking suite for tabular deep learning models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = []
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
    def load_benchmark_datasets(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str], str]]:
        """Load all benchmark datasets."""
        datasets = {}
        
        # Classification datasets
        print("Loading classification datasets...")
        
        # Breast Cancer
        data = load_breast_cancer()
        datasets['breast_cancer'] = (
            data.data, data.target, list(data.feature_names), 'classification'
        )
        
        # Wine
        data = load_wine()
        datasets['wine'] = (
            data.data, data.target, list(data.feature_names), 'classification'
        )
        
        # Digits (reduced for speed)
        data = load_digits()
        datasets['digits'] = (
            data.data, data.target, [f'pixel_{i}' for i in range(data.data.shape[1])], 'classification'
        )
        
        # Synthetic classification
        X, y = make_classification(
            n_samples=2000, n_features=20, n_informative=15, n_redundant=5,
            n_classes=3, random_state=self.random_state
        )
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        datasets['synthetic_clf'] = (X, y, feature_names, 'classification')
        
        # Regression datasets
        print("Loading regression datasets...")
        
        # California Housing
        data = fetch_california_housing()
        datasets['california_housing'] = (
            data.data, data.target, list(data.feature_names), 'regression'
        )
        
        # Synthetic regression
        X, y = make_regression(
            n_samples=2000, n_features=15, n_informative=10, noise=0.1,
            random_state=self.random_state
        )
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        datasets['synthetic_reg'] = (X, y, feature_names, 'regression')
        
        print(f"Loaded {len(datasets)} datasets")
        return datasets
    
    def get_baseline_models(self, task_type: str) -> Dict[str, Any]:
        """Get baseline models for comparison."""
        if task_type == 'classification':
            return {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                'LogisticRegression': LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                'SVM': SVC(
                    random_state=self.random_state, probability=True
                ),
                'MLP': MLPClassifier(
                    hidden_layer_sizes=(128, 64), random_state=self.random_state,
                    max_iter=500, early_stopping=True
                )
            }
        else:  # regression
            return {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state
                ),
                'LinearRegression': LinearRegression(),
                'SVR': SVR(),
                'MLP': MLPRegressor(
                    hidden_layer_sizes=(128, 64), random_state=self.random_state,
                    max_iter=500, early_stopping=True
                )
            }
    
    def create_rcanet_config(self, input_dim: int, output_dim: int, 
                           task_type: str, dataset_size: int) -> RCANetConfig:
        """Create optimized RCANet configuration based on dataset characteristics."""
        
        # Adjust model size based on dataset size
        if dataset_size < 1000:
            hidden_dim = 64
            num_heads = 4
            num_layers = 2
            batch_size = 16
        elif dataset_size < 5000:
            hidden_dim = 128
            num_heads = 6
            num_layers = 3
            batch_size = 32
        else:
            hidden_dim = 256
            num_heads = 8
            num_layers = 4
            batch_size = 64
            
        config = RCANetConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            output_dim=output_dim,
            task_type=task_type,
            
            # Training configuration
            learning_rate=0.001,
            batch_size=batch_size,
            num_epochs=100,  # Reduced for benchmarking speed
            
            # Regularization
            dropout=0.1,
            weight_decay=0.01,
            
            # Contrastive learning (only for classification)
            use_contrastive=task_type == 'classification',
            contrastive_weight=0.1 if task_type == 'classification' else 0.0,
            
            # Early stopping
            early_stopping=True,
            patience=10,
            
            # Optimization
            optimizer='adam',
            scheduler='plateau'
        )
        
        return config
    
    def train_baseline_model(self, model, model_name: str, X_train: np.ndarray, 
                           X_test: np.ndarray, y_train: np.ndarray, 
                           y_test: np.ndarray, task_type: str) -> BenchmarkResult:
        """Train and evaluate a baseline model."""
        print(f"  Training {model_name}...")
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        if task_type == 'classification':
            test_score = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            additional_metrics = {'f1_score': f1}
            test_loss = None
        else:
            test_score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            additional_metrics = {'mse': mse, 'rmse': np.sqrt(mse)}
            test_loss = mse
            
        # Count parameters (approximate for sklearn models)
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            n_features = X_train.shape[1]
            
        if model_name == 'MLP':
            # Approximate parameter count for MLP
            if task_type == 'classification':
                n_outputs = len(np.unique(y_train))
            else:
                n_outputs = 1
            model_params = n_features * 128 + 128 * 64 + 64 * n_outputs
        else:
            model_params = None
            
        return BenchmarkResult(
            model_name=model_name,
            dataset_name="",  # Will be filled later
            task_type=task_type,
            train_time=train_time,
            test_score=test_score,
            test_loss=test_loss,
            additional_metrics=additional_metrics,
            model_params=model_params
        )
    
    def train_rcanet_model(self, X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray,
                         feature_names: List[str], task_type: str) -> BenchmarkResult:
        """Train and evaluate RCANet model."""
        print("  Training RCANet...")
        
        # Create configuration
        output_dim = len(np.unique(y_train)) if task_type == 'classification' else 1
        config = self.create_rcanet_config(
            X_train.shape[1], output_dim, task_type, X_train.shape[0]
        )
        
        # Preprocess data
        preprocessor = TabularPreprocessor(
            numerical_features=list(range(len(feature_names))),
            categorical_features=[],
            scale_features=True,
            handle_missing='mean'
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Create datasets
        train_dataset = TabularDataset(
            X_train_processed, y_train,
            feature_names=feature_names,
            task_type=task_type
        )
        
        test_dataset = TabularDataset(
            X_test_processed, y_test,
            feature_names=feature_names,
            task_type=task_type
        )
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_dataset,
            batch_size=config.batch_size,
            validation_split=0.2,
            shuffle=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # Create model and trainer
        model = RCANet(config)
        trainer = RCANetTrainer(
            model=model,
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Train model
        start_time = time.time()
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs
        )
        train_time = time.time() - start_time
        
        # Evaluate model
        test_metrics = trainer.evaluate(test_loader)
        
        # Get predictions for additional metrics
        predictions, targets, _ = trainer.predict(test_loader)
        
        # Compute metrics
        if task_type == 'classification':
            test_score = test_metrics['accuracy']
            f1 = f1_score(targets, predictions, average='weighted')
            additional_metrics = {'f1_score': f1}
        else:
            test_score = test_metrics['r2_score']
            mse = test_metrics['mse']
            additional_metrics = {'mse': mse, 'rmse': np.sqrt(mse)}
            
        # Count model parameters
        model_params = sum(p.numel() for p in model.parameters())
        
        return BenchmarkResult(
            model_name='RCANet',
            dataset_name="",  # Will be filled later
            task_type=task_type,
            train_time=train_time,
            test_score=test_score,
            test_loss=test_metrics['loss'],
            additional_metrics=additional_metrics,
            model_params=model_params
        )
    
    def benchmark_dataset(self, dataset_name: str, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str], task_type: str) -> List[BenchmarkResult]:
        """Benchmark all models on a single dataset."""
        print(f"\nBenchmarking dataset: {dataset_name}")
        print(f"Shape: {X.shape}, Task: {task_type}")
        
        # Train-test split
        test_size = 0.2
        if task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        dataset_results = []
        
        # Standardize features for baseline models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train baseline models
        baseline_models = self.get_baseline_models(task_type)
        
        for model_name, model in baseline_models.items():
            try:
                result = self.train_baseline_model(
                    model, model_name, X_train_scaled, X_test_scaled,
                    y_train, y_test, task_type
                )
                result.dataset_name = dataset_name
                dataset_results.append(result)
                
                metric_name = 'Accuracy' if task_type == 'classification' else 'R²'
                print(f"    {model_name}: {metric_name}={result.test_score:.4f}, Time={result.train_time:.2f}s")
                
            except Exception as e:
                print(f"    {model_name}: Failed - {e}")
                
        # Train RCANet
        try:
            result = self.train_rcanet_model(
                X_train, X_test, y_train, y_test, feature_names, task_type
            )
            result.dataset_name = dataset_name
            dataset_results.append(result)
            
            metric_name = 'Accuracy' if task_type == 'classification' else 'R²'
            print(f"    RCANet: {metric_name}={result.test_score:.4f}, Time={result.train_time:.2f}s, Params={result.model_params}")
            
        except Exception as e:
            print(f"    RCANet: Failed - {e}")
            import traceback
            traceback.print_exc()
            
        return dataset_results
    
    def run_benchmark(self, dataset_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Run comprehensive benchmark on all datasets."""
        print("Starting RCANet Benchmark Suite")
        print("=" * 50)
        
        # Load datasets
        datasets = self.load_benchmark_datasets()
        
        # Filter datasets if specified
        if dataset_names:
            datasets = {name: data for name, data in datasets.items() 
                       if name in dataset_names}
            
        all_results = []
        
        # Benchmark each dataset
        for dataset_name, (X, y, feature_names, task_type) in datasets.items():
            try:
                dataset_results = self.benchmark_dataset(
                    dataset_name, X, y, feature_names, task_type
                )
                all_results.extend(dataset_results)
                
            except Exception as e:
                print(f"Failed to benchmark {dataset_name}: {e}")
                
        # Store results
        self.results = all_results
        
        # Create results DataFrame
        results_df = self.create_results_dataframe()
        
        # Print summary
        self.print_benchmark_summary(results_df)
        
        return results_df
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from benchmark results."""
        data = []
        
        for result in self.results:
            row = {
                'Dataset': result.dataset_name,
                'Model': result.model_name,
                'Task': result.task_type,
                'Test_Score': result.test_score,
                'Train_Time': result.train_time,
                'Parameters': result.model_params
            }
            
            # Add additional metrics
            if result.additional_metrics:
                row.update(result.additional_metrics)
                
            data.append(row)
            
        return pd.DataFrame(data)
    
    def print_benchmark_summary(self, results_df: pd.DataFrame):
        """Print comprehensive benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        # Group by task type
        for task_type in results_df['Task'].unique():
            task_results = results_df[results_df['Task'] == task_type]
            
            print(f"\n{task_type.upper()} TASKS:")
            print("-" * 40)
            
            # Group by dataset
            for dataset in task_results['Dataset'].unique():
                dataset_results = task_results[task_results['Dataset'] == dataset]
                
                print(f"\n{dataset}:")
                
                # Sort by test score (descending)
                dataset_results = dataset_results.sort_values('Test_Score', ascending=False)
                
                metric_name = 'Accuracy' if task_type == 'classification' else 'R² Score'
                
                for _, row in dataset_results.iterrows():
                    model_name = row['Model']
                    score = row['Test_Score']
                    time_taken = row['Train_Time']
                    params = row['Parameters']
                    
                    params_str = f", {params:,} params" if params else ""
                    print(f"  {model_name:15s}: {metric_name}={score:.4f}, Time={time_taken:6.2f}s{params_str}")
                    
        # Overall statistics
        print(f"\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)
        
        # RCANet vs best baseline comparison
        rcanet_results = results_df[results_df['Model'] == 'RCANet']
        baseline_results = results_df[results_df['Model'] != 'RCANet']
        
        wins = 0
        total = 0
        
        for dataset in results_df['Dataset'].unique():
            dataset_results = results_df[results_df['Dataset'] == dataset]
            
            if 'RCANet' in dataset_results['Model'].values:
                rcanet_score = dataset_results[dataset_results['Model'] == 'RCANet']['Test_Score'].iloc[0]
                baseline_scores = dataset_results[dataset_results['Model'] != 'RCANet']['Test_Score']
                
                if len(baseline_scores) > 0:
                    best_baseline = baseline_scores.max()
                    if rcanet_score > best_baseline:
                        wins += 1
                    total += 1
                    
        print(f"\nRCANet wins: {wins}/{total} datasets ({wins/total*100:.1f}%)")
        
        # Average performance
        if len(rcanet_results) > 0:
            avg_rcanet_score = rcanet_results['Test_Score'].mean()
            avg_rcanet_time = rcanet_results['Train_Time'].mean()
            avg_rcanet_params = rcanet_results['Parameters'].mean()
            
            print(f"\nRCANet averages:")
            print(f"  Score: {avg_rcanet_score:.4f}")
            print(f"  Training time: {avg_rcanet_time:.2f}s")
            print(f"  Parameters: {avg_rcanet_params:,.0f}")
            
        if len(baseline_results) > 0:
            avg_baseline_score = baseline_results['Test_Score'].mean()
            avg_baseline_time = baseline_results['Train_Time'].mean()
            
            print(f"\nBaseline averages:")
            print(f"  Score: {avg_baseline_score:.4f}")
            print(f"  Training time: {avg_baseline_time:.2f}s")
    
    def save_results(self, filename: str = 'rcanet_benchmark_results.csv'):
        """Save benchmark results to CSV file."""
        if self.results:
            results_df = self.create_results_dataframe()
            results_df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
        else:
            print("No results to save. Run benchmark first.")


def run_quick_benchmark():
    """Run a quick benchmark on selected datasets."""
    benchmark = TabularBenchmark(random_state=42)
    
    # Run on a subset of datasets for speed
    quick_datasets = ['breast_cancer', 'wine', 'synthetic_clf', 'synthetic_reg']
    
    results_df = benchmark.run_benchmark(dataset_names=quick_datasets)
    
    # Save results
    benchmark.save_results('quick_benchmark_results.csv')
    
    return results_df


def run_full_benchmark():
    """Run full benchmark on all datasets."""
    benchmark = TabularBenchmark(random_state=42)
    
    results_df = benchmark.run_benchmark()
    
    # Save results
    benchmark.save_results('full_benchmark_results.csv')
    
    return results_df


def main():
    """Main benchmarking function."""
    print("RCANet Benchmarking Suite")
    print("========================\n")
    
    print("Choose benchmark type:")
    print("1. Quick benchmark (4 datasets)")
    print("2. Full benchmark (all datasets)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == '1':
            print("\nRunning quick benchmark...")
            results_df = run_quick_benchmark()
        elif choice == '2':
            print("\nRunning full benchmark...")
            results_df = run_full_benchmark()
        else:
            print("Invalid choice. Running quick benchmark by default.")
            results_df = run_quick_benchmark()
            
        print("\nBenchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # For automated runs without input
    benchmark = TabularBenchmark(random_state=42)
    results_df = run_quick_benchmark()