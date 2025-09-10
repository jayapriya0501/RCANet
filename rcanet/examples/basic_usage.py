"""Basic usage example for RCANet.

This script demonstrates how to train RCANet on a simple tabular dataset
with minimal configuration.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# RCANet imports
from rcanet.models import RCANet
from rcanet.data import TabularDataset, TabularPreprocessor, create_data_loaders
from rcanet.training import RCANetTrainer
from rcanet.utils import RCANetConfig, compute_metrics


def create_sample_classification_data(n_samples=1000, n_features=20, n_classes=3):
    """Create a sample classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names


def create_sample_regression_data(n_samples=1000, n_features=15):
    """Create a sample regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        noise=0.1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names


def train_rcanet_classification():
    """Train RCANet on a classification task."""
    print("=== RCANet Classification Example ===")
    
    # Create sample data
    df, feature_names = create_sample_classification_data(n_samples=1000, n_features=20, n_classes=3)
    
    # Split features and target
    X = df[feature_names].values
    y = df['target'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Create configuration
    config = RCANetConfig(
        input_dim=X_train.shape[1],
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        output_dim=len(np.unique(y)),
        task_type='classification',
        dropout=0.1,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        use_contrastive=True,
        contrastive_weight=0.1
    )
    
    print(f"Model configuration: {config}")
    
    # Create preprocessor
    preprocessor = TabularPreprocessor(
        numerical_features=list(range(len(feature_names))),
        categorical_features=[],
        scale_features=True,
        handle_missing='mean'
    )
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create datasets
    train_dataset = TabularDataset(
        X_train_processed, y_train,
        feature_names=feature_names,
        task_type='classification'
    )
    
    test_dataset = TabularDataset(
        X_test_processed, y_test,
        feature_names=feature_names,
        task_type='classification'
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
    
    # Create model
    model = RCANet(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = RCANetTrainer(
        model=model,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Training on device: {trainer.device}")
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Get predictions for detailed analysis
    predictions, targets, attention_weights = trainer.predict(test_loader, return_attention=True)
    
    # Compute additional metrics
    accuracy = accuracy_score(targets, predictions)
    print(f"\nDetailed Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print training history summary
    print(f"\nTraining Summary:")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    return model, trainer, history, attention_weights


def train_rcanet_regression():
    """Train RCANet on a regression task."""
    print("\n=== RCANet Regression Example ===")
    
    # Create sample data
    df, feature_names = create_sample_regression_data(n_samples=1000, n_features=15)
    
    # Split features and target
    X = df[feature_names].values
    y = df['target'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Create configuration
    config = RCANetConfig(
        input_dim=X_train.shape[1],
        hidden_dim=64,
        num_heads=4,
        num_layers=3,
        output_dim=1,
        task_type='regression',
        dropout=0.1,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        use_contrastive=False  # Disable contrastive learning for regression
    )
    
    # Create preprocessor
    preprocessor = TabularPreprocessor(
        numerical_features=list(range(len(feature_names))),
        categorical_features=[],
        scale_features=True,
        handle_missing='mean'
    )
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Create datasets
    train_dataset = TabularDataset(
        X_train_processed, y_train,
        feature_names=feature_names,
        task_type='regression'
    )
    
    test_dataset = TabularDataset(
        X_test_processed, y_test,
        feature_names=feature_names,
        task_type='regression'
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
    
    # Create model
    model = RCANet(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = RCANetTrainer(
        model=model,
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Get predictions for detailed analysis
    predictions, targets, _ = trainer.predict(test_loader)
    
    # Compute additional metrics
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\nDetailed Test Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Print training history summary
    print(f"\nTraining Summary:")
    print(f"Best validation MSE: {min(history['val_mse']):.4f}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    return model, trainer, history


def demonstrate_attention_analysis(model, trainer, attention_weights, feature_names):
    """Demonstrate attention analysis capabilities."""
    print("\n=== Attention Analysis ===")
    
    try:
        from rcanet.utils.visualization import plot_attention_maps, plot_attention_statistics
        
        # Plot attention maps
        print("Generating attention visualizations...")
        
        # Convert attention weights to proper format
        if isinstance(attention_weights, dict):
            row_attention = attention_weights.get('row_attention')
            col_attention = attention_weights.get('col_attention')
        else:
            row_attention = attention_weights
            col_attention = attention_weights
            
        if row_attention is not None:
            print(f"Row attention shape: {row_attention.shape}")
            print(f"Column attention shape: {col_attention.shape if col_attention is not None else 'None'}")
            
            # Create attention analysis
            attention_dict = {
                'row': row_attention,
                'col': col_attention if col_attention is not None else row_attention
            }
            
            # Plot attention maps (first few samples)
            fig1 = plot_attention_maps(
                attention_dict,
                feature_names=feature_names[:min(10, len(feature_names))],  # Limit for visualization
                max_samples=2
            )
            
            # Plot attention statistics
            fig2 = plot_attention_statistics(attention_dict)
            
            if fig1 or fig2:
                print("Attention visualizations created successfully!")
                print("Note: Figures are displayed if matplotlib backend supports it.")
            else:
                print("Visualization libraries not available. Install matplotlib and seaborn for plots.")
                
    except ImportError as e:
        print(f"Visualization not available: {e}")
        print("Install matplotlib and seaborn for attention analysis.")


def main():
    """Run basic usage examples."""
    print("RCANet Basic Usage Examples")
    print("===========================\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run classification example
    try:
        model_cls, trainer_cls, history_cls, attention_weights = train_rcanet_classification()
        
        # Demonstrate attention analysis
        feature_names = [f'feature_{i}' for i in range(20)]
        demonstrate_attention_analysis(model_cls, trainer_cls, attention_weights, feature_names)
        
    except Exception as e:
        print(f"Classification example failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Run regression example
    try:
        model_reg, trainer_reg, history_reg = train_rcanet_regression()
        
    except Exception as e:
        print(f"Regression example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Examples Complete ===")
    print("Check the output above for training results and metrics.")
    print("For more advanced usage, see advanced_usage.py")


if __name__ == "__main__":
    main()