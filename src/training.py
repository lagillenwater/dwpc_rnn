"""
Training Utilities for Neural Network Models

This module contains functions for training neural networks, including
training loops, validation, and performance evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import numpy as np

from .models import EdgePredictionNN
from .data_processing import prepare_edge_prediction_data


def train_edge_prediction_model(features, labels, test_size=0.2, epochs=100, batch_size=1024, learning_rate=0.001):
    """
    Train the edge prediction neural network.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature matrix with source and target degrees
    labels : numpy.ndarray
        Binary labels for edge existence
    test_size : float
        Proportion of data for testing
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for optimizer
    
    Returns:
    --------
    model : EdgePredictionNN
        Trained neural network model
    train_history : dict
        Training history with losses and metrics
    test_metrics : dict
        Test set performance metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = EdgePredictionNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    val_losses = []
    val_aucs = []
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    print(f"Feature shapes: {X_train.shape}, Labels shape: {y_train.shape}")
    
    # Training loop
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        val_auc = roc_auc_score(val_true, val_predictions)
        
        # Store history
        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(val_loss / len(test_loader))
        val_aucs.append(val_auc)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_losses[-1]:.4f}, "
                  f"Val Loss: {val_losses[-1]:.4f}, Val AUC: {val_auc:.4f}")
        
        model.train()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy()
    
    test_auc = roc_auc_score(y_test, test_predictions)
    test_ap = average_precision_score(y_test, test_predictions)
    
    print(f"Final Test AUC: {test_auc:.4f}")
    print(f"Final Test AP: {test_ap:.4f}")
    
    # Prepare return values
    train_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs
    }
    
    test_metrics = {
        'auc': test_auc,
        'average_precision': test_ap,
        'predictions': test_predictions,
        'true_labels': y_test,
        'scaler': scaler
    }
    
    return model, train_history, test_metrics


def train_across_permutations(all_perm_data, epochs=20):
    """
    Train neural network models across multiple permutations.
    
    Parameters:
    -----------
    all_perm_data : dict
        Dictionary of permutation data
    epochs : int
        Number of epochs for training each model
    
    Returns:
    --------
    results : dict
        Dictionary containing results for each permutation
    """
    results = {}
    
    print("=" * 60)
    print("TRAINING ACROSS ALL PERMUTATIONS")
    print("=" * 60)
    
    for perm_name, perm_data in all_perm_data.items():
        try:
            print(f"\nTraining on {perm_name}...")
            
            # Prepare data
            features, labels = prepare_edge_prediction_data(
                perm_data, sample_negative_ratio=1.0
            )
            
            # Train model
            model, train_history, test_metrics = train_edge_prediction_model(
                features, labels,
                epochs=epochs,
                batch_size=512,
                learning_rate=0.001
            )
            
            results[perm_name] = {
                'model': model,
                'train_history': train_history,
                'test_metrics': test_metrics,
                'features': features,
                'labels': labels
            }
            
            print(f"✓ {perm_name}: AUC = {test_metrics['auc']:.4f}, AP = {test_metrics['average_precision']:.4f}")
            
        except Exception as e:
            print(f"✗ Failed on {perm_name}: {e}")
            results[perm_name] = None
    
    return results
