# DWPC RNN Analysis - Helper Functions

This document describes the refactored codebase where functions have been moved from notebooks to organized helper scripts in the `/src` directory.

## Directory Structure

```
dwpc_rnn/
├── src/                           # Helper functions and utilities
│   ├── __init__.py               # Package initialization
│   ├── models.py                 # Neural network architectures
│   ├── data_processing.py        # Data preparation utilities
│   ├── training.py               # Training and evaluation functions
│   ├── visualization.py          # Plotting and analysis tools
│   └── download_utils.py         # File download and organization
├── notebooks/                    # Jupyter notebooks (now use helper functions)
│   ├── 2_learn_null_edge.ipynb  # Neural network edge prediction
│   └── 3_download_null_graphs.ipynb # Download external permutations
├── scripts/                      # Shell scripts for HPC execution
└── example_usage.py              # Example of using helper functions
```

## Helper Modules

### 1. `models.py` - Neural Network Architectures

Contains the neural network models for edge prediction:

- `EdgePredictionNN`: Neural network for predicting edge probability based on source and target degrees

**Example usage:**
```python
from src.models import EdgePredictionNN

model = EdgePredictionNN(input_dim=2, hidden_dims=[64, 32], dropout_rate=0.2)
```

### 2. `data_processing.py` - Data Preparation

Functions for preparing and processing data:

- `prepare_edge_prediction_data()`: Prepare training data from permutation data

**Example usage:**
```python
from src.data_processing import prepare_edge_prediction_data

features, labels = prepare_edge_prediction_data(permutation_data, sample_negative_ratio=1.0)
```

### 3. `training.py` - Training and Evaluation

Training loops and model management:

- `train_edge_prediction_model()`: Train a single neural network model
- `train_across_permutations()`: Train models across multiple permutations

**Example usage:**
```python
from src.training import train_edge_prediction_model

model, train_history, test_metrics = train_edge_prediction_model(
    features, labels, epochs=100, batch_size=512
)
```

### 4. `visualization.py` - Plotting and Analysis

Visualization and analysis tools:

- `plot_training_history()`: Plot training curves
- `evaluate_model_performance()`: Create performance evaluation plots
- `create_probability_heatmap()`: Generate edge probability heatmaps
- `plot_permutation_comparison()`: Compare results across permutations

**Example usage:**
```python
from src.visualization import plot_training_history, evaluate_model_performance

plot_training_history(train_history)
threshold_results = evaluate_model_performance(test_metrics)
```

### 5. `download_utils.py` - File Management

Download and file organization utilities:

- `download_file()`: Download files with progress tracking
- `extract_zip()`: Extract archives with progress tracking
- `organize_permutations()`: Organize downloaded permutations
- `download_hetionet_permutations()`: Complete download workflow

**Example usage:**
```python
from src.download_utils import download_hetionet_permutations

success = download_hetionet_permutations(
    download_url=url,
    data_dir=data_dir,
    permutations_dir=permutations_dir,
    hetio_permutations_dir=hetio_permutations_dir
)
```

## Using Helper Functions in Notebooks

To use the helper functions in notebooks, add this to the first cell:

```python
import sys
from pathlib import Path

# Add src directory to path
repo_dir = Path().cwd().parent
src_dir = repo_dir / "src"
sys.path.insert(0, str(src_dir))

# Import helper functions
from models import EdgePredictionNN
from training import train_edge_prediction_model
from visualization import plot_training_history
# ... etc
```

## Benefits of This Refactoring

1. **Code Reusability**: Functions can be used across multiple notebooks and scripts
2. **Maintainability**: Centralized functions are easier to update and debug
3. **Testing**: Helper functions can be unit tested independently
4. **Documentation**: Clear separation of concerns with focused modules
5. **Collaboration**: Team members can work on different modules independently
6. **Version Control**: Changes to functions are tracked separately from notebook execution

## Migration Notes

### What Changed:
- Function definitions moved from notebooks to `/src` modules
- Notebooks now import and use helper functions
- Download workflow simplified using helper functions
- All interactive prompts removed for papermill compatibility

### What Stayed the Same:
- Notebook execution flow and outputs
- Analysis logic and algorithms
- HPC shell scripts (updated to use helper functions)
- Data structures and file formats

## Running the Code

### Notebooks:
The notebooks can be run as before, but now use helper functions:

```bash
jupyter notebook notebooks/2_learn_null_edge.ipynb
```

### HPC Scripts:
The shell scripts work the same way:

```bash
sbatch scripts/2_learn_null_edge.sh
sbatch scripts/3_download_null_graphs.sh
```

### Direct Python:
You can also use the helper functions directly:

```bash
python example_usage.py
```

## Dependencies

The helper functions require the same dependencies as before:
- PyTorch
- scikit-learn
- matplotlib
- pandas
- numpy
- tqdm
- requests
- hetmatpy

Make sure your conda environment includes all required packages.
