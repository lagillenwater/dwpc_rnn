"""
Data Processing Utilities for DWPC Analysis

This module contains functions for preparing and processing data for neural network training,
including edge prediction data preparation and negative sampling.
"""

import numpy as np
import pandas as pd
import scipy.sparse
import pathlib


def prepare_edge_prediction_data(permutation_data, sample_negative_ratio=1.0):
    """
    Prepare training data for edge prediction based on source and target degrees.
    
    Parameters:
    -----------
    permutation_data : dict
        Dictionary containing anatomy nodes, gene nodes, and adjacency matrix
    sample_negative_ratio : float
        Ratio of negative to positive examples to sample
    
    Returns:
    --------
    features : numpy.ndarray
        Feature matrix with source and target degrees
    labels : numpy.ndarray
        Binary labels (1 for existing edges, 0 for non-existing)
    """
    anatomy_nodes = permutation_data['anatomy_nodes']
    gene_nodes = permutation_data['gene_nodes']
    aeg_edges = permutation_data['aeg_edges']
    
    # Get node degrees
    anatomy_degrees = np.array(aeg_edges.sum(axis=1)).flatten()
    gene_degrees = np.array(aeg_edges.sum(axis=0)).flatten()
    
    print(f"Anatomy degree range: {anatomy_degrees.min()} - {anatomy_degrees.max()}")
    print(f"Gene degree range: {gene_degrees.min()} - {gene_degrees.max()}")
    
    # Prepare positive examples (existing edges)
    rows, cols = aeg_edges.nonzero()
    positive_features = []
    positive_labels = []
    
    for anatomy_idx, gene_idx in zip(rows, cols):
        positive_features.append([anatomy_degrees[anatomy_idx], gene_degrees[gene_idx]])
        positive_labels.append(1)
    
    positive_features = np.array(positive_features)
    positive_labels = np.array(positive_labels)
    
    print(f"Number of positive examples (existing edges): {len(positive_labels)}")
    
    # Prepare negative examples (non-existing edges)
    num_negative = int(len(positive_labels) * sample_negative_ratio)
    negative_features = []
    negative_labels = []
    
    attempts = 0
    max_attempts = num_negative * 10  # Prevent infinite loops
    
    while len(negative_labels) < num_negative and attempts < max_attempts:
        # Random sample of anatomy and gene indices
        anatomy_idx = np.random.randint(0, len(anatomy_nodes))
        gene_idx = np.random.randint(0, len(gene_nodes))
        
        # Check if this pair doesn't have an edge
        if aeg_edges[anatomy_idx, gene_idx] == 0:
            negative_features.append([anatomy_degrees[anatomy_idx], gene_degrees[gene_idx]])
            negative_labels.append(0)
        
        attempts += 1
    
    negative_features = np.array(negative_features)
    negative_labels = np.array(negative_labels)
    
    print(f"Number of negative examples (non-existing edges): {len(negative_labels)}")
    
    # Combine positive and negative examples
    all_features = np.vstack([positive_features, negative_features])
    all_labels = np.concatenate([positive_labels, negative_labels])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(all_labels))
    all_features = all_features[shuffle_idx]
    all_labels = all_labels[shuffle_idx]
    
    return all_features, all_labels


def load_permutation_data(permutation_name, permutations_dir):
    """
    Load AeG edges, Anatomy nodes, and Gene nodes for a specific permutation.
    
    Parameters:
    -----------
    permutation_name : str
        Name of the permutation directory (e.g., '001.hetmat')
    permutations_dir : pathlib.Path
        Path to the permutations directory
    
    Returns:
    --------
    dict : Dictionary containing loaded data with keys:
           - 'aeg_edges': scipy sparse matrix for AeG edges
           - 'anatomy_nodes': pandas DataFrame of anatomy nodes
           - 'gene_nodes': pandas DataFrame of gene nodes
           - 'permutation_path': pathlib.Path to the permutation directory
    """
    # Set up paths for this permutation
    perm_dir = permutations_dir / permutation_name
    edges_dir = perm_dir / 'edges'
    nodes_dir = perm_dir / 'nodes'
    
    if not perm_dir.exists():
        raise FileNotFoundError(f"Permutation directory not found: {perm_dir}")
    
    print(f"Loading data from permutation: {permutation_name}")
    print(f"Permutation path: {perm_dir}")
    
    # Load AeG edges (Anatomy-expresses-Gene)
    aeg_path = edges_dir / 'AeG.sparse.npz'
    if not aeg_path.exists():
        raise FileNotFoundError(f"AeG edges file not found: {aeg_path}")
    
    aeg_edges = scipy.sparse.load_npz(aeg_path)
    print(f"Loaded AeG edges: {aeg_edges.shape} matrix with {aeg_edges.nnz} non-zero entries")
    
    # Load Anatomy nodes
    anatomy_path = nodes_dir / 'Anatomy.tsv'
    if not anatomy_path.exists():
        raise FileNotFoundError(f"Anatomy nodes file not found: {anatomy_path}")
    
    anatomy_nodes = pd.read_csv(anatomy_path, sep='\t')
    print(f"Loaded Anatomy nodes: {len(anatomy_nodes)} nodes")
    print(f"Anatomy columns: {list(anatomy_nodes.columns)}")
    
    # Load Gene nodes
    gene_path = nodes_dir / 'Gene.tsv'
    if not gene_path.exists():
        raise FileNotFoundError(f"Gene nodes file not found: {gene_path}")
    
    gene_nodes = pd.read_csv(gene_path, sep='\t')
    print(f"Loaded Gene nodes: {len(gene_nodes)} nodes")
    print(f"Gene columns: {list(gene_nodes.columns)}")
    
    return {
        'aeg_edges': aeg_edges,
        'anatomy_nodes': anatomy_nodes,
        'gene_nodes': gene_nodes,
        'permutation_path': perm_dir
    }


def load_all_permutations(available_permutations, permutations_dir):
    """
    Load AeG edges, Anatomy nodes, and Gene nodes from all available permutations.
    
    Parameters:
    -----------
    available_permutations : list
        List of permutation directory names
    permutations_dir : pathlib.Path
        Path to the permutations directory
    
    Returns:
    --------
    dict : Dictionary with permutation names as keys and loaded data as values
    """
    all_permutations = {}
    
    for perm_name in available_permutations:
        try:
            print(f"\nLoading permutation: {perm_name}")
            perm_data = load_permutation_data(perm_name, permutations_dir)
            all_permutations[perm_name] = perm_data
            print(f"✓ Successfully loaded {perm_name}")
        except Exception as e:
            print(f"✗ Failed to load {perm_name}: {e}")
    
    return all_permutations
