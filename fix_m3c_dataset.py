#!/usr/bin/env python3
"""
Complete fix for Max-3-Cut dataset compatibility with DiffILO.
This script directly creates a BipartiteNodeData dataset structure compatible with DiffILO.
"""

import os
import sys
import glob
import pickle
import numpy as np
import torch
import argparse
from tqdm import tqdm
import pyscipopt as scip

# Add the current directory to the path to import the src.model module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import BipartiteNodeData, GraphDataset

def fix_existing_datasets(samples_dir, output_dir):
    """
    Fix existing dataset files by converting them to BipartiteNodeData objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .pkl files in the directory
    pkl_files = glob.glob(os.path.join(samples_dir, "*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {samples_dir}")
        return
    
    print(f"Found {len(pkl_files)} files to fix")
    
    for file_path in tqdm(pkl_files, desc="Fixing dataset files"):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, BipartiteNodeData):
                # Already a BipartiteNodeData object, just make sure tensors are contiguous
                data.constraint_features = data.constraint_features.contiguous()
                data.edge_index = data.edge_index.contiguous()
                data.edge_attr = data.edge_attr.contiguous()
                data.variable_features = data.variable_features.contiguous()
            elif isinstance(data, list) and len(data) == 4:
                # Convert from list format to BipartiteNodeData
                constraint_features, edge_indices, edge_features, variable_features = data
                
                # Ensure proper tensor types and contiguity
                if not isinstance(constraint_features, torch.Tensor):
                    constraint_features = torch.tensor(constraint_features, dtype=torch.float32)
                if not isinstance(edge_indices, torch.Tensor):
                    edge_indices = torch.tensor(edge_indices, dtype=torch.long)
                if not isinstance(edge_features, torch.Tensor):
                    edge_features = torch.tensor(edge_features, dtype=torch.float32)
                if not isinstance(variable_features, torch.Tensor):
                    variable_features = torch.tensor(variable_features, dtype=torch.float32)
                
                # Create a BipartiteNodeData object
                data = BipartiteNodeData(
                    constraint_features=constraint_features.contiguous(),
                    edge_indices=edge_indices.contiguous(),
                    edge_features=edge_features.contiguous(),
                    variable_features=variable_features.contiguous()
                )
            else:
                print(f"Unknown data format in {file_path}")
                continue
            
            # Save the fixed data
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Fixed dataset files saved to {output_dir}")

def create_sample_batch_file(data_dir, batch_size=2):
    """Create a sample batch file to verify batchability"""
    output_file = os.path.join(data_dir, "sample_batch.pkl")
    
    # Find all .pkl files in the directory
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    
    if len(pkl_files) < batch_size:
        print(f"Not enough files in {data_dir} to create a batch of size {batch_size}")
        return
    
    print(f"Creating a sample batch from {batch_size} files")
    
    # Load data from the first batch_size files
    data_list = []
    for i in range(batch_size):
        with open(pkl_files[i], 'rb') as f:
            data = pickle.load(f)
            data_list.append(data)
    
    try:
        # Use PyTorch Geometric's Batch class to create a batch
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(data_list)
        
        # Save the batch
        with open(output_file, 'wb') as f:
            pickle.dump(batch, f)
        
        print(f"Successfully created a batch from {batch_size} files")
        print(f"Sample batch saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error creating batch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading(data_dir, batch_size=2):
    """Test if the dataset can be loaded and batched with DiffILO's data loader"""
    
    # Create a small GraphDataset
    try:
        # Find all .pkl files in the directory
        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        
        if not pkl_files:
            print(f"No .pkl files found in {data_dir}")
            return False
        
        print(f"Testing dataset loading with {len(pkl_files)} files")
        
        # Create a GraphDataset
        dataset = GraphDataset(pkl_files[:5])  # Use at most 5 files for testing
        
        # Create a DataLoader
        from torch_geometric.loader import DataLoader
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            follow_batch=["constraint_features", "variable_features"]
        )
        
        # Try to load a batch
        batch = next(iter(loader))
        
        print("Successfully loaded a batch!")
        print(f"Batch type: {type(batch)}")
        print(f"Batch size (constraint features): {batch.constraint_features.shape}")
        print(f"Batch size (variable features): {batch.variable_features.shape}")
        
        return True
    except Exception as e:
        print(f"Error testing dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix Max-3-Cut dataset for DiffILO")
    parser.add_argument("--data_dir", type=str, default="data/preprocess/M3C/samples",
                        help="Directory containing preprocessed data (samples)")
    parser.add_argument("--fixed_data_dir", type=str, default="data/preprocess/M3C/fixed_samples",
                        help="Directory to store fixed dataset files")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Explicit training directory (default: data_dir/train)")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Explicit test directory (default: data_dir/test)")
    
    args = parser.parse_args()
    
    # Determine training and test directories
    train_dir = args.train_dir or os.path.join(args.data_dir, "train")
    test_dir = args.test_dir or os.path.join(args.data_dir, "test")
    
    # Create output directories
    fixed_train_dir = os.path.join(args.fixed_data_dir, "train")
    fixed_test_dir = os.path.join(args.fixed_data_dir, "test")
    os.makedirs(fixed_train_dir, exist_ok=True)
    os.makedirs(fixed_test_dir, exist_ok=True)
    
    # Fix training dataset
    print("\n=== Fixing training dataset ===")
    if os.path.exists(train_dir):
        fix_existing_datasets(train_dir, fixed_train_dir)
        
        # Test if the fixed dataset can be batched
        print("\nTesting training dataset batchability...")
        if test_dataset_loading(fixed_train_dir):
            print("✅ Training dataset can be successfully batched")
            print("\nYou can now use this dataset with DiffILO:")
            print(f"  - Update data_samples_dir in config/paths/M3C.yaml to: {fixed_train_dir}")
        else:
            print("❌ Training dataset still has issues with batching")
    else:
        print(f"Training directory not found: {train_dir}")
    
    # Fix test dataset
    print("\n=== Fixing test dataset ===")
    if os.path.exists(test_dir):
        fix_existing_datasets(test_dir, fixed_test_dir)
        
        # Test if the fixed dataset can be batched
        print("\nTesting test dataset batchability...")
        if test_dataset_loading(fixed_test_dir):
            print("✅ Test dataset can be successfully batched")
            print("\nYou can now use this dataset with DiffILO:")
            print(f"  - Update test_data_samples_dir in config/paths/M3C.yaml to: {fixed_test_dir}")
        else:
            print("❌ Test dataset still has issues with batching")
    else:
        print(f"Test directory not found: {test_dir}")
    
    print("\n=== Complete ===")
    print("If tests were successful, update your config/paths/M3C.yaml file with the fixed dataset paths.")
    print("Then run training with: python train.py dataset=M3C cuda=0")

if __name__ == "__main__":
    main() 