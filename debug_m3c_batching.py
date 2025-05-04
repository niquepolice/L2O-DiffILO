import os
import pickle
import glob
import torch
from torch_geometric.loader import DataLoader
from src.model import BipartiteNodeData, GraphDataset

def test_batching(data_dir, batch_size=2):
    """Test loading and batching data to ensure it works with PyTorch Geometric."""
    # Look for pickle files
    pickle_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    
    if not pickle_files:
        print(f"No pickle files found in {data_dir}")
        return False
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Try loading one file to check its format
    with open(pickle_files[0], 'rb') as f:
        data = pickle.load(f)
    
    print("\nLoaded data type:", type(data))
    
    if isinstance(data, BipartiteNodeData):
        print("Data is already a BipartiteNodeData object (good)")
    else:
        print("WARNING: Data is not a BipartiteNodeData object")
        return False
    
    # Check tensor properties
    print("\nTensor properties:")
    print(f"constraint_features: {data.constraint_features.shape}, {data.constraint_features.dtype}")
    print(f"edge_index: {data.edge_index.shape}, {data.edge_index.dtype}")
    print(f"edge_attr: {data.edge_attr.shape}, {data.edge_attr.dtype}")
    print(f"variable_features: {data.variable_features.shape}, {data.variable_features.dtype}")
    
    # Check if tensors are contiguous
    print("\nContiguity check:")
    print(f"constraint_features contiguous: {data.constraint_features.is_contiguous()}")
    print(f"edge_index contiguous: {data.edge_index.is_contiguous()}")
    print(f"edge_attr contiguous: {data.edge_attr.is_contiguous()}")
    print(f"variable_features contiguous: {data.variable_features.is_contiguous()}")
    
    # Create a dataset and try to batch
    try:
        # Create a custom dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, file_paths):
                self.file_paths = file_paths
            
            def __len__(self):
                return len(self.file_paths)
            
            def __getitem__(self, idx):
                with open(self.file_paths[idx], 'rb') as f:
                    return pickle.load(f)
        
        dataset = SimpleDataset(pickle_files[:min(5, len(pickle_files))])
        
        # Try to create a data loader and batch
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            follow_batch=["constraint_features", "variable_features"]
        )
        
        print(f"\nTrying to batch {batch_size} graphs...")
        batch = next(iter(loader))
        
        print("Batching successful!")
        print(f"Batched data type: {type(batch)}")
        print(f"Batched constraint_features: {batch.constraint_features.shape}")
        print(f"Batched edge_index: {batch.edge_index.shape}")
        print(f"Batched edge_attr: {batch.edge_attr.shape}")
        print(f"Batched variable_features: {batch.variable_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"\nError during batching: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the training data
    data_dir = "data/preprocess/M3C/samples/train"
    
    if not os.path.exists(data_dir):
        print(f"Directory does not exist: {data_dir}")
    else:
        success = test_batching(data_dir)
        
        if success:
            print("\n✅ Batching test successful! The dataset should work with DiffILO.")
        else:
            print("\n❌ Batching test failed. The dataset needs further fixes.") 