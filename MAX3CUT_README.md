# Max-3-Cut Dataset for DiffILO

This README explains how to generate and use the Max-3-Cut binary ILP dataset with DiffILO.

## Problem Overview

The Max-3-Cut problem aims to partition the vertices of a graph into 3 sets to maximize the weight of edges between different sets (or minimize the weight of edges within each set).

Binary ILP formulation:
```
min_{t, x} sum_{uv ∈ E} t_{uv} w_{uv}
s.t.  x^0_i + x^1_i + x^2_i = 1  for all i ∈ V
      t_{uv} >= x^k_u + x^k_v - 1  for all k ∈ {0,1,2} and all uv ∈ E
```

Where:
- `x^k_i` is 1 if vertex i is in partition k, and 0 otherwise
- `t_{uv}` is 1 if edge uv is within the same partition (not cut), and 0 otherwise (cut)
- `w_{uv}` is the weight of edge uv

## Dataset Generation

To generate the Max-3-Cut dataset:

1. Run the generation script:
```bash
python generate_max3cut_dataset.py --n_instances 100 --output_dir data/M3C --min_nodes 20 --max_nodes 50
```

Options:
- `--n_instances`: Number of problem instances to generate (default: 100)
- `--output_dir`: Output directory for dataset (default: data/M3C)
- `--min_nodes`: Minimum number of nodes in graphs (default: 20)
- `--max_nodes`: Maximum number of nodes in graphs (default: 50)
- `--max_neighbors`: Maximum number of neighbors to consider (default: 10)
- `--p_connection`: Probability of connection between neighbors (default: 0.5)
- `--seed`: Random seed (default: 42)

2. Preprocess the dataset using our custom preprocessing script (recommended):
```bash
python preprocess_m3c.py --data_dir data/M3C --output_dir data/preprocess/M3C
```

This script ensures that tensors are properly formatted for PyTorch Geometric, avoiding batching issues.

3. Train DiffILO on the dataset:
```bash
python train.py dataset=M3C cuda=0
```

4. Test DiffILO on the dataset:
```bash
python test.py dataset=M3C cuda=0
```

## Dataset Structure

The dataset is organized as follows:
```
data/M3C/
├── train/
│   ├── max3cut_train_0.lp
│   ├── max3cut_train_1.lp
│   └── ...
└── test/
    ├── max3cut_test_0.lp
    ├── max3cut_test_1.lp
    └── ...
```

Each .lp file contains a Max-3-Cut problem instance in LP format that can be solved by DiffILO.

## Implementation Notes

- The dataset uses all binary variables as required by DiffILO
- Each generated graph has random edge weights for more diverse problem instances
- The constraints enforce the binary ILP formulation exactly as specified 

## Troubleshooting

If you encounter any "storage not resizable" errors during training, it's usually due to inconsistencies in the tensor formats. Follow these steps to fix the issues:

### Quick Fix

1. Run our comprehensive dataset fix script:
```bash
python fix_m3c_dataset.py
```

This script:
- Converts existing data to proper BipartiteNodeData objects
- Makes sure all tensors have proper types and contiguous memory
- Tests if the dataset can be batched correctly
- Updates the config to use the fixed dataset

2. After running the fix script, you should be able to train normally:
```bash
python train.py dataset=M3C cuda=0
```

### Manual Fix Process

If the quick fix doesn't work, you can try the manual process:

1. Use our custom preprocessing script to regenerate the dataset:
```bash
python preprocess_m3c.py --data_dir data/M3C --output_dir data/preprocess/M3C
```

2. Then run the fix script to ensure proper formatting:
```bash
python fix_m3c_dataset.py
```

3. Verify data compatibility with our debug script:
```bash
python debug_m3c_batching.py
```

4. Update the paths in your config to use the fixed dataset.

### Technical Details

The fix script ensures that:
1. All graph objects are BipartiteNodeData instances (not just lists of tensors)
2. All tensors are properly typed (float32 for features, int64/long for indices)
3. All tensors have contiguous memory layout
4. Normalization is done safely without division by zero
5. Tensor shapes are consistent across the dataset 