import os
import pyscipopt as scip
import numpy as np
import time
import re

def check_lp_file(file_path):
    """Check if an LP file can be properly read and solved by SCIP"""
    print(f"\nChecking file: {file_path}")
    
    # Create a SCIP model and read the LP file
    model = scip.Model()
    model.hideOutput()
    
    try:
        model.readProblem(file_path)
        print(f"Successfully loaded model with {model.getNVars()} variables and {model.getNConss()} constraints")
        
        # Check variable types
        vars = model.getVars()
        binary_count = 0
        for v in vars:
            if v.vtype() == 'BINARY':
                binary_count += 1
        
        print(f"Binary variables: {binary_count} (should be all variables)")
        
        # Analyze variable names to verify problem structure
        x_vars = 0
        t_vars = 0
        x_vars_dict = {}
        t_vars_dict = {}
        
        for v in vars:
            name = v.name
            if name.startswith('x_'):
                x_vars += 1
                parts = name.split('_')
                if len(parts) >= 3:
                    k = int(parts[1])  # Partition (0, 1, or 2)
                    i = int(parts[2])  # Node
                    x_vars_dict[(k, i)] = v
            elif name.startswith('t_'):
                t_vars += 1
                parts = name.split('_')
                if len(parts) >= 3:
                    u = int(parts[1])
                    v_idx = int(parts[2])
                    t_vars_dict[(u, v_idx)] = v
        
        # A Max-3-Cut problem with n nodes should have 3n x-variables and |E| t-variables
        n_nodes = x_vars // 3
        print(f"Problem has {n_nodes} nodes ({x_vars} x-variables) and {t_vars} edges (t-variables)")
        
        # Verify constraint structure
        cons = model.getConss()
        node_partition_cons = 0
        edge_cons = 0
        
        # Check a sample of constraints
        sample_size = min(10, len(cons))
        print(f"\nAnalyzing {sample_size} sample constraints:")
        for i in range(sample_size):
            con = cons[i]
            name = con.name
            print(f"  Constraint {i+1}: {name}")
            
            # Get the constraint type
            con_type = "unknown"
            if name.startswith("node_"):
                con_type = "node partition"
                node_partition_cons += 1
            elif name.startswith("edge_"):
                con_type = "edge constraint"
                edge_cons += 1
            
            print(f"    Type: {con_type}")
        
        # Print constraint counts
        print(f"\nNode partition constraints: {node_partition_cons} (sample)")
        print(f"Edge constraints: {edge_cons} (sample)")
        
        # Try to solve the problem
        print("\nAttempting to solve...")
        start_time = time.time()
        model.setParam('limits/time', 10)  # Set a time limit of 10 seconds
        model.optimize()
        solve_time = time.time() - start_time
        
        # Report solution status
        status = model.getStatus()
        print(f"Solution status: {status}")
        print(f"Solve time: {solve_time:.2f} seconds")
        
        if status == 'optimal':
            # Get objective value
            obj_val = model.getObjVal()
            print(f"Objective value: {obj_val}")
            
            # Check solution - extract node assignments
            partition = [None] * n_nodes
            for v in vars:
                name = v.name
                if name.startswith('x_') and model.getVal(v) > 0.5:
                    parts = name.split('_')
                    k = int(parts[1])  # Partition number (0, 1, or 2)
                    i = int(parts[2])  # Node number
                    partition[i] = k
            
            # Check if all nodes have exactly one partition
            if None in partition:
                print("ERROR: Some nodes are not assigned to any partition")
            else:
                print(f"Solution partitions nodes into: {np.bincount(partition)} groups")
            
            # Verify that the t-variables match the partition assignments
            t_correct = 0
            t_incorrect = 0
            for v in vars:
                name = v.name
                if name.startswith('t_'):
                    parts = name.split('_')
                    u = int(parts[1])
                    v_idx = int(parts[2])
                    
                    if u < len(partition) and v_idx < len(partition):
                        # t_uv should be 1 if u and v are in the same partition
                        t_expected = 1 if partition[u] == partition[v_idx] else 0
                        t_actual = round(model.getVal(v))
                        
                        if t_expected == t_actual:
                            t_correct += 1
                        else:
                            t_incorrect += 1
            
            if t_incorrect > 0:
                print(f"WARNING: {t_incorrect} t-variables don't match partition assignments")
            else:
                print(f"All {t_correct} t-variables correctly match partition assignments")
                
            # Read the LP file directly to find objective coefficients
            try:
                with open(file_path, 'r') as f:
                    lp_content = f.read()
                
                # Find the Minimize section
                obj_section = re.search(r"Minimize\s*\n(.*?)\n\n", lp_content, re.DOTALL)
                if obj_section:
                    obj_text = obj_section.group(1).strip()
                    
                    # Calculate edges within partition
                    edges_within_partition = 0
                    obj_calc = 0
                    
                    for v in vars:
                        name = v.name
                        if name.startswith('t_') and model.getVal(v) > 0.5:
                            edges_within_partition += 1
                            
                            # Find this t-variable in the objective function
                            parts = name.split('_')
                            u = parts[1]
                            v_idx = parts[2]
                            pattern = r"([\d\.]+)\s*t_{}_{}".format(u, v_idx)
                            
                            weight_match = re.search(pattern, obj_text)
                            if weight_match:
                                weight = float(weight_match.group(1))
                                obj_calc += weight
                    
                    print(f"Edges within same partition: {edges_within_partition}")
                    print(f"Calculated objective: {obj_calc:.6f} (should match {obj_val:.6f})")
            except Exception as e:
                print(f"Error calculating objective: {e}")
                
            return True
        return status != 'infeasible'
    
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    # Get a few sample files from the dataset
    train_dir = "data/M3C/train"
    test_dir = "data/M3C/test"
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Dataset directories not found. Generate the dataset first.")
        return
    
    # Get sample files
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)
    
    print(f"Found {len(train_files)} training files and {len(test_files)} test files")
    
    # Check a few training files
    train_samples = min(2, len(train_files))
    print(f"\n=== Checking {train_samples} training files ===")
    for i in range(train_samples):
        file_path = os.path.join(train_dir, train_files[i])
        success = check_lp_file(file_path)
        if not success:
            print(f"WARNING: Issue with file {file_path}")
    
    # Check a few test files
    test_samples = min(1, len(test_files))
    print(f"\n=== Checking {test_samples} test files ===")
    for i in range(test_samples):
        file_path = os.path.join(test_dir, test_files[i])
        success = check_lp_file(file_path)
        if not success:
            print(f"WARNING: Issue with file {file_path}")
    
    print("\nDataset check complete!")

if __name__ == "__main__":
    main() 