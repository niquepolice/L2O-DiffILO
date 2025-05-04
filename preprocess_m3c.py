import os.path
import pickle
import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging
import sys
import glob
import torch
import pyscipopt as scip
import ecole

# Add the project root to the path so we can import DiffILO modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import BipartiteNodeData

def preprocess_instance(file_path, config: DictConfig):
    """
    Preprocess a single Max-3-Cut LP file into the format expected by DiffILO.
    """
    filename = os.path.basename(file_path).split('.')[0]
    
    # Extract features using SCIP
    m = scip.Model()
    m.hideOutput(True)
    m.readProblem(file_path)
    
    # Get problem dimensions
    ncons = m.getNConss()
    nvars = m.getNVars()
    mvars = m.getVars()
    
    # Initialize variable features
    variable_features = []
    for i in range(len(mvars)):
        # DiffILO uses 5 features per variable
        tp = [0] * 5
        tp[3] = 0      # Min coefficient 
        tp[4] = 1e+20  # Max coefficient
        variable_features.append(tp)
    
    # Create variable mapping
    v_map = {}
    for indx, v in enumerate(mvars):
        v_map[v.name] = indx
    
    # Extract objective coefficients
    obj = m.getObjective()
    indices_spr = [[], []]
    values_spr = []
    
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            values_spr.append(1)
        variable_features[v_indx][0] = v
    
    # Process constraints
    cons = m.getConss()
    new_cons = []
    for c in cons:
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            continue
        new_cons.append(c)
    
    cons = new_cons
    ncons = len(cons)
    lcons = ncons
    
    # Extract constraint features
    constraint_features = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        
        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)
            variable_features[v_indx][2] += 1
            variable_features[v_indx][1] += coeff[k] / lcons
            variable_features[v_indx][3] = max(variable_features[v_indx][3], coeff[k])
            variable_features[v_indx][4] = min(variable_features[v_indx][4], coeff[k])
            summation += coeff[k]
        
        llc = max(len(coeff), 1)
        constraint_features.append([summation / llc, llc, rhs])
    
    # Convert to PyTorch tensors with proper types and contiguous memory
    variable_features = torch.tensor(variable_features, dtype=torch.float32).contiguous()
    constraint_features = torch.tensor(constraint_features, dtype=torch.float32).contiguous()
    
    # Normalize variable features
    clip_max = [20000, 1, variable_features[:, 2].max().item()]
    clip_min = [0, -1, 0]
    variable_features[:, 0] = torch.clamp(variable_features[:, 0], clip_min[0], clip_max[0])
    
    maxs = variable_features.max(dim=0)[0]
    mins = variable_features.min(dim=0)[0]
    diff = maxs - mins
    diff[diff == 0] = 1.0  # Avoid division by zero
    
    variable_features = (variable_features - mins) / diff
    
    # Normalize constraint features
    maxs = constraint_features.max(dim=0)[0]
    mins = constraint_features.min(dim=0)[0]
    diff = maxs - mins
    diff[diff == 0] = 1.0  # Avoid division by zero
    
    constraint_features = (constraint_features - mins) / diff
    
    # Create edge indices and features
    edge_indices = torch.tensor(indices_spr, dtype=torch.long).contiguous()
    edge_features = torch.tensor(values_spr, dtype=torch.float32).view(-1, 1).contiguous()
    
    # Create a BipartiteNodeData object (DiffILO's graph format)
    graph = BipartiteNodeData(
        constraint_features=constraint_features, 
        edge_indices=edge_indices, 
        edge_features=edge_features, 
        variable_features=variable_features
    )
    
    # Save the graph
    sample_path = os.path.join(config.paths.data_samples_dir, f"{filename}.pkl")
    with open(sample_path, 'wb') as f:
        pickle.dump(graph, f)
    
    # Use ecole to extract standard form matrices (A, b, c)
    try:
        model = ecole.scip.Model.from_file(file_path)
        obs = ecole.observation.MilpBipartite().extract(model, True)
        
        A_i = torch.tensor(np.array(obs.edge_features.indices, dtype=int), dtype=torch.long)
        A_e = torch.tensor(obs.edge_features.values, dtype=torch.float32)
        A = torch.sparse_coo_tensor(A_i, A_e).to_dense()
        
        b = torch.tensor(obs.constraint_features, dtype=torch.float32)
        c = torch.tensor(obs.variable_features[:, 0].reshape(-1, 1), dtype=torch.float32)
        
        tensor_path = os.path.join(config.paths.data_tensors_dir, f"{filename}.pkl")
        with open(tensor_path, 'wb') as f:
            pickle.dump((A, b, c), f)
    
    except Exception as e:
        logging.error(f"Error with ecole extraction for {filename}: {e}")
    
    return True

@hydra.main(version_base=None, config_path="config", config_name="preprocess")
def preprocess(config: DictConfig):
    """
    Main function to preprocess the Max-3-Cut dataset.
    """
    logging.basicConfig(
        format="[%(asctime)s]: %(message)s",
        level=logging.INFO
    )

    # Create output directories
    os.makedirs(config.paths.data_samples_dir, exist_ok=True)
    os.makedirs(config.paths.data_tensors_dir, exist_ok=True)
    os.makedirs(config.paths.data_solution_dir, exist_ok=True)
    os.makedirs(config.paths.data_solve_log_dir, exist_ok=True)
    
    if config.mode == "train":
        data_dir = config.paths.train_data_dir
    else:
        data_dir = config.paths.test_data_dir
    
    # Find all LP files
    files = [f for f in os.listdir(data_dir) if f.endswith('.lp')]
    
    logging.info(f"Preprocessing the Max-3-Cut dataset {config.dataset.name} ({config.dataset.full_name}).")
    logging.info(f"Found {len(files)} LP files in {data_dir}")
    
    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(data_dir, file)
        preprocess_instance(file_path, config)
    
    logging.info(f"Preprocessing done.")
    logging.info(f"The preprocessed data files are saved in {config.paths.preprocess_dir}.")

if __name__ == '__main__':
    preprocess() 