import os.path
import pickle
import multiprocessing as mp
import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from functools import partial
import logging
import ecole
import torch
import pyscipopt as scip

def preprocess_(file: str, config: DictConfig):
    """
    Preprocess a single instance file.
    """
    
    file_path = os.path.join(config.paths.train_data_dir, file)  
    
    # extract the features from the instance file
    m = scip.Model()
    m.hideOutput(True)
    m.readProblem(file_path)

    ncons = m.getNConss()
    nvars = m.getNVars()
    mvars = m.getVars()

    variable_features = []

    for i in range(len(mvars)):
        tp = [0] * 5
        tp[3] = 0
        tp[4] = 1e+20

        variable_features.append(tp)
        
    v_map = {}
    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            values_spr.append(1)
        variable_features[v_indx][0] = v
        obj_node[0] += v
        obj_node[1] += 1
        
    if obj_node[1] > 0:
        obj_node[0] /= obj_node[1]

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)

    lcons = ncons
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
    
    variable_features = torch.as_tensor(variable_features, dtype=torch.float32)
    constraint_features = torch.as_tensor(constraint_features, dtype=torch.float32)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons, nvars))
    clip_max = [20000, 1, torch.max(variable_features, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    variable_features[:, 0] = torch.clamp(variable_features[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(variable_features, 0)[0]
    mins = torch.min(variable_features, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    variable_features = variable_features - mins
    variable_features = variable_features / diff

    maxs = torch.max(constraint_features, 0)[0]
    mins = torch.min(constraint_features, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    constraint_features = constraint_features - mins
    constraint_features = constraint_features / diff
    
    # save the preprocessed data
    edge_indices= torch.LongTensor(np.array(indices_spr, dtype=int))
    edge_features = torch.FloatTensor(values_spr).reshape(-1,1)
    graph = [constraint_features, edge_indices, edge_features, variable_features]
    sample_path = os.path.join(config.paths.data_samples_dir, file.split(".")[0]+'.pkl')
    pickle.dump(graph, open(sample_path, 'wb'))
    
    # save the tensor data
    # A, b, c in the standard form min c^T x s.t. Ax <= b 
    model = ecole.scip.Model.from_file(file_path)
    obs = ecole.observation.MilpBipartite().extract(model, True)
    A_i = torch.LongTensor(np.array(obs.edge_features.indices, dtype=int))
    A_e = torch.FloatTensor(obs.edge_features.values)
    A = torch.sparse_coo_tensor(A_i, A_e).to_dense()         
    c = torch.FloatTensor(obs.variable_features[:,0].reshape(-1,1))
    b = torch.FloatTensor(obs.constraint_features)
    sample_tensor_path = os.path.join(config.paths.data_tensors_dir, file.split(".")[0]+'.pkl')
    pickle.dump((A, b, c), open(sample_tensor_path, 'wb'))

@hydra.main(version_base=None, config_path="config", config_name="preprocess")
def preprocess(config: DictConfig):
    logging.basicConfig(
        format="[%(asctime)s]: %(message)s",
        level=logging.DEBUG
    )

    os.makedirs(config.paths.data_samples_dir, exist_ok=True)
    os.makedirs(config.paths.data_solution_dir, exist_ok=True)
    os.makedirs(config.paths.data_solve_log_dir, exist_ok=True)
    os.makedirs(config.paths.data_tensors_dir, exist_ok=True)
    
    files = os.listdir(config.paths.train_data_dir)
    
    logging.info(f"Preprocessing the dataset {config.dataset.name} ({config.dataset.full_name}).")
    
    func = partial(preprocess_, config=config)
    with mp.Pool(config.num_workers) as pool:
        for _ in tqdm(pool.imap(func, files), total=len(files), desc="Collect Sample"):
            pass
    
    logging.info(f"Preprocessing done.")
    logging.info(f"The preprocessed data files are saved in {config.paths.preprocess_dir}.")


if __name__ == '__main__':
    preprocess()