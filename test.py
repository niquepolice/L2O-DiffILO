
import gurobipy as gp
import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from src.utils import set_seed, set_cpu_num
import src.tb_writter as tb_writter
from src.model import GNNPredictor, BipartiteNodeData
import logging
from functools import partial
import torch.multiprocessing as mp
from tqdm import tqdm
import ecole
import json
import pyscipopt as scip

mp.set_start_method('spawn', force=True)

delta = 200

def solve(test_ins_name, model: GNNPredictor, test_data_dir, log_dir, json_dir, mu):
    ins_name_to_read = os.path.join(test_data_dir, test_ins_name)
    
    m = scip.Model()
    m.hideOutput(True)
    m.readProblem(ins_name_to_read)

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
    
    edge_indices= torch.LongTensor(np.array(indices_spr, dtype=int))
    edge_features = torch.FloatTensor(values_spr).reshape(-1,1)
    
    graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features).cuda()
    
    m = ecole.scip.Model.from_file(ins_name_to_read)
    obs = ecole.observation.MilpBipartite().extract(m, True)

    constraint_features = torch.FloatTensor(obs.constraint_features)
    edge_indices = torch.LongTensor(np.array(obs.edge_features.indices, dtype=int))
    edge_features = torch.FloatTensor(obs.edge_features.values.reshape((-1,1)))
    variable_features = torch.FloatTensor(obs.variable_features)

    b = torch.FloatTensor(obs.constraint_features).numpy()
    A_i = torch.LongTensor(np.array(obs.edge_features.indices, dtype=int))
    A_e = torch.FloatTensor(obs.edge_features.values.reshape(-1))
    c = torch.FloatTensor(obs.variable_features[:,0].reshape(-1,1)).numpy()
    A = torch.sparse_coo_tensor(A_i, A_e).to_dense().numpy()

    with torch.no_grad():
        model = model.cuda()
        logits = model.forward(graph)[0]
        
    model = model.cpu()
    pred = logits.sigmoid().cpu().numpy()
    x = np.random.binomial(1, pred.squeeze(), size=(1000, len(pred)))
    cons = np.maximum(A @ x.T - b, 0).sum(0)
    idx = np.where(cons == 0)[0]
    if len(idx) > 0:
        best_idx = np.argmin(x[idx] @ c)
        best_x = x[idx][best_idx]
    else:
        best_x = np.argmin((x @ c).squeeze() + mu * cons)
        best_x = x[best_x]
    

    gp.setParam('LogToConsole', 1)
    m = gp.read(ins_name_to_read)
    m.Params.TimeLimit = 1000
    m.Params.Threads = 1
    m.Params.MIPFocus = 1
    m.Params.LogFile = os.path.join(log_dir, f'{test_ins_name}.log')

    error = 0
    
    for i, v in enumerate(m.getVars()):
        v_0 = best_x[i]
        v.Start = v_0
        
        tmp_var = m.addVar(name=f'alp_{v}', vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, obj=0)
        if v_0 == 0:
            m.addConstr(tmp_var == v)
        elif v_0 == 1:
            m.addConstr(tmp_var == 1 - v)
        error += tmp_var
    
    m.addConstr(error <= delta, name="sum_alpha")
    
    m.optimize()
    
    result = {
        "obj": m.ObjVal,
        "time": m.Runtime,
        "nnodes": m.NodeCount,
        "gap": m.MIPGap,
        "stat": m.status
    }
    
    print(result)
    
    json_path = os.path.join(json_dir, f'{test_ins_name}.json')
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

@hydra.main(version_base=None, config_path="config", config_name="test")
def test(config: DictConfig):
    # Initialize settings
    set_seed(config.seed)
    set_cpu_num(config.num_workers + 1)
    tb_writter.set_logger(config.paths.tensorboard_dir)
    
    # Create output directories
    test_dir = config.paths.test_dir
    log_dir = os.path.join(test_dir, "logs") 
    json_dir = os.path.join(test_dir, "jsons")
    for directory in [test_dir, log_dir, json_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load and prepare model
    model_path = os.path.join(config.model_dir, "models", "model.pth")
    model = GNNPredictor(config.model)
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'), strict=False)
    model.eval()

    # Get test files and run parallel solving
    files = os.listdir(config.paths.test_data_dir)
    solve_func = partial(solve, 
                        model=model,
                        test_data_dir=config.paths.test_data_dir,
                        log_dir=log_dir,
                        json_dir=json_dir,
                        mu=config.mu)
                        
    with mp.Pool(config.num_workers) as pool:
        list(tqdm(pool.imap(solve_func, files), 
                 total=len(files),
                 desc="Solving"))
        

if __name__ == "__main__":
    test()