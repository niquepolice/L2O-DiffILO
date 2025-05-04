import os
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
import argparse

def generate_graph(n_nodes=50, max_neighbors=10, p_connection=0.5, seed=42):
    """Generate a random geometric graph for Max-3-Cut problem."""
    rs = np.random.RandomState(seed)
    
    # Set positions
    xy = rs.rand(n_nodes, 2)
    
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=max_neighbors+1).fit(xy)
    _, indices = nbrs.kneighbors(xy)
    
    # Create adjacency matrix and edge list
    adj = lil_matrix((n_nodes, n_nodes))
    e_list = []
    weights = []
    
    for i, neighbors in enumerate(indices):
        neighs = np.asarray(neighbors[1:])
        edge_mask = (rs.random(len(neighs)) > p_connection) & (neighs > i)
        w = np.exp(rs.randn(len(neighs)))
        weights += w[edge_mask].tolist()
        adj[i, neighs] = w * edge_mask
        for n in neighs[edge_mask]:
            e_list.append([i, n])
    
    adj = adj + adj.T  # Make the graph undirected
    weights = np.asarray(weights)
    
    # Create NetworkX graph
    G = nx.from_scipy_sparse_array(adj)
    
    # Add edge weights to NetworkX graph
    for i, (u, v) in enumerate(e_list):
        G[u][v]['weight'] = weights[i]
    
    return G

def write_max3cut_lp(G, output_file):
    """
    Write a Max-3-Cut problem in LP format based on the binary ILP formulation:
    min sum_{uv in E} t_uv * w_uv
    s.t. x^0_i + x^1_i + x^2_i = 1 for all i in V
         t_uv >= x^k_u + x^k_v - 1 for all k in {0,1,2} and all uv in E
    """
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    
    with open(output_file, 'w') as f:
        # Write objective function (minimize)
        f.write("Minimize\n")
        obj_terms = []
        for u, v, data in edges:
            weight = data.get('weight', 1.0)
            obj_terms.append(f"{weight} t_{u}_{v}")
        f.write(" + ".join(obj_terms))
        f.write("\n\n")
        
        # Write constraints
        f.write("Subject To\n")
        
        # Each node must be in exactly one partition
        for i in nodes:
            f.write(f"node_{i}: x_0_{i} + x_1_{i} + x_2_{i} = 1\n")
        
        # Edge constraints for each partition
        constraint_idx = 1
        for u, v, _ in edges:
            for k in range(3):
                f.write(f"edge_{constraint_idx}: t_{u}_{v} - x_{k}_{u} - x_{k}_{v} >= -1\n")
                constraint_idx += 1
        
        # Variable types (all binary)
        f.write("\nBinary\n")
        
        # Node partition variables
        for i in nodes:
            for k in range(3):
                f.write(f"x_{k}_{i}\n")
        
        # Edge variables
        for u, v, _ in edges:
            f.write(f"t_{u}_{v}\n")
        
        # End of file
        f.write("\nEnd\n")

def generate_dataset(n_instances, output_dir, min_nodes=20, max_nodes=50, 
                     max_neighbors=10, p_connection=0.5, seed=42):
    """Generate a dataset of LP files for Max-3-Cut problems."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    rs = np.random.RandomState(seed)
    
    # Generate training instances (80%)
    train_count = int(0.8 * n_instances)
    test_count = n_instances - train_count
    
    print(f"Generating {train_count} training instances...")
    for i in range(train_count):
        n_nodes = rs.randint(min_nodes, max_nodes + 1)
        G = generate_graph(n_nodes=n_nodes, max_neighbors=max_neighbors, 
                           p_connection=p_connection, seed=rs.randint(10000))
        output_file = os.path.join(output_dir, "train", f"max3cut_train_{i}.lp")
        write_max3cut_lp(G, output_file)
        print(f"Generated {output_file} with {n_nodes} nodes and {G.number_of_edges()} edges")
    
    print(f"Generating {test_count} test instances...")
    for i in range(test_count):
        n_nodes = rs.randint(min_nodes, max_nodes + 1)
        G = generate_graph(n_nodes=n_nodes, max_neighbors=max_neighbors, 
                           p_connection=p_connection, seed=rs.randint(10000))
        output_file = os.path.join(output_dir, "test", f"max3cut_test_{i}.lp")
        write_max3cut_lp(G, output_file)
        print(f"Generated {output_file} with {n_nodes} nodes and {G.number_of_edges()} edges")
    
    print(f"Dataset generation complete. Files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Max-3-Cut dataset for DiffILO")
    parser.add_argument("--n_instances", type=int, default=100, 
                        help="Number of problem instances to generate")
    parser.add_argument("--output_dir", type=str, default="data/M3C", 
                        help="Output directory for dataset")
    parser.add_argument("--min_nodes", type=int, default=20, 
                        help="Minimum number of nodes in graphs")
    parser.add_argument("--max_nodes", type=int, default=50, 
                        help="Maximum number of nodes in graphs")
    parser.add_argument("--max_neighbors", type=int, default=10, 
                        help="Maximum number of neighbors to consider")
    parser.add_argument("--p_connection", type=float, default=0.5, 
                        help="Probability of connection between neighbors")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    args = parser.parse_args()
    
    generate_dataset(
        n_instances=args.n_instances,
        output_dir=args.output_dir,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        max_neighbors=args.max_neighbors,
        p_connection=args.p_connection,
        seed=args.seed
    ) 