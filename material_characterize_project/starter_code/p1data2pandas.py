import json
import pandas as pd
import os
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np 
import random 
import torch
import mendeleev

DIRECTED_GRAPH = False

MEND_DICT = dict()

PAULING_EN_STATIC = {
    'H':   2.20,   # Hydrogen
    'He':  0.00,   # no defined EN, set to 0.0
    'Li':  0.98,
    'Be':  1.57,
    'B':   2.04,
    'C':   2.55,
    'N':   3.04,
    'O':   3.44,
    'F':   3.98,
    'Ne':  0.00,   # no defined EN, set to 0.0
    'Na':  0.93,
    'Mg':  1.31,
    'Al':  1.61,
    'Si':  1.90,
    'P':   2.19,
    'S':   2.58,
    'Cl':  3.16,
    'Ar':  0.00,   # no defined EN, set to 0.0
    'K':   0.82,
    'Ca':  1.00,
    'Sc':  1.36,
    'Ti':  1.54,
    'V':   1.63,
    'Cr':  1.66,
    'Mn':  1.55,
    'Fe':  1.83,
    'Co':  1.88,
    'Ni':  1.91,
    'Cu':  1.90,
    'Zn':  1.65,
    'Ga':  1.81,
    'Ge':  2.01,
    'As':  2.18,
    'Se':  2.55,
    'Br':  2.96,
    'Kr':  3.00,   # krypton has a small EN
    'Rb':  0.82,
    'Sr':  0.95,
    'Y':   1.22,
    'Zr':  1.33,
    'Nb':  1.60,
    'Mo':  2.16,
    'Tc':  1.90,
    'Ru':  2.20,
    'Rh':  2.28,
    'Pd':  2.20,
    'Ag':  1.93,
    'Cd':  1.69,
    'In':  1.78,
    'Sn':  1.96,
    'Sb':  2.05,
    'Te':  2.10,
    'I':   2.66,
    'Xe':  2.60,   # xenon has a reported EN
}

def get_atom_pauling_en(atom_sym: str) -> float:
    """
    1. Cache the mendeleev.Element in mend_dict.
    2. Try elem.electronegativity_pauling (might be a method).
    3. If missing/None, fall back to PAULING_EN_STATIC or 0.0.
    Always returns a float.
    """
    #gets the element
    if atom_sym not in MEND_DICT:
        MEND_DICT[atom_sym] = mendeleev.element(atom_sym)
    elem = MEND_DICT[atom_sym]

    # attempt to get the attribute is the value exists in mendeleev
    en_attr = getattr(elem, 'electronegativity_pauling', None)

    # it might be a method, so have to call it and extract the value
    if callable(en_attr):
        val = en_attr()
    else:
        val = en_attr

    # if we still have nothing, this means it doesnt exist in mendeleev so use use the static table
    if val is None or not isinstance(val, (int, float)):
        return PAULING_EN_STATIC.get(atom_sym, 0.0)

    return float(val)

def calculate_electronegativity_stats(g: nx.Graph):
    """
    Returns (avg_en, min_en, max_en) over all nodes in graph g,
    looking up each node’s data['atom_type'] as the symbol.
    """
    # extract every node’s Pauling EN
    ens = [
        get_atom_pauling_en(data['atom_type'])
        for _, data in g.nodes(data=True)
    ]
    # guard against empty graphs
    if not ens:
        return 0.0, 0.0, 0.0
    return sum(ens)/len(ens), min(ens), max(ens)

def get_atoms(g):
    res = {}

    for node, data in g.nodes(data=True): # loop through all of the nodes explicitly
        if (data['atom_type'] in res): res[data['atom_type']] += 1 # iterate dictionary vals
        else: res[data['atom_type']] = 1
    return res

def get_atom_weight(s): 
    if s not in MEND_DICT: 
        MEND_DICT[s] = mendeleev.element(s)
    return MEND_DICT[s].atomic_weight

def get_atom_group(s): 
    if s not in MEND_DICT: 
        MEND_DICT[s] = mendeleev.element(s)
    return MEND_DICT[s].group_number

def get_atom_period(s): 
    if s not in MEND_DICT: 
        MEND_DICT[s] = mendeleev.element(s)
    return MEND_DICT[s].period

def get_atom_volume(s): 
    if s not in MEND_DICT: 
        MEND_DICT[s] = mendeleev.element(s)
    return MEND_DICT[s].atomic_volume

def get_atom_num_protons(s): 
    if s not in MEND_DICT: 
        MEND_DICT[s] = mendeleev.element(s)
    return MEND_DICT[s].atomic_number

def calculate_molecular_weight(g): 
    return sum(list( get_atom_weight(data['atom_type']) for _, data in g.nodes(data=True)))

def calculate_average_degree(g):
    seen = set()
    node_to_degree = dict()
    for u,v in g.edges(): 
        if (u in seen and v in seen): 
            node_to_degree[u] += 1
            node_to_degree[v] += 1
        elif u in seen: 
            node_to_degree[u] += 1
            seen.add(v)
            node_to_degree[v] = 1
        elif v in seen: 
            node_to_degree[v] += 1
            seen.add(u)
            node_to_degree[u] = 1
        else: 
            seen.add(u)
            node_to_degree[u] = 1
            node_to_degree[v] = 1
    return sum(node_to_degree.values()) / max(len(node_to_degree),1) 

def add_weights_and_connectivity_to_df(g):
    w = calculate_molecular_weight(g)
    d = calculate_average_degree(g)
    l = len(g.nodes())

    return w, d, l


def add_global_node(g, data):
    
    global_node = "global"

    pre_global = list(g.nodes)

    g.add_node(global_node)



    for key, value in data.items():
        g.nodes["global"][key] = value

    # Connect the global node to all existing nodes (outgoing edges only)
    g.add_edges_from([(global_node, n) for n in pre_global])

def add_node_features_to_graph(g): 

    to_add = {'atomic_weight' : get_atom_weight,
              'pauling_en' : get_atom_pauling_en,
              'period' : get_atom_period,
              'volume': get_atom_volume,
              'numprotons' : get_atom_num_protons,
              } 
    for node, data in g.nodes(data=True):
        for key, value in to_add.items(): 
            data[key] = value(data['atom_type'])
            print(data)

def load_data_from_file(filename, directed):
    """
    Load a dictionary of graphs from JSON file.
    """
    with open(filename, "r") as file_handle:
        string_dict = json.load(file_handle)
    return _load_data_from_string_dict(string_dict, directed)

def load_data_from_string(json_string):
    """
    Load a dictionary of graphs from JSON string.
    """
    string_dict = json.loads(json_string)
    return _load_data_from_string_dict(string_dict)

def _load_data_from_string_dict(string_dict, directed):
	result_dict = {}
	for key in string_dict:
		graph = nx.node_link_graph(string_dict[key], edges="edges", directed = directed)
		result_dict[key] = graph
	return result_dict

def write_data_to_json_string(graph_dict, **kwargs):
    """
    Write dictionary of graphs to JSON string.
    """
    json_string = json.dumps(graph_dict, default=nx.node_link_data, **kwargs)
    return json_string


def write_data_to_json_file(graph_dict, filename, **kwargs):
    """
    Write dictionary of graphs to JSON file.
    """
    with open(filename, "w") as file_handle:
        file_handle.write(write_data_to_json_string(graph_dict, **kwargs))

def smiles_to_iupac(smiles):
    # Convert SMILES string to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "NOT FOUND"

    # Generate IUPAC name
    iupac_name = rdMolDescriptors.CalcMolFormula(mol)  # Molecular formula
    return iupac_name
         
def load_data_frame_from_file(file_name, directed):
    res = load_data_from_file(file_name, directed) 
    dat = []

    for i, (smile, graph) in enumerate(res.items()):
        molecule_name = smiles_to_iupac(smile)
        connectivity = calculate_average_degree(graph)
        molecule_weight = calculate_molecular_weight(graph)
        avg_pauling, min_pauling, max_pauling = calculate_electronegativity_stats(graph)
        add_node_features_to_graph(graph)
        
        graph_data = {'molecule_weight' : molecule_weight,
                      'average_connectivity' : connectivity,
                      'avg_pauling': avg_pauling,
                      'min_pauling': min_pauling,
                      'max_pauling': max_pauling,
                      }

        add_global_node(graph, graph_data)


        dat.append({
            'smile': smile,
            'graph': graph,
            'name': molecule_name,
            'average_connectivity': connectivity,
            'molecular_weight': molecule_weight,
            'avg_pauling': avg_pauling,
            'min_pauling': min_pauling,
            'max_pauling': max_pauling,
            })
        
    df = pd.DataFrame(dat)
    return df

def build_node_feature_tensor(G):
    features = []

    # Iterate nodes in ascending ID for consistent order
    for node_id, node_data in sorted(G.nodes(data=True), key=lambda x: x[0]):
        atom_type = node_data['atom_type']
        charge = float(node_data.get('formal_charge', 0))
        
        # One-hot encode atom type
        one_hot = np.zeros(num_atom_types, dtype=np.float32)
        one_hot[type_to_index[atom_type]] = 1.0
        
        
        # Concatenate
        feature_vector = np.concatenate([one_hot, [charge]])
        
        features.append(feature_vector)

    feature_matrix = np.stack(features, axis=0)

    return torch.from_numpy(feature_matrix)

def encode_bond_features(G):
    """
    Takes a NetworkX graph G and returns:
      - edge_index: a [2 × E] torch.LongTensor of source/target node pairs
      - edge_attr: a [E × num_bond_types] torch.FloatTensor of one-hot bond types

    Prints debug info so you can verify shapes and contents.
    """
    # How many “real” edges are in G?
    n_edges = G.number_of_edges()
    print(f"\nEncoding bonds for graph with {n_edges} edges (before doubling)")

    # CASE A: no edges at all
    if n_edges == 0:
        # We still need to return an “empty” edge_index and edge_attr of the right shape:
        ei = torch.empty((2, 0), dtype=torch.long)
        ea = torch.empty((0, num_bond_types), dtype=torch.float32)
        print(" No edges present. ✔")
        print(" edge_index shape:", ei.shape, "(should be [2, 0])")
        print(" edge_attr shape:", ea.shape, "(should be [0, 4])")
        return ei, ea

    # CASE B: some edges exist
    edge_list = []
    attr_list = []
    for u, v, data in G.edges(data=True):
        btype = data.get("bond_type", "NONE")
        idx   = bond_to_index.get(btype, bond_to_index["NONE"])
        onehot = np.zeros(num_bond_types, dtype=np.float32)
        onehot[idx] = 1.0

        # add both directions for undirected
        edge_list.append([u, v])
        attr_list.append(onehot)
        edge_list.append([v, u])
        attr_list.append(onehot)

        print(f" Edge {u} ↔ {v}: type={btype}, idx={idx}, one-hot={onehot.tolist()}")

    # Turn into numpy arrays, then into torch
    edge_index = torch.tensor(np.array(edge_list, dtype=int).T, dtype=torch.long)
    edge_attr  = torch.tensor(np.array(attr_list, dtype=np.float32), dtype=torch.float32)

    print(" edge_index shape:", edge_index.shape, "(2 × #edges×2)")
    print(" edge_attr shape: ", edge_attr.shape, "(#edges×2 × 4)")
    return edge_index, edge_attr

def nx_to_pyg(G, smile):
    """
    Convert a NetworkX graph G into a PyG Data object,
    including:
      • x         – node features ([N, num_node_features])
      • edge_index– bond connectivity ([2, 2*E])
      • edge_attr – one-hot bond types ([2*E, num_bond_types])
      • y         – per-node binding energies ([N])
      • mask      – boolean mask for valid y entries ([N])
    Prints shapes so you can verify.
    """
    print(f"\n--- Converting graph: {smile} ---")
    # 1) Node features
    x = build_node_feature_tensor(G)
    # 2) Edge features
    edge_index, edge_attr = encode_bond_features(G)
    # 3) Targets + mask
    y, mask = extract_node_targets_with_mask(G)
    # 4) Bundle into Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask)
    # 5) Diagnostics
    return data

if __name__ == "__main__":
 
    file_path = '/home/ivan/Notes/material_characterization/ai-sci-hackathon-2025/material_characterize_project/graph_data.json'

    df = load_data_frame_from_file(file_path, directed = directed)

    



    
