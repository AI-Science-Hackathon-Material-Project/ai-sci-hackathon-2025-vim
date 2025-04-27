# %%
import torch 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
import random
from p1data2pandas import load_data_frame_from_file

# %%

# loads the data into a dataframe, currently the name and type columns are unassigned
file_name = '../graph_data.json'
df = load_data_frame_from_file(file_name)
df

# %%
import json
import networkx as nx
from networkx.readwrite.json_graph import node_link_graph

# 1) Load the JSON
with open('../graph_data.json', 'r') as f:
    raw_data = json.load(f)

# 2) Convert to NetworkX, explicitly naming the edges field
graphs = {}
for smile, gdata in raw_data.items():
    G = node_link_graph(
        gdata,
        directed=False,
        multigraph=False,
        edges="edges"      # ← tell it to use the "edges" list in your JSON
    )
    graphs[smile] = G

# 3) Sanity check
print(f"Loaded {len(graphs)} molecules.")
for i, (smile, G) in enumerate(list(graphs.items())[:5], 1):
    print(f"{i}. {smile} → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# %%
atom_types = set()
for G in graphs.values():
    for _, node_data in G.nodes(data=True):
        atom_types.add(node_data['atom_type'])
atom_types = sorted(atom_types)
num_atom_types = len(atom_types)
print(f"Found {num_atom_types} unique atom types")  
# Expected: Found 55 unique atom types
print("Sample atom types:", atom_types[:5], "...")  
# Expected sample: ['Ag', 'Al', 'As', 'B', 'Br', ...]

type_to_index = {atom: idx for idx, atom in enumerate(atom_types)}
print("Index of some atom types:", {k: type_to_index[k] for k in ['C', 'O', 'Ag']})  
# Expected: {'C': some_index, 'O': some_index, 'Ag': some_index}

# --- Step 3: Build node feature tensor with debug prints ---
def build_node_feature_tensor(G):
    features = []
    print(f"\nBuilding node features for graph with {G.number_of_nodes()} nodes")
    # Iterate nodes in ascending ID for consistent order
    for node_id, node_data in sorted(G.nodes(data=True), key=lambda x: x[0]):
        atom_type = node_data['atom_type']
        charge = float(node_data.get('formal_charge', 0))
        
        # One-hot encode atom type
        one_hot = np.zeros(num_atom_types, dtype=np.float32)
        one_hot[type_to_index[atom_type]] = 1.0
        
        print(f" Node {node_id}: atom_type={atom_type}, index={type_to_index[atom_type]}")
        print(f"  One-hot vector (first 10 dims): {one_hot[:10]} ...")
        print(f"  Formal charge: {charge}")
        
        # Concatenate
        feature_vector = np.concatenate([one_hot, [charge]])
        print(f"  Feature vector length: {feature_vector.shape[0]}; sample values: {feature_vector[:10]} ...")
        
        features.append(feature_vector)
    
    # Stack into matrix
    feature_matrix = np.stack(features, axis=0)
    print(f"Built feature matrix of shape: {feature_matrix.shape}")  
    # Expected shape: [num_nodes, 56]
    print("Matrix sample rows (first 2):\n", feature_matrix[:2])
    feature_matrix = np.stack(features, axis=0).astype(np.float32)
    return torch.from_numpy(feature_matrix)

# --- Step 4: Example on first graph ---
first_smile, first_graph = next(iter(graphs.items()))
print(f"\nExample for SMILES: {first_smile}")
X_first = build_node_feature_tensor(first_graph)
print(f"Resulting tensor shape: {X_first.shape}")  

print("All atom types in dataset:", atom_types)


# %%
import numpy as np
import torch  # in your environment, you’ll convert these numpy arrays to torch tensors

# 1. Define the bond types and mapping (same as before)
bond_types = ["SINGLE", "DOUBLE", "TRIPLE", "NONE"]
bond_to_index = {b: i for i, b in enumerate(bond_types)}
num_bond_types = len(bond_types)

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

# Example usage:
# Pick the second graph in your `graphs` dictionary
second_smile, second_graph = list(graphs.items())[1]
print(f"\nRunning on second graph: {second_smile}")

# Encode its edges
ei2, ea2 = encode_bond_features(second_graph)

# Inspect the first few entries
print("edge_index (first 6 columns):\n", ei2[:, :6])
print("edge_attr (first 6 rows):\n", ea2[:6])



# %%
import torch

def extract_node_targets_with_mask(G):
    """
    Returns:
      - y: a FloatTensor of length = #nodes, with 0.0 where label is missing
      - mask: a BoolTensor of same length, True for valid labels, False otherwise
    """
    targets = []
    mask_vals = []
    print(f"\nExtracting targets (with mask) for graph with {G.number_of_nodes()} nodes")
    
    for node_id, data in sorted(G.nodes(data=True), key=lambda x: x[0]):
        bes = data.get("binding_energies", [])
        # Valid if exactly one energy and not placeholder -1
        if len(bes) == 1 and bes[0] != -1:
            val = float(bes[0])
            valid = True
        else:
            val = 0.0    # placeholder value
            valid = False
        print(f" Node {node_id}: binding_energies={bes} → y={val}, mask={valid}")
        targets.append(val)
        mask_vals.append(valid)
    
    y = torch.tensor(targets, dtype=torch.float32)
    mask = torch.tensor(mask_vals, dtype=torch.bool)
    print(f"Built y of shape {y.shape}, mask of shape {mask.shape}")
    print(f" Valid labels: {mask.sum().item()}/{mask.numel()}")
    return y, mask

# Example test on the second graph
second_smile, second_graph = list(graphs.items())[1]
print(f"\n>>> Testing on {second_smile}")
y2, mask2 = extract_node_targets_with_mask(second_graph)


# %%
from torch_geometric.data import Data   # make sure you’ve installed torch_geometric
import torch

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
    print("Created Data object:")
    print(f" x.shape       = {data.x.shape}    # nodes × features")
    print(f" edge_index    = {data.edge_index.shape}  # 2 × (2×original edges)")
    print(f" edge_attr     = {data.edge_attr.shape}   # (2×original edges) × bond-types")
    print(f" y.shape       = {data.y.shape}    # nodes")
    print(f" mask.shape    = {data.mask.shape} # nodes")
    print(f" valid targets = {data.mask.sum().item()}/{data.mask.numel()}")
    return data

# --- Test on second graph ---
second_smile, second_graph = list(graphs.items())[1]
print(f"\n>>> Running nx_to_pyg on second graph: {second_smile}")
data2 = nx_to_pyg(second_graph, second_smile)


# %%
import torch
from torch_geometric.loader import DataLoader

# Number of graphs to demonstrate on (you can set this = len(graphs) later)
n_graphs = 4  

# 1) Convert the first n_graphs into Data objects
print(f"Converting the first {n_graphs} graphs...")
data_list = []
for idx, (smile, G) in enumerate(list(graphs.items())[:n_graphs]):
    print(f" Graph {idx+1}/{n_graphs}: {smile}")
    data = nx_to_pyg(G, smile)
    data_list.append(data)
print(f"→ Prepared {len(data_list)} Data objects.\n")

# 2) Create a DataLoader for batching
batch_size = 2
loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
print(f"DataLoader ready: batch_size = {batch_size}, shuffle = True\n")

# 3) Peek at one batch
batch = next(iter(loader))
print("Batch contents:")
print(" • Keys:", batch.keys)
print(" • x.shape       =", batch.x.shape,
      "(total_nodes × feature_dim)")
print(" • edge_index    =", batch.edge_index.shape,
      "(2 × total_edges×2)")
print(" • edge_attr     =", batch.edge_attr.shape,
      "(total_edges×2 × bond_types)")
print(" • y.shape       =", batch.y.shape,
      "(total_nodes)")
print(" • mask.shape    =", batch.mask.shape,
      "(total_nodes)")
print(" • batch.batch   =", batch.batch.shape,
      "(total_nodes)  # graph assignment for each node)\n")

# Sample a few values for sanity
print(" Sample y[:6]    =", batch.y[:6].tolist())
print(" Sample mask[:6] =", batch.mask[:6].tolist())
print(" Sample batch IDs[:6] =", batch.batch[:6].tolist())


# %%
# === Final Cell: Tiny GCN with Casting and No Debug Prints ===

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# 1) Build and cast Data objects
data_list = []
for smile, G in graphs.items():
    data = nx_to_pyg(G, smile)
    data.x         = data.x.float()         # ensure float32
    data.edge_attr = data.edge_attr.float() # ensure float32
    data_list.append(data.to(device))

# Quick sanity check on first graph
print("Example x dtype/shape:", data_list[0].x.dtype, data_list[0].x.shape)

# 2) Split into train/test (80/20)
torch.manual_seed(0)
n_train      = int(0.8 * len(data_list))
train_loader = DataLoader(data_list[:n_train], batch_size=16, shuffle=True)
test_loader  = DataLoader(data_list[n_train:], batch_size=16)

# 3) Define a minimal 2-layer GCN
class BasicGCN(torch.nn.Module):
    def __init__(self, in_feats, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,  1)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index).view(-1)

model     = BasicGCN(in_feats=data_list[0].x.size(1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = torch.nn.MSELoss()

# 4) Training loop (20 epochs)
for epoch in range(1, 21):
    model.train()
    total_loss, total_pts = 0.0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        out   = model(batch.x, batch.edge_index)
        loss  = loss_fn(out[batch.mask], batch.y[batch.mask])
        loss.backward()
        optimizer.step()
        n = int(batch.mask.sum().item())
        total_loss += loss.item() * n
        total_pts  += n
    print(f"Epoch {epoch:02d} — Train MSE: {total_loss/total_pts:.4f}")

# 5) Evaluation on test set
model.eval()
test_loss, test_pts = 0.0, 0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out[batch.mask], batch.y[batch.mask])
        n = int(batch.mask.sum().item())
        test_loss += loss.item() * n
        test_pts  += n

print(f"Test MSE: {test_loss/test_pts:.4f}")


# %%



