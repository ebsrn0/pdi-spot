# Generating Embeddings for a Protein–DNA Interface Residue

This example demonstrates the complete feature extraction pipeline of PDI-Spot
using the **Zif268 zinc finger-DNA complex (PDB: 1AAY, Chain A)** as a worked case.
Residue **R118** is used as the target interface residue.

---

## Requirements

```bash
# Create and activate the ESM-DBP environment
source esm_dbp_env/bin/activate

# Other dependencies
pip install biopython igraph node2vec pandas numpy
```


All scripts and precomputed example output are available under `Feature Exctraction` 
`example` in the GitHub repository.

---

## Step 1 — Download the PDB Structure

```python
import urllib.request
urllib.request.urlretrieve(
    "https://files.rcsb.org/download/1AAY.pdb",
    "1AAY.pdb"
)
```

---

## Step 2 — Generate ESM-DBP Sequence Embeddings

ESM-DBP takes the full UniProt protein **sequence** as input (FASTA format)
and returns a 1280-dimensional vector per residue, saved as `.fea` files.

**2a. Prepare the FASTA file:**

```python
import numpy as np

# UniProt sequence for 1AAY Chain A (NFκB p50, UniProt: P08046)
sequence = ">1AAY_A\nMAAAKAEMQLMSPLQISDPFGSFPHSPTMDNYPKLEEMMLLSNGAPQFLGAAGTPEGSGGNSSSSTSSGGGGGGGSNSGSSAFNPQGEPSEQPYEHLTTESFSDIALNNEKAMVETSYPSQTTRLPPITYTGRFSLEPAPNSGNTLWPEPLFSLVSGLVSMTNPPTSSSSAPSPAASSSSSASQSPPLSCAVPSNDSSPIYSAAPTFPTPNTDIFPEPQSQAFPGSAGTALQYPPPAYPATKGGFQVPMIPDYLFPQQQGDLSLGTPDQKPFQGLENRTQQPSLTPLSTIKAFATQSGSQDLKALNTTYQSQLIKPSRMRKYPNRPSKTPPHERPYACPVESCDRRFSRSDELTRHIRIHTGQKPFQCRICMRNFSRSDHLTTHIRTHTGEKPFACDICGRKFARSDERKRHTKIHLRQKDKKADKSVVASPAASSLSSYPSPVATSYPSPATTSFPSPVPTSYSSPGSSTYPSPAHSGFPSPSVATTFASVPPAFPTQVSSFPSAGVSSSFSTSTGLSDMTATFSPRTIEIC"
# Full sequence available at: https://www.uniprot.org/uniprot/P08046

with open("1AAY_A.fasta", "w") as f:
    f.write(sequence)
```

**2b. Run ESM-DBP prediction script:**

```bash
# Activate environment
source esm_dbp_env/bin/activate

# Run on GPU (recommended)
CUDA_VISIBLE_DEVICES=0 python prediction.py \
    ../ESM-DBP/ \
    1AAY_A.fasta \
    1AAY_embeddings/ \
    cuda:0

# Run on CPU (if GPU unavailable)
python prediction.py \
    ../ESM-DBP/ \
    1AAY_A.fasta \
    1AAY_embeddings/ \
    cpu
```

**2c. Load the output `.fea` file and extract the embedding for R118:**

```python
import pandas as pd
import numpy as np

def extract_sequence_embeddings(row, path):
    uniprotid = row['uniprot_acc']       
    target    = row['uniprot_resid'] - 1 # because of starting with 0-index

    embedding_file_path = f'{path}/{uniprotid}.fea'
    embedding_df        = pd.read_csv(embedding_file_path, header=None, sep=" ")
    embedding_vector    = embedding_df.iloc[target].values
    return embedding_vector

# Example row for R118
example_row = {
    'uniprot_acc':   'P08046',
    'uniprot_resid': 118        # 1-indexed UniProt position
}

r118_esm = extract_sequence_embeddings(example_row, "features/sequence_embedding/")
print(f"ESM-DBP embedding shape: {r118_esm.shape}")  # (1280,)
print(f"ESM-DBP embeddings: {r118_esm}")
pd.DataFrame([r118_esm], columns=[f"sequence_embedding{i}" for i in range(len(r118_esm))]).to_csv("example/1aay_esmdbp_embeddings.csv", index=False)
```



**Output:** `1aay_esmdbp_embeddings.csv` — 1280-dimensional vector for R118.
Precomputed file available at `example/1aay_esmdbp_embeddings.csv`.

---

## Step 3 — Build the Amino Acid Network (AAN) and Generate Node2Vec Embeddings

The AAN is constructed from the PDB structure: nodes = residues,
edges = Cα–Cα contacts within 7 Å.

**3a. Build the contact graph (igraph) and save as .pkl:**

```python
from Bio.PDB import PDBParser
import numpy as np
import igraph as ig
import pickle
import os

def parse_structure(pdb_file):
    parser = PDBParser(QUIET=True)
    return parser.get_structure('protein', pdb_file)

def get_ca_atoms_for_chain(structure, chain_id):
    ca_atoms = []
    for residue in structure[0][chain_id]:
        if ('CA' in residue and
            residue.id[2] == ' ' and
            residue.id[0] == ' '):
            ca_atoms.append(residue['CA'])
    return ca_atoms

def calculate_distances(ca_atoms, threshold=7.0):
    n = len(ca_atoms)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = ca_atoms[i] - ca_atoms[j]
            if dist <= threshold:
                distances[i, j] = dist
                distances[j, i] = dist
    return distances

def create_graph(ca_atoms, distances):
    g      = ig.Graph(directed=False)
    labels = []
    for atom in ca_atoms:
        residue = atom.get_parent()
        resname = residue.get_resname()        # three-letter code e.g. "ARG"
        resnum  = residue.get_id()[1]
        chain   = residue.get_parent().id
        label   = f"{resname}{resnum}_{chain}" # e.g. "ARG118_A"
        labels.append(label)
        g.add_vertex(name=label, label=label)
    edges = [(labels[i], labels[j])
             for i in range(len(ca_atoms))
             for j in range(i + 1, len(ca_atoms))
             if distances[i, j] > 0]
    g.add_edges(edges)
    return g

os.makedirs("graphs", exist_ok=True)

structure = parse_structure("1AAY.pdb")
ca_atoms  = get_ca_atoms_for_chain(structure, chain_id="A")
distances = calculate_distances(ca_atoms, threshold=7.0)
g         = create_graph(ca_atoms, distances)

with open("graphs/1AAY_A.pkl", "wb") as f:
    pickle.dump(g, f)

print(f"Graph saved: {g.vcount()} nodes, {g.ecount()} edges")
```

**3b. Extract the embedding for R118:**

```python
import pickle
import networkx as nx
from node2vec import Node2Vec
from pathlib import Path
import pandas as pd

output_dir = Path("features/network_embedding")
output_dir.mkdir(parents=True, exist_ok=True)

# Load igraph .pkl
with open("graphs/1AAY_A.pkl", "rb") as f:
    g = pickle.load(f)

# Convert igraph → networkx
residue_labels = list(map(str, g.vs["name"]))
nx_graph = nx.Graph()
for edge in g.get_edgelist():
    nx_graph.add_edge(residue_labels[edge[0]], residue_labels[edge[1]])

print(f"Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")

# Run Node2Vec
node2vec_model = Node2Vec(
    nx_graph,
    dimensions=64,
    walk_length=32,
    num_walks=200,
    p=0.5,    # return parameter — favors BFS / local structure
    q=2,      # in-out parameter
    workers=4,
    seed=42
)
model = node2vec_model.fit(window=2, min_count=1, batch_words=4)

# Save as CSV — index = node label (compatible with extract_network_embeddings)
emb_df = pd.DataFrame(
    [model.wv[n] for n in model.wv.index_to_key],
    index=model.wv.index_to_key
)
emb_df.to_csv(output_dir / "1AAY_A.csv", header=False)
```

**3c. Extract the embedding for R118:**

```python
def extract_network_embeddings(row, path):
    pdb_file = row['pdb']      # e.g. "1AAY"
    chain_id = row['chain']    # e.g. "A"
    target   = row['resname'] + str(row['resid']) + "_" + row['chain']
               # e.g. "ARG118_A"

    embedding_file_path = f'{path}/{pdb_file}_{chain_id}.csv'
    embedding_df        = pd.read_csv(embedding_file_path, header=None, sep=",")
    embedding_vector    = embedding_df[embedding_df[0] == target].iloc[:, 1:].values[0]
    return embedding_vector

example_row_n2v = {
    'pdb':     '1AAY',
    'chain':   'A',
    'resname': 'ARG',   # three-letter code
    'resid':   118
}

r118_n2v = extract_network_embeddings(example_row_n2v, "features/network_embedding/")
print(f"Node2Vec embedding shape: {r118_n2v.shape}")  # (64,)
print(f"Node2Vec embeddings: {r118_n2v}")
pd.DataFrame([r118_n2v ], columns=[f"network_embedding{i}" for i in range(len(r118_n2v ))]).to_csv("example/1aay_node2vec_embeddings.csv", index=False)

```

**Output:** `1aay_node2vec_embeddings.csv` — 64-dimensional vector for R118.
Precomputed file available at `example/1aay_node2vec_embeddings.csv`.

---





## Step 4 — Concatenate and Apply Feature Selection

The ESM-DBP (1280-dim) and Node2Vec (64-dim) embeddings are concatenated
to form the full 1344-dimensional feature vector.

```python
import numpy as np
import pandas as pd

# Load precomputed embeddings
esm_df = pd.read_csv("examples/1AAY/r118_esm_embedding.csv")   # (1280,)
n2v_df = pd.read_csv("examples/1AAY/r118_node2vec_embedding.csv")   # (64,)

# Concatenate column-wise — ESM first, Node2Vec second
full_df = pd.concat([esm_df, n2v_df], axis=1)
print(f"Full feature vector shape: {full_df.shape}")  # (1, 1344)

# PDI-Spot selected 13 features — select by column name directly
selected_col_names = [
    'sequence_embedding1018', 'sequence_embedding202',
    'sequence_embedding8',    'sequence_embedding30',
    'network_embedding32',    'sequence_embedding469',
    'sequence_embedding1044', 'sequence_embedding1042',
    'sequence_embedding518',  'sequence_embedding375',
    'sequence_embedding1135', 'sequence_embedding468',
    'network_embedding39'
]

1aay_example = full_df[selected_col_names]
print(1aay_example)


```

---
