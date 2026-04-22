from Bio import PDB
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
from dictionaries import * 
import igraph as ig
from tqdm import tqdm
import pickle
import os 

def parse_structure(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    return structure

def get_ca_atoms_for_chain(structure, chain_id):
    ca_atoms = []
    model = structure[0]
    for chain in model:
        if chain.id == chain_id:
            for residue in chain:
                
                # residue.id[2] is for removal of residues with insertional code and residue.id[0] is for removal of hetatoms.
                if 'CA' in residue and residue.id[2] == ' ' and residue.id[0] == " ": 
                    ca_atoms.append(residue['CA'])
    return ca_atoms

def calculate_distances(ca_atoms, threshold=7.0):
    n = len(ca_atoms)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = ca_atoms[i] - ca_atoms[j]
            if dist <= threshold:
                distances[i, j] = dist
                distances[j, i] = dist
    return distances

def create_graph(ca_atoms, distances):
    
    g = ig.Graph(directed=False)
    labels = []
    for atom in ca_atoms:
        residue = atom.get_parent()
        resname = three2one(residue.get_resname())
        resnum = residue.get_id()[1]
        chain_id = residue.get_parent().id
        label = f"{resname}{resnum}_{chain_id}"
        labels.append(label)
        g.add_vertex(name=label,label=label)
    
    edges = []
    weights = []
    for i in range(len(ca_atoms)):
        for j in range(i+1, len(ca_atoms)):
            if distances[i, j] > 0:
                edges.append((labels[i], labels[j]))
    
    g.add_edges(edges)
    return g

os.makedirs("graphs", exist_ok=True)

all_sets_filtered = pd.read_csv("./all_sets_filtered.txt",sep="\t")

for pdb in tqdm((all_sets_filtered.pdb+all_sets_filtered.chain).unique()):
    pdb_file = f'pdb_files/{pdb[:4]}.pdb'  
    chain_id = f'{pdb[4]}'   
    structure = parse_structure(pdb_file)
    ca_atoms = get_ca_atoms_for_chain(structure, chain_id)
    distances = calculate_distances(ca_atoms, threshold=7.0)
    g = create_graph(ca_atoms, distances)
    with open(f'graphs/{pdb[:4]}_{pdb[4]}.pkl', 'wb') as f:
        pickle.dump(g, f)
