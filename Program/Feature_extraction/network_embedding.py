import re
import os
import requests
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from get_protein_residues import * 
import pickle
import igraph as ig
from node2vec import Node2Vec
import networkx as nx
from pathlib import Path


output_dir = Path("network_embeddings")
output_dir.mkdir(parents=True, exist_ok=True)

all_sets_filtered = pd.read_csv("./all_sets_filtered.txt",sep="\t")


for index, row in all_sets_filtered[["pdb", "chain"]].drop_duplicates().iterrows():
    with open(f"./graphs/{row['pdb']}_{row['chain']}.pkl", "rb") as f:
        g = pickle.load(f)

    # Convert igraph to networkx
    residue_labels = list(map(str, g.vs["name"])) 
    nx_graph = nx.Graph()
    for edge in g.get_edgelist():
        nx_graph.add_edge(residue_labels[edge[0]], residue_labels[edge[1]])


    # applying Node2Vec with BFS-favoring params
    node2vec = Node2Vec(
        nx_graph,
        dimensions=64,
        walk_length=32,
        num_walks=200,
        p=0.5,
        q=2,
        workers=4,
        seed =42
    )
    model = node2vec.fit(window=2, min_count=1, batch_words=4, seed =42)

    # saving embeddings as csv
    emb_df = pd.DataFrame([model.wv[n] for n in model.wv.index_to_key], index=model.wv.index_to_key)
    emb_df.to_csv(output_dir / f"{row['pdb']}_{row['chain']}.csv", header=False)

    #model.wv.save_word2vec_format(output_dir / f"{row['pdb']}_{row['chain']}.txt")

    # saving full model (.model)
    model.save(f"{output_dir}/{row['pdb']}_{row['chain']}.model")

 

