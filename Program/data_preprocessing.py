

import pandas as pd
from uniprot_pdb_res_mapping import *
from write_uniprot_seqs_fasta import *
import os
from tqdm.auto import tqdm
import requests



def uniprot_acc_column(dataset):
    for pdb_id in dataset['pdb'].unique():
        response = get_response(pdb_id)
        uniprot = get_uniprot_acc(response, pdb_id)

        for i in range(len(uniprot)):
            index = dataset.index[(dataset["pdb"]==pdb_id) & (dataset["chain"]==uniprot[i][0])]
            dataset.loc[index, ["uniprot_acc"]] = uniprot[i][1]
    return dataset



all_sets=pd.read_csv("./all_sets.csv",sep="\t")



all_sets_test = all_sets[all_sets["category"] == "test"]
all_sets_train = all_sets[all_sets["category"] == "training"]

test_uniprots = all_sets_test["uniprot_acc"].unique()

all_sets_train = all_sets_train[~all_sets_train["uniprot_acc"].isin(test_uniprots)]

all_sets_train = all_sets_train.drop_duplicates(subset=["uniprot_acc", "resname", "resid"])

all_sets_filtered = pd.concat([all_sets_test, all_sets_train], ignore_index=True)


sequences_all_sets_filtered={}
for each in all_sets_filtered.uniprot_acc.unique():
    sequences_all_sets_filtered[each]=fetch_uniprot_sequence(each)

sequences_all_sets_filtered


write_fasta(sequences_all_sets_filtered, "./cdhit/sequences_all_sets_filtered.fasta")


### filtration after CD-HIT

#./cd-hit -i sequences_all_sets_filtered.fasta -o output_sequences_all_sets_filtered.fasta -c 0.4 -n 2


all_sets_filtered=all_sets_filtered.query("uniprot_acc != 'Q6TDU5' and uniprot_acc != 'P06766' and uniprot_acc !='P84131' and uniprot_acc !='P63159'")


all_sets_filtered.loc[:, 'label'] = np.where(all_sets_filtered['ddg'] >= 2, "H", "NH")


all_sets_filtered.to_csv("all_sets_filtered.csv", index=False)



### check if resid is based on pdb numbering

tuples = list(zip(all_sets_filtered['pdb'], all_sets_filtered['chain'], all_sets_filtered['resname'], all_sets_filtered['resid']))

def check_residue_match(pdb_id, chain_id, expected_resname, resid):
    pdb_file = f"pdb_files/{pdb_id}.pdb"  
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_file)
        residue = structure[0][chain_id][(' ', int(resid), ' ')]
        actual_resname = three_to_one(residue.get_resname())
        return actual_resname == expected_resname
    except Exception as e:
        return f"Error: {e}"

results = []
for pdb_id, chain_id, expected_resname, resid in tuples:
    result = check_residue_match(pdb_id, chain_id, expected_resname, resid)
    results.append(result)



def get_uniprot_resnum(row):
    pdb_id = row['pdb'].lower()
    chain_id = row['chain']
    uniprot_id = row['uniprot_acc']
    pdb_resid = row['resid']
    pdb_resname = row['resname']

    key = (pdb_id, chain_id, uniprot_id)
    residues_uniprot_only_y = mapping_cache.get(key)

    if residues_uniprot_only_y is None:
        return None

    for res in residues_uniprot_only_y:
        pdb = res[0][:4]
        chain = res[0][-1]
        uniprot = res[3]
        pdb_res_num = res[7]
        pdb_res_name = res[8]
        uniprot_res_num = res[4]
        uniprot_res_name = res[5]

        if pdb_id == pdb and chain_id == chain and uniprot_id == uniprot and pdb_resid == pdb_res_num and pdb_resname == pdb_res_name and pdb_resname == uniprot_res_name:
            return int(uniprot_res_num)

    return None

all_sets_filtered['uniprot_resid'] = all_sets_filtered.apply(get_uniprot_resnum, axis=1).astype('Int64')


all_sets_filtered_drop_na=all_sets_filtered.dropna()
all_sets_filtered_drop_na.to_csv("all_sets_filtered_drop_na.csv", index=False)





