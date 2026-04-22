import sys
from get_protein_residues import * 
from tqdm.auto import tqdm
from uniprot_pdb_res_mapping import *
from write_uniprot_seqs_fasta import *
from dictionaries import *



all_sets_filtered = pd.read_csv("./all_sets_filtered.txt",sep="\t")

sequences_all_sets_filtered=[]
for each in all_sets_filtered.uniprot_acc.unique():
    sequence=fetch_uniprot_sequence(each)
    sequences_all_sets_filtered.append(f">{each}\n{sequence}")

sequences_all_sets_filtered


np.savetxt("./sequences_all_sets_filtered.fasta", sequences_all_sets_filtered, fmt="%s")


# run on the cluster bash 
# code is "python prediction.py ../ESM-DBP/ training_sequences.fasta training_embedding/ cpu"
# features are under embedding outputs and they are .fea files.


def get_entity_id_and_res_start_ends(pdb_residues, chain_id):
  # get entity id of a chain, residue number of the first residue and last residue
    try:
        for entry in pdb_residues:
            entity_id = entry['entity_id']
            for chain in entry['chains']:

                pdb_res_name_num=[(d['author_residue_number'],three2one(d['residue_name']),d["observed_ratio"]) for d in chain['residues'] if is_aa(d['residue_name'])==True]
                dict_residues = [d['residue_number'] for d in chain['residues']]
                res_start = int(min(dict_residues))
                res_end = int(max(dict_residues))

                return entity_id, res_start, res_end,pdb_res_name_num
    except TypeError:
        entity_id, res_start, res_end = 0, 0, 0
        return entity_id, res_start, res_end,pdb_res_name_num
        

all_sets_filtered_w_uniprot_maping= all_sets_filtered.copy()


unique_keys = all_sets_filtered_w_uniprot_maping[['pdb', 'chain', 'uniprot_acc']].drop_duplicates().values.tolist()
mapping_cache = {}

for pdb_id, chain_id, uniprot_id in tqdm(unique_keys):
    pdb_id_lower = pdb_id.lower()
    try:
        pdb_residues = get_pdb_residues(pdb_id_lower, chain_id)
        entity_id, res_start, res_end, pdb_res_name_num = get_entity_id_and_res_start_ends(pdb_residues, chain_id)
        residues_uniprot = get_pdb_uniprot_residue_mapping(pdb_id_lower, uniprot_id, chain_id, entity_id, res_start, res_end, pdb_res_name_num)
        residues_uniprot_only_y = [r for r in residues_uniprot if r[9] == "Y"]
        mapping_cache[(pdb_id_lower, chain_id, uniprot_id)] = residues_uniprot_only_y
        
    except Exception as e:
        print(f"Error for {pdb_id} {chain_id} {uniprot_id}: {e}")
        mapping_cache[(pdb_id_lower, chain_id, uniprot_id)] = None


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

all_sets_filtered_w_uniprot_maping['uniprot_resid'] = all_sets_filtered_w_uniprot_maping.apply(get_uniprot_resnum, axis=1).astype('Int64')



