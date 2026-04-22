from heapq import heappush, heappop
from sklearn.preprocessing import StandardScaler
import joblib
from autogluon.tabular import TabularDataset, TabularPredictor
from collections import defaultdict
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from mrmr import mrmr_classif
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.metrics import make_scorer, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm



all_sets_filtered = pd.read_csv("./all_sets_filtered.txt",sep="\t")



# Function to extract the embedding vector for each row
def extract_sequence_embeddings(row,path):
    # Extract protein, chain, and residue number
    uniprotid = row['uniprot_acc']
    
    target = row['uniprot_resid']-1

    # Load the corresponding embedding file based on the protein and chain
    embedding_file_path = f'{path}/{uniprotid}.fea'
    

    embedding_df = pd.read_csv(embedding_file_path,header=None, sep=" ")
        
    # Filter for the row matching the residue ID
    embedding_vector = embedding_df.iloc[target].values

    return embedding_vector



# Function to extract the embedding vector for each row
def extract_network_embeddings(row,path):
    # Extract protein, chain, and residue number
    pdb_file = row['pdb']
    chain_id = row['chain']
    
    target = row['resname']+str(row['resid'])+"_"+row['chain']

    # Load the corresponding embedding file based on the protein and chain
    embedding_file_path = f'{path}/{pdb_file}_{chain_id}.csv'
    

    embedding_df = pd.read_csv(embedding_file_path,header=None, sep=",")
        
    # Filter for the row matching the residue ID
    embedding_vector = embedding_df[embedding_df[0] == target].iloc[:, 1:].values[0]

    return embedding_vector



def features(data):
    
    dataset=data.copy()
    
    
    dataset[[f"sequence_embedding{i}" for i in range(len(dataset.sequence_embedding[0]))]] = pd.DataFrame(dataset.sequence_embedding.tolist(), index= dataset.index)

    
    dataset[[f"network_embedding{i}" for i in range(len(dataset.network_embedding[0]))]] = pd.DataFrame(dataset.network_embedding.tolist(), index= dataset.index)
    
    return dataset



def train_test_w_features( df):
    
    # Training set
    df_training = df[df['category'] == 'training']

    # Test set
    df_test = df[df['category'] == 'test']
    
    X_train = df_training.iloc[:,np.r_[0,12:df_training.shape[1]]]
    y_train = df_training["label"]
    
    X_indep =df_test.iloc[:, 12:]
    y_indep=df_test["label"]
    
    return X_train, X_indep, y_train, y_indep



all_sets_filtered_w_features = all_sets_filtered.copy()
all_sets_filtered_w_features['sequence_embedding'] =all_sets_filtered_w_features.apply(lambda row: extract_sequence_embeddings(row,"features/sequence_embedding/"), axis=1)
all_sets_filtered_w_features['network_embedding'] =all_sets_filtered_w_features.apply(lambda row: extract_network_embeddings(row,"features/network_embedding/"), axis=1)




filtered_table_unzip_features = features(all_sets_filtered_w_features)
X_train, X_indep, y_train, y_indep = train_test_w_features(filtered_table_unzip_features)




def equal_fold_split(dataframe,num_groups):

    df = pd.DataFrame(dataframe.protein.value_counts().reset_index())


    # creating groups 
    num_groups = num_groups
    groups = [[] for _ in range(num_groups)]
    group_sums = [0] * num_groups

    protein_to_group = {}

    # adding it into the smallest group by Min-heap 
    heap = [(0, i) for i in range(num_groups)]  # total number, group indexes

    for _, row in df.iterrows():
        smallest_sum, group_idx = heappop(heap)
        groups[group_idx].append((row['protein'], row['count']))
        protein_to_group[row['protein']] = group_idx + 1 
        new_sum = smallest_sum + row['count']
        heappush(heap, (new_sum, group_idx))
        
     
    """
    # print part showing each group belonging to what proteins and how many mutations.
    for i, group in enumerate(groups, 1):
        print(f"Group {i}:")
        total = 0
        for protein, count in group:
            print(f"   {protein}     {count}")
            total += count
        print(f"Total count: {total}\n")
    """
    return protein_to_group



scaler = StandardScaler()
X_train_scaled = X_train.copy()

feature_cols = X_train_scaled.columns[1:]  # fist column is pdb
X_train_scaled[feature_cols] = scaler.fit_transform(X_train_scaled[feature_cols])


# Combine features and target 
train_data = pd.concat([X_train_scaled, y_train], axis=1)
train_data['label'] = train_data['label'].map({'H': 1, 'NH': 0})
train_data = train_data.rename(columns={'pdb': 'protein'})
num_bag_folds=5

# equal_fold_split is used for equal fold 
protein_to_group=equal_fold_split(train_data,num_bag_folds)

train_data["split"] = train_data["protein"].map(protein_to_group)




# MLP feature selection
ps = PredefinedSplit(test_fold=train_data["split"])

X_df = train_data.iloc[:,1:-2].copy()
y_series = train_data.iloc[:,-2].copy()


f1_scores_mlp = []
features_list_mlp = []

weights_array = compute_sample_weight(class_weight="balanced", y=y_series)


while X_df.shape[1] > 0:
    print(f"Training MLP on the {X_df.shape[1]} most important features.")

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    f1 = cross_val_score(mlp, X_df, y_series, scoring='f1', cv=ps).mean()

    f1_scores_mlp.append(f1)
    features_list_mlp.append(X_df.columns.tolist())

    if X_df.shape[1] == 1:
        break

    mlp.fit(X_df, y_series, sample_weight=weights_array)

    importances = abs(mlp.coefs_[0]).sum(axis=1)  # input-layer weights
    least_important_idx = importances.argmin()

    # remove the least contributing feature
    X_df = X_df.drop(X_df.columns[least_important_idx], axis=1)




# mRMR feature selection
ps = PredefinedSplit(test_fold=train_data["split"])

X_df = train_data.iloc[:,1:-2].copy()
y_series = train_data.iloc[:,-2].copy()

selected_features_mrmr = mrmr_classif(X=X_df, y=y_series, K=len(X_df.columns))

f1_scores_mrmr   = []
features_list_mrmr = []

for k in tqdm(range(1, len(selected_features_mrmr)+1)):
    
    selected_k = selected_features_mrmr[:k]
    
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)

    f1 = cross_val_score(mlp, X_df[selected_k], y_series, scoring='f1', cv=ps).mean()
    f1_scores_mrmr.append(f1)
    
    features_list_mrmr.append(X_df[selected_k].columns.tolist())
   



set(features_list_mrmr[:50]).intersection(set(features_list_mlp[:60]))
# features: ['sequence_embedding1018', 'sequence_embedding202', 'sequence_embedding8', 'sequence_embedding30', 'network_embedding32', 'sequence_embedding469', 'sequence_embedding1044', 'sequence_embedding1042', 'sequence_embedding518', 'sequence_embedding375', 'sequence_embedding1135', 'sequence_embedding468', 'network_embedding39']

