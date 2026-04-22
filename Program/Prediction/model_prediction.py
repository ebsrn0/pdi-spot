from heapq import heappush, heappop
from sklearn.preprocessing import StandardScaler
import joblib
from autogluon.tabular import TabularDataset, TabularPredictor
from collections import defaultdict
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def equal_fold_split(dataframe,num_groups):

    df = pd.DataFrame(dataframe.protein.value_counts().reset_index())


    # creating groups 
    num_groups = num_groups
    groups = [[] for _ in range(num_groups)]
    group_sums = [0] * num_groups

    protein_to_group = {}

    # adding it into the smallest group by Min-heap 
    heap = [(0, i) for i in range(num_groups)]  

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


def autogluon_run(X_train, y_train, X_indep, y_indep,num_bag_folds=5,num_bag_sets=1):
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_indep_scaled = X_indep.copy()

    
    feature_cols = [col for col in X_train.columns if col != 'pdb']  # fist column is pdb
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_indep_scaled[feature_cols] = scaler.transform(X_indep[feature_cols])
    
    
    # Combine features and target for AutoGluon
    train_data = pd.concat([X_train_scaled, y_train], axis=1)
    train_data['label'] = train_data['label'].map({'H': 1, 'NH': 0})
    train_data = train_data.rename(columns={'pdb': 'protein'})
    group=None
    
    if num_bag_folds>0:
        
        # equal_fold_split is used for equal fold 
        protein_to_group=equal_fold_split(train_data,num_bag_folds)

        train_data["split"] = train_data["protein"].map(protein_to_group)
        group="split"
        
    
    selected_features_training = ['network_embedding32','network_embedding39','sequence_embedding1018','sequence_embedding1042',
 'sequence_embedding1044','sequence_embedding1135','sequence_embedding202','sequence_embedding30','sequence_embedding375',
 'sequence_embedding468','sequence_embedding469','sequence_embedding518','sequence_embedding8']+["label","split"]
    
    

    train_data_wout_proteins=train_data[selected_features_training]

    # Train the AutoGluon predictor
    predictor = TabularPredictor(label="label", problem_type='binary',eval_metric="roc_auc",sample_weight="balance_weight",groups=group).fit(train_data=train_data_wout_proteins,num_bag_folds=num_bag_folds, num_bag_sets=num_bag_sets ,presets=["best_quality"], num_stack_levels=0) 
    
    # Combine features and target for AutoGluon
    indep_data = pd.concat([X_indep_scaled, y_indep], axis=1)
    indep_data=indep_data[selected_features_training[:-1]]
    indep_data['label'] = indep_data['label'].map({'H': 1, 'NH': 0})
    

    return predictor, train_data, indep_data




X_train = pd.read_csv("./X_train.csv",sep="\t")
X_indep = pd.read_csv("./X_indep.csv",sep="\t")
y_train = pd.read_csv("./y_train.csv",sep="\t")
y_indep = pd.read_csv("./y_indep.csv",sep="\t")


predictor, train_data, indep_data= autogluon_run(X_train, y_train, X_indep, y_indep, num_bag_folds=5,num_bag_sets=1)



# predictions on the independent data
y_pred = predictor.predict(indep_data, model="NeuralNetTorch_r185_BAG_L1").map({1: "H", 0: "NH"})
y_true = indep_data['label'].map({1: "H", 0: "NH"}) 

conf_matrix = confusion_matrix(y_true, y_pred)

# visualization of the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["H","NH"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Training Data")
plt.show()

