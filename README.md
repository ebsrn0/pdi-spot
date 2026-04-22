# PDI-Spot
PDI-Spot is a machine learning method using representation learning for predicting hot spots on protein-DNA interfaces.

Some dependencies are required as follows:

* python==3.10
* autogluon==1.3.0
* numpy==2.0.0
* pandas==2.3.2
* scikit-learn==1.6.1
* node2vec==0.5.0
* joblib
* torch
* ESM-DBP
   

<ins><strong>1. Feature Extraction </strong><ins>

   1. Sequence embedding: To generate sequence embeddings, run sequence_embedding.py in the feature extraction folder. Use "sequences.fasta" as input to produce "*.fea" as output for each protein.    
   
   2. Network embedding: To generate network embeddings, run network_embedding.py in the feature extraction folder.Use "* .pkl" as input to produce "*.csv" as output one per protein.    


<ins><strong>2. Feature Selection </strong><ins>

   After extracting all feature files, run feature_selection.py to obtain the optimal feature subset, which is then used in the prediction process.
   
<ins><strong>3. Prediction </strong><ins>

   Run model_prediction.py with the model in the model folder, which gives the predicted hot spots as a result.

The authors: Esra Basaran, Nurcan Tuncbag, Attila Gürsoy, and Özlem Keskin

