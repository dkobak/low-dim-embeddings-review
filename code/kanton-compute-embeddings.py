#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# Running embedding methods on Kanton et al data, and evaluate their quality.

########################################################################################################
########################################################################################################

import numpy as np, utils.run_embs as run_embs, paths

# Name of this file
module_name = "kanton-compute-embeddings.py"

# Set to True to compute the proportion of preserved variance by the first 50 PCs
compute_pca_preserved_var = False

# Set to True to check whether there are duplicated samples in the data set
check_duplicates = False

##############################
############################## 
# Loading and processing Kanton et al data
####################

# Number of samples: 20,272
# Raw number of features: 32,856
# Number of features after feature selection: 1,000
# Number of features after PCA preprocessing: 50

print('Loading {v} data'.format(v=paths.kanton_name))
X_hd = np.load('{p}human-409b2/preprocessed-data.npy'.format(p=paths.kanton_data))
X_hd_nopca = np.load('{p}human-409b2/gene-selected-data.npy'.format(p=paths.kanton_data)) if compute_pca_preserved_var else None

run_embs.compute_embs_and_quality(X_hd=X_hd, pca_preproc=True, data_name=paths.kanton_name, res_path_emb=paths.kanton_emb, res_path_qa=paths.kanton_qa, check_duplicates=check_duplicates, compute_pca_preserved_var=compute_pca_preserved_var, X_hd_nopca=X_hd_nopca, genomes=False)

print('*********************')
print('***** Done! :-) *****')
print('*********************')




