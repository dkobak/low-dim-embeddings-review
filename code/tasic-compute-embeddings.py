#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# Running embedding methods on Tasic et al data, and evaluate their quality. 
# Runtimes are also estimated and embeddings are computed using several hyper-parameter values. 

########################################################################################################
########################################################################################################

import numpy as np, utils.run_embs as run_embs, paths

# Name of this file
module_name = "tasic-compute-embeddings.py"

# Set to True to compute the proportion of preserved variance by the first 50 PCs
compute_pca_preserved_var = False

# Set to True to check whether there are duplicated samples in the data set
check_duplicates = False

##############################
############################## 
# Loading and processing Tasic et al data
####################

# Number of samples: 23,822
# Raw number of genes: 45,768
# Number of genes after feature selection: 3,000
# Number of features after PCA preprocessing: 50

print('Loading {v} data'.format(v=paths.tasic_name))
X_hd = np.load('{p}preprocessed-data.npy'.format(p=paths.tasic_data))
X_hd_nopca = np.load('{p}gene-selected-data.npy'.format(p=paths.tasic_data)) if compute_pca_preserved_var else None

# Computing embeddings
run_embs.compute_embs_and_quality(X_hd=X_hd, pca_preproc=True, data_name=paths.tasic_name, res_path_emb=paths.tasic_emb, res_path_qa=paths.tasic_qa, check_duplicates=check_duplicates, compute_pca_preserved_var=compute_pca_preserved_var, X_hd_nopca=X_hd_nopca, genomes=False)

# Estimating runtimes of embedding computation
run_embs.compute_runtimes(X_hd=X_hd, pca_preproc=True, data_name=paths.tasic_name, res_path_emb=paths.tasic_emb, X_hd_nopca=X_hd_nopca, check_duplicates=check_duplicates, genomes=False)

# Computing embeddings for several hyper-parameter values
run_embs.compute_embs_quality_sev_hps(X_hd=X_hd, pca_preproc=True, data_name=paths.tasic_name, res_path_emb=paths.tasic_emb, res_path_qa=paths.tasic_qa, check_duplicates=check_duplicates, genomes=False)

print('*********************')
print('***** Done! :-) *****')
print('*********************')
