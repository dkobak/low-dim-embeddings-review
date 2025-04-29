#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# Running embedding methods on 1000 Genomes Project data, and evaluate their quality.

########################################################################################################
########################################################################################################

import numpy as np, utils.run_embs as run_embs, paths, copy

# Name of this file
module_name = "genomes-compute-embeddings.py"

# Set to True to compute the proportion of preserved variance by the first 50 PCs
compute_pca_preserved_var = False

# Set to True to check whether there are duplicated samples in the data set
check_duplicates = False

##############################
############################## 
# Loading and processing 1000 Genomes Project data
####################

# Number of samples: 3,450
# Number of features: 53,999

print('Loading {v} data'.format(v=paths.genomes_name))
X_hd = np.loadtxt('{p}gt_sum_thinned.npy.gz'.format(p=paths.genomes_data))
X_hd_nopca = copy.deepcopy(X_hd) if compute_pca_preserved_var else None

run_embs.compute_embs_and_quality(X_hd=X_hd, pca_preproc=False, data_name=paths.genomes_name, res_path_emb=paths.genomes_emb, res_path_qa=paths.genomes_qa, check_duplicates=check_duplicates, compute_pca_preserved_var=compute_pca_preserved_var, X_hd_nopca=X_hd_nopca, genomes=True)

print('*********************')
print('***** Done! :-) *****')
print('*********************')
