#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# Computing PHATE embedding of Simple English Wikipedia data.

########################################################################################################
########################################################################################################

import numpy as np, datasets, utils.plot_fcts as plot_fcts, paths, params, utils.run_embs as run_embs, sklearn.preprocessing

# Name of this file
module_name = "wikipedia-compute-phate-embedding.py"

##############################
############################## 
# Parameters
####################

# Set to True to compute the proportion of preserved variance by the first 50 PCs
compute_pca_preserved_var = False

##############################
############################## 
# Loading Simple English Wikipedia data
####################

print('===')
print("Processing {v} data".format(v=paths.wiki_name))
print("===")

# Checking whether the folder where to store the results exists
plot_fcts.check_create_dir(paths.wiki_emb)

print('Loading HD data')
wiki_docs = datasets.load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
X_hd = np.asarray(wiki_docs['emb'])
print('- Number of samples: ', X_hd.shape[0]) # 485,859
print('- Number of features:', X_hd.shape[1]) # 768
print("===")

# Normalizing the samples
X_hd = sklearn.preprocessing.normalize(X_hd, norm='l2', axis=1, copy=False, return_norm=False)

if compute_pca_preserved_var:
    run_embs.preserved_variance_PCs(X=X_hd)

print('Computing PHATE embedding')
run_embs.apply_meth(X_hd=X_hd, meth_name=paths.phate_name, meth_name4path=paths.phate_path, pca_preproc=False, compute_dist_HD=None, compute_dist_LD_qa=None, seed=params.seed, res_path_emb=paths.wiki_emb, res_path_qa=None, dim_LDS=params.dim_LDS, perp_tsne=params.perp_tsne, nn_umap=params.nn_umap, nn_phate=params.nn_phate, nn_LE=params.nn_LE, skip_qa=True)
print("===")


###
###
###
print('*********************')
print('***** Done! :-) *****')
print('*********************')
