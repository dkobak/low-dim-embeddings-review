#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# This file defines parameters for embedding methods as well as for some quality scores. 

########################################################################################################
########################################################################################################

# Name of this file
module_name = "params.py"

# Targeted dimension of the LD embeddings
dim_LDS = 2

# Random seed
seed = 40

# Neighborhood size to compute K-NN recall (local quality assessment). Must be a strictly positive integer. 
K_qa = 10

# Number of neighbors in Laplacian eigenmaps (LE)
nn_LE = 100

# Perplexity in t-SNE
perp_tsne = 30.0

# Number of neighbors in UMAP
nn_umap = 15

# Number of neighbors in PHATE
nn_phate = 5

# Format to save the figures
f_format = 'png'

# Number of processors to use when running parallel jobs
n_jobs = 25
