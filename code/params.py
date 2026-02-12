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

# List of number of neighbors and perplexity values considered to illustrate the influence of these hyper-parameters on LE, t-SNE, UMAP and PHATE.
L_nn_perp = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 200, 500, 1000, 1500, 2000]

# Format to save the figures
f_format = 'png'

# Number of runs of a given embedding method that are conducted on a given data set to estimate its average computation time. 
n_runs_meth_timing = 5

# Number of processors to use when running parallel jobs (not used when estimating runtimes)
n_jobs = 8
