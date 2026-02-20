#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# This file defines parameters for embedding methods as well as for some quality scores. 

########################################################################################################
########################################################################################################

import numpy as np

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
L_nn_perp = [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000]#, 2500]#, 3000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

# Format to save the figures
f_format = 'png'

# Number of runs of a given embedding method that are conducted on a given data set to estimate its average computation time. 
n_runs_meth_timing = 30

# Number of processors to use when running parallel jobs (not used when estimating runtimes)
n_jobs = 7

# Proportions of the full data set that are considered as subsamplings
proportions = np.arange(start=0.05, step=0.05, stop=1.01, dtype=np.float64)