#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# This file defines names of data, embedding methods and quality scores for prints, as well as paths that are used to read data and save results/figures. 

########################################################################################################
########################################################################################################

import params

# Name of this file
module_name = "paths.py"

# Name of the data sets in prints
tasic_name = 'Tasic et al'
genomes_name = '1000 Genomes Project'
kanton_name = 'Kanton et al'
wiki_name = 'Simple English Wikipedia'

# Name of the data sets in paths
tasic_path = 'Tasic-et-al'
genomes_path = '1000-Genomes-Project'
kanton_path = 'Kanton-et-al'
wiki_path = 'Simple-English-Wikipedia'

# Paths where data are stored
res_data = '../data/'
tasic_data = '{p}{n}/'.format(p=res_data, n=tasic_path)
genomes_data = '{p}{n}/'.format(p=res_data, n=genomes_path)
kanton_data = '{p}{n}/'.format(p=res_data, n=kanton_path)
wiki_data = '{p}{n}/'.format(p=res_data, n=wiki_path)

# Path where figures are saved
res_fig = '../figures/'
tasic_fig = '{p}{n}'.format(p=res_fig, n=tasic_path)
genomes_fig = '{p}{n}'.format(p=res_fig, n=genomes_path)
kanton_fig = '{p}{n}'.format(p=res_fig, n=kanton_path)
wiki_fig = '{p}{n}'.format(p=res_fig, n=wiki_path)

# Path where embeddings are saved
res_emb = '../results/embeddings/'
tasic_emb = '{p}{n}/'.format(p=res_emb, n=tasic_path)
genomes_emb = '{p}{n}/'.format(p=res_emb, n=genomes_path)
kanton_emb = '{p}{n}/'.format(p=res_emb, n=kanton_path)
wiki_emb = '{p}{n}/'.format(p=res_emb, n=wiki_path)

# Path where quality scores are saved
res_qa = '../results/quality_scores/'
tasic_qa = '{p}{n}/'.format(p=res_qa, n=tasic_path)
genomes_qa = '{p}{n}/'.format(p=res_qa, n=genomes_path)
kanton_qa = '{p}{n}/'.format(p=res_qa, n=kanton_path)
wiki_qa = '{p}{n}/'.format(p=res_qa, n=wiki_path)

# Name of some embedding methods in prints without indicating their parameter values
LE_name_no_param = 'LE'
tsne_name_no_param = 't-SNE'
umap_name_no_param = 'UMAP'
phate_name_no_param = 'PHATE'

# Name of the embedding methods in prints
pca_name = 'PCA'
mds_name = "MDS"
LE_name = '{v} ({n} neighbors)'.format(n=params.nn_LE, v=LE_name_no_param)
tsne_name = '{v} (perplexity: {p})'.format(p=params.perp_tsne, v=tsne_name_no_param)
umap_name = '{v} ({n} neighbors)'.format(n=params.nn_umap, v=umap_name_no_param)
phate_name = '{v} ({n} neighbors)'.format(n=params.nn_phate, v=phate_name_no_param)

# Name of some embedding methods in paths without indicating their parameter values
LE_path_no_param = 'LE'
tsne_path_no_param = 'tsne'
tsne_sklearn_path_no_param = 'tsne_sklearn'
umap_path_no_param = 'umap'
phate_path_no_param = 'phate'

# Name of the embedding methods in paths
pca_path = 'pca'
mds_path = 'mds'
mds_sklearn_path = 'mds_sklearn'
LE_path = '{v}-n{n}'.format(n=params.nn_LE, v=LE_path_no_param)
tsne_path = '{v}-p{p}'.format(v=tsne_path_no_param, p=int(round(params.perp_tsne)))
tsne_sklearn_path = '{v}-p{p}'.format(v=tsne_sklearn_path_no_param, p=int(round(params.perp_tsne)))
umap_path = '{v}-n{n}'.format(n=params.nn_umap, v=umap_path_no_param)
phate_path = '{v}-n{n}'.format(n=params.nn_phate, v=phate_path_no_param)

# Name of the quality scores in prints
auc_name = 'AUC'
sigma_d_name = 'sigma distortion'
pearson_corr_name = 'Pearson corr(HD dists., LD dists.)'
pearson_corr_name_long = 'Pearson correlation between HD and LD distances'
knn_recall_name = '{Knn}-NN recall'.format(Knn=params.K_qa)

# Name of the quality scores in paths
auc_path = 'auc'
sigma_d_path = 'sigmad'
pearson_corr_path = 'pearsonr'
knn_recall_path = '{K_qa}-NN_recall'.format(K_qa=params.K_qa)
