#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# Create figure depicting 2x3 embeddings of Tasic et al data.

########################################################################################################
########################################################################################################

import numpy as np, utils.plot_fcts as plot_fcts, paths, params, pandas as pd

# Name of this file
module_name = "tasic-figure.py"

##############################
############################## 
####################

print('Loading {v} data'.format(v=paths.tasic_name))
X_PCs = np.load('{p}preprocessed-data.npy'.format(p=paths.tasic_data))[:,:2]

# Loading metadata
clusterInfo = pd.read_csv('{v}tasic-sample_heatmap_plot_data.csv'.format(v=paths.tasic_data))
ids        = clusterInfo['cluster_id'].values
labels_orig     = clusterInfo['cluster_label'].values
colors     = clusterInfo['cluster_color'].values
clusterNames  = np.array([labels_orig[ids==i+1][0] for i in range(np.max(ids))])
clusterColors = np.array([colors[ids==i+1][0] for i in range(np.max(ids))])
clusters   = np.copy(ids) - 1

# Dictionary that will contain entries to annotate some 2-D embeddings.
D_viz_emb = dict()
D_viz_emb['clusters'] = clusters
D_viz_emb['clusterColors'] = clusterColors
D_viz_emb['clusterNames'] = clusterNames

plot_fcts.create_2x3_figure(data_name=paths.tasic_name, emb_path=paths.tasic_emb, fig_path=paths.tasic_fig, arr_colors=colors, f_format=params.f_format, X_PCs=X_PCs, D_viz_emb=D_viz_emb)

print('*********************')
print('***** Done! :-) *****')
print('*********************')
