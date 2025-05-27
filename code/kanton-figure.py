#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# Create figure depicting 2x3 embeddings of Kanton et al data.

# This file contains elements of code written by Sebastian Damrich and made publicly available following this link: https://github.com/berenslab/ne_spectrum_scRNAseq/blob/main/utils/utils.py#L99

########################################################################################################
########################################################################################################

import numpy as np, utils.plot_fcts as plot_fcts, paths, params

# Name of this file
module_name = "kanton-figure.py"

##############################
##############################
####################

print('Loading {v} data'.format(v=paths.kanton_name))
X_PCs = np.load('{p}human-409b2/preprocessed-data.npy'.format(p=paths.kanton_data))[:,:2]

# Loading metadata
labels = np.load('{v}human-409b2/labels.npy'.format(v=paths.kanton_data))

d = {"label_colors": {
    "iPSCs": "navy",
    "EB": "royalblue",
    "Neuroectoderm": "skyblue",
    "Neuroepithelium": "lightgreen",
    "Organoid-1M": "gold",
    "Organoid-2M": "tomato",
    "Organoid-3M": "firebrick",
    "Organoid-4M": "maroon",
}, "time_colors": {
    "  0 days": "navy",
    "  4 days": "royalblue",
    "10 days": "skyblue",
    "15 days": "lightgreen",
    "  1 month": "gold",
    "  2 months": "tomato",
    "  3 months": "firebrick",
    "  4 months": "maroon",
}, "colors_time": {
    "navy":"0 days",
    "royalblue":"4 days",
    "skyblue":"10 days",
    "lightgreen":"15 days",
    "gold":"1 month",
    "tomato":"2 months",
    "firebrick":"3 months",
    "maroon":"4 months",
}}

for i, v in enumerate(labels):
    labels[i] = d['label_colors'][v]

D_samp_by_time = {d["colors_time"][k]:[] for k in d["colors_time"]}
for i, v in enumerate(labels):
    D_samp_by_time[d["colors_time"][v]].append(i)
D_samp_by_time_arr = {}
for k in D_samp_by_time:
    if len(D_samp_by_time[k]) > 0:
        D_samp_by_time_arr[k] = np.asarray(D_samp_by_time[k], dtype=np.int64)

plot_fcts.create_2x3_figure(data_name=paths.kanton_name, emb_path=paths.kanton_emb, fig_path=paths.kanton_fig, arr_colors=labels, f_format=params.f_format, X_PCs=X_PCs, D_viz_emb={'D_samp_by_time':D_samp_by_time_arr})

print('*********************')
print('***** Done! :-) *****')
print('*********************')
