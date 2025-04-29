#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# Create figure depicting 2x3 embeddings of 1000 Genomes Project data.

# This file contains elements of code written by Alex Diaz-Papkovich.

########################################################################################################
########################################################################################################

import numpy as np, utils.plot_fcts as plot_fcts, paths, params, pandas as pd

# Name of this file
module_name = "genomes-figure.py"

##############################
############################## 
####################

# Loading metadata
labels_vec = pd.read_csv("{v}affy_samples.20141118.panel".format(v=paths.genomes_data), delimiter="\t")["pop"].values

color_dict = {"ACB":"#bd9e39","ASW":"#8c6d31","BEB":"#637939","CDX":"#393b79","CEU":"#d6604d","CHB":"#5254a3","CHS":"#9e9ac8","CLM":"#7b4173","ESN":"#e7ba52","FIN":"#ad494a","GBR":"#843c39","GIH":"#8ca252","GWD":"#e7cb94","IBS":"#d6616b","ITU":"#b5cf6b","JPT":"#6b6ecf","KHV":"#9c9ede","LWK":"#7f3b08","MSL":"#b35806","MXL":"#a55194","PEL":"#ce6dbd","PJL":"#cedb9c","PUR":"#de9ed6","STU":"#c7e9c0","TSI":"#e7969c","YRI":"#e08214"}

labels = np.empty(shape=labels_vec.size, dtype="U7")
for i, v in enumerate(labels_vec):
    labels[i] = color_dict[v]

label_descr = pd.read_csv("{v}20131219.populations.tsv".format(v=paths.genomes_data), sep='\t')

D_superpop = {"AFR":{'superpop_descr':"African", 'idx':[], 'L_colors':[]}, "EUR":{'superpop_descr':"European", 'idx':[], 'L_colors':[]}, "AMR":{'superpop_descr':"Central/South American", 'idx':[], 'L_colors':[]}, "SAS":{'superpop_descr':"South Asian", 'idx':[], 'L_colors':[]}, "EAS":{'superpop_descr':"East Asian", 'idx':[], 'L_colors':[]}}

D_pop = {}
for df_idx, df_row in label_descr.iterrows():
    D_pop[df_row["Population Code"]] = {'pop_descr':df_row["Population Description"], 'idx':[]}
    for i, v in enumerate(labels_vec):
        if v == df_row["Population Code"]:
            D_pop[df_row["Population Code"]]['idx'].append(i)
            D_superpop[df_row["Super Population"]]['idx'].append(i)
            D_superpop[df_row["Super Population"]]['L_colors'].append(color_dict[df_row["Population Code"]])

D_pop_arr = {}
for k in D_pop.keys():
    if len(D_pop[k]['idx']) > 0:
        D_pop_arr[k] = {'pop_descr':D_pop[k]['pop_descr'], 'color':color_dict[k], 'idx':np.asarray(D_pop[k]['idx'], dtype=np.int64)}

D_superpop_arr = {}
for k in D_superpop.keys():
    if len(D_superpop[k]['idx']) > 0:
        superpop_color = plot_fcts.rgb_to_hex(plot_fcts.arr_hex_to_rgb(np.asarray(D_superpop[k]['L_colors'])).mean(axis=0).astype(np.int32))
        D_superpop_arr[k] = {'superpop_descr':D_superpop[k]['superpop_descr'], 'idx':np.asarray(D_superpop[k]['idx'], dtype=np.int64), 'color':superpop_color}

# Dictionary that will contain entries to annotate some 2-D embeddings.
D_viz_emb = dict()
D_viz_emb['D_pop'] = D_pop_arr
D_viz_emb['D_superpop'] = D_superpop_arr

plot_fcts.create_2x3_figure(data_name=paths.genomes_name, emb_path=paths.genomes_emb, fig_path=paths.genomes_fig, arr_colors=labels, f_format=params.f_format, D_viz_emb=D_viz_emb)

print('*********************')
print('***** Done! :-) *****')
print('*********************')
