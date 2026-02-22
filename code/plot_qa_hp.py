#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# This file contains a function to produce figures to illustrate the influence of the hyper-parameters of the embedding methods.

########################################################################################################
########################################################################################################

import numpy as np, os
import pandas as pd
import pylab as plt
import json

#plt.style.use("mpl_style.txt")
import matplotlib.pyplot as plt
import colormaps as cmaps

plt.rcParams['text.usetex'] = False
plt.style.use('./utils/mpl_style.txt') 

#import opinionated
#plt.style.use("opinionated_rc")

import paths, params, utils.plot_fcts as plot_fcts

def plot_sev_hps(data_name, res_path_qa, genomes, fig_path):
    """
    Creates figures to illustrate the influence of the hyper-parameters of the embedding methods on some data set.
    In: similar inputs as the ones of the compute_embs_quality_sev_hps function in ./utils/run_embs and of the create_2x3_figure function in ./utils/plot_fcts.
    """
    
    print('===')
    print("=== Creating figures for {v} data illustrating the influence of the hyper-parameters".format(v=data_name))
    print("===")
    
    with open('./method_colors.json', 'r') as f:
        base_colors = json.load(f)

    colors = {
        "pca": base_colors["PCA (1901)"]['color'],
        "mds": base_colors["MDS (1938)"]['color'],
        "le": base_colors["Lapl. Eig. (2003)"]['color'],
        "tsne": base_colors["t-SNE (2008)"]['color'],
        "umap": base_colors["UMAP (2018)"]['color'],
        "phate": base_colors["PHATE (2019)"]['color'],
    }
    
    labels = {
        "pca": "PCA",
        "mds": "MDS",
        "le": "Lapl. Eig.",
        "tsne": "t-SNE",
        "umap": "UMAP",
        "phate": "PHATE",
    }
    
    markers = {
        "pca": "s",
        "mds": "P",
        "le": "v",
        "tsne": "o",
        "umap": "*",
        "phate": "D",
    }
    
    if data_name == paths.tasic_name:
        labels_pos = {
            "pca": (0.02, 0.96),
            "mds": (0.08, 0.9),
            "le": (0.1, 0.18),
            "tsne": (0.35, 0.525),
            "umap": (0.23, 0.65),
            "phate": (0.2, 0.28),
        }
    elif data_name == paths.kanton_name:
        labels_pos = {
            "pca": (0.02, 0.96),
            "mds": (0.08, 0.9),
            "le": (0.1, 0.35),
            "tsne": (0.3, 0.65),
            "umap": (0.18, 0.55),
            "phate": (0.12, 0.78),
        }
    elif data_name == paths.genomes_name:
        labels_pos = {
            "pca": (0.001, 0.8),
            "mds": (0.001, 0.7),
            "le": (0.05, 0.18),
            "tsne": (0.3, 0.5),
            "umap": (0.15, 0.38),
            "phate": (0.15, 0.6),
        }
    else:
        labels_pos = {
            "pca": (0.02, 0.96),
            "mds": (0.08, 0.9),
            "le": (0.1, 0.18),
            "tsne": (0.35, 0.5),
            "umap": (0.23, 0.43),
            "phate": (0.2, 0.28),
        }
    
    for x, y in [(paths.knn_recall_path, paths.pearson_corr_path), (paths.auc_path, paths.pearson_corr_path)]: # , (paths.auc_path, paths.sigma_d_path)
        # Loading quality scores
        D_qa = dict()
        L_meths = list()
        
        # PCA: 
        L_meths.append('pca')
        D_qa['pca'] = dict()
        D_qa['pca'][x] = np.asarray([np.load('{rp}{npath}-{k}.npy'.format(rp=res_path_qa, npath=paths.pca_path, k=x))], dtype=np.float64)
        D_qa['pca'][y] = np.asarray([np.load('{rp}{npath}-{k}.npy'.format(rp=res_path_qa, npath=paths.pca_path, k=y))], dtype=np.float64)
        
        # MDS: 
        L_meths.append('mds')
        D_qa['mds'] = dict()
        D_qa['mds'][x] = np.asarray([np.load('{rp}{npath}-{k}.npy'.format(rp=res_path_qa, npath=paths.mds_sklearn_path if genomes else paths.mds_path, k=x))], dtype=np.float64)
        D_qa['mds'][y] = np.asarray([np.load('{rp}{npath}-{k}.npy'.format(rp=res_path_qa, npath=paths.mds_sklearn_path if genomes else paths.mds_path, k=y))], dtype=np.float64)
        
        # LE: 
        L_meths.append('le')
        D_qa['le'] = dict()
        D_qa['le'][x] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.LE_path_no_param), k=x)) for nnp in params.L_nn_perp if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.LE_path_no_param), k=x))], dtype=np.float64)
        D_qa['le'][y] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.LE_path_no_param), k=y)) for nnp in params.L_nn_perp if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.LE_path_no_param), k=y))], dtype=np.float64)
        
        # t-SNE: 
        L_meths.append('tsne')
        D_qa['tsne'] = dict()
        D_qa['tsne'][x] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-p{p}'.format(p=int(round(nnp)), v=paths.tsne_path_no_param), k=x)) for nnp in params.L_nn_perp if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-p{p}'.format(p=int(round(nnp)), v=paths.tsne_path_no_param), k=x))], dtype=np.float64)
        D_qa['tsne'][y] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-p{p}'.format(p=int(round(nnp)), v=paths.tsne_path_no_param), k=y)) for nnp in params.L_nn_perp if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-p{p}'.format(p=int(round(nnp)), v=paths.tsne_path_no_param), k=y))], dtype=np.float64)
        
        # UMAP: 
        L_meths.append('umap')
        D_qa['umap'] = dict()
        D_qa['umap'][x] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.umap_path_no_param), k=x)) for nnp in params.L_nn_perp if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.umap_path_no_param), k=x))], dtype=np.float64)
        D_qa['umap'][y] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.umap_path_no_param), k=y)) for nnp in params.L_nn_perp if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.umap_path_no_param), k=y))], dtype=np.float64)
        
        # PHATE: 
        L_meths.append('phate')
        D_qa['phate'] = dict()
        D_qa['phate'][x] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.phate_path_no_param), k=x)) for nnp in params.L_nn_perp if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.phate_path_no_param), k=x))], dtype=np.float64)
        D_qa['phate'][y] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.phate_path_no_param), k=y)) for nnp in params.L_nn_perp if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.phate_path_no_param), k=y))], dtype=np.float64)
        
        plt.figure(figsize=(8,8))
        
        for algo in L_meths:
            plt.scatter(
                D_qa[algo][x],
                D_qa[algo][y],
                c=colors[algo],
                marker=markers[algo],
                s=40,
                clip_on=False,
            )
            
            if D_qa[algo][x].size > 1:
                plt.plot(
                    D_qa[algo][x],
                    D_qa[algo][y],
                    c=colors[algo],
                    lw=2,
                    clip_on=False,
                )
            
            plt.text(
                *labels_pos[algo],
                labels[algo],
                ha="center",
                va="center",
                c="w",
                fontsize=13,
                bbox=dict(
                    facecolor=colors[algo], edgecolor="none", boxstyle="round", pad=0.2
                )
            )
        
        plt.xlim([0, 0.5])
        plt.ylim([0, 1])
        plt.xlabel("Local quality (neighbor preservation)")
        plt.ylabel("Global quality (distance preservation)")
        
        ax = plt.gca()
        
        ax.spines.left.set_position(("data", -0.025))
        ax.spines.bottom.set_position(("data", -0.05))
        
        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig_name = "{v}_hps/{x}-{y}".format(v=fig_path, x=x, y=y)
        plot_fcts.check_create_dir(fig_name)
        
        #plt.savefig("{v}.pdf".format(v=fig_name), bbox_inches="tight")
        plt.savefig("{v}.png".format(v=fig_name), dpi=300, facecolor="white", bbox_inches="tight")

