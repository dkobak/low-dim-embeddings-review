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
from matplotlib.gridspec import GridSpec
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

def plot_hps_qa_emb(data_name, res_path_qa, genomes, fig_path, arr_colors, emb_path):
    """
    Creates a figure to illustrate the influence of the hyper-parameters of the embedding methods on some data set.
    In: similar inputs as the ones of the compute_embs_quality_sev_hps function in ./utils/run_embs and of create_2x3_figure and create_2x3_figures_hps functions in ./utils/plot_fcts.
    """
    
    print('===')
    print("=== Creating a figure for {v} data illustrating the influence of the hyper-parameters".format(v=data_name))
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
            "pca": (0, 0.99),
            "mds": (0.08, 0.95),
            "le": (0.1, 0.15),
            "tsne": (0.35, 0.6),
            "umap": (0.23, 0.675),
            "phate": (0.2, 0.2),
        }
    elif data_name == paths.kanton_name:
        labels_pos = {
            "pca": (0.0, 1.0),
            "mds": (0.07, 0.95),
            "le": (0.1, 0.4),
            "tsne": (0.3, 0.65),
            "umap": (0.18, 0.5),
            "phate": (0.01, 0.65), # (0.12, 0.525),
        }
    elif data_name == paths.genomes_name:
        labels_pos = {
            "pca": (0.001, 0.9),
            "mds": (0.001, 0.6),
            "le": (0.05, 0.18),
            "tsne": (0.3, 0.5),
            "umap": (0.15, 0.35),
            "phate": (0.09, 0.9),
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
    
    L_show_nnp_le = [10,500]
    L_show_nnp_tsne = [10,500]
    L_show_nnp_umap = [5,500]
    L_show_nnp_phate = [5,500]
    
    arr_nn_perp = np.asarray(a=params.L_nn_perp, dtype=np.int64)
    
    for x, y in [(paths.knn_recall_path, paths.pearson_corr_path)]: # , (paths.auc_path, paths.sigma_d_path), (paths.auc_path, paths.pearson_corr_path)
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
        D_qa['le'][x] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.LE_path_no_param), k=x)) for nnp in params.L_nn_perp], dtype=np.float64) #  if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.LE_path_no_param), k=x))
        D_qa['le'][y] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.LE_path_no_param), k=y)) for nnp in params.L_nn_perp], dtype=np.float64) #  if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.LE_path_no_param), k=y))
        
        # t-SNE: 
        L_meths.append('tsne')
        D_qa['tsne'] = dict()
        D_qa['tsne'][x] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-p{p}'.format(p=int(round(nnp)), v=paths.tsne_path_no_param), k=x)) for nnp in params.L_nn_perp], dtype=np.float64) #  if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-p{p}'.format(p=int(round(nnp)), v=paths.tsne_path_no_param), k=x))
        D_qa['tsne'][y] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-p{p}'.format(p=int(round(nnp)), v=paths.tsne_path_no_param), k=y)) for nnp in params.L_nn_perp], dtype=np.float64) #  if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-p{p}'.format(p=int(round(nnp)), v=paths.tsne_path_no_param), k=y))
        
        # UMAP: 
        L_meths.append('umap')
        D_qa['umap'] = dict()
        D_qa['umap'][x] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.umap_path_no_param), k=x)) for nnp in params.L_nn_perp], dtype=np.float64) #  if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.umap_path_no_param), k=x))
        D_qa['umap'][y] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.umap_path_no_param), k=y)) for nnp in params.L_nn_perp], dtype=np.float64) #  if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.umap_path_no_param), k=y))
        
        # PHATE: 
        L_meths.append('phate')
        D_qa['phate'] = dict()
        D_qa['phate'][x] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.phate_path_no_param), k=x)) for nnp in params.L_nn_perp], dtype=np.float64) #  if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.phate_path_no_param), k=x))
        D_qa['phate'][y] = np.asarray([np.load('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.phate_path_no_param), k=y)) for nnp in params.L_nn_perp], dtype=np.float64) #  if os.path.exists('{rp}hps/{npath}-{k}.npy'.format(rp=res_path_qa, npath='{v}-n{n}'.format(n=nnp, v=paths.phate_path_no_param), k=y))
        
        fig = plt.figure(figsize=(7, 5.5))# plt.figure(figsize=(7, 3.5))
        gs = GridSpec(nrows=4, ncols=4, wspace=0.0, figure=fig)
        
        ax = plt.subplot(gs[0:2,1:3])
        
        s_highlight = 30
        
        for algo in L_meths:
            ax.scatter(
                D_qa[algo][x],
                D_qa[algo][y],
                c=colors[algo],
                marker=markers[algo],
                s=5 if D_qa[algo][x].size > 1 else s_highlight,
                clip_on=False,
            )
            
            if (algo != 'pca') and (algo != 'mds'):
                
                if algo == 'le':
                    L_show_nnp = L_show_nnp_le
                elif algo == 'tsne':
                    L_show_nnp = L_show_nnp_tsne
                elif algo == 'umap':
                    L_show_nnp = L_show_nnp_umap
                elif algo == 'phate':
                    L_show_nnp = L_show_nnp_phate
                else: 
                    L_show_nnp = []
                
                L_idx_show_nnp = [np.argwhere(np.isclose(arr_nn_perp-v,0))[0][0] for v in L_show_nnp]
                
                ax.scatter(
                    D_qa[algo][x][L_idx_show_nnp],
                    D_qa[algo][y][L_idx_show_nnp],
                    c=colors[algo],
                    marker=markers[algo],
                    s=s_highlight,
                    clip_on=False,
                )
                
                if algo == 'le':
                    default_nn = params.nn_LE
                elif algo == 'tsne':
                    default_nn = params.perp_tsne
                elif algo == 'umap':
                    default_nn = params.nn_umap
                elif algo == 'phate':
                    default_nn = params.nn_phate
                
                idx_def_nn = [np.argwhere(np.isclose(arr_nn_perp-default_nn,0))[0][0]]
                
                ax.scatter(
                    D_qa[algo][x][idx_def_nn],
                    D_qa[algo][y][idx_def_nn],
                    c=colors[algo],
                    marker=markers[algo],
                    s=s_highlight,
                    clip_on=False,
                )
            
            if D_qa[algo][x].size > 1:
                ax.plot(
                    D_qa[algo][x],
                    D_qa[algo][y],
                    c=colors[algo],
                    lw=.5,
                    clip_on=False,
                )
            
            ax.text(
                *labels_pos[algo],
                labels[algo],
                ha="center",
                va="center",
                c="w",
                fontsize=7,
                bbox=dict(
                    facecolor=colors[algo], edgecolor="none", boxstyle="round", pad=0.2
                )
            )
        
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Local quality (neighbor preservation)")
        ax.set_ylabel("Global quality (distance preservation)")
        
        ax.spines.left.set_position(("data", -0.025))
        ax.spines.bottom.set_position(("data", -0.05))
        
        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ###
        ###
        ###
        
        res_path_emb_hps = "{v}hps/".format(v=emb_path)
        
        for i_nnp in range(2):
            
            nnp_le = L_show_nnp_le[i_nnp]
            cur_LE_name = 'Lapl. Eig. ({n} neighbors)'.format(n=nnp_le, v=paths.LE_name_no_param)
            cur_LE_path = '{v}-n{n}'.format(n=nnp_le, v=paths.LE_path_no_param)
            
            nnp_tsne = L_show_nnp_tsne[i_nnp]
            cur_tsne_name = r't-SNE (perplexity: {p})'.format(p=nnp_tsne)#, v=paths.tsne_name_no_param)
            cur_tsne_path = '{v}-p{p}'.format(v=paths.tsne_path_no_param, p=int(round(nnp_tsne)))
            
            nnp_umap = L_show_nnp_umap[i_nnp]
            cur_umap_name = '{v} ({n} neighbors)'.format(n=nnp_umap, v=paths.umap_name_no_param)
            cur_umap_path = '{v}-n{n}'.format(n=nnp_umap, v=paths.umap_path_no_param)
            
            nnp_phate = L_show_nnp_phate[i_nnp]
            cur_phate_name = '{v} ({n} neighbors)'.format(n=nnp_phate, v=paths.phate_name_no_param)
            cur_phate_path = '{v}-n{n}'.format(n=nnp_phate, v=paths.phate_path_no_param)
            
            ##############################
            ############################## 
            # Laplacian eigenmaps (LE)
            ####################
            
            X_LE = np.load('{rp}{npath}.npy'.format(rp=res_path_emb_hps, npath=cur_LE_path))
            
            if data_name == paths.tasic_name:
                flipx = False
                flipy = False
            elif data_name == paths.genomes_name:
                flipx = False
                flipy = False
            elif data_name == paths.kanton_name:
                flipx = True
                flipy = False
            else:
                flipx = False
                flipy = False
            #tit=paths.LE_name_no_param
            plot_fcts.viz_2d_emb(X=X_LE, vcol=arr_colors, tit=cur_LE_name, ax_def=gs[2+i_nnp,0], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, LE_tasic=False, LE_genomes=False, LE_kanton=False, D_viz_emb=None, loc_center=True, ylab=None) #ylab="# neighbors / perplexity: {n}".format(n=nnp_le)
            
            ##############################
            ############################## 
            # PHATE
            ####################
            
            X_phate = np.load('{rp}{npath}.npy'.format(rp=res_path_emb_hps, npath=cur_phate_path))
            
            if data_name == paths.tasic_name:
                flipx = True
                flipy = False
            elif data_name == paths.genomes_name:
                flipx = True
                flipy = False
            elif data_name == paths.kanton_name:
                flipx = False
                flipy = False
            else:
                flipx = False
                flipy = False
            #tit=paths.phate_name_no_param
            plot_fcts.viz_2d_emb(X=X_phate, vcol=arr_colors, tit=cur_phate_name, ax_def=gs[2+i_nnp,1], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, phate_kanton=False, D_viz_emb=None, loc_center=True)
            
            ##############################
            ############################## 
            # t-SNE
            ####################
            
            X_tsne = np.load('{rp}{npath}.npy'.format(rp=res_path_emb_hps, npath=cur_tsne_path))
            
            if data_name == paths.tasic_name:
                flipx = True
                flipy = True
            elif data_name == paths.genomes_name:
                flipx = False
                flipy = False
            elif data_name == paths.kanton_name:
                flipx = False
                flipy = False
            else:
                flipx = False
                flipy = False
            
            plot_fcts.viz_2d_emb(X=X_tsne, vcol=arr_colors, tit=cur_tsne_name, ax_def=gs[2+i_nnp,2], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, tsne_tasic=False, D_viz_emb=None, loc_center=True)
            
            ##############################
            ############################## 
            # UMAP
            ####################
            
            X_umap = np.load('{rp}{npath}.npy'.format(rp=res_path_emb_hps, npath=cur_umap_path))
            
            if data_name == paths.tasic_name:
                flipx = False
                flipy = True
            elif data_name == paths.genomes_name:
                flipx = False
                flipy = False
            elif data_name == paths.kanton_name:
                flipx = True
                flipy = True
            else:
                flipx = False
                flipy = False
            #tit=paths.umap_name_no_param
            plot_fcts.viz_2d_emb(X=X_umap, vcol=arr_colors, tit=cur_umap_name, ax_def=gs[2+i_nnp,3], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, umap_tasic=False, umap_genomes=False, D_viz_emb=None, loc_center=True)
        
        fig_name = "{v}_hps/{x}-{y}_embs".format(v=fig_path, x=x, y=y)
        plot_fcts.check_create_dir(fig_name)
        
        #plt.savefig("{v}.pdf".format(v=fig_name), bbox_inches="tight")
        plt.savefig("{v}.png".format(v=fig_name), dpi=300, facecolor="white", bbox_inches="tight")
        plt.close()
