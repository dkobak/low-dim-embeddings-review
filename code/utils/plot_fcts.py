#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# This file contains functions to produce figures.

########################################################################################################
########################################################################################################

import numpy as np, matplotlib.pyplot as plt, os, paths, copy
from matplotlib.gridspec import GridSpec
from matplotlib import style 

# Name of this file
module_name = "plot_fcts.py"

plt.rcParams['text.usetex'] = True
plt.style.use('./utils/mpl_style.txt') 

##############################
##############################

def rstr(v, d=2):
    """
    Rounds v with d digits and returns it as a string. If it starts with 0, it is omitted. 
    In:
    - v: a number. 
    - d: number of digits to keep.
    Out:
    A string representing v rounded with d digits. If it starts with 0, it is omitted. 
    """
    p = 10.0**d
    v = str(int(round(v*p))/p)
    if v[0] == '0':
        v = v[1:]
    elif (len(v) > 3) and (v[:3] == '-0.'):
        v = "-.{a}".format(a=v[3:])
    
    if v[-2:] == '.0':
        v = v[:-2]
    
    if v == '':
        v = '0'
    
    return v

def check_create_dir(path):
    """
    Create a directory at the specified path only if it does not already exist.
    """
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def save_show_fig(fname=None, f_format=None, dpi=300):
    """
    Save or show a figure.
    In:
    - fname: filename to save the figure, without the file extension. If None, the figure is shown.
    - f_format: format to save the figure. If None, set to pdf. 
    - dpi: DPI to save the figure.
    Out: 
    A figure is shown if fname is None, and saved otherwise.
    """
    if fname is None:
        plt.show()
    else:
        if f_format is None:
            f_format = 'png'
        # Checking whether a folder needs to be created
        check_create_dir(fname)
        # Saving the figure
        plt.savefig("{fname}.{f_format}".format(fname=fname, f_format=f_format), format=f_format, bbox_inches='tight') 

def hex_to_rgb(hex_color):
    """
    Given an hexadecimal color description, translate it in RGB. 
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def arr_hex_to_rgb(arr_hex):
    """
    Similar to hex_to_rgb, but for an array of hexadecimal descriptions. 
    """
    arr_rgb = np.empty(shape=(arr_hex.size, 3), dtype=np.int32)
    for i, hex_color in enumerate(arr_hex):
        rgb = hex_to_rgb(hex_color)
        for j in range(3):
            arr_rgb[i,j] = rgb[j]
    return arr_rgb

def rgb_to_hex(arr_rgb_color):
    """
    Given an RGB color description, translate it in hexadecimal. 
    arr_rgb_color is assumed to be a 1-D numpy array with 3 integer values between 0 and 255.
    """
    rgb_color = (arr_rgb_color[0], arr_rgb_color[1], arr_rgb_color[2])
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)

def viz_2d_emb(X, vcol, tit='', fname=None, f_format=None, ax_def=None, flipx=False, flipy=False, genomes=False, show=False, pca_tasic=False, tsne_tasic=False, umap_tasic=False, LE_tasic=False, pca_genomes=False, umap_genomes=False, LE_genomes=False, LE_kanton=False, phate_kanton=False, D_viz_emb=None):
    """
    Plot a 2-D embedding of a data set.
    In:
    - X: a 2-D numpy array with shape (N, 2), where N is the number of data points to represent in the 2-D embedding.
    - vcol: a 1-D numpy array with N elements, indicating the colors of the data points.
    - tit: title of the figure.
    - fname, f_format: same as in save_show_fig.
    - ax_def: if not None, define ax object for the figure 
    - flipx: flip the x-axis. 
    - flipy: flip the y-axis. 
    - genomes: boolean. Set to True if 1000 Genomes Project data are used. 
    - show: boolean. If True, the figure is shown. 
    - pca_tasic: boolean. Set to True if PCA embedding of Tasic et al data is displayed. 
    - tsne_tasic: boolean. Set to True if t-SNE embedding of Tasic et al data is displayed. 
    - umap_tasic: boolean. Set to True if UMAP embedding of Tasic et al data is displayed. 
    - LE_tasic: boolean. Set to True if LE embedding of Tasic et al data is displayed. 
    - pca_genomes: boolean. Set to True if PCA embedding of 1000 Genomes Project data is displayed. 
    - umap_genomes: boolean. Set to True if UMAP embedding of 1000 Genomes Project data is displayed. 
    - LE_genomes: boolean. Set to True if LE embedding of 1000 Genomes Project data is displayed. 
    - LE_kanton: boolean. Set to True if LE embedding of Kanton et al data is displayed. 
    - phate_kanton: boolean. Set to True if PHATE embedding of Kanton et al data is displayed. 
    - D_viz_emb: dictionary with entries enabling to annotate the figure, depending on the embedding which is currently displayed. 
    Out:
    Same as save_show_fig.
    """  
    global module_name
    
    # Checking X
    if X.ndim != 2:
        raise ValueError("Error in function viz_2d_emb of {module_name}: X must be a numpy array with shape (N, 2), where N is the number of data points to plot in the 2-D embedding.".format(module_name=module_name))
    if X.shape[1] != 2:
        raise ValueError("Error in function viz_2d_emb of {module_name}: X must have 2 columns.".format(module_name=module_name))
    
    # Flipping the x and y axes
    if flipx:
        X[:,0] = -X[:,0]
    if flipy:
        X[:,1] = -X[:,1]
    
    # Moving back to the origin
    for i in range(2):
        X[:,i] -= X[:,i].min()
    
    # Scaling
    maxmax = max(X[:,0].max(), X[:,1].max())
    X /= maxmax
    
    # Margin fraction to define the axes limits
    ax_margin = 0.025
    
    # Limits of the axes
    xmin = 0.0 
    xmax = 1.0 
    ev = (xmax-xmin)*ax_margin
    x_lim = np.asarray([xmin-ev, xmax+ev])
    
    ymin = 0.0 
    ymax = 1.0 
    ev = (ymax-ymin)*ax_margin
    y_lim = np.asarray([ymin-ev, ymax+ev])
    
    if ax_def is None:
        fig = plt.figure(figsize=(7.0/4.0, 3.5/2.0))
        ax = fig.add_subplot(111)
    else:
        ax = plt.subplot(ax_def)
    
    # Setting the limits of the axes
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    # Plotting the data points
    if LE_tasic or LE_genomes:
        emb_markersize = 5
        emb_alpha = .1
    else:
        emb_alpha = None
        emb_markersize = 0.1
    if genomes:
        emb_markersize += 0.4
    ax.scatter(X[:,0], X[:,1], c=vcol, s=emb_markersize, alpha=emb_alpha, linewidths=0.0)
    
    # Removing the ticks
    for minor_v in [True, False]:
        ax.set_xticks([], minor=minor_v)
        ax.set_yticks([], minor=minor_v)
    ax.set_xticklabels([], minor=False)
    ax.set_yticklabels([], minor=False)
    
    ax.set_title(tit, color="black", fontweight="bold")
    
    # Making the axes invisible
    for side in ['bottom', 'top', 'left', 'right']:
        ax.spines[side].set_linewidth(0.0)
    
    ax.set_aspect('equal', adjustable='box')
    
    if pca_tasic:
        cold_col_hex = vcol[np.logical_and(np.logical_and(X[:,0] > 0.0 , X[:,0] < 0.4 ), np.logical_and(X[:,1] > 0.25, X[:,1] < 0.55))]
        warm_col_hex = vcol[np.logical_and(np.logical_and(X[:,0] > 0.35, X[:,0] < 0.65), np.logical_and(X[:,1] > 0.0 , X[:,1] < 0.25))]
        grey_col_hex = vcol[np.logical_and(np.logical_and(X[:,0] > 0.75, X[:,0] < 1.0 ), np.logical_and(X[:,1] > 0.65, X[:,1] < 1.0 ))]
        
        cold_col_avg_hex = rgb_to_hex(arr_hex_to_rgb(cold_col_hex).mean(axis=0).astype(np.int32))
        warm_col_avg_hex = rgb_to_hex(arr_hex_to_rgb(warm_col_hex).mean(axis=0).astype(np.int32))
        grey_col_avg_hex = rgb_to_hex(arr_hex_to_rgb(grey_col_hex).mean(axis=0).astype(np.int32))
        
        plt.text(0.05, 0.55, 'Excitatory neurons', ha="left", va="center", c=cold_col_avg_hex, fontsize=6)
        plt.text(0.56, 0.05, 'Inhibitory neurons', ha="left", va="center", c=warm_col_avg_hex, fontsize=6)
        plt.text(0.55, 0.9, 'Non-neurons', ha="left", va="center", c=grey_col_avg_hex, fontsize=6)
    
    if tsne_tasic or umap_tasic:
        clusters = D_viz_emb['clusters']
        clusterColors = D_viz_emb['clusterColors']
        clusterNames = D_viz_emb['clusterNames']
        
        L_bigC = ['Lamp5', 'Sncg', 'Serpinf1', 'Vip', 'Sst', 'Pvalb', 'L2/3 IT', 'L4 IT', 'L5 IT', 'L6 IT', 'L5 PT', 'L5 NP', 'L6 NP', 'L6 CT', 'L6b'] 
        D_bigC = {k:{'idx':np.zeros(shape=X.shape[0], dtype=bool), 'Lcolors':[]} for k in L_bigC}
        
        n_clusters = clusterColors.size
        for i in range(n_clusters):
            for bigC in L_bigC:
                min_len = min(len(bigC), len(clusterNames[i]))
                if clusterNames[i][:min_len] == bigC:
                    D_bigC[bigC]['idx'] = np.logical_or(D_bigC[bigC]['idx'], clusters == i)
                    D_bigC[bigC]['Lcolors'].append(clusterColors[i])
                    break
        
        for bigC in L_bigC:
            D_bigC[bigC]['avg_color'] = rgb_to_hex(arr_hex_to_rgb(np.asarray(D_bigC[bigC]['Lcolors'])).mean(axis=0).astype(np.int32))
            D_bigC[bigC]['avg_pos'] = X[D_bigC[bigC]['idx'],:].mean(axis=0)
            
            # Adjusting labels
            if tsne_tasic:
                if bigC == 'Lamp5': #
                    D_bigC[bigC]['avg_pos'][0] += 0.02
                    D_bigC[bigC]['avg_pos'][1] -= 0.065
                elif bigC == 'Sncg': #
                    D_bigC[bigC]['avg_pos'][0] += 0.0
                    D_bigC[bigC]['avg_pos'][1] -= 0.05
                elif bigC == 'Serpinf1': #
                    D_bigC[bigC]['avg_pos'][0] += 0.0175
                    D_bigC[bigC]['avg_pos'][1] += 0.01
                elif bigC == 'Vip': #
                    D_bigC[bigC]['avg_pos'][0] -= 0.005
                    D_bigC[bigC]['avg_pos'][1] += 0.015
                elif bigC == 'Sst': #
                    D_bigC[bigC]['avg_pos'][0] += 0.018
                    D_bigC[bigC]['avg_pos'][1] -= 0.02
                elif bigC == 'Pvalb': #
                    D_bigC[bigC]['avg_pos'][0] -= 0.075
                    D_bigC[bigC]['avg_pos'][1] -= 0.015
                elif bigC == 'L2/3 IT': #
                    D_bigC[bigC]['avg_pos'][0] -= 0.095
                    D_bigC[bigC]['avg_pos'][1] += 0.0
                elif bigC == 'L4 IT': #
                    D_bigC[bigC]['avg_pos'][0] += 0.115
                    D_bigC[bigC]['avg_pos'][1] += 0.0
                elif bigC == 'L5 IT': #
                    D_bigC[bigC]['avg_pos'][0] -= 0.11
                    D_bigC[bigC]['avg_pos'][1] += 0.02
                elif bigC == 'L6 IT': #
                    D_bigC[bigC]['avg_pos'][0] += 0.015
                    D_bigC[bigC]['avg_pos'][1] += 0.105
                elif bigC == 'L5 PT': #
                    D_bigC[bigC]['avg_pos'][0] -= 0.12
                    D_bigC[bigC]['avg_pos'][1] -= 0.04
                elif bigC == 'L5 NP': #
                    D_bigC[bigC]['avg_pos'][0] -= 0.075
                    D_bigC[bigC]['avg_pos'][1] += 0.04
                elif bigC == 'L6 NP': #
                    D_bigC[bigC]['avg_pos'][0] += 0.01
                    D_bigC[bigC]['avg_pos'][1] -= 0.04
                elif bigC == 'L6 CT': #
                    D_bigC[bigC]['avg_pos'][0] += 0.0
                    D_bigC[bigC]['avg_pos'][1] += 0.085
                elif bigC == 'L6b': #
                    D_bigC[bigC]['avg_pos'][0] -= 0.05
                    D_bigC[bigC]['avg_pos'][1] += 0.04
            elif umap_tasic:
                if bigC == 'Lamp5': 
                    D_bigC[bigC]['avg_pos'][0] += 0.085
                    D_bigC[bigC]['avg_pos'][1] -= 0.0
                elif bigC == 'Sncg': 
                    D_bigC[bigC]['avg_pos'][0] -= 0.07
                    D_bigC[bigC]['avg_pos'][1] -= 0.0
                elif bigC == 'Serpinf1': 
                    D_bigC[bigC]['avg_pos'][0] -= 0.0825
                    D_bigC[bigC]['avg_pos'][1] += 0.0
                elif bigC == 'Vip': 
                    D_bigC[bigC]['avg_pos'][0] += 0.06
                    D_bigC[bigC]['avg_pos'][1] -= 0.055
                elif bigC == 'Sst': 
                    D_bigC[bigC]['avg_pos'][0] -= 0.0
                    D_bigC[bigC]['avg_pos'][1] += 0.055
                elif bigC == 'Pvalb': 
                    D_bigC[bigC]['avg_pos'][0] -= 0.0
                    D_bigC[bigC]['avg_pos'][1] -= 0.06
                elif bigC == 'L2/3 IT': 
                    D_bigC[bigC]['avg_pos'][0] += 0.0
                    D_bigC[bigC]['avg_pos'][1] -= 0.055
                elif bigC == 'L4 IT': 
                    D_bigC[bigC]['avg_pos'][0] -= 0.0
                    D_bigC[bigC]['avg_pos'][1] += 0.05
                elif bigC == 'L5 IT': 
                    D_bigC[bigC]['avg_pos'][0] += 0.035
                    D_bigC[bigC]['avg_pos'][1] += 0.0
                elif bigC == 'L6 IT': 
                    D_bigC[bigC]['avg_pos'][0] -= 0.05
                    D_bigC[bigC]['avg_pos'][1] += 0.005
                elif bigC == 'L5 PT': 
                    D_bigC[bigC]['avg_pos'][0] -= 0.0
                    D_bigC[bigC]['avg_pos'][1] += 0.0
                elif bigC == 'L5 NP': 
                    D_bigC[bigC]['avg_pos'][0] += 0.0
                    D_bigC[bigC]['avg_pos'][1] += 0.05
                elif bigC == 'L6 NP': 
                    D_bigC[bigC]['avg_pos'][0] += 0.0
                    D_bigC[bigC]['avg_pos'][1] -= 0.04
                elif bigC == 'L6 CT': 
                    D_bigC[bigC]['avg_pos'][0] += 0.0
                    D_bigC[bigC]['avg_pos'][1] -= 0.06
                elif bigC == 'L6b': 
                    D_bigC[bigC]['avg_pos'][0] -= 0.0
                    D_bigC[bigC]['avg_pos'][1] += 0.04
            
            plt.text(D_bigC[bigC]['avg_pos'][0], D_bigC[bigC]['avg_pos'][1], bigC, ha="center", va="center", c=D_bigC[bigC]['avg_color'], fontsize=4)
    
    if LE_kanton or phate_kanton:
        D_samp_by_time = D_viz_emb['D_samp_by_time']
        
        for k_time in D_samp_by_time:
            idx_samp = D_samp_by_time[k_time]
            mean_samp = X[idx_samp,:].mean(axis=0)
            mean_x, mean_y = mean_samp[0], mean_samp[1]
            
            # Adjusting labels
            if LE_kanton:
                if k_time == "0 days":
                    mean_x += 0.125
                    mean_y += 0.0
                elif k_time == "4 days":
                    mean_x += 0.115
                    mean_y += 0.0
                elif k_time == "10 days":
                    mean_x -= 0.14
                    mean_y += 0.0
                elif k_time == "15 days":
                    mean_x += 0.0
                    mean_y += 0.06
                elif k_time == "1 month":
                    mean_x -= 0.125
                    mean_y += 0.0
                elif k_time == "2 months":
                    mean_x -= 0.16
                    mean_y += 0.0
                elif k_time == "4 months":
                    mean_x -= 0.16
                    mean_y += 0.0
            elif phate_kanton:
                if k_time == "0 days":
                    mean_x += 0.135
                    mean_y += 0.025
                elif k_time == "4 days":
                    mean_x += 0.11
                    mean_y -= 0.025
                elif k_time == "10 days":
                    mean_x -= 0.0
                    mean_y += 0.14
                elif k_time == "15 days":
                    mean_x += 0.075
                    mean_y -= 0.1
                elif k_time == "1 month":
                    mean_x += 0.0
                    mean_y -= 0.07
                elif k_time == "2 months":
                    mean_x -= 0.125
                    mean_y += 0.0
                elif k_time == "4 months":
                    mean_x -= 0.175
                    mean_y += 0.0
            
            plt.text(mean_x, mean_y, k_time, ha="center", va="center", c=vcol[idx_samp[0]], fontsize=7)
    
    if pca_genomes or umap_genomes: 
        D_pop = D_viz_emb['D_pop']
        D_superpop = D_viz_emb['D_superpop']
        
        if umap_genomes: 
            for k_pop in D_pop:
                idx_samp = D_pop[k_pop]['idx']
                mean_samp = X[idx_samp,:].mean(axis=0)
                mean_x, mean_y = mean_samp[0], mean_samp[1]
                cur_text = k_pop
                
                # Adjusting labels
                if k_pop == "ACB": # African Caribbean in Barbados
                    mean_x -= 0.075
                    mean_y += 0.05
                elif k_pop == "ASW": # African Ancestry in Southwest US
                    mean_x -= 0.125
                    mean_y -= 0.025
                elif k_pop == "BEB": # Bengali in Bangladesh
                    mean_x += 0.0
                    mean_y -= 0.05
                elif k_pop == "CDX": # Chinese Dai in Xishuangbanna, China
                    mean_x -= 0.075
                    mean_y -= 0.04
                elif k_pop == "CEU": # Utah residents with N and W European ancestry
                    mean_x -= 0.075
                    mean_y += 0.04
                elif k_pop == "CHB": # Han Chinese in Bejing, China
                    mean_x += 0.0
                    mean_y -= 0.05
                elif k_pop == "CHS": # Southern Han Chinese, China
                    mean_x += 0.0
                    mean_y += 0.05
                elif k_pop == "CLM": # Colombian in Medellin, Colombia
                    mean_x += 0.08
                    mean_y -= 0.0
                elif k_pop == "ESN": # Esan in Nigeria
                    mean_x += 0.09
                    mean_y -= 0.005
                elif k_pop == "FIN": # Finnish in Finland
                    mean_x += 0.0
                    mean_y += 0.035
                elif k_pop == "GBR": # British in England and Scotland
                    mean_x += 0.05
                    mean_y += 0.04
                elif k_pop == "GIH": # Gujarati Indian in Houston, TX
                    mean_x += 0.075
                    mean_y += 0.0
                elif k_pop == "GWD": # Gambian in Western Division
                    mean_x += 0.0
                    mean_y -= 0.05
                elif k_pop == "IBS": # Iberian populations in Spain
                    mean_x -= 0.075
                    mean_y += 0.0
                elif k_pop == "ITU": # Indian Telugu in the UK
                    mean_x += 0.0
                    mean_y += 0.02
                elif k_pop == "JPT": # Japanese in Tokyo, Japan
                    mean_x += 0.07
                    mean_y += 0.0
                elif k_pop == "KHV": # Kinh in Ho Chi Minh City, Vietnam
                    mean_x -= 0.08
                    mean_y += 0.015
                elif k_pop == "LWK": # Luhya in Webuye, Kenya
                    mean_x += 0.0
                    mean_y += 0.04
                elif k_pop == "MSL": # Mende in Sierra Leone
                    mean_x += 0.075
                    mean_y += 0.0
                elif k_pop == "MXL": # Mexican Ancestry in LA, California
                    mean_x -= 0.08
                    mean_y += 0.0
                elif k_pop == "PEL": # Peruvian in Lima, Peru
                    mean_x += 0.0
                    mean_y -= 0.05
                elif k_pop == "PJL": # Punjabi in Lahore, Pakistan
                    mean_x -= 0.055
                    mean_y += 0.01
                elif k_pop == "PUR": # Puerto Rican in Puerto Rico
                    mean_x += 0.1
                    mean_y += 0.0
                elif k_pop == "STU": # Sri Lankan Tamil in the UK
                    mean_x += 0.075
                    mean_y -= 0.01
                elif k_pop == "TSI": # Toscani in Italy
                    mean_x += 0.08
                    mean_y += 0.0
                elif k_pop == "YRI": # Yoruba in Ibadan, Nigeria
                    mean_x -= 0.055
                    mean_y -= 0.05
                
                plt.text(mean_x, mean_y, cur_text, ha="center", va="center", c=D_pop[k_pop]['color'], fontsize=6)
        
        if pca_genomes:
            for k_superpop in D_superpop:
                idx_samp = D_superpop[k_superpop]['idx']
                mean_samp = X[idx_samp,:].mean(axis=0)
                mean_x, mean_y = mean_samp[0], mean_samp[1]
                
                # Adjusting labels
                if pca_genomes:
                    if k_superpop == "AFR":
                        mean_x += 0.0
                        mean_y += 0.075
                    elif k_superpop == "EUR":
                        mean_x += 0.19
                        mean_y += 0.0
                    elif k_superpop == "AMR":
                        mean_x += 0.4
                        mean_y -= 0.2
                    elif k_superpop == "SAS":
                        mean_x += 0.19
                        mean_y += 0.05
                    elif k_superpop == "EAS":
                        mean_x += 0.18
                        mean_y += 0.0
                
                plt.text(mean_x, mean_y, D_superpop[k_superpop]['superpop_descr'], ha="center", va="center", c=D_superpop[k_superpop]['color'], fontsize=6)
    
    if show:
        plt.show()
    
    # Saving or showing the figure, and closing
    if ax_def is None:
        save_show_fig(fname=fname, f_format=f_format)
        plt.close()

def create_2x3_figure(data_name, emb_path, fig_path, arr_colors, f_format='png', X_PCs=None, D_viz_emb=None):
    """
    Create a figure with 2 x 3 subplots, depicting the results of 6 embedding methods applied on a data set. 
    The embedding methods that are considered are PCA, MDS, Laplacian eigenmaps, PHATE, t-SNE and UMAP. 
    It is assumed that 2-D embeddings of the data were previously computed and saved.
    In:
    - data_name: name of the currently considered data set in prints, as specified in the paths.py file. 
    - emb_path: path where the embeddings of the data are stored. 
    - fig_path: path where to save the figure. 
    - arr_colors: an array or list with one entry per example, specifying the color to use to plot it in 2-D. 
    - f_format: format of the figure to be saved. 
    - X_PCs: a 2-D numpy array with one example per row and two columns, the first (resp. second) one storing the first (resp. second) principal component of the data set. Can be None except if Tasic et al or Kanton et al data are considered; in this case, X_PCs will be deepcopied and hence not modified. 
    - D_viz_emb: same as in viz_2d_emb.
    Out: a figure is produced and saved. 
    """
    print('===')
    print("=== Creating the 2x3 figure for {v} data".format(v=data_name))
    print("===")
    
    fig = plt.figure(figsize=(7, 3.5))
    gs = GridSpec(nrows=2, ncols=3, wspace=0.0, figure=fig)
    
    ##############################
    ############################## 
    # PCA
    ####################
    
    if (data_name == paths.tasic_name) or (data_name == paths.kanton_name):
        X_pca = copy.deepcopy(X_PCs)
    else:
        X_pca = np.load('{rp}{npath}.npy'.format(rp=emb_path, npath=paths.pca_path))
    
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
    
    viz_2d_emb(X=X_pca, vcol=arr_colors, tit=paths.pca_name, ax_def=gs[0,0], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, pca_tasic=data_name == paths.tasic_name, pca_genomes=data_name == paths.genomes_name, D_viz_emb=D_viz_emb)
    
    ##############################
    ############################## 
    # MDS
    ####################
    
    X_mds = np.load('{rp}{npath}.npy'.format(rp=emb_path, npath=paths.mds_sklearn_path if data_name == paths.genomes_name else paths.mds_path))
    
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
    
    viz_2d_emb(X=X_mds, vcol=arr_colors, tit=paths.mds_name, ax_def=gs[1,0], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name)
    
    ##############################
    ############################## 
    # Laplacian eigenmaps (LE)
    ####################
    
    X_LE = np.load('{rp}{npath}.npy'.format(rp=emb_path, npath=paths.LE_path))
    
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
    
    viz_2d_emb(X=X_LE, vcol=arr_colors, tit=paths.LE_name_no_param, ax_def=gs[0,1], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, LE_tasic=data_name == paths.tasic_name, LE_genomes=data_name == paths.genomes_name, LE_kanton=data_name == paths.kanton_name, D_viz_emb=D_viz_emb)
    
    ##############################
    ############################## 
    # PHATE
    ####################
    
    X_phate = np.load('{rp}{npath}.npy'.format(rp=emb_path, npath=paths.phate_path))
    
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
    
    viz_2d_emb(X=X_phate, vcol=arr_colors, tit=paths.phate_name_no_param, ax_def=gs[1,1], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, phate_kanton=data_name == paths.kanton_name, D_viz_emb=D_viz_emb)
    
    ##############################
    ############################## 
    # t-SNE
    ####################
    
    X_tsne = np.load('{rp}{npath}.npy'.format(rp=emb_path, npath=paths.tsne_path))
    
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
    
    viz_2d_emb(X=X_tsne, vcol=arr_colors, tit=r'\textit{t}-SNE', ax_def=gs[0,2], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, tsne_tasic=data_name == paths.tasic_name, D_viz_emb=D_viz_emb)
    
    ##############################
    ############################## 
    # UMAP
    ####################
    
    X_umap = np.load('{rp}{npath}.npy'.format(rp=emb_path, npath=paths.umap_path))
    
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
    
    viz_2d_emb(X=X_umap, vcol=arr_colors, tit=paths.umap_name_no_param, ax_def=gs[1,2], flipx=flipx, flipy=flipy, genomes=data_name == paths.genomes_name, umap_tasic=data_name == paths.tasic_name, umap_genomes=data_name == paths.genomes_name, D_viz_emb=D_viz_emb)
    
    ##############################
    ##############################
    # Creating figure
    ####################
    
    save_show_fig(fname=fig_path, f_format=f_format)
    plt.close()
