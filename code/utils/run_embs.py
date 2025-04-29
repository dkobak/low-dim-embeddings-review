#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# This file contains functions to run LD embeddings on a given data set. Scores evaluating the quality of these LD embeddings are also computed. 

########################################################################################################
########################################################################################################

import numpy as np, numba, sklearn.decomposition, scipy.spatial.distance, time, os, sklearn.manifold, umap, copy, phate, scipy.stats, utils.SQuaD_MDS as SQuaD_MDS, utils.plot_fcts as plot_fcts, utils.dr_quality as dr_quality, paths, params

# Name of this file
module_name = "run_embs.py"

##############################
##############################

@numba.jit(nopython=True)
def contains_duplicates(X):
    """
    Returns True if the data set contains two identical samples, False otherwise.
    In:
    - X: a 2-D numpy array with one example per row and one feature per column.
    Out:
    A boolean being True if and only if X contains two identical rows.
    """
    # Number of samples and of features
    N, M = X.shape
    # Tolerance
    atol = 10.0**(-8.0)
    # For each sample
    for i in range(N):
        if np.any(np.absolute(np.dot((np.absolute(X[i,:]-X[i+1:,:]) > atol).astype(np.float64), np.ones(shape=M, dtype=np.float64)))<=atol):
            return True
    return False

def eucl_dist_matr(X):
    """
    Compute pairwise Euclidean distances in a data set. 
    In:
    - X: a 2-D np.ndarray with shape (N,M) containing one example per row and one feature per column.
    Out:
    A 2-D np.ndarray dm with shape (N,N) containing the pairwise Euclidean distances between the data points in X, such that dm[i,j] stores the Euclidean distance between X[i,:] and X[j,:].
    """
    return scipy.spatial.distance.squareform(X=scipy.spatial.distance.pdist(X=X, metric='euclidean'), force='tomatrix')

def preserved_variance_PCs(X, nPCs=50):
    """
    Computes PCA of data set X and show ratios of preserved variance of the first nPCs principal components (PCs). Their sum is also displayed, as well as the sum of preserved variance ratios of the first 2 PCs. 
    In:
    - X: a 2-D numpy array with one example per row and one feature per column.
    - nPCs: number of principal components to consider. Must be a strictly positive integer. 
    Out: /
    """
    print("Computing {v} to display preserved variance ratios of the first {nPCs} PCs.".format(nPCs=nPCs, v=paths.pca_name))
    t0 = time.time()
    pca_model = sklearn.decomposition.PCA(n_components=nPCs, copy=True, whiten=False, svd_solver='arpack', tol=0.0, iterated_power='auto', n_oversamples=10, power_iteration_normalizer='auto', random_state=42).fit(X)
    tf = time.time() - t0
    print('- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
    for i in range(nPCs):
        str_expl_var_ratio = plot_fcts.rstr(pca_model.explained_variance_ratio_[i], d=4)
        print('---> Explained variance ratio PC{i}: {v}'.format(i=i+1, v=str_expl_var_ratio))
    print('- Sum explained variance ratios first {nPCs} PCs: {v}'.format(nPCs=nPCs, v=plot_fcts.rstr(pca_model.explained_variance_ratio_.sum(), d=4)))
    print('- Sum explained variance ratios first 2 PCs: {v}'.format(v=plot_fcts.rstr(pca_model.explained_variance_ratio_[:2].sum(), d=4)))
    print('===')

def apply_meth(X_hd, meth_name, meth_name4path, pca_preproc, compute_dist_HD, compute_dist_LD_qa, seed, res_path_emb, res_path_qa, dim_LDS=2, perp_tsne=None, nn_umap=None, nn_phate=None, nn_LE=None, nn_hd=None, dm_hd=None):
    """
    Apply an embedding method on a data set and evaluates its quality. 
    In:
    - X_hd: a 2-D np.ndarray with shape (N,M) containing one example per row and one feature per column. It stores the data set to be embedded. 
    - meth_name: embedding method name. 
    - meth_name4path: embedding method name to use in paths. It also defines the embedding method that is employed. It is assumed to be equal either to 'pca', 'mds', 'mds_sklearn' or to begin with 'tsne', 'umap', 'phate' or 'LE', otherwise an error is raised, indicating that the method to employ is unknown. 
    - pca_preproc: boolean. Set to True if X_hd consists in principale components; in this case, it is assumed that X_hd[:,0] contains the first PC, X_hd[:,1] contains the second one, etc. 
    - compute_dist_HD: a function such as eucl_dist_matr in this file, enabling to compute pairwise distances in X_hd. Only used if dm_hd is None. 
    - compute_dist_LD_qa: similar to compute_dist_HD, but to compute pairwise distances in the embedding. These distances are only used for the quality assessment of the embedding, not for its computation. 
    - seed: random seed. 
    - res_path_emb: the embedding will be saved in '{rp}{npath}.npy'.format(rp=res_path_emb, npath=meth_name4path).
    - res_path_qa: path where to save the quality scores. 
    - dim_LDS: dimension of the embedding. 
    - perp_tsne: t-SNE perplexity. 
    - nn_umap: number of neighbors in UMAP.
    - nn_phate: number of neighbors in PHATE. 
    - nn_LE: number of neighbors in Laplacian Eigenmaps. 
    - nn_hd: same as in dr_quality.eval_knn_recall. If several embeddings of the same data set are computed, this parameter enables avoiding to compute the HD neighbors several times. 
    - dm_hd: a 2-D np.ndarray with shape (N,N) containing the pairwise distances in X_hd. If None, it is computed using compute_dist_HD. 
    Out: 
    - X_ld: a 2-D np.ndarray with shape (N, dim_LDS) containing one example per row and one feature per column. It stores embedding of X_hd. 
    - nn_hd: same as in dr_quality.eval_knn_recall. It cannot be None at the output. 
    - dm_hd: same as at the input, but it cannot be None at the output. Returning dm_hd avoids computing it several times if multiple embeddings of the same data set are computed. 
    """
    global module_name
    
    path_emb = '{rp}{npath}.npy'.format(rp=res_path_emb, npath=meth_name4path)
    plot_fcts.check_create_dir(path_emb)
    
    path_auc = '{rp}{npath}-{a}.npy'.format(rp=res_path_qa, npath=meth_name4path, a=paths.auc_path)
    path_Knn_recall = '{rp}{npath}-{k}.npy'.format(rp=res_path_qa, npath=meth_name4path, k=paths.knn_recall_path)
    path_sigmad = '{rp}{npath}-{s}.npy'.format(rp=res_path_qa, npath=meth_name4path, s=paths.sigma_d_path)
    path_pearsonr = '{rp}{npath}-{p}.npy'.format(rp=res_path_qa, npath=meth_name4path, p=paths.pearson_corr_path)
    plot_fcts.check_create_dir(path_auc)
    
    if (meth_name4path == paths.pca_path) and pca_preproc:
        # X_hd has been preprocessed into its principal components => just extract the first PCs. 
        X_ld = copy.deepcopy(X_hd[:,:dim_LDS])
    elif os.path.exists(path_emb):
        X_ld = np.load(path_emb)
    else:
        print('Applying {n} on the data set to obtain a {dim_LDS}-D embedding'.format(dim_LDS=dim_LDS, n=meth_name))
        
        t0 = time.time()
        if meth_name4path == paths.pca_path:
            X_ld = sklearn.decomposition.PCA(n_components=dim_LDS, copy=True, whiten=False, svd_solver='full', random_state=seed).fit_transform(X_hd)
        elif meth_name4path == paths.mds_path:
            X_ld = SQuaD_MDS.run_SQuaD_MDS(X_hd, {'in python':True})
        elif (len(meth_name4path) > 4) and (meth_name4path[:4] == paths.tsne_path_no_param):
            # The same 'early_exaggeration' is used as in 'van der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. The journal of machine learning research, 15(1), 3221-3245.'. 
            X_ld = sklearn.manifold.TSNE(n_components=dim_LDS, perplexity=perp_tsne, early_exaggeration=12.0, metric='euclidean', init='pca', random_state=seed, method='barnes_hut', angle=0.5).fit_transform(X_hd)
        elif (len(meth_name4path) > 4) and (meth_name4path[:4] == paths.umap_path_no_param):
            X_ld = umap.UMAP(n_neighbors=nn_umap, n_components=dim_LDS, metric='euclidean', output_metric='euclidean', min_dist=0.1, random_state=seed, init='spectral').fit_transform(X_hd)
        elif (len(meth_name4path) > 5) and (meth_name4path[:5] == paths.phate_path_no_param):
            X_ld = phate.PHATE(n_components=dim_LDS, knn=nn_phate, decay=40, n_landmark=2000, t='auto', gamma=1, n_pca=100, mds_solver='sgd', knn_dist='euclidean', knn_max=None, mds_dist='euclidean', mds='metric', n_jobs=1, random_state=seed).fit_transform(X_hd)
        elif (len(meth_name4path) > 2) and (meth_name4path[:2] == paths.LE_path_no_param):
            X_ld = sklearn.manifold.SpectralEmbedding(n_components=dim_LDS, affinity='nearest_neighbors', random_state=seed, n_neighbors=nn_LE).fit_transform(X_hd)
        elif meth_name4path == paths.mds_sklearn_path:
            # Using PCA init
            if pca_preproc:
                X_init = copy.deepcopy(X_hd[:,:dim_LDS])
            else:
                X_init = sklearn.decomposition.PCA(n_components=dim_LDS, copy=True, whiten=False, svd_solver='arpack', random_state=seed).fit_transform(X_hd)
            
            mds_model = sklearn.manifold.MDS(n_components=dim_LDS, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=seed, dissimilarity='euclidean', normalized_stress='auto')
            X_ld = mds_model.fit_transform(X_hd, init=X_init) 
        else:
            raise ValueError('In apply_meth of {module_name}: unknown method "{npath}"'.format(module_name=module_name, npath=meth_name4path))
        t = time.time() - t0
        print('- Done. It took {t} seconds.'.format(t=plot_fcts.rstr(t)))
        
        np.save(path_emb, X_ld)
    
    if not (os.path.exists(path_auc) and os.path.exists(path_sigmad) and os.path.exists(path_pearsonr)): 
        if dm_hd is None:
            print('- Computing HD distances')
            t0 = time.time()
            dm_hd = compute_dist_HD(X_hd)
            tf = time.time() - t0
            print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
        
        print('- Computing LD distances in {n} {dim_LDS}-D embedding'.format(n=meth_name, dim_LDS=dim_LDS))
        t0 = time.time()
        dld = compute_dist_LD_qa(X_ld)
        tf = time.time() - t0
        print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
    
    if os.path.exists(path_auc):
        auc = np.load(path_auc)
    else:
        print('- Evaluating the {a} of the LD embedding obtained by {n}'.format(n=meth_name, a=paths.auc_name))
        t0 = time.time()
        rnx, auc = dr_quality.eval_dr_quality(d_hd=dm_hd, d_ld=dld)
        tf = time.time() - t0
        print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
        
        np.save(path_auc, auc)
    
    if os.path.exists(path_sigmad):
        sigma_d = np.load(path_sigmad)
    else:
        print('- Evaluating the {s} of the LD embedding obtained by {n}'.format(n=meth_name, s=paths.sigma_d_name))
        t0 = time.time()
        sigma_d = dr_quality.eval_sigma_distortion(d_hd=dr_quality.make_dist_vector(dm_hd), d_ld=dr_quality.make_dist_vector(dld))
        tf = time.time() - t0
        print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
        
        np.save(path_sigmad, sigma_d)
    
    if os.path.exists(path_pearsonr):
        pr = np.load(path_pearsonr)
    else:
        print('- Evaluating {p} in the LD embedding obtained by {n}'.format(n=meth_name, p=paths.pearson_corr_name_long))
        t0 = time.time()
        
        dv_hd = dr_quality.make_dist_vector(dm_hd)
        dv_ld = dr_quality.make_dist_vector(dld)
        
        pr = scipy.stats.pearsonr(dv_hd, dv_ld).statistic
        
        tf = time.time() - t0
        print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
        
        np.save(path_pearsonr, pr)
    
    if os.path.exists(path_Knn_recall):
        knn_recall = np.load(path_Knn_recall)
    else:
        knn_recall, nn_hd = dr_quality.eval_knn_recall(X_hd=X_hd, X_ld=X_ld, nn_hd=nn_hd)
        np.save(path_Knn_recall, knn_recall)
    
    D_qa = dict()
    D_qa[paths.auc_name] = auc
    D_qa[paths.sigma_d_name] = sigma_d
    D_qa[paths.knn_recall_name] = knn_recall
    D_qa[paths.pearson_corr_name] = pr
    
    print("**********")
    print('Results of the quality assessment of the {dim_LDS}-D embedding obtained by {n}'.format(n=meth_name, dim_LDS=dim_LDS))
    print('---')
    
    for score in D_qa.keys():
        if np.isfinite(D_qa[score]):
            str_score = plot_fcts.rstr(D_qa[score], d=4)
        else:
            str_score = D_qa[score]
        print("{n} [ {score} ] = {v}".format(n=meth_name, score=score, v=str_score))
    
    print('---')
    
    return X_ld, nn_hd, dm_hd

def compute_embs_and_quality(X_hd, pca_preproc, data_name, res_path_emb, res_path_qa, check_duplicates=False, compute_pca_preserved_var=False, X_hd_nopca=None, genomes=False):
    """
    Apply PCA, MDS, Laplacian Eigenmaps, t-SNE, UMAP and PHATE on a data set. Quality scores of the obtained embeddings are also computed. 
    In:
    - X_hd, pca_preproc, res_path_emb, res_path_qa: same as in apply_meth. 
    - data_name: a string storing the name of the data to be embedded. 
    - check_duplicates: boolean. If True, X_hd is checked for duplicated examples. 
    - compute_pca_preserved_var: boolean. If True, the function preserved_variance_PCs (in this file) is applied on X_hd_nopca. 
    - X_hd_nopca: a version of X_hd before PCA preprocessing, if any. Only used if compute_pca_preserved_var is True, to compute the ratio of preserved variance by the first few principal components; can be None otherwise. 
    - genomes: boolean. Set to True if 1000 Genomes data are employed. In this case, MDS implementation from scikit-learn is employed because this data set contains much less examples. It is then not necessary to use a fast implementation such as SQuadMDS. 
    Out: /
    """
    print('===')
    print("=== Processing {v} data".format(v=data_name))
    print("===")
    
    n_samples = X_hd.shape[0]
    n_features = X_hd.shape[1]
    
    print('Number of samples:  ', n_samples)
    print('Number of features: ', n_features)
    
    ###
    ###
    ###
    if check_duplicates:
        print('- Checking if there are duplicated examples in the data set')
        if contains_duplicates(X_hd):
            print(' *** !!! Warning !!! *** The data set contains duplicated examples...')
        else:
            print('There are no duplicates.')
        print('===')
        print('===')
        print('===')
    
    ###
    ###
    ###
    # Function to compute distances in HD space. 
    compute_dist_HD = eucl_dist_matr
    # Function to compute distances in LD space (only used for quality assessment of the LD embeddings). 
    compute_dist_LD_qa = eucl_dist_matr
    # Targeted dimension of the LD embedding
    dim_LDS = params.dim_LDS
    # Random seed. 
    seed = 40
    
    ##############################
    ############################## 
    # Applying PCA
    ####################
    X_ld_pca, nn_hd, dm_hd = apply_meth(X_hd=X_hd, meth_name=paths.pca_name, meth_name4path=paths.pca_path, pca_preproc=pca_preproc, compute_dist_HD=compute_dist_HD, compute_dist_LD_qa=compute_dist_LD_qa, seed=seed, res_path_emb=res_path_emb, res_path_qa=res_path_qa, dim_LDS=dim_LDS)
    
    if compute_pca_preserved_var:
        preserved_variance_PCs(X=X_hd_nopca)
    
    ##############################
    ##############################
    # Applying MDS
    ####################
    apply_meth(X_hd=X_hd, meth_name=paths.mds_name, meth_name4path=paths.mds_sklearn_path if genomes else paths.mds_path, pca_preproc=pca_preproc, compute_dist_HD=compute_dist_HD, compute_dist_LD_qa=compute_dist_LD_qa, seed=seed, res_path_emb=res_path_emb, res_path_qa=res_path_qa, dim_LDS=dim_LDS, nn_hd=nn_hd, dm_hd=dm_hd)
    
    ##############################
    ############################## 
    # Applying Laplacian eigenmaps (LE)
    ####################
    apply_meth(X_hd=X_hd, meth_name=paths.LE_name, meth_name4path=paths.LE_path, pca_preproc=pca_preproc, compute_dist_HD=compute_dist_HD, compute_dist_LD_qa=compute_dist_LD_qa, seed=seed, res_path_emb=res_path_emb, res_path_qa=res_path_qa, dim_LDS=dim_LDS, nn_LE=params.nn_LE, nn_hd=nn_hd, dm_hd=dm_hd)
    
    ##############################
    ############################## 
    # Applying t-SNE
    ####################
    apply_meth(X_hd=X_hd, meth_name=paths.tsne_name, meth_name4path=paths.tsne_path, pca_preproc=pca_preproc, compute_dist_HD=compute_dist_HD, compute_dist_LD_qa=compute_dist_LD_qa, seed=seed, res_path_emb=res_path_emb, res_path_qa=res_path_qa, dim_LDS=dim_LDS, perp_tsne=params.perp_tsne, nn_hd=nn_hd, dm_hd=dm_hd)
    
    ##############################
    ############################## 
    # Applying UMAP
    ####################
    apply_meth(X_hd=X_hd, meth_name=paths.umap_name, meth_name4path=paths.umap_path, pca_preproc=pca_preproc, compute_dist_HD=compute_dist_HD, compute_dist_LD_qa=compute_dist_LD_qa, seed=seed, res_path_emb=res_path_emb, res_path_qa=res_path_qa, dim_LDS=dim_LDS, nn_umap=params.nn_umap, nn_hd=nn_hd, dm_hd=dm_hd)
    
    ##############################
    ############################## 
    # Applying PHATE
    ####################
    apply_meth(X_hd=X_hd, meth_name=paths.phate_name, meth_name4path=paths.phate_path, pca_preproc=pca_preproc, compute_dist_HD=compute_dist_HD, compute_dist_LD_qa=compute_dist_LD_qa, seed=seed, res_path_emb=res_path_emb, res_path_qa=res_path_qa, dim_LDS=dim_LDS, nn_phate=params.nn_phate, nn_hd=nn_hd, dm_hd=dm_hd)
    

