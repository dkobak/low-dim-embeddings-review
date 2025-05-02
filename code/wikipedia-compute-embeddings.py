#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# Assessing DR quality and distortions of LD embeddings of Simple English Wikipedia data.

########################################################################################################
########################################################################################################

import numpy as np, time, os, scipy.stats, datasets, utils.plot_fcts as plot_fcts, utils.dr_quality as dr_quality, paths, params, utils.run_embs as run_embs

# Name of this file
module_name = "wikipedia_quality.py"

##############################
############################## 
# Parameters
####################

# Set to True to compute the proportion of preserved variance by the first 50 PCs
compute_pca_preserved_var = False

# Number of landmarks
n_landmarks = 1000

# Number of runs with different sets of landmarks. Results can then be averaged over the runs and standard deviations can be estimated. 
n_runs = 1

##############################
############################## 
# Loading Simple English Wikipedia data and computing quality scores
####################

print('===')
print("Starting quality assessment of {v} embeddings ({n_landmarks} landmarks)".format(n_landmarks=n_landmarks, v=paths.wiki_name))
print("===")

# Checking whether the folder where to store the results exists
plot_fcts.check_create_dir(paths.wiki_qa)

print('Loading HD data')
wiki_docs = datasets.load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
X_hd = np.asarray(wiki_docs['emb'])
print('- Number of samples: ', X_hd.shape[0]) # 485,859
print('- Number of features:', X_hd.shape[1]) # 768
print("===")

if compute_pca_preserved_var:
    run_embs.preserved_variance_PCs(X=X_hd)

print('Loading the LD embeddings')
L_X_LD = list()
L_names = [(paths.pca_name, paths.pca_path), (paths.mds_name, paths.mds_path), (paths.LE_name, paths.LE_path), (paths.phate_name, paths.phate_path), (paths.tsne_name, paths.tsne_path), (paths.umap_name, paths.umap_path)]
for name, npath in L_names:
    L_X_LD.append(run_embs.apply_meth(X_hd=X_hd, meth_name=name, meth_name4path=npath, pca_preproc=False, compute_dist_HD=None, compute_dist_LD_qa=None, seed=params.seed, res_path_emb=paths.wiki_emb, res_path_qa=None, dim_LDS=params.dim_LDS, perp_tsne=params.perp_tsne, nn_umap=params.nn_umap, nn_phate=params.nn_phate, nn_LE=params.nn_LE, skip_qa=True))
print("===")
n_embs = len(L_X_LD)

# Dictionary with one entry per type of quality score. Each entry contains a list with as many elements as LD embeddings in L_X_LD. Each element of each list is a np.array with n_runs elements. 
D_L_score_meths = {}
D_L_score_meths[paths.auc_name] = [np.empty(shape=n_runs, dtype=np.float64) for i in range(n_embs)]
D_L_score_meths[paths.sigma_d_name] = [np.empty(shape=n_runs, dtype=np.float64) for i in range(n_embs)]
D_L_score_meths[paths.pearson_corr_name] = [np.empty(shape=n_runs, dtype=np.float64) for i in range(n_embs)]

# List with the K-NN recall for each embedding
L_Knn_recalls = np.empty(shape=n_embs, dtype=np.float64)

# Path where to save the quality scores computed using landmarks
res_path_landmarks = '{rp}{n_landmarks}-landmarks/'.format(rp=paths.wiki_qa, n_landmarks=n_landmarks)
plot_fcts.check_create_dir(res_path_landmarks)
# Path where to save K-NN recalls
res_path_Knn_recall = '{rp}{k}/'.format(rp=paths.wiki_qa, k=paths.knn_recall_path)
plot_fcts.check_create_dir(res_path_Knn_recall)

# Looping over the runs
for i_run in range(n_runs):
    
    # Path where to save the sampled landmarks
    res_path_landmarks_ids = '{res_path_landmarks}run-{i_run}_landmarks.npy'.format(res_path_landmarks=res_path_landmarks, i_run=i_run+1)
    
    if os.path.exists(res_path_landmarks_ids):
        landmarks = np.load(res_path_landmarks_ids)
        landmarks_isNone = False
    else:
        landmarks = None
        landmarks_isNone = True
    
    if i_run == 0:
        nn_hd = None
    
    # For each embedding
    for i_meth in range(n_embs):
        
        print('Starting {n} embedding (run {i}/{n_runs})'.format(n=L_names[i_meth][0], i=i_run+1, n_runs=n_runs))
        
        res_path_meth = '{res_path_landmarks}run-{i_run}_{meth}'.format(res_path_landmarks=res_path_landmarks, i_run=i_run+1, meth=L_names[i_meth][1])
        res_path_meth_Knn_recall = '{res_path_Knn_recall}{meth}.npy'.format(res_path_Knn_recall=res_path_Knn_recall, meth=L_names[i_meth][1])
        res_path_meth_auc = '{res_path_meth}-{a}.npy'.format(res_path_meth=res_path_meth, a=paths.auc_path)
        res_path_meth_sigmad = '{res_path_meth}-{s}.npy'.format(res_path_meth=res_path_meth, s=paths.sigma_d_path)
        res_path_meth_pearsonr = '{res_path_meth}-{p}.npy'.format(res_path_meth=res_path_meth, p=paths.pearson_corr_path)
        
        if os.path.exists(res_path_meth_auc):
            auc = np.load(res_path_meth_auc)
            if landmarks_isNone:
                raise ValueError('In {module_name}: landmarks cannot be None in this case.'.format(module_name=module_name))
        else:
            print('- Estimating {v}'.format(v=paths.auc_name))
            
            t0 = time.time()
            qnx, rnx, auc, landmarks = dr_quality.fast_eval_dr_quality(X_hd=X_hd, X_ld=L_X_LD[i_meth], dist_hd=dr_quality.cos_dist, dist_ld=dr_quality.eucl_dist, n=n_landmarks, seed=3+i_run, pow2K=False, vp_samp=False, samp=landmarks)
            tf = time.time() - t0
            print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
            
            np.save(res_path_meth_auc, auc)
            if landmarks_isNone:
                np.save(res_path_landmarks_ids, landmarks)
                landmarks_isNone = False
        
        if os.path.exists(res_path_meth_sigmad) and os.path.exists(res_path_meth_pearsonr):
            sigma_d = np.load(res_path_meth_sigmad)
            pr = np.load(res_path_meth_pearsonr)
        else:
            
            print('- Computing HD distances')
            t0 = time.time()
            dhds = dr_quality.dist_matr(X=X_hd, samp=landmarks, dist_fct=dr_quality.cos_dist)
            tf = time.time() - t0
            print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
            
            print('- Computing LD distances')
            t0 = time.time()
            dlds = dr_quality.dist_matr(X=L_X_LD[i_meth], samp=landmarks, dist_fct=dr_quality.eucl_dist)
            tf = time.time() - t0
            print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
            
            print('- Computing {v}'.format(v=paths.sigma_d_name))
            t0 = time.time()
            sigma_d = dr_quality.eval_sigma_distortion(d_hd=dhds, d_ld=dlds)
            tf = time.time() - t0
            print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
            
            np.save(res_path_meth_sigmad, sigma_d)
            
            print('- Computing {v}'.format(v=paths.pearson_corr_name_long))
            t0 = time.time()
            pr = scipy.stats.pearsonr(dhds.flatten(), dlds.flatten()).statistic
            tf = time.time() - t0
            print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
            
            np.save(res_path_meth_pearsonr, pr)
        
        if i_run == 0:
            if os.path.exists(res_path_meth_Knn_recall):
                knn_recall = np.load(res_path_meth_Knn_recall)
            else:
                knn_recall, nn_hd = dr_quality.eval_knn_recall(X_hd=X_hd, X_ld=L_X_LD[i_meth], nn_hd=nn_hd, metric_hd='cosine')
                np.save(res_path_meth_Knn_recall, knn_recall)
            
            L_Knn_recalls[i_meth] = knn_recall
        
        D_L_score_meths[paths.auc_name][i_meth][i_run] = auc
        D_L_score_meths[paths.sigma_d_name][i_meth][i_run] = sigma_d
        D_L_score_meths[paths.pearson_corr_name][i_meth][i_run] = pr
        
        print('===')

print("**********")
print("Results of the quality assessment (n_runs={v}):".format(v=n_runs))
print('---')

for i_meth in range(n_embs):
    
    str_score = plot_fcts.rstr(L_Knn_recalls[i_meth], d=3) if np.isfinite(L_Knn_recalls[i_meth]) else L_Knn_recalls[i_meth]
    print("- {name} [ {k} ] = {v}".format(name=L_names[i_meth][0], v=str_score, k=paths.knn_recall_name))
    
    for score in D_L_score_meths.keys():
        D_qa = dict()
        D_qa['mean [{score}]'.format(score=score)] = D_L_score_meths[score][i_meth].mean()
        D_qa['std dev [{score}]'.format(score=score)] = D_L_score_meths[score][i_meth].std()
        
        for stat in ['mean', 'std dev']:
            cur_val = D_qa['{stat} [{score}]'.format(score=score, stat=stat)]
            str_score = plot_fcts.rstr(cur_val, d=3) if np.isfinite(cur_val) else cur_val
            print("- {name} - {stat} [{score}] = {v}".format(name=L_names[i_meth][0], stat=stat, score=score, v=str_score))
    
    print('---')

###
###
###
print('*********************')
print('***** Done! :-) *****')
print('*********************')
