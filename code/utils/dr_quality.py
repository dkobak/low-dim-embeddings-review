#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

# This file gathers functions implementing scores to assess distortions and quality of low-dimensional embeddings. 

########################################################################################################
########################################################################################################

import numpy as np, numba, scipy.spatial.distance, sklearn.neighbors, time, utils.plot_fcts as plot_fcts, params, paths

# Name of this file
module_name = "dr_quality.py"

##############################
############################## 
# Sigma-distortion, from 'Chennuru Vankadara, L., & von Luxburg, U. (2018). Measures of distortion for machine learning. Advances in Neural Information Processing Systems, 31.'.
####################

def make_dist_vector(d):
    """
    If d is 2-D, returns scipy.spatial.distance.squareform(X=d, force='tovector').
    If d is 1-D, returns d.
    Otherwise, an error is raised. 
    """
    global module_name
    s = len(d.shape)
    if s == 2:
        return scipy.spatial.distance.squareform(X=d, force='tovector')
    elif s == 1:
        return d
    else:
        raise ValueError('In make_dist_vector of {module_name}: d must have 1 or 2 dimensions.'.format(module_name=module_name))

def eval_sigma_distortion(d_hd, d_ld):
    """
    Computes sigma-distortion.
    In:
    - d_hd: 1-D or 2-D np.ndarray with pairwise HD distances to consider to estimate the sigma-distortion. It cannot contain zeros. 
    - d_ld: 1-D or 2-D np.ndarray with pairwise LD distances to consider to estimate the sigma-distortion.
    Out:
    - sigma-distortion
    """
    
    rho = d_ld/d_hd
    
    rho_mean = rho.mean()
    rho_tilde = rho/rho_mean
    
    sigma_d = ((rho_tilde-1)**2).mean()
    
    return sigma_d

##############################
############################## 
# K-NN recall. 
####################

def eval_knn_recall(X_hd, X_ld, nn_hd=None, metric_hd='minkowski', p_metric_hd=2, n_jobs=5):
    """
    Compute KNN recall of an embedding with respect to a data set. Definition from 'Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. Nature communications, 10(1), 5416.': "The fraction of k-nearest neighbours in the original high-dimensional data that are preserved as k-nearest neighbours in the embedding. KNN quantifies preservation of the local, or microscopic structure."
    As in 'Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. Nature communications, 10(1), 5416.', we compute the average across all points. 
    The K value that is used is K_qa, which is a global variable defined at the beginning of this file. 
    In:
    - X_hd: a 2-D np.ndarray with shape (N,M) containing one example per row and one feature per column.
    - X_ld: a 2-D np.ndarray with shape (N,P) storing an embedding of X_hd. X_ld[i,:] is the embedded representation of X_hd[i,:].
    - metric_hd: metric parameter of sklearn.neighbors.NearestNeighbors.
    - p_metric_hd: p parameter of sklearn.neighbors.NearestNeighbors.
    - n_jobs: n_jobs parameter of sklearn.neighbors.NearestNeighbors.
    - nn_hd: if None, set to the output of sklearn.neighbors.NearestNeighbors(n_neighbors=K_qa + 1, algorithm='auto', metric=metric_hd, p=p_metric_hd).fit(X_hd).kneighbors_graph(X_hd), with K_qa defined as a global variable of this file. Otherwise, it is assumed to be the ooutput of this call. 
    Out:
    - knn_recall: a number between 0 and 1 being the K-NN recall. 
    - nn_hd: as defined above, but it cannot be None at the output. 
    """
    if nn_hd is None:
        print('- Computing the {K_qa}-NN graph in HD'.format(K_qa=params.K_qa))
        t0 = time.time()
        nn_hd = sklearn.neighbors.NearestNeighbors(n_neighbors=params.K_qa + 1, algorithm='auto', metric=metric_hd, p=p_metric_hd, n_jobs=n_jobs).fit(X_hd).kneighbors_graph(X_hd)
        tf = time.time() - t0
        print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
    
    print('- Computing the {K_qa}-NN graph in LD'.format(K_qa=params.K_qa))
    t0 = time.time()
    nn_ld = sklearn.neighbors.NearestNeighbors(n_neighbors=params.K_qa + 1, algorithm='auto', n_jobs=n_jobs).fit(X_ld).kneighbors_graph(X_ld)
    tf = time.time() - t0
    print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
    
    print('- Computing the {k}'.format(k=paths.knn_recall_name))
    t0 = time.time()
    knn_recall = (nn_hd.multiply(nn_ld).sum() / nn_hd.shape[0] - 1) / params.K_qa
    tf = time.time() - t0
    print('-- Done. It took {tf} seconds.'.format(tf=plot_fcts.rstr(tf)))
    
    return knn_recall, nn_hd

##############################
############################## 
# Unsupervised DR quality assessment: rank-based criteria measuring the HD neighborhood preservation in the LD embedding. 
# The main function is 'eval_dr_quality'. 
# The computation of the criteria is exact and has O(N**2 log(N)) time complexity, where N is the number of data points. 
# References:
# [1] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
# [2] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
####################

def coranking(d_hd, d_ld):
    """
    Computation of the co-ranking matrix, as defined in 'Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.'. 
    The time complexity of this function is O(N**2 log(N)), where N is the number of data points.
    In:
    - d_hd: 2-D numpy array representing the redundant matrix of pairwise distances in the HD space (HDS).
    - d_ld: 2-D numpy array representing the redundant matrix of pairwise distances in the LD space (LDS).
    Out:
    The (N-1)x(N-1) co-ranking matrix, where N = d_hd.shape[0].
    """
    # Computing the permutations to sort the rows of the distance matrices in HDS and LDS. 
    perm_hd = d_hd.argsort(axis=-1, kind='mergesort')
    perm_ld = d_ld.argsort(axis=-1, kind='mergesort')
    
    N = d_hd.shape[0]
    i = np.arange(N, dtype=np.int64)
    # Computing the ranks in the LDS
    R = np.empty(shape=(N,N), dtype=np.int64)
    for j in range(N):
        R[perm_ld[j,i],j] = i
    # Computing the co-ranking matrix
    Q = np.zeros(shape=(N,N), dtype=np.int64)
    for j in range(N):
        Q[i,R[perm_hd[j,i],j]] += 1
    # Returning
    return Q[1:,1:]

@numba.jit(nopython=True)
def eval_auc(arr):
    """
    Evaluates the AUC, as defined in [2].
    In:
    - arr: 1-D numpy array storing the values of a curve from K=1 to arr.size.
    Out:
    The AUC under arr, as defined in [2], with a log scale for K=1 to arr.size. 
    """
    i_all_k = 1.0/(np.arange(arr.size)+1.0)
    return np.float64(np.dot(arr, i_all_k))/(i_all_k.sum())

@numba.jit(nopython=True)
def eval_rnx(qnx):
    """
    Computing R_{NX}(K) based on Q_{NX}(K).
    In:
    - qnx: a 1-D numpy array with N-1 elements, where N is the number of points in the data sets. qnx(i) contains Q_{NX}(i+1).
    Out:
    A 1-D numpy array with N-2 elements, in which the ith element contains R_{NX}(i+1).
    """
    N_1 = qnx.size
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    return (N_1*qnx[:N_1-1]-arr_K)/(N_1-arr_K)

@numba.jit(nopython=True)
def eval_rnx_coranking(Q):
    """
    Evaluate R_NX(K) for K = 1 to N-2, as defined in [2]. N is the number of data points in the data set.
    The time complexity of this function is O(N^2).
    In:
    - Q: a 2-D numpy array representing the (N-1)x(N-1) co-ranking matrix of the embedding. 
    Out:
    A 1-D numpy array with N-2 elements. Element i contains R_NX(i+1).
    """
    N_1 = Q.shape[0]
    N = N_1 + 1
    # Computing Q_NX
    qnxk = np.empty(shape=N_1, dtype=np.float64)
    acc_q = 0.0
    for K in range(N_1):
        acc_q += (Q[K,K] + np.sum(Q[K,:K]) + np.sum(Q[:K,K]))
        qnxk[K] = acc_q/((K+1)*N)
    # Computing R_NX
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    rnxk = (N_1*qnxk[:N_1-1]-arr_K)/(N_1-arr_K)
    # Returning
    return rnxk

def eval_dr_quality(d_hd, d_ld):
    """
    Compute the DR quality assessment criteria R_{NX}(K) and AUC, as defined in [2].
    These criteria measure the neighborhood preservation around the data points from the HDS to the LDS. 
    Based on the HD and LD distances, the sets v_i^K (resp. n_i^K) of the K nearest neighbors of data point i in the HDS (resp. LDS) can first be computed. 
    Their average normalized agreement develops as Q_{NX}(K) = (1/N) * sum_{i=1}^{N} |v_i^K cap n_i^K|/K, where N refers to the number of data points and cap to the set intersection operator. 
    Q_{NX}(K) ranges between 0 and 1; the closer to 1, the better.
    As the expectation of Q_{NX}(K) with random LD coordinates is equal to K/(N-1), which is increasing with K, R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K) enables more easily comparing different neighborhood sizes K. 
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1. 
    The R_{NX}(K) values for K=1 to N-2 can be displayed as a curve with a log scale for K, as closer neighbors typically prevail. 
    The area under the resulting curve (AUC) is a scalar score which grows with DR quality, quantified at all scales with an emphasis on small ones.
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random. 
    In: 
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    Out: a tuple with
    - a 1-D numpy array with N-2 elements. Element i contains R_{NX}(i+1).
    - the AUC of the R_{NX}(K) curve with a log scale for K, as defined in [2].
    Remark:
    - The time complexity to evaluate the quality criteria is O(N**2 log(N)). It is the time complexity to compute the co-ranking matrix. R_{NX}(K) can then be evaluated for all K=1, ..., N-2 in O(N**2). 
    """
    # Computing the co-ranking matrix of the embedding, and the R_{NX}(K) curve.
    rnxk = eval_rnx_coranking(Q=coranking(d_hd=d_hd, d_ld=d_ld))
    # Computing the AUC, and returning.
    return rnxk, eval_auc(rnxk)

##############################
############################## 
# Fast computation of the unsupervised DR quality assessment.
# The main function which should be used is 'fast_eval_dr_quality'. 
# The quality curves may be efficiently estimated for neighborhood sizes K either ranging from 1 to N-1, where N is the number of data points, or equal to 2^0, 2^1, 2^2, 2^3, ..., 2^(int(np.log2(N-1))). 
# The fast computation of the quality curves for K = 1 to N-1 yields O(n*N*log(N)) time complexity, where n is the number of representative data points (= landmarks) sampled to approximate Q_{NX}(K). n typically scales as O(np.log(N)), for instance n = 10*np.log(N) by default in the fast_eval_qnx function. 
# If the curves are only estimated for K = 2^0, 2^1, 2^2, 2^3, ..., 2^(int(np.log2(N-1))), N-1, the time complexity drops to O(n*N).
# Reference: Novak, D., de Bodt, C., Lambert, P., Lee, J. A., Van Gassen, S., & Saeys, Y. (2023). Interpretable models for scRNA-seq data embedding with multi-scale structure preservation. bioRxiv, 2023-11.
##############################

@numba.jit(nopython=True)
def eucl_dist(X, x):
    """
    Compute the Euclidean distances between a vector and a bunch of others. 
    In:
    - X: a 2-D numpy array with one example per row and one feature per column. 
    - x: a 1-D numpy array such that x.size = X.shape[1].
    Out:
    A 1-D numpy array in which element i is the Euclidean distance between x and X[i,:].
    """
    M = X-x
    return np.sqrt((M*M).sum(axis=1))

@numba.jit(nopython=True)
def cos_dist(X, x):
    """
    Compute the Cosine distances between a vector and a bunch of others. 
    In:
    - X: a 2-D numpy array with one example per row and one feature per column. 
    - x: a 1-D numpy array such that x.size = X.shape[1].
    Out:
    A 1-D numpy array in which element i is the Cosine distance between x and X[i,:].
    """
    return 1.0 - (np.dot(X, x) / (np.sqrt((X*X).sum(axis=1)) * np.sqrt(np.dot(x, x))))

@numba.jit(nopython=True)
def dist_matr(X, samp, dist_fct):
    """
    Given a distance function, evaluate the distances between a set of samples and all the others in a data set. 
    In:
    - X: a 2-D numpy array with one example per row and one feature per column. 
    - samp: a 1-D numpy array containing integers between 0 and X.shape[0]-1, storing indexes of examples in X.
    - dist_fct: a distance function such as eucl_dist or cos_dist in this file. 
    Out:
    A 2-D numpy array with shape (samp.size, X.shape[0]-1), where entries at indexes [idx,:] consists in the output of dist_fct(np.vstack((X[:samp[idx],:], X[samp[idx]+1:,:])), X[samp[idx],:]).
    """
    res = np.empty(shape=(samp.size, X.shape[0]-1), dtype=np.float64)
    
    for idx, i in enumerate(samp):
        res[idx,:] = dist_fct(np.vstack((X[:i,:], X[i+1:,:])), X[i,:])
    
    return res

@numba.jit(nopython=True)
def randomized_partition(a, ind, p, r, tmp):
    """
    Implements a variant of the RANDOMIZED-PARTITION procedure described in 'Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). Introduction to algorithms. MIT press.', page 179.
    In:
    - a: a 1-D numpy array. 
    - ind: a 1-D numpy array being a permutation of np.arange(start=0, step=1, stop=a.size).
    - p: an integer >=0 and < a.size.
    - r: an integer >=p and < a.size.
    - tmp: a 1-D numpy array of integer with at least r-p+1 entries, which can be modified. The purpose of tmp is to serve as a container to store intermediate computations of this function, to avoid the overhead of allocating memory each time this function is called in recursive functions. tmp will hence be modified in an undefined way.
    Out:
    An integer i is returned and ind is modified such that:
    - for k such that p <= k <= i, a[ind[k]] <= a[ind[i]].
    - for k such that i < k <= r, a[ind[k]] > a[ind[i]].
    a is not modified. tmp is modified. 
    This function is stable, meaning that the relative order of the elements in a with equal values is preserved. 
    """
    r_1 = r + 1
    # Sampling the pivot element
    i = np.random.randint(p, r_1)
    x = a[ind[i]]
    # Increasing i as much as possible while keeping the same value of a[ind[i]]. This ensures the stability of the function. 
    for j in range(r, i, -1):
        if a[ind[j]] == x:
            i = j
            break
    r = ind[i]
    # Current index in tmp
    i_tmp = 0
    # Current index in ind
    i_ind = p
    # For each element in ind[p:r_1], but skipping the i^th one
    for lb, ub in [(p, i), (i+1, r_1)]:
        for j in range(lb, ub, 1):
            if a[ind[j]] > x:
                tmp[i_tmp] = ind[j]
                i_tmp += 1
            else:
                ind[i_ind] = ind[j]
                i_ind += 1
    # Inserting the pivot in ind
    ind[i_ind] = r
    # Transfering from tmp to ind
    ind[i_ind+1:r_1] = tmp[:i_tmp]
    # Returning the index of the pivot in ind
    return i_ind

@numba.jit(nopython=True)
def randomized_partition_unstable(a, ind, p, r):
    """
    Implements a variant of the RANDOMIZED-PARTITION procedure described in 'Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). Introduction to algorithms. MIT press.', page 179.
    In:
    - a: a 1-D numpy array. 
    - ind: a 1-D numpy array being a permutation of np.arange(start=0, step=1, stop=a.size).
    - p: an integer >=0 and < a.size.
    - r: an integer >=p and < a.size.
    Out:
    An integer i is returned and ind is modified such that:
    - for k such that p <= k <= i, a[ind[k]] <= a[ind[i]].
    - for k such that i < k <= r, a[ind[k]] > a[ind[i]].
    a is not modified.
    This function is NOT stable. 
    """
    # Sampling the pivot element
    i = np.random.randint(p, r+1)
    x = a[ind[i]]
    ind[r], ind[i] = ind[i], ind[r]
    i = p
    for j in range(p, r, 1):
        if a[ind[j]] <= x:
            ind[i], ind[j] = ind[j], ind[i]
            i += 1
    ind[r], ind[i] = ind[i], ind[r]
    return i

@numba.jit(nopython=True)
def pow2_sort(a, ind, p, r, tmp):
    """
    Find a permutation of the indexes of an array to partially sort it at the powers of 2. 
    This code is inspired from the RANDOMIZED-SELECT procedure described in 'Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). Introduction to algorithms. MIT press.', page 216.
    In:
    - a, ind, p, r, tmp: as in randomized_partition.
    Nothing is returned but ind is modified such that for u and v such that p <= u < v <= r and such that int(np.log2(u)) != int(np.log2(v)), a[ind[u]] <= a[ind[v]].
    a is not modified. 
    """
    if p != r:
        q = randomized_partition(a, ind, p, r, tmp) - 1
        if (q > p) and ((p == 0) or (int(np.log2(p)) != int(np.log2(q)))):
            pow2_sort(a, ind, p, q, tmp)
        q += 2
        if (q < r) and (int(np.log2(q)) != int(np.log2(r))):
            pow2_sort(a, ind, q, r, tmp)

@numba.jit(nopython=True)
def pow2_rank(d, tmp):
    """
    Determine the ranks of the data points, in terms of powers of 2.
    In:
    - d: a 1-D numpy array. 
    - tmp: a 1-D numpy array of integers with d.size elements, which can be modified. It will serve as a container to store intermediate computation, and will thus be modified in an undefined way. It avoids the overhead of allocating it each time this function is called. 
    Out:
    A 1-D numpy array with as many elements as d and in which the element at index i equals the ranking of d[i] in d in terms of powers of 2, i.e. the smallest element in d has ranking 0, the second smallest has ranking 1, the 3rd and 4th smallest have ranking 2, the 5th to 8th smallest have ranking 3, the 9th to 16th smallest have ranking 4, etc.
    """
    N = d.size
    # ind will contain the index associated with each rank
    ind = np.arange(N)
    pow2_sort(d, ind, 0, N-1, tmp)
    # Computing the rank, in terms of powers of 2, associated with each index
    r = np.empty(shape=N, dtype=np.int64)
    r[ind[0]] = 0
    lb = 1
    ub = 2
    cr = 1
    while lb < N:
        r[ind[lb:ub]] = cr
        cr += 1
        lb = ub
        ub = min(N, ub * 2)
    # Returning
    return r

@numba.jit(nopython=True)
def rank_from_dist(d):
    """
    Compute ranks from distances. 
    In: 
    - d: a 1-D numpy array with distances between data points and a reference point. 
    Out:
    A 1-D numpy array with the same size as d and with the ranks of the data points with respect to the reference point. 
    """
    # Number of elements in d
    nd = d.size
    # Computing the index associated with each rank. We use mergesort since it is stable. 
    v = d.argsort(kind='mergesort')
    # Computing the rank associated with each index.
    r = np.empty(shape=nd, dtype=np.int64)
    for i in range(nd):
        r[v[i]] = i
    # Returning 
    return r

@numba.jit(nopython=True)
def lower_median_partition(a, ind):
    """
    Apply randomized_partition_unstable until the indexes in an array are partitioned around their lower median.
    This function is inspired from the RANDOMIZED-SELECT procedure described in 'Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). Introduction to algorithms. MIT press.', page 216.
    In:
    - a: a 1-D numpy array.
    - ind: a 1-D numpy array of integer being indexes of entries in a.
    Out:
    - The index of the lower median, which is equal to (ind.size-1)//2.
    This function modifies ind so that a[ind] is partionned around its lower median, at index (ind.size-1)//2.
    a is not modified.
    """
    p = 0
    r = ind.size - 1
    # Index of the lower median
    imed = r // 2
    while p < r:
        # Unstable partition is employed as stability is not necessary.
        q = randomized_partition_unstable(a, ind, p, r)
        if q == imed:
            break
        elif q > imed:
            r = q-1
        else:
            p = q+1
    return imed

@numba.jit(nopython=True)
def vp_repr(X_hd, dist_hd, n):
    """
    Sample representative data points (= landmarks) in a data set by partionning it using a vantage-point tree. 
    In:
    - X_hd, dist_hd, n: same as in fast_eval_qnx function.
    Out:
    A 1-D numpy array of integer with n elements, indicating the indexes of the representative data points in X_hd.
    Remark:
    If X_hd contains N samples and n representatives (= landmarks) are sampled, the time complexity of this function is O(N log(n)).
    """
    # Number of data points and dimension of the HDS
    N, M = X_hd.shape
    # Indexes of the data points
    ind = np.arange(N)
    # Determining the depth of the tree and its number of leafs as a function of n
    depth = int(np.log2(n))
    nleafs = 2**depth
    # The ith leaf will span ind[leafs[i]:leafs[i+1]].
    leafs = np.empty(shape=nleafs+1, dtype=np.int64)
    leafs[0] = 0
    leafs[nleafs] = N
    # Current depth
    depth_cur = 0
    # The ith current leaf is spanning ind[leafs[i*jump]:leafs[(i+1)*jump]]
    jump = nleafs
    # Array to store the mean of the data points stored in a leaf
    mean_cur = np.empty(shape=M, dtype=np.float64)
    # Array to store the distances to the vantage point
    d_vp = np.empty(shape=N, dtype=np.float64)
    # Digging in the tree
    while depth_cur < depth:
        jump_2 = jump // 2
        # Splitting the current leafs
        leaf_start = 0
        while leaf_start < nleafs:
            # The currently considered leaf is spanning ind[leafs[leaf_start]:leafs[leaf_stop]].
            leaf_stop = leaf_start + jump
            # Mean of the data points indexed in ind[leafs[leaf_start]:leafs[leaf_stop]].
            for j in range(M):
                mean_cur[j] = X_hd[ind[leafs[leaf_start]:leafs[leaf_stop]],j].mean()
            # Index in ind[leafs[leaf_start]:leafs[leaf_stop]] of the vantage point, which is the farthest point to the central one, which is the closest point to the mean in ind[leafs[leaf_start]:leafs[leaf_stop]]
            vp = dist_hd(X_hd[ind[leafs[leaf_start]:leafs[leaf_stop]],:], X_hd[ind[leafs[leaf_start]:leafs[leaf_stop]][dist_hd(X_hd[ind[leafs[leaf_start]:leafs[leaf_stop]],:], mean_cur).argmin()],:]).argmax()
            # Distances to vp
            d_vp[ind[leafs[leaf_start]:leafs[leaf_stop]]] = dist_hd(X_hd[ind[leafs[leaf_start]:leafs[leaf_stop]],:], X_hd[ind[leafs[leaf_start]:leafs[leaf_stop]][vp],:])
            # Partitioning ind[leafs[leaf_start]:leafs[leaf_stop]] around the lower median of d_vp[ind[leafs[leaf_start]:leafs[leaf_stop]]], and updating leafs according to the new lower median partition
            leafs[leaf_start + jump_2] = lower_median_partition(d_vp, ind[leafs[leaf_start]:leafs[leaf_stop]]) + leafs[leaf_start] + 1
            leaf_start = leaf_stop
        # Updating the current depth
        depth_cur += 1
        jump = jump_2
    # Array to store the n selected representatives
    repr = np.empty(shape=n, dtype=np.int64)
    # Current number of sampled representatives
    n_cur = 0
    # Order in which the leafs are visited
    id_leafs = np.arange(nleafs)
    # Current position in id_leafs
    iid_leafs = 0
    # Copying leafs and updating it so that leafs[i] and leafs_stop[i] respectively point to the start (included) and the end (not included) of the ith leaf in ind. 
    leafs_stop = leafs[1:].copy()
    leafs = leafs[:nleafs]
    # Sampling the representatives in the leafs
    while n_cur < n:
        if iid_leafs == nleafs:
            # Shuffling the order of id_leafs and setting iid_leafs back to 0
            np.random.shuffle(id_leafs)
            iid_leafs = 0
        # Index of the current leaf
        cL = id_leafs[iid_leafs]
        # Checking whether the current leaf still contains data points which can be sampled
        if leafs_stop[cL] - leafs[cL] > 0:
            # Sampling a representative in the current leaf
            id_samp = np.random.randint(leafs[cL], leafs_stop[cL])
            repr[n_cur] = ind[id_samp]
            # Replacing the sampled data point with leafs[cL] to avoid sampling it again
            ind[id_samp] = ind[leafs[cL]]
            leafs[cL] += 1
            # Updating n_cur
            n_cur += 1
        # Updating iid_leafs
        iid_leafs += 1
    # Returning
    return repr

@numba.jit(nopython=True)
def fast_eval_qnx_impl(X_hd, X_ld, dist_hd, dist_ld, n, pow2K, samp, tmp, qnx, den_qnx):
    """
    Used by fast_eval_qnx in this file, to gather all instructions that can be managed by numba. 
    """
    # For each representative data point (= landmark)
    for i in samp:
        i_1 = i+1
        # Computing the HD and LD distances between i and the other data points
        d_hd = dist_hd(np.vstack((X_hd[:i,:], X_hd[i_1:,:])), X_hd[i,:])
        d_ld = dist_ld(np.vstack((X_ld[:i,:], X_ld[i_1:,:])), X_ld[i,:])
        # Computing the HD and LD ranks. 
        if pow2K:
            r_hd = pow2_rank(d_hd, tmp)
            r_ld = pow2_rank(d_ld, tmp)
        else:
            r_hd = rank_from_dist(d_hd)
            r_ld = rank_from_dist(d_ld)
        # Updating qnx
        for j, rj_hd in enumerate(r_hd):
            qnx[max(rj_hd, r_ld[j])] += 1
    # Computing the cumulative sum of qnx, normalizing and returning
    return qnx.cumsum().astype(np.float64)/(n*den_qnx), den_qnx, samp

def fast_eval_qnx(X_hd, X_ld, dist_hd, dist_ld, n=-1, seed=-1, pow2K=False, vp_samp=False, samp=None):
    """
    Fast evaluation of Q_{NX}(K).
    If pow2K is True, the time complexity of this function is O(n*N), where N is the number of data points. 
    If pow2K is False, the time complexity is O(n*N*log(N)).
    In:
    - X_hd: a 2-D numpy array storing the HD samples, with one example per row and one feature per column.
    - X_ld: a 2-D numpy array storing the LD samples, with one example per row and one feature per column.
    - dist_hd: a callable taking as input a 2-D numpy array such as X_hd and a 1-D numpy array v with as many elements as columns in X_hd, and returning a 1-D numpy array with the HD distances between v and the data points in the rows of X_hd. 
    - dist_ld: similar to dist_hd, but for the LD distances. 
    - n: an integer indicating the number of representative data points to sample (without replacement) to compute the approximation of Q_{NX}(K). Higher values provide more accurate estimations, but slower computation time. If n is < 1, it is set to int(round(10*np.log(N))), where N = X_hd.shape[0]. If n > N, it is set to N.
    - seed: integer to seed the random number generator. Used only if >= 0.
    - pow2K: boolean. If False, the Q_NX(K) curve is estimated for all K = 1 to N-1, which yields O(n*N*log(N)) time complexity. If True, it is only estimated for K equal to 2^0, 2^1, 2^2, ..., 2^(int(np.log2(N-1))), N-1, which results in O(n*N) time complexity. 
    - vp_samp: boolean. If False, the n representative data points are sampled uniformly at random. Otherwise, vp_repr function is employed. 
    - samp: if not None and vp_samp is False, than must be a 1-d numpy array with n elements, containing the index of the representative data points. It is not modified and returned as the last element of the output tuple. 
    Out:
    If pow2K is True, a tuple with:
    - A 1-D numpy array with the estimated values of Q_{NX}(K) at K = 2^0, 2^1, 2^2, 2^3, ..., 2^(int(np.log2(N-1))), N-1.
    - A 1-D numpy array containing 2^0, 2^1, 2^2, 2^3, ..., 2^(int(np.log2(N-1))), N-1.
    - A 1-D numpy array of integer with n elements, indicating the indexes of the representative data points in X_hd.
    If pow2K is False, a tuple with:
    - A 1-D numpy array with N-1 elements, in which element at index i is the estimation of Q_{NX}(i+1).
    - A 1-D numpy array equal to np.arange(N-1) + 1.0.
    - A 1-D numpy array of integer with n elements, indicating the indexes of the representative data points in X_hd.
    """
    # Seeding the random number generator
    if seed >= 0:
        np.random.seed(seed)
    # Number of samples
    N = X_hd.shape[0]
    N_1 = N-1
    # Checking n
    if n < 1:
        n = int(round(10 * np.log(N)))
    if n > N:
        n = N
    # Checking pow2K
    if pow2K:
        size_qnx = int(np.log2(N_1))
        if 2**size_qnx < N_1:
            size_qnx += 2
        else:
            size_qnx += 1
        den_qnx = np.empty(shape=size_qnx, dtype=np.float64)
        den_qnx[0] = 1
        for i in range(1, size_qnx-1, 1):
            den_qnx[i] = den_qnx[i-1]*2
        den_qnx[size_qnx-1] = N_1
    else:
        size_qnx = N_1
        den_qnx = np.arange(N_1)+1.0
    # Allocating space
    qnx = np.zeros(shape=size_qnx, dtype=np.int64)
    # Container to store intermediate computations in pow2_rank, to avoid the overhead of allocating it each time the function is called.
    tmp = np.empty(shape=N_1, dtype=np.int64)
    # Sampling the representative data points
    if vp_samp:
        samp = vp_repr(X_hd, dist_hd, n)
    else:
        if samp is None:
            samp = np.random.choice(N, size=n, replace=False)
    return fast_eval_qnx_impl(X_hd=X_hd, X_ld=X_ld, dist_hd=dist_hd, dist_ld=dist_ld, n=n, pow2K=pow2K, samp=samp, tmp=tmp, qnx=qnx, den_qnx=den_qnx)

@numba.jit(nopython=True)
def fast_eval_dr_quality_pow2K(X_hd, qnx, den_qnx, samp):
    """
    Used by fast_eval_dr_quality, to gather all instructions that can be managed by numba. 
    """
    srnx = qnx.size - 1
    N_1 = X_hd.shape[0] - 1
    rnx = (N_1*qnx[:srnx]-den_qnx[:srnx])/(N_1-den_qnx[:srnx])
    srnx -= 1
    i_K_cur = (1.0/np.arange(den_qnx[srnx], N_1, 1, np.float64)).sum()
    auc = rnx[0] - rnx[1] + (N_1*rnx[srnx]*i_K_cur)/(N_1 - den_qnx[srnx])
    i_K = 1.0 + i_K_cur
    j = 2
    for i in range(1, srnx, 1):
        i_K_cur = (1.0/np.arange(den_qnx[i], den_qnx[j], 1, np.float64)).sum()
        auc += (2.0*rnx[i]-rnx[j])*i_K_cur
        i_K += i_K_cur
        j += 1
    return qnx, rnx, auc/i_K, samp

def fast_eval_dr_quality(X_hd, X_ld, dist_hd, dist_ld, n=-1, seed=-1, pow2K=False, vp_samp=False, samp=None):
    """
    Fast version of eval_dr_quality, to efficiently compute estimations of Q_{NX}(K), R_{NX}(K) and the AUC.
    In: 
    - Same parameters as fast_eval_qnx. 
    Out: a tuple with
    - a 1-D numpy array with the Q_{NX} curve as in the first element of the tuple returned by fast_eval_qnx.
    - a 1-D numpy array with the R_{NX} curve computed from the above Q_{NX} curve.
    - the AUC of the above R_{NX}(K) curve with a log scale for K, as defined in [2]. If pow2K is True, the AUC is estimated using a linear approximation of R_NX(K) for K between 2^i and 2^(i+1).
    - A 1-D numpy array of integer with n elements, indicating the indexes of the representative data points in X_hd.
    Remark:
    - The time complexity of this function is determined by the function fast_eval_qnx. See its documentation for details. 
    """
    # Fast computation of Q_{NX}(K)
    qnx, den_qnx, samp = fast_eval_qnx(X_hd=X_hd, X_ld=X_ld, dist_hd=dist_hd, dist_ld=dist_ld, n=n, seed=seed, pow2K=pow2K, vp_samp=vp_samp, samp=samp)
    if pow2K:
        return fast_eval_dr_quality_pow2K(X_hd=X_hd, qnx=qnx, den_qnx=den_qnx, samp=samp)
    else:
        # Computing R_{NX}(K) from qnx
        rnx = eval_rnx(qnx)
        # Computing the AUC of rnx, and returning.
        return qnx, rnx, eval_auc(rnx), samp
