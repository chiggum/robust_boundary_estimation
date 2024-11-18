import sys
sys.path.insert(0, '../pyLDLE2/')

import numpy as np
from pyLDLE2 import util_, visualize_, datasets
from scipy.sparse import coo_matrix
from scipy import optimize
from scipy.special import erf, erfinv
from matplotlib import pyplot as plt
from scipy.stats import chi2, spearmanr
from sklearn.decomposition import PCA
from scipy.linalg import svd
from umap.umap_ import compute_membership_strengths
from numba import jit
import pdb
from scipy.linalg import svd
from scipy.sparse.linalg import svds

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf
EPS = 1e-30

default_opts = {
    'd': 2,
    'k_nn': 128,
    'k_tune': 128,
    'h': None,
    's': 0.1,
    'optimizer': 'newton',
    'optimizer_maxiter': 5000,
    'maxiter_for_selecting_bw': 30,
    'lr': 0.03,
    'reg': 0.01,
    'W': None,
    'local_subspace': None,
    'only_kde': False,
    'ds': True
}

def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each sample. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho

def umap_kernel(knn_indices, knn_dists, k_tune):
    n = knn_indices.shape[0]
    sigmas, rhos = smooth_knn_dist(knn_dists, k_tune, local_connectivity=0)
    rows, cols, vals, _ = compute_membership_strengths(knn_indices, knn_dists.astype('float32'), sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    return result, sigmas

# def sinkhorn_old(K, maxiter=20000, delta=1e-15, eps=0, print_freq=1000):
#     """https://epubs.siam.org/doi/pdf/10.1137/20M1342124 """
#     D = np.array(K.sum(axis=1)).squeeze()
#     d0 = 1./(D+eps)
#     d1 = 1./(K.dot(d0)+eps)
#     d2 = 1./(K.dot(d1)+eps)
#     for tau in range(maxiter):
#         error = np.max(np.abs(d0 / d2 - 1))
#         if tau%print_freq==0:
#             print('Error:', error, flush=True)
#         if error < delta:
#             print('Sinkhorn converged at iter:', tau)
#             break
#         d3 = 1. / (K.dot(d2) + eps)
#         d0=d1.copy()
#         d1=d2.copy()
#         d2=d3.copy()
#     d = np.sqrt(d2 * d1)
#     K.data = K.data*d[K.row]*d[K.col]
#     return K, d

def sinkhorn(K, maxiter=20000, delta=1e-15, eps=0, boundC = 1e-8, print_freq=1000):
    """https://epubs.siam.org/doi/pdf/10.1137/20M1342124 """
    n = K.shape[0]
    r = np.ones((n,1))
    u = np.ones((n,1))
    v = r/(K.dot(u))
    x = np.sqrt(u*v)

    assert np.min(x) > boundC, 'assert min(x) > boundC failed.'
    for tau in range(maxiter):
        error =  np.max(np.abs(u*(K.dot(v)) - r))
        if tau%print_freq:
            print('Error:', error, flush=True)
        
        if error < delta:
            print('Sinkhorn converged at iter:', tau)
            break

        u = r/(K.dot(v))
        v = r/(K.dot(u))
        x = np.sqrt(u*v)
        if np.sum(x<boundC) > 0:
            print('boundC not satisfied at iter:', tau)
            x[x < boundC] = boundC
        
        u=x
        v=x
    x = x.flatten()
    K.data = K.data*x[K.row]*x[K.col]
    return K, x

# def sinkhorn(K, maxiter=20000, delta=1e-15, eps=0, boundC = 1e-8, print_freq=1000):
#     K_inds, K_vals = K
#     """https://epubs.siam.org/doi/pdf/10.1137/20M1342124 """
#     n = K_inds.shape[0]
#     r = np.ones((n,1))
#     u = np.ones((n,1))
#     v = r/K_vals.sum(axis=1) # r/Ku
#     x = np.sqrt(u*v)

#     assert np.min(x) > boundC, 'assert min(x) > boundC failed.'
#     for tau in range(maxiter):
#         #error =  np.max(np.abs(u*(K.dot(v)) - r))
#         Kv = (K_vals * v[K_inds]).sum(axis=1)
#         uKV = u*Kv
#         error =  np.max(np.abs(uKv) - r))
#         if tau%print_freq:
#             print('Error:', error, flush=True)
        
#         if error < delta:
#             print('Sinkhorn converged at iter:', tau)
#             break

#         u = r/(Kv)
#         Ku = (K_vals * u[K_inds]).sum(axis=1)
#         v = r/(Ku)
#         x = np.sqrt(u*v)
#         if np.sum(x<boundC) > 0:
#             print('boundC not satisfied at iter:', tau)
#             x[x < boundC] = boundC
        
#         u=x
#         v=x

#     K_vals = (K_vals * x[:,None]) * x[K_inds]
#     return K_vals, x


# @jit(nopython=True, parallel=True)
# def sinkhorn_numpy(K, maxiter=10000, delta=1e-20, eps=0):
#     """https://epubs.siam.org/doi/pdf/10.1137/20M1342124 """
#     D = K.sum(axis=1)[:,None]
#     d0 = 1./(D+eps)
#     d1 = 1./(K.dot(d0)+eps)
#     d2 = 1./(K.dot(d1)+eps)
#     for tau in range(maxiter):
#         if np.max(np.abs(d0 / d2 - 1)) < delta:
#             #print('Sinkhorn converged at iter:', tau)
#             break
#         d3 = 1. / (K.dot(d2) + eps)
#         d0=d1.copy()
#         d1=d2.copy()
#         d2=d3.copy()
#     d = np.sqrt(d2 * d1)
#     K = K*d[:,None]*d[None,:]
#     return K, d

def compute_m0(bx, h, d=2):
    return 0.5*(np.pi**(d/2))*(1+erf(bx/h))

def compute_dm0(bx, h, d=2):
    return (np.pi**((d-1)/2))*np.exp(-(bx/h)**2)/h

def compute_m1(bx, h, d=2):
    return -0.5*(np.pi**((d-1)/2))*np.exp(-(bx/h)**2)

def compute_dm1(bx, h, d=2):
    return (np.pi**((d-1)/2))*np.exp(-(bx/h)**2)*(bx/h)/h

def compute_m2(bx, h, d=2):
    return (bx/h)*compute_m1(bx,h,d) + 0.5*compute_m0(bx,h,d)

def compute_dm2(bx, h, d=2):
    return (bx/h)*compute_dm1(bx,h,d) + compute_m1(bx,h,d)/h + 0.5*compute_dm0(bx,h,d)

def compute_m(bx, h):
    return compute_m1(bx,h)/compute_m0(bx,h)

def compute_A(bx, h, d=2, H=0):
    m1 = compute_m1(bx, h, d) 
    m0 = compute_m0(bx, h, d) 
    m2 = compute_m2(bx, h, d)
    return 2*(m1**2 - m2*(m0 + 0.5*H*h*(d-1)*m1))

def compute_B(bx, h, d=2, H=0, eta_q_by_q=0):
    m1 = compute_m1(bx, h, d) 
    return (np.pi**(d/2))*m1*(h*(eta_q_by_q + 0.5*(d-1)*H) - 2*bx/h)

def compute_ABD(bx, h, d=2, H=0, eta_q_by_q=0):
    A = compute_A(bx, h, d, H)
    B = compute_B(bx, h, d, H, eta_q_by_q)
    D = np.sqrt(B**2-4*(np.pi**d)*A)
    return A, B, D

def compute_rho(Dd, h, d=2):
    n = len(Dd)
    return Dd * np.sqrt((n-1)*(np.pi**(d/2))*(h**d))

def compute_dzeta(bx, h, d=2):
    m0 = compute_m0(bx, h, d)
    m1 = compute_m1(bx, h, d)
    m2 = compute_m2(bx, h, d)
    dm0 = compute_dm0(bx, h, d)
    dm1 = compute_dm1(bx, h, d)
    dm2 = compute_dm2(bx, h, d)

    c = (np.pi**(d/2))/2
    bxhm1 = (bx/h)*m1
    bxhm1pm0 = bxhm1+m0
    m12 = m1**2
    a1 = np.sqrt((bxhm1pm0)**2 - 2*(m12))
    m1dm1 = m1*dm1
    
    num = bxhm1 + a1
    denom = m2*m0-m12

    t1 = m1/h + (bx/h)*dm1
    t2 = 0.5*(2*(bxhm1pm0)*(t1 + dm0) - 4*m1dm1)/a1
    
    dnum = t1 + t2
    ddenom = dm2*m0 + m2*dm0 - 2*m1dm1
    
    return c*(dnum*denom - ddenom*num)/(denom**2)

def compute_zeta(bx, h, d=2):
    m0 = compute_m0(bx, h, d)
    m1 = compute_m1(bx, h, d)
    bxhm1 = (bx/h)*m1
    m2 = bxhm1 + m0/2
    temp = ((np.pi**(d/2))/2)*(bxhm1 + np.sqrt((bxhm1+m0)**2 - 2*(m1**2)))/(m2*m0-m1**2)
    return temp

def compute_drho2q_first_order2(bx, h, d=2):
    m0 = compute_m0(bx, h, d)
    dm0 = compute_dm0(bx, h, d)
    return -dm0*(np.pi**(d/2))/(m0**2)

def compute_rho2q_first_order2(bx, h, d=2):
    m0 = compute_m0(bx, h, d)
    return (np.pi**(d/2))/m0

def compute_rho_first_order(bx, h, d=2, q=1):
    rho2q = compute_rho2q_first_order(bx, h, d=2)
    return np.sqrt(rho2q/q)

def compute_xi(bx, h, d=2):
    m0 = compute_m0(bx, h, d)
    m1 = compute_m1(bx, h, d)
    m2 = compute_m2(bx, h, d)
    bxbyhm1 = (bx/h)*m1
    return -0.5*(bxbyhm1 + np.sqrt((bxbyhm1+m0)**2-2*(m1**2))) + m2

def compute_dxi(bx, h, d=2):
    m0 = compute_m0(bx, h, d)
    m1 = compute_m1(bx, h, d)
    m2 = compute_m2(bx, h, d)
    dm0 = compute_dm0(bx, h, d)
    dm1 = compute_dm1(bx, h, d)
    dm2 = compute_dm2(bx, h, d)
    
    bxbyhm1 = (bx/h)*m1
    dbxbyhm1 = m1/h + (bx/h)*dm1
    sqrt_term = np.sqrt((bxbyhm1+m0)**2-2*(m1**2))
    return -0.5*(dbxbyhm1 + ((dbxbyhm1+dm0)*(bxbyhm1+m0) - 2*m1*dm1)/sqrt_term) + dm2
    
def epanechnikov_kernel(dist, eps):
    return (1-dist**2/eps)*(dist > 0)*(dist < np.sqrt(eps))

# def compute_nu_norm(X, neigh_ind, K, d=None, local_pca=False):
#     nu_norm = np.zeros(X.shape[0])
#     for i in range(X.shape[0]):
#         if local_pca:
# #             n_i = neigh_ind[i]
# #             X_i = X[n_i,:].T # p x N_i
# #             X_i = X_i - X[i,:][:,None]
# #             X_i_norm = np.linalg.norm(X_i, axis=0) # N_i dimensional
# #             D_i = np.sqrt(epanechnikov_kernel(X_i_norm, eps_pca)) #N_i dimensional
# #             B_i = X_i * D_i[None,:]
# #             U_i, Sigma_i, V_iT = svd(B_i)
# #             temp = Sigma_i[:d][:,None]*V_iT[:d,:]
# #             temp = temp.T
#             X_i_nbrs = X[neigh_ind[i,:].tolist()+[i],:]
#             pca = PCA(n_components=d)
#             y = pca.fit_transform(X_i_nbrs)
#             temp = y[:-1,:] - y[-1,:][None,:]
#         else:
#             temp = X[neigh_ind[i,:],:]-X[i,:][None,:]
#         nu_norm[i] = np.linalg.norm(K.getrow(i)[0,neigh_ind[i,:]].dot(temp))
#     return nu_norm

def compute_nu_norm(X, neigh_ind, K, d=None, local_subspace=None):
    nu_norm = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if local_subspace is not None:
            if type(local_subspace) == str:
                X_i_nbrs = X[neigh_ind[i,:].tolist()+[i],:]
                pca = PCA(n_components=d)
                y = pca.fit_transform(X_i_nbrs)
                temp = y[:-1,:] - y[-1,:][None,:]
            else:
                if d in local_subspace.shape:
                    Q_k,Sigma_k,_ = svd(local_subspace[:,i,:].T)
                else:
                    Q_k,Sigma_k,_ = svds(local_subspace[:,i,:].T, d, which='LM')
                Q_k = Q_k[:,:d]
                temp = (X[neigh_ind[i,:],:] -  X[i,:][None,:]).dot(Q_k)
        else:
            temp = X[neigh_ind[i,:],:]-X[i,:][None,:]
        nu_norm[i] = np.linalg.norm(K.getrow(i)[0,neigh_ind[i,:]].dot(temp))
        #nu_norm[i] = np.linalg.norm(np.sum(K[i,:][:,None]*temp, axis=0))
    return nu_norm

# def compute_mu(X, neigh_ind, K, d=None, local_pca=False):
#     mu = np.zeros(X.shape[0])
#     for i in range(X.shape[0]):
#         if local_pca:
# #             n_i = neigh_ind[i]
# #             X_i = X[n_i,:].T # p x N_i
# #             X_i = X_i - X[i,:][:,None]
# #             X_i_norm = np.linalg.norm(X_i, axis=0) # N_i dimensional
# #             D_i = np.sqrt(epanechnikov_kernel(X_i_norm, eps_pca)) #N_i dimensional
# #             B_i = X_i * D_i[None,:]
# #             U_i, Sigma_i, V_iT = svd(B_i)
# #             temp = Sigma_i[:d][:,None]*V_iT[:d,:]
# #             temp = temp.T
#             X_i_nbrs = X[neigh_ind[i,:].tolist()+[i],:]
#             pca = PCA(n_components=d)
#             y = pca.fit_transform(X_i_nbrs)
#             temp = y[:-1,:] - y[-1,:][None,:]
#         else:
#             temp = X[neigh_ind[i,:],:]-X[i,:][None,:]
#         mu[i] = K.getrow(i)[0,neigh_ind[i,:]].dot((np.linalg.norm(temp, axis=-1)**2)[:,None])
#     return mu

def compute_mu(X, neigh_ind, K, d=None, local_subspace=None):
    mu = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if local_subspace is not None:
            if d in local_subspace.shape:
                Q_k,Sigma_k,_ = svd(local_subspace[:,i,:].T)
            else:
                Q_k,Sigma_k,_ = svds(local_subspace[:,i,:].T, d, which='LM')
            Q_k = Q_k[:,:d]
            temp = (X[neigh_ind[i,:],:] -  X[i,:][None,:]).dot(Q_k)
        else:
            temp = X[neigh_ind[i,:],:]-X[i,:][None,:]
        mu[i] = K.getrow(i)[0,neigh_ind[i,:]].dot((np.linalg.norm(temp, axis=-1)**2)[:,None])
    return mu

def compute_autotuned_bandwidth(neigh_ind, neigh_dist, k_tune, maxiter_for_selecting_bw):
    h_per_point = neigh_dist[:,k_tune-1]
    cnvgd = False
    it = 0
    while (not cnvgd) and (it<maxiter_for_selecting_bw):
        h_per_point_new = np.median(h_per_point[neigh_ind], axis=1)
        temp = np.mean(np.abs(h_per_point_new-h_per_point))
        print(temp)
        if temp < 1e-6:
            cnvgd = True
        h_per_point = h_per_point_new
        it += 1
    print('h_per_point min max median', np.min(h_per_point), np.max(h_per_point), np.median(h_per_point))
    return h_per_point

def compute_self_tuned_kernel(neigh_ind, neigh_dist, h, ds=False):
    if type(h) == np.ndarray:
        K = np.exp(-neigh_dist**2/(h[:,None]*h[neigh_ind]))+EPS
    else:
        K = np.exp(-(neigh_dist/h)**2)+EPS
    n = neigh_ind.shape[0]
    source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
    K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    # symmetrize the kernel
    K = K + K.T
    ones_K_like = ones_K_like + ones_K_like.T
    K.data /= ones_K_like.data

    K = K.tocoo()
    #Dinvsqrt = 1/np.sqrt(np.array(K.sum(axis=1)).flatten())
    #K.data = K.data * Dinvsqrt[K.row] * Dinvsqrt[K.col]
    
    # make it doubly stochastic
    if ds:
        K, D = sinkhorn(K)
    else:
        D = 1/np.sqrt(np.array(K.sum(axis=1)).flatten())

    K = K.tocsr()
    return K, D

def newtons_method(x0, F, F_prime, maxiter, tol=1e-12, print_freq=250):
    x0 = x0.copy()
    for i in range(maxiter):
        Fx0 = F(x0)
        x = x0 - Fx0/(F_prime(x0)+1e-30)
        #x = np.maximum(0, np.minimum(x, 1))
        x = np.maximum(0, x)
        #pdb.set_trace()
        error = np.max(np.abs(x-x0))
        #error = np.max(np.abs(Fx0))
        if i%print_freq == 0:
            print('Error at iter:', i, 'is:', error, flush=True) 
        if error < tol:
            print('newton method converged at iter:', i, flush=True)
            return x
        x0 = x.copy()
    print('newton method did not converge.', flush=True)
    return x

def secants_method(x0, x1, F, maxiter, tol=1e-8):
    x0 = x0.copy()
    x1 = x1.copy()
    for i in range(maxiter):
        Fx1 = F(x1)
        Fx0 = F(x0)
        F_prime = (Fx1-Fx0)/(x1-x0+1e-12)
        x2 = x1 - Fx1/(F_prime+1e-12)
        print(x0[0], x1[0], x2[0], Fx1[0], Fx0[0], F_prime[0])
        #x = np.maximum(0, np.minimum(x, 1))
        x2 = np.maximum(0, x2)
        pdb.set_trace()
        #pdb.set_trace()
        if np.max(np.abs(x2-x1)) < tol:
            print('secants method converged at iter:', i, flush=True)
            return x2
        x0 = x1.copy()
        x1 = x2.copy()
    print('secants method did not converge.', flush=True)
    return x2

def compute_bx(bx_init, F, F_prime, maxiter, optimizer='newton'):
    if optimizer == 'newton':
        #bx = optimize.newton(F, bx_init, F_prime, maxiter=maxiter)
        bx = newtons_method(bx_init, F, F_prime, maxiter)
        bx = np.maximum(bx, 0)
    else:
        bx = secants_method(bx_init, F, maxiter)
        bx = np.maximum(bx, 0)
    return bx

def compute_jaccard_index(ddX, bx, prctile=10):
    mask0 = ddX <= np.percentile(ddX, prctile)
    mask = bx <= np.percentile(bx, prctile)
    return np.sum(mask*mask0)/np.sum((mask+mask0) > 0)

def compute_f1_score(ddX, bx, prctile=10):
    mask0 = ddX < np.percentile(ddX, prctile)
    mask = bx < np.percentile(bx, prctile)
    recall = np.sum(mask0*mask)/np.sum(mask0)
    precision = np.sum(mask0*mask)/np.sum(mask)
    f1 = 2*precision*recall/(precision+recall)
    return f1

# Berry and Sauer's method to estimate bx
# If opts['local_pca'] = True then uses
# local pca to compute ||\nu||
def estimate_bx_berry_and_sauer(X, opts=default_opts):
    d = opts['d']
    h = opts['h']
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]

    if h is None:
        # Compute bandwidth for each point
        h = compute_autotuned_bandwidth(neigh_ind, neigh_dist, opts['k_tune'], opts['maxiter_for_selecting_bw'])

    # Compute kernel
    K, D = compute_self_tuned_kernel(neigh_ind, neigh_dist, h)

    # Setup the function whose root is needed
    g_hat = np.array(K.sum(axis=1)).flatten()
    if opts['only_kde']:
        return g_hat
        
    nu_norm = compute_nu_norm(X, neigh_ind, K, d, opts['local_subspace'])

    def F(bx):
        return h * g_hat * compute_m1(bx, h, d) + nu_norm * compute_m0(bx, h, d) # plus because m1 is negative
    def F_prime(bx):
        return h * g_hat * compute_dm1(bx, h, d) + nu_norm * compute_dm0(bx, h, d)

    # Initialized bx
    bx_init = h*np.sqrt(np.maximum(0, np.log(h*g_hat + EPS) - np.log(2*np.sqrt(np.pi)*nu_norm + EPS)))

    # Optimize for bx
    bx = compute_bx(bx_init, F, F_prime, opts['optimizer_maxiter'])
    return bx, bx_init

# Uses doubly stochastic kernel and local pca to compute bx
# v1 = uses ||\nu||
def estimate_bx_ours_v1(X, opts=default_opts):
    d = opts['d']
    h = opts['h']
    s = opts['s']
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]

    if h is None:
        # Compute bandwidth for each point
        h = compute_autotuned_bandwidth(neigh_ind, neigh_dist, opts['k_tune'], opts['maxiter_for_selecting_bw'])

    if ('W' in opts) and (opts['W'] is not None):
        W = opts['W']
        D = opts['D']
    else:
        # Compute kernel
        W, D = compute_self_tuned_kernel(neigh_ind, neigh_dist, h, ds=opts['ds'])

    # Setup the function whose root is needed
    g_hat = np.array(W.power(s).sum(axis=1)).flatten()
    if opts['only_kde']:
        return g_hat, W, D
    
    nu_norm = compute_nu_norm(X, neigh_ind, W.power(s), d, opts['local_subspace'])

    h_by_sqrt_s = h/np.sqrt(s)
    # Initialized bx
    bx_init = h_by_sqrt_s*np.sqrt(np.maximum(0, np.log(h_by_sqrt_s*g_hat + EPS) - np.log(2*np.sqrt(np.pi)*nu_norm + EPS)))

    def F(bx):
        return h_by_sqrt_s * g_hat * compute_m1(bx, h_by_sqrt_s, d) + nu_norm * compute_m0(bx, h_by_sqrt_s, d) # plus because m1 is negative
    def F_prime(bx):
        return h_by_sqrt_s * g_hat * compute_dm1(bx, h_by_sqrt_s, d) + nu_norm * compute_dm0(bx, h_by_sqrt_s, d)

    # Optimize for bx
    if opts['optimizer'] == 'newton':
        bx = compute_bx(bx_init, F, F_prime, opts['optimizer_maxiter'])
    else:
        bx = bx_init.copy()
        lr = opts['lr']
        reg = opts['reg']
        n = bx.shape[0]
        # minimize F(bx)^2 + bx^T(I-W)bx
        for i in range(opts['optimizer_maxiter']):
            Fbx = F(bx)
            Wbx = W.dot(bx)
            bx_minus_Wbx = bx - Wbx
            #loss = np.mean(Fbx**2) + reg * np.sum(bx*bx_minus_Wbx)
            loss = np.mean(Fbx**2) 
            if loss < 1e-12:
                print('Converged at iter:', i+1)
                break
            print('Iter:', i+1, ':: loss:', loss)
            grad_bx = 2*Fbx*F_prime(bx)/n + reg*bx_minus_Wbx
            bx = bx - lr * grad_bx
            bx = np.maximum(bx, 0)
        
    return bx, bx_init, W, D

# Uses doubly stochastic kernel and local pca to compute bx
# v2 = uses mu
def estimate_bx_ours_v2(X, opts=default_opts):
    d = opts['d']
    h = opts['h']
    s = opts['s']
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]

    if h is None:
        # Compute bandwidth for each point
        h = compute_autotuned_bandwidth(neigh_ind, neigh_dist, opts['k_tune'], opts['maxiter_for_selecting_bw'])

    if ('W' in opts) and (opts['W'] is not None):
        W = opts['W']
        D = opts['D']
    else:
        # Compute kernel
        W, D = compute_self_tuned_kernel(neigh_ind, neigh_dist, h, ds=True)

    # Setup the function whose root is needed
    g_hat = np.array(W.power(s).sum(axis=1)).flatten()
    if opts['only_kde']:
        return g_hat, W, D
        
    mu = compute_mu(X, neigh_ind, W.power(s), d, opts['local_pca'])
    nu_norm = compute_nu_norm(X, neigh_ind, W.power(s), d, opts['local_subspace'])
    
    h_by_sqrt_s = h/np.sqrt(s)
    m2 = np.power(np.pi, d/2.0)/2

    def F(bx):
        return (h_by_sqrt_s**2)*(m2*(d-1)+compute_m2(bx, h_by_sqrt_s, d))*g_hat - mu * compute_m0(bx, h_by_sqrt_s, d)
    def F_prime(bx):
        return (h_by_sqrt_s**2)*compute_dm2(bx, h_by_sqrt_s, d)*g_hat - mu * compute_dm0(bx, h_by_sqrt_s, d)

    # step_num = 1
    # def callbackF(bx):
    #     global step_num
    #     print('{0:4d}   {1: 3.6f}   {2: 3.6f}'.format(step_num, bx, F(bx))
    #     step_num += 1

    # Initialized bx
    bx_init = h_by_sqrt_s*np.sqrt(np.maximum(0, np.log(h_by_sqrt_s*g_hat + EPS) - np.log(2*np.sqrt(np.pi)*nu_norm + EPS)))

    # Optimize for bx
    bx = compute_bx(bx_init, F, F_prime, opts['newton_maxiter'])
    return bx, bx_init, W, D


# Uses doubly stochastic kernel and local pca to compute bx
# v1 = uses ||\nu||
def estimate_bx_ours_v3(X, opts=default_opts):
    d = opts['d']
    h = opts['h']
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]

    if h is None:
        # Compute bandwidth for each point
        h = compute_autotuned_bandwidth(neigh_ind, neigh_dist, opts['k_tune'], opts['maxiter_for_selecting_bw'])

    if ('W' in opts) and (opts['W'] is not None):
        W = opts['W']
        D = opts['D']
    else:
        # Compute kernel
        W, D = compute_self_tuned_kernel(neigh_ind, neigh_dist, h, ds=opts['ds'])

    nu_norm = compute_nu_norm(X, neigh_ind, W, d, opts['local_subspace'])
    # Initialized bx
    #bx_init = h*np.sqrt(np.maximum(0, np.log(h + EPS) - np.log(2*np.sqrt(np.pi)*nu_norm + EPS)))
    bx_init = np.zeros(len(nu_norm))
    # s_ = 0.1
    # g_hat1 = np.array(W.power(s_).sum(axis=1)).flatten()
    # nu_norm1 = compute_nu_norm(X, neigh_ind, W.power(s_), d, opts['local_subspace'])

    # h_by_sqrt_s = h/np.sqrt(s_)
    # # Initialized bx
    # bx_init1 = h_by_sqrt_s*np.sqrt(np.maximum(0, np.log(h_by_sqrt_s*g_hat1 + EPS) - np.log(2*np.sqrt(np.pi)*nu_norm1 + EPS)))
    n = len(bx_init)

    # def F(bx):
    #     rho2q = compute_rho2q_first_order(bx, h, d)
    #     m1 = compute_m1(bx, h, d)
    #     return nu_norm + (rho2q*h*m1)/(np.pi**(d/2))

    # def F_prime(bx):
    #     rho2q = compute_rho2q_first_order(bx, h, d)
    #     drho2q = compute_drho2q_first_order(bx, h, d)
    #     m1 = compute_m1(bx, h, d)
    #     dm1 = compute_dm1(bx, h, d)
    #     return (drho2q*h*m1 + rho2q*h*dm1)/(np.pi**(d/2))

    def F(bx):
        xi = compute_xi(bx, h, d)
        m1 = compute_m1(bx, h, d)
        return nu_norm*m1 + h*xi

    def F_prime(bx):
        dxi = compute_dxi(bx, h, d)
        dm1 = compute_dm1(bx, h, d)
        return nu_norm*dm1 + h*dxi

    #bx = secants_method(bx_init, bx_init+1e-2, F, opts['optimizer_maxiter'])
    
    # Optimize for bx
    if opts['optimizer'] == 'newton':
        bx = compute_bx(bx_init, F, F_prime, opts['optimizer_maxiter'])
    else:
        bx = bx_init.copy()
        lr = opts['lr']
        reg = opts['reg']
        n = bx.shape[0]
        # minimize F(bx)^2 + bx^T(I-W)bx
        for i in range(opts['optimizer_maxiter']):
            Fbx = F(bx)
            Wbx = W.dot(bx)
            bx_minus_Wbx = bx - Wbx
            #loss = np.mean(Fbx**2) + reg * np.sum(bx*bx_minus_Wbx)
            loss = np.mean(Fbx**2) 
            if loss < 1e-12:
                print('Converged at iter:', i+1)
                break
            print('Iter:', i+1, ':: loss:', loss)
            grad_bx = 2*Fbx*F_prime(bx)/n + reg*bx_minus_Wbx
            bx = bx - lr * grad_bx
            bx = np.maximum(bx, 0)
        
    return bx, bx_init, W, D, nu_norm
    
# def estimate_bx(X, opts=default_opts, ret_K_D=False):
#     d = opts['d']
#     h = opts['h']
#     ds = opts['ds']
    
#     # compute nearest neighbors
#     neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
#     neigh_dist = neigh_dist[:,1:]
#     neigh_ind = neigh_ind[:,1:]
    
#     # set h
#     if h <= 0:
#         h_cand = neigh_dist[:,opts['k_tune']-1]
#         h = np.median(h_cand)

#     print('h:', h, flush=True)
    
#     if ('K' in opts) and (opts['K'] is not None):
#         K = opts['K']
#         D = opts['D']
#     else:
#         # standard kernel
#         K = np.exp(-neigh_dist**2/h**2)+1e-24
#         n = X.shape[0]
#         source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
#         K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
#         ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        
#         # symmetrize the kernel
#         K = K + K.T
#         ones_K_like = ones_K_like + ones_K_like.T
#         K.data /= ones_K_like.data
#         # if doubly stochastic
#         if ds:
#             K = K.tocoo()
#             K, D = sinkhorn(K)
#             K = K.tocsr()
#         else:
#             D = None
    
#     # Compute ||mu_i||
    
#     if ds:
#         s = opts['s']
#     else:
#         s = None

#     mu_norm = compute_mu_norm(X, neigh_ind, K, s, ds, opts['local_pca'], d)
            
#     if ds:
#         c_num = h*(np.array((K.power(s)).sum(axis=1)).flatten())/(np.sqrt(np.pi)*np.sqrt(s))
#     else:
#         c_num = h*np.array(K.sum(axis=1)).flatten()/np.sqrt(np.pi)

#     c_denom = mu_norm
#     #c = c_num/(c_denom+1e-20)
#     #print(c_num, c_denom)

#     if ds:
#         def F(x):
#             #return c_denom*((1+erf(np.sqrt(s)*x/h)))*np.exp(s*(x**2/h**2))-c_num
#             return c_denom*((1+erf(np.sqrt(s)*x/h)))-c_num*np.exp(-s*(x**2/h**2))
#             #return np.log(c_denom/(c_num+1e-12)+1e-12) + np.log((1+erf(np.sqrt(s)*x/h)) + 1e-12) + s*(x**2/h**2)
#         def F_prime(x):
#             #return (c_denom/h)*(2*np.sqrt(s)/np.sqrt(np.pi) + 2*s*(1+erf(np.sqrt(s)*x/h))*np.exp(s*x**2/h**2)*x/h)
#             return (2*np.sqrt(s)/h)*np.exp(-s*(x**2/h**2))*(c_denom/np.sqrt(np.pi) + np.sqrt(s)*x*c_num/h) + 1e-12
#             #return (1/h)*((2*np.sqrt(s)*np.exp(-s*x**2/h**2))/(np.sqrt(np.pi)*(1+erf(np.sqrt(s)*x/h))) + 2*s*x/h)
#     else:
#         def F(x):
#             return c_denom*(1+erf(x/h))*np.exp(x**2/h**2)-c_num
#         def F_prime(x):
#             return (c_denom/h)*(2/np.sqrt(np.pi) + 2*(1+erf(x/h))*np.exp(x**2/h**2)*x/h)

#     if ds:
#         bx_init = h*np.sqrt(np.maximum(0, (np.log(c_num/(2*c_denom+1e-12)+1e-12))/s))
#     else:
#         bx_init = h*np.sqrt(np.maximum(0, -np.log(2*c_denom+1e-30)+np.log(c_num+1e-30)))
        
#     bx = optimize.newton(F, bx_init, F_prime, maxiter=opts['newton_maxiter'])
#     bx = np.maximum(bx, 0)
    
#     if ret_K_D:
#         return bx, bx_init, K, D
#     return bx, bx_init
    
def estimate_bx_self_tuned(X, opts=default_opts, ret_K_D=False, max_iter=0):
    d = opts['d']
    h = opts['h']
    ds = opts['ds']
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]
    

    #hs = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
    hs = neigh_dist[:,opts['k_tune']-1]
    #hs,_ = smooth_knn_dist(neigh_dist, opts['k_nn'], local_connectivity=d)
    cnvgd = False
    it = 0
    while (not cnvgd) and (it<max_iter):
        hs_new = np.median(hs[neigh_ind], axis=1)
        temp = np.mean(np.abs(hs_new-hs))
        print(temp)
        if temp < 1e-6:
            cnvgd = True
        hs = hs_new
        it += 1
    print('hs min max median', np.min(hs), np.max(hs), np.median(hs))
    opts['h'] = hs
        
    
    if ('K' in opts) and (opts['K'] is not None):
        K = opts['K']
        D = opts['D']
    else:
        # standard kernel
        K = np.exp(-neigh_dist**2/(hs[:,None]*hs[neigh_ind]))+1e-24
        n = X.shape[0]
        source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
        K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        
        # symmetrize the kernel
        K = K + K.T
        ones_K_like = ones_K_like + ones_K_like.T
        K.data /= ones_K_like.data
        # if doubly stochastic
        #K, hs = umap_kernel(neigh_ind, neigh_dist, opts['k_tune'])
        if ds:
            K = K.tocoo()
            K, D = sinkhorn(K)
            K = K.tocsr()
        else:
            D = None
    
    # Compute ||mu_i||
    
    if ds:
        s = opts['s']
    else:
        s = None

    mu_norm = compute_mu_norm(X, neigh_ind, K, s, ds, opts['local_pca'], d)
            
    if ds:
        c_num = hs*(np.array((K.power(s)).sum(axis=1)).flatten())/(np.sqrt(np.pi)*np.sqrt(s))
    else:
        c_num = hs*np.array(K.sum(axis=1)).flatten()/np.sqrt(np.pi)

    c_denom = mu_norm
    #c = c_num/(c_denom+1e-20)
    #print(c_num, c_denom)

    if ds:
        def F(x):
            #return c_denom*((1+erf(np.sqrt(s)*x/h)))*np.exp(s*(x**2/h**2))-c_num
            return c_denom*((1+erf(np.sqrt(s)*x/hs)))-c_num*np.exp(-s*(x**2/hs**2))
            #return np.log(c_denom/(c_num+1e-12)+1e-12) + np.log((1+erf(np.sqrt(s)*x/h)) + 1e-12) + s*(x**2/h**2)
        def F_prime(x):
            #return (c_denom/h)*(2*np.sqrt(s)/np.sqrt(np.pi) + 2*s*(1+erf(np.sqrt(s)*x/h))*np.exp(s*x**2/h**2)*x/h)
            return (2*np.sqrt(s)/hs)*np.exp(-s*(x**2/hs**2))*(c_denom/np.sqrt(np.pi) + np.sqrt(s)*x*c_num/hs) + 1e-12
            #return (1/h)*((2*np.sqrt(s)*np.exp(-s*x**2/h**2))/(np.sqrt(np.pi)*(1+erf(np.sqrt(s)*x/h))) + 2*s*x/h)
    else:
        def F(x):
            return c_denom*(1+erf(x/hs))*np.exp(x**2/hs**2)-c_num
        def F_prime(x):
            return (c_denom/hs)*(2/np.sqrt(np.pi) + 2*(1+erf(x/hs))*np.exp(x**2/hs**2)*x/hs)

    if ds:
        bx_init = hs*np.sqrt(np.maximum(0, (np.log(c_num/(2*c_denom+1e-12)+1e-12))/s))
    else:
        bx_init = hs*np.sqrt(np.maximum(0, -np.log(2*c_denom+1e-30)+np.log(c_num+1e-30)))
        
    bx = optimize.newton(F, bx_init, F_prime, maxiter=opts['newton_maxiter'])
    bx = np.maximum(bx, 0)
    
    if ret_K_D:
        return bx, bx_init, K, D
    return bx, bx_init

def estimate_bx_self_tuned_and_PTP(X, opts=default_opts, ret_K_D=False, max_iter=0):
    d = opts['d']
    h = opts['h']
    ds = opts['ds']
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]
    

    #hs = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
    hs = neigh_dist[:,opts['k_tune']-1]
    #hs,_ = smooth_knn_dist(neigh_dist, opts['k_nn'], local_connectivity=d)
    cnvgd = False
    it = 0
    while (not cnvgd) and (it<max_iter):
        hs_new = np.median(hs[neigh_ind], axis=1)
        temp = np.mean(np.abs(hs_new-hs))
        print(temp)
        if temp < 1e-6:
            cnvgd = True
        hs = hs_new
        it += 1
    print('hs min max median', np.min(hs), np.max(hs), np.median(hs))
    opts['h'] = hs
        
    
    if ('K' in opts) and (opts['K'] is not None):
        K = opts['K']
        D = opts['D']
    else:
        # standard kernel
        K = np.exp(-neigh_dist**2/(hs[:,None]*hs[neigh_ind]))+1e-24
        n = X.shape[0]
        source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
        K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        
        # symmetrize the kernel
        K = K + K.T
        ones_K_like = ones_K_like + ones_K_like.T
        K.data /= ones_K_like.data
        # if doubly stochastic
        #K, hs = umap_kernel(neigh_ind, neigh_dist, opts['k_tune'])
        if ds:
            K = K.tocoo()
            K, D = sinkhorn(K)
            K = K.tocsr()
        else:
            D = None
    
    # Compute ||mu_i||
    
    if ds:
        s = opts['s']
    else:
        s = None

    mu = compute_mu(X, neigh_ind, K, s, ds, opts['local_pca'], d)
            
    if ds:
        c_num = hs*hs*(np.array((K.power(s)).sum(axis=1)).flatten())/(np.sqrt(np.pi)*s)
    else:
        c_num = hs*np.array(K.sum(axis=1)).flatten()/np.sqrt(np.pi)

    c_denom = mu
    #c = c_num/(c_denom+1e-20)
    #print(c_num, c_denom)

    if ds:
        def F(x):
            #return c_denom*((1+erf(np.sqrt(s)*x/h)))*np.exp(s*(x**2/h**2))-c_num
            return c_denom*((1+erf(np.sqrt(s)*x/hs)))-c_num*np.exp(-s*(x**2/hs**2))
            #return np.log(c_denom/(c_num+1e-12)+1e-12) + np.log((1+erf(np.sqrt(s)*x/h)) + 1e-12) + s*(x**2/h**2)
        def F_prime(x):
            #return (c_denom/h)*(2*np.sqrt(s)/np.sqrt(np.pi) + 2*s*(1+erf(np.sqrt(s)*x/h))*np.exp(s*x**2/h**2)*x/h)
            return (2*np.sqrt(s)/hs)*np.exp(-s*(x**2/hs**2))*(c_denom/np.sqrt(np.pi) + np.sqrt(s)*x*c_num/hs) + 1e-12
            #return (1/h)*((2*np.sqrt(s)*np.exp(-s*x**2/h**2))/(np.sqrt(np.pi)*(1+erf(np.sqrt(s)*x/h))) + 2*s*x/h)
    else:
        def F(x):
            return c_denom*(1+erf(x/hs))*np.exp(x**2/hs**2)-c_num
        def F_prime(x):
            return (c_denom/hs)*(2/np.sqrt(np.pi) + 2*(1+erf(x/hs))*np.exp(x**2/hs**2)*x/hs)

    if ds:
        bx_init = hs*np.sqrt(np.maximum(0, (np.log(c_num/(2*c_denom+1e-12)+1e-12))/s))
    else:
        bx_init = hs*np.sqrt(np.maximum(0, -np.log(2*c_denom+1e-30)+np.log(c_num+1e-30)))
        
    bx = optimize.newton(F, bx_init, F_prime, maxiter=opts['newton_maxiter'])
    bx = np.maximum(bx, 0)
    
    if ret_K_D:
        return bx, bx_init, K, D
    return bx, bx_init

def compute_SRCC(x, y, prctile=10):
    x = x.copy()
    y = y.copy()

    mask_x = x < np.percentile(x, prctile)
    
    x = x[mask_x]
    y = y[mask_x]
    #print('len(x) =',len(x), 'len(y) =', len(y))
    #min_len = min(len(x), len(y))
    
    assert len(x) == len(y)
    return spearmanr(x, y)


def estimate_q(X, opts=default_opts, bx=None):
    ds = opts['ds']
    s = opts['s']
    h = opts['h']
    d = opts['d']
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]
    
    # set h
    if h <= 0:
        h_cand = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
        h = np.min(h_cand)
        
    print('h:', h, flush=True)
    
    # standard kernel
    K = np.exp(-neigh_dist**2/h**2)
    n = X.shape[0]
    source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
    K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
    ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))

    # symmetrize the kernel
    K = K + K.T
    ones_K_like = ones_K_like + ones_K_like.T
    K.data /= ones_K_like.data
    
    K_old = K.copy()
    
    # if doubly stochastic
    if ds:
        K = K.tocoo()
        K, D = sinkhorn(K)
        K = K.tocsr()
        print('s:', s, flush=True)
        
    if bx is not None:
        if ds:
            if opts['q_est_type'] == 1:
                m0_h = compute_m0(bx, h)
                m0_h_s = compute_m0(bx, h/np.sqrt(s))
                m0 = ((m0_h**(-s))*m0_h_s)
            elif opts['q_est_type'] == 2:
                m0_h = compute_m0(bx, h/np.sqrt(s))
            else:
                m0_h = compute_m0(bx, h)
        else:
            m0 = compute_m0(bx, h)
    else:
        m0 = 1
    
    n = X.shape[0]
    if ds:
        if opts['q_est_type'] == 1:
            Z = ((n-1)**s)*(h**(d*(s-1)))*(s**(d/2))
            #Z = (((m0_h**(d/2))*(n-1))**s)*(h**(d*(s-1)))*(s**(d/2))
            f = (np.array((K.power(s)).sum(axis=1)).flatten()/(n-1))*Z/m0
            q = f**(1/(1-s))
        elif opts['q_est_type'] == 1:
            f = (s**(d/2)) * np.array((K.power(s)).sum(axis=1)).flatten()/(D.power(2*s))
            q = f/((np.pi**(d*s*0.5)) * (h**d) * m0_h)
        else:
            rho = D*np.sqrt((n-1)*((np.pi*h**2)**(d/2)))
            q = (rho**2)*(np.pi**(d/2))*m0_h
    else:
        q = np.array(K_old.sum(axis=1)).flatten()/((n-1)*h**d)
        q = q/m0
        
    return q

def estimate_bx_new(X, opts=default_opts, ret_K_D=False):
    d = opts['d']
    h = opts['h']
    H = opts['H']
    
    def compute_D(h_scale):
        h = opts['h']
        # compute nearest neighbors
        neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
        neigh_dist = neigh_dist[:,1:]
        neigh_ind = neigh_ind[:,1:]

        # set h
        if h <= 0:
            h_cand = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
            h = np.min(h_cand)

        h = h*h_scale
        print('h:', h, flush=True)

        # standard kernel
        K = np.exp(-neigh_dist**2/h**2)
        n = X.shape[0]
        source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
        K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))

        # symmetrize the kernel
        K = K + K.T
        ones_K_like = ones_K_like + ones_K_like.T
        K.data /= ones_K_like.data
        
        K = K.tocoo()
        K, D = sinkhorn(K)
        return D, h
    
    D_h, h = compute_D(1)
    D_hb2, hb2 = compute_D(0.5)
    
    D_h = (h**(d/2)) * D_h
    D_hb2 = (hb2**(d/2)) * D_hb2
    
    def F(bx):
        C_hb2_num, C_hb2_denom = compute_C(bx, hb2, d, H)
        C_h_num, C_h_denom = compute_C(bx, h, d, H)
        return D_h*C_hb2_num*C_h_denom - D_hb2*C_h_num*C_hb2_denom
        
    np.random.seed(42)
    bx_init = 0.01 * np.random.uniform(0, 1, D_h.shape[0])
    bx = optimize.newton(F, bx_init, maxiter=opts['newton_maxiter'])
    bx = np.maximum(bx, 0)
    return bx

def estimate_Hx(X, bx, opts=default_opts, ret_K_D=False):
    d = opts['d']
    h = opts['h']

    def compute_D(h_scale):
        h = opts['h']
        # compute nearest neighbors
        neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
        neigh_dist = neigh_dist[:,1:]
        neigh_ind = neigh_ind[:,1:]

        # set h
        if h <= 0:
            h_cand = np.sqrt(2)*neigh_dist[:,opts['k_tune']-2]/np.sqrt(chi2.ppf(opts['p'], df=d))
            h = np.min(h_cand)

        h = h*h_scale
        print('h:', h, flush=True)

        # standard kernel
        K = np.exp(-neigh_dist**2/h**2)
        n = X.shape[0]
        source_ind = np.repeat(np.arange(n),neigh_ind.shape[1])
        K = coo_matrix((K.flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))
        ones_K_like = coo_matrix((np.ones(neigh_dist.shape).flatten(),(source_ind, neigh_ind.flatten())),shape=(n,n))

        # symmetrize the kernel
        K = K + K.T
        ones_K_like = ones_K_like + ones_K_like.T
        K.data /= ones_K_like.data
        
        K = K.tocoo()
        K, D = sinkhorn(K)
        return D, h
    
    D_h, h = compute_D(1)
    D_hb2, hb2 = compute_D(0.5)
    
    D_h = (h**(d/2)) * D_h
    D_hb2 = (hb2**(d/2)) * D_hb2
    
    def F(H):
        C_hb2_num, C_hb2_denom = compute_C(bx, hb2, d, H)
        C_h_num, C_h_denom = compute_C(bx, h, d, H)
        return D_h*C_hb2_num*C_h_denom - D_hb2*C_h_num*C_hb2_denom
        
    H_init = np.zeros(D_h.shape[0])
    H = optimize.newton(F, H_init, maxiter=opts['newton_maxiter'])
    return H