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
import time
import torch
import torch.nn as nn
from scipy.sparse.csgraph import dijkstra

import multiprocess as mp

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
    'ds': True,
    'no_newton': True
}

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


def compute_m0(bx, h, d=2):
    return 0.5*(np.pi**(d/2))*(1+erf(bx/h))

def compute_dm0(bx, h, d=2):
    return (np.pi**((d-1)/2))*np.exp(-(bx/h)**2)/h

def compute_m1(bx, h, d=2):
    return -0.5*(np.pi**((d-1)/2))*np.exp(-(bx/h)**2)

def compute_logmm1(bx, h, d=2):
    return np.log(0.5*(np.pi**((d-1)/2))) - ((bx/h)**2)

def compute_dm1(bx, h, d=2):
    return (np.pi**((d-1)/2))*np.exp(-(bx/h)**2)*(bx/h)/h

def compute_m2(bx, h, d=2):
    return (bx/h)*compute_m1(bx,h,d) + 0.5*compute_m0(bx,h,d)

def compute_dm2(bx, h, d=2):
    return (bx/h)*compute_dm1(bx,h,d) + compute_m1(bx,h,d)/h + 0.5*compute_dm0(bx,h,d)

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

def compute_rho_from_zeta(bx, h, d=2, q=1):
    rho2q = compute_zeta(bx, h, d=2)
    return np.sqrt(rho2q/q)

def compute_rho_naive(bx, h, d=2):
    m0 = compute_m0(bx, h, d)
    return (np.pi**(d/2))/m0

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

def compute_beta(bx, h, d=2):
    m1 = compute_m1(bx, h, d)
    m2 = compute_m2(bx, h, d)
    zeta = compute_zeta(bx, h, d)
    return (m1*zeta)/(np.pi**(d/2) + 2*m2*zeta)

def compute_dbeta(bx, h, d=2):
    m1 = compute_m1(bx, h, d)
    m2 = compute_m2(bx, h, d)
    zeta = compute_zeta(bx, h, d)

    dm1 = compute_dm1(bx, h, d)
    dm2 = compute_dm2(bx, h, d)
    dzeta = compute_dzeta(bx, h, d)

    num = m1*zeta
    dnum = (dm1*zeta + m1*dzeta)
    denom = np.pi**(d/2) + 2*m2*zeta
    ddenom = 2*(dm2*zeta + zeta*dm2)
    
    return (dnum*denom-ddenom*num)/(denom**2 + 1e-20)
    
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

# def compute_nu_norm(X, neigh_ind, K, d=None, local_subspace=None, n_proc=32):
#     n = X.shape[0]
#     nu_norm = np.zeros(n)
#     chunk_sz = int(n/n_proc)
#     def target_proc(p_num, q_):
#         start_ind = p_num*chunk_sz
#         if p_num == (n_proc-1):
#             end_ind = n
#         else:
#             end_ind = (p_num+1)*chunk_sz
#         nu_norm_ = np.zeros(end_ind-start_ind)
#         for i in range(start_ind, end_ind):
#             if local_subspace is not None:
#                 if type(local_subspace) == str:
#                     X_i_nbrs = X[neigh_ind[i,:].tolist()+[i],:]
#                     pca = PCA(n_components=d)
#                     y = pca.fit_transform(X_i_nbrs)
#                     temp = y[:-1,:] - y[-1,:][None,:]
#                 else:
#                     if d in local_subspace.shape:
#                         Q_k,Sigma_k,_ = svd(local_subspace[:,i,:].T)
#                     else:
#                         Q_k,Sigma_k,_ = svds(local_subspace[:,i,:].T, d, which='LM')
#                     Q_k = Q_k[:,:d]
#                     temp = (X[neigh_ind[i,:],:] -  X[i,:][None,:]).dot(Q_k)
#             else:
#                 temp = X[neigh_ind[i,:],:]-X[i,:][None,:]
#             nu_norm_[i-start_ind] = np.linalg.norm(K.getrow(i)[0,neigh_ind[i,:]].dot(temp))
#             #nu_norm[i] = np.linalg.norm(np.sum(K[i,:][:,None]*temp, axis=0))
        
#         q_.put((start_ind, end_ind, nu_norm_))
    
#     q_ = mp.Queue()
#     proc = []
#     for p_num in range(n_proc):
#         proc.append(mp.Process(target=target_proc, args=(p_num, q_)))
#         proc[-1].start()
        
#     for p_num in range(n_proc):
#         start_ind, end_ind, nu_norm_ = q_.get()
#         nu_norm[start_ind:end_ind] = nu_norm_
#     q_.close()
    
#     for p_num in range(n_proc):
#         proc[p_num].join()
#     return nu_norm
    
def compute_nu_norm(X, neigh_ind, K, d=None, local_subspace=None, n_proc=32):
    n = X.shape[0]
    nu_norm = np.zeros(n)
    for i in range(n):
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
    # make it doubly stochastic
    if ds:
        K, D = sinkhorn(K)
    else:
        D = 1/np.sqrt(np.array(K.sum(axis=1)).flatten())

    K = K.tocsr()
    return K, D

def newtons_method(x0, F, F_prime, maxiter, tol=1e-8, print_freq=250):
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

def optimize_for_bx(bx_init, F, F_prime, maxiter):
    #bx = optimize.newton(F, bx_init, F_prime, maxiter=maxiter)
    bx = newtons_method(bx_init, F, F_prime, maxiter)
    bx = np.maximum(bx, 0)
    return bx

def compute_jaccard_index(ddX, bx, prctile=10):
    mask0 = ddX <= np.percentile(ddX, prctile)
    mask = bx <= np.percentile(bx, prctile)
    return np.sum(mask*mask0)/np.sum((mask+mask0) > 0)

def compute_distances_from_boundary(d_e, bx, percentiles=[10]):
    max_percentile = np.max(percentiles)
    mask = bx <= np.percentile(bx, max_percentile)
    pts_on_boundary = np.where(mask)[0]
    dist = dijkstra(d_e, directed=False, indices=pts_on_boundary)
    inv_ind_map = np.zeros(len(mask), dtype=int)+len(bx)+1
    inv_ind_map[pts_on_boundary] = np.arange(len(pts_on_boundary))
                                             
    dist_from_pts_on_boundary = []
    for i in range(len(percentiles)):
        mask_i = bx <= np.percentile(bx, percentiles[i])
        pts_on_boundary_i = np.where(mask_i)[0]
        dist_from_pts_on_boundary.append(np.min(dist[inv_ind_map[pts_on_boundary_i],:], axis=0))
    return np.array(dist_from_pts_on_boundary)

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
        h = np.median(h)

    # Compute kernel
    K, D = compute_self_tuned_kernel(neigh_ind, neigh_dist, h)

    # Setup the function whose root is needed
    g_hat = np.array(K.sum(axis=1)).flatten()
    if opts['only_kde']:
        return g_hat
        
    nu_norm = compute_nu_norm(X, neigh_ind, K, d, opts['local_subspace'])
    #nu_norm = opts['h']*nu_norm/(np.max(nu_norm/g_hat) * np.sqrt(np.pi))
    # if opts['no_newton']:
    #     return 1/(nu_norm/g_hat+1), g_hat, nu_norm

    def F(bx):
        return h * g_hat * compute_m1(bx, h, d) + nu_norm * compute_m0(bx, h, d) # plus because m1 is negative
    def F_prime(bx):
        return h * g_hat * compute_dm1(bx, h, d) + nu_norm * compute_dm0(bx, h, d)

    # Initialized bx
    bx_init = h*np.sqrt(np.maximum(0, np.log(h*g_hat + EPS) - np.log(2*np.sqrt(np.pi)*nu_norm + EPS)))

    # Optimize for bx
    bx = optimize_for_bx(bx_init, F, F_prime, opts['optimizer_maxiter'])
    return bx, bx_init, nu_norm

def estimate_bx(X, opts=default_opts):
    d = opts['d']
    h = opts['h']
    n = X.shape[0]
    
    # compute nearest neighbors
    neigh_dist, neigh_ind = util_.nearest_neighbors(X, opts['k_nn'], metric='euclidean')
    neigh_dist = neigh_dist[:,1:]
    neigh_ind = neigh_ind[:,1:]

    if h is None:
        # Compute bandwidth for each point
        h = compute_autotuned_bandwidth(neigh_ind, neigh_dist, opts['k_tune'], opts['maxiter_for_selecting_bw'])
        h = np.median(h)

    if ('W' in opts) and (opts['W'] is not None):
        W = opts['W']
        D = opts['D']
    else:
        # Compute kernel
        W, D = compute_self_tuned_kernel(neigh_ind, neigh_dist, h, ds=opts['ds'])

    if opts['only_kde']:
        g_hat = np.array(W.power(opts['s']).sum(axis=1)).flatten()/(n-1)
        return g_hat
    
    nu_norm = compute_nu_norm(X, neigh_ind, W, d, opts['local_subspace'])

    if opts['no_newton']:
        return 1/(nu_norm+1), 1/(nu_norm+1), W, D, nu_norm
    
    nu_norm = nu_norm*opts['h']*0.5*(np.sqrt(np.pi)-np.sqrt(np.pi-2))/np.max(nu_norm)
    # Initialized bx
    bx_init = np.zeros(n)

    def F(bx):
        m1 = compute_m1(bx, h, d)
        m2 = compute_m2(bx, h, d)
        zeta = compute_zeta(bx, h, d)
        c = np.pi**(d/2)
        return nu_norm*(c+2*m2*zeta) + h*m1*zeta

    def F_prime(bx):
        m1 = compute_m1(bx, h, d)
        m2 = compute_m2(bx, h, d)
        zeta = compute_zeta(bx, h, d)
        dm1 = compute_dm1(bx, h, d)
        dm2 = compute_dm2(bx, h, d)
        dzeta = compute_dzeta(bx, h, d)
        return 2*nu_norm*(dm2*zeta+m2*dzeta) + h*(dm1*zeta+m1*dzeta)

    # def F(bx):
    #     xi = compute_xi(bx, h, d)
    #     m1 = compute_m1(bx, h, d)
    #     return nu_norm*m1 + h*xi

    # def F_prime(bx):
    #     dxi = compute_dxi(bx, h, d)
    #     dm1 = compute_dm1(bx, h, d)
    #     return nu_norm*dm1 + h*dxi
    # def F(bx):
    #     xi = compute_xi(bx, h, d)
    #     m1 = compute_m1(bx, h, d)
    #     logmm1 = np.log(-m1)
    #     logxi = np.log(xi)
    #     return nu_norm - h*np.exp(logxi - logmm1)

    # def F_prime(bx):
    #     dxi = compute_dxi(bx, h, d)
    #     dm1 = compute_dm1(bx, h, d)
    #     xi = compute_xi(bx, h, d)
    #     m1 = compute_m1(bx, h, d)
    #     logmm1 = np.log(-m1)
    #     logxi = np.log(xi)
    #     return h*(np.exp(np.log(-dxi)-logmm1) - dm1 * np.exp(logxi - 2*logmm1))

    # Optimize for bx
    if opts['optimizer'] == 'newton':
        bx = optimize_for_bx(bx_init, F, F_prime, opts['optimizer_maxiter'])
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