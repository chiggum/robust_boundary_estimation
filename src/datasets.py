import pdb

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
from sklearn.datasets import fetch_openml
import scipy
from scipy.spatial.distance import cdist
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
import os
import scipy.misc
import matplotlib.image as mpimg
from scipy.stats import truncnorm

def read_img(fpath, grayscale=False, bbox=None):
    if grayscale:
        img = ImageOps.grayscale(Image.open(fpath))
    else:
        img = Image.open(fpath)
    if bbox is not None:
        return np.asarray(img.crop(bbox).reduce(2))
    else:
        return np.asarray(img.reduce(2))
    
def do_pca(X, n_pca):
    print('Applying PCA')
    pca = PCA(n_components=n_pca, random_state=42)
    pca.fit(X)
    #print('explained_variance_ratio:', pca.explained_variance_ratio_)
    print('sum(explained_variance_ratio):', np.sum(pca.explained_variance_ratio_))
    #print('singular_values:', pca.singular_values_)
    X = pca.fit_transform(X)
    return X

class Datasets:
    def __init__(self):
        pass
    
    def circular_disk(self, RES=100, noise=0, noise_type='uniform'):
        sideLx = 2
        sideLy = 2
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)-sideLx/2
        y = np.linspace(0, sideLy, RESy)-sideLy/2
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')
        yv = yv.flatten('F')
        mask = (xv**2 + yv**2) < 1
        xv = xv[mask][:,None]
        yv = yv[mask][:,None]
        X = np.concatenate([xv,yv], axis=1)
        if noise:
            np.random.seed(42)
            n = xv.shape[0]
            if noise_type == 'normal':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = 1-np.sqrt(X[:,0]**2+X[:,1]**2)
        return X, labelsMat, ddX
    
    def circular_disk_uniform(self, n=10000, noise=0, noise_type='uniform', seed=42):
        np.random.seed(seed)
        xv = np.random.uniform(-1, 1, n)
        yv = np.random.uniform(-1, 1, n)
        mask = (xv**2 + yv**2) < 1
        xv = xv[mask][:,None]
        yv = yv[mask][:,None]
        X = np.concatenate([xv,yv], axis=1)
        if noise:
            np.random.seed(42)
            n = xv.shape[0]
            if noise_type == 'gaussian':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X[:,-1] = X[:,-1] + noise*np.random.normal(0,1,(n))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = 1-np.sqrt(X[:,0]**2+X[:,1]**2)
        return X, labelsMat, ddX
    
    def annulus_non_uniform(self, n=1000, inner_r=0.3, seed=42, noise=0, noise_type='uniform'):
        np.random.seed(seed)
        sigma = 0.5*np.pi
        theta = np.mod(sigma*np.random.normal(0, 1, n), 2*np.pi)
        X = np.zeros((n,2))
        X[:,0] = np.cos(theta)
        X[:,1] = np.sin(theta)
        r = np.random.uniform(inner_r,1,n)
        ddX = np.minimum(1 - r, r - 0.3)
        X= X*r[:,None]
        q_true = np.exp(-(np.minimum(theta, 2*np.pi-theta)**2)/(2*sigma**2))/r
        labelsMat = X.copy()
        if noise:
            np.random.seed(42)
            if noise_type == 'gaussian':
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X[:,-1] = X[:,-1] + noise*np.random.normal(0,1,(n))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        return X, labelsMat, ddX, q_true 
    
    def annulus(self, n=1000, inner_r=0.3, outer_r=1, seed=42, noise=0, noise_type='uniform'):
        X = []
        labelsMat = []
        ddX = []
        cur_n = 0
        while cur_n < n:
            X_, labelsMat_, ddX_ = self.circular_disk_uniform(n=n, noise=0, noise_type=noise_type, seed=cur_n)
            r = np.linalg.norm(X_, axis=1)
            max_r = np.max(r)
            X_ = outer_r*X_/max_r
            r = outer_r*r/max_r

            mask = r > inner_r
            X_ = X_[mask,:]
            labelsMat_ = labelsMat_[mask,:]
            r = r[mask]
            ddX_ = np.minimum(outer_r - r, r - inner_r)
            X.append(X_)
            labelsMat.append(labelsMat_)
            ddX.append(ddX_)

            cur_n += X_.shape[0]

        X = np.concatenate(X, axis=0)[:n,:]
        labelsMat = np.concatenate(labelsMat, axis=0)[:n,:]
        ddX = np.concatenate(ddX)[:n]
        print(X.shape)

        if noise:
            np.random.seed(42)
            if noise_type == 'gaussian':
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X[:,-1] = X[:,-1] + noise*np.random.normal(0,1,(n))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        return X, labelsMat, ddX 
    
    def wave_on_circle(self, RES=25, R_in=2, R_out=3, r=1/2, f=8):
        sideLx = int(np.ceil(2*(R_out+r)))
        sideLy = int(np.ceil(2*(R_out+r)))
        RESx = sideLx*RES+1
        RESy = sideLy*RES+1
        x = np.linspace(-sideLx/2,sideLx/2,RESx);
        y = np.linspace(-sideLy/2,sideLy/2,RESy);
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')
        yv = yv.flatten('F')
        
        theta = np.arctan2(yv, xv)
        radius = np.sqrt(xv**2 + yv**2)
        R_out_at_theta = R_out+r*np.sin(f*theta)
        R_in_at_theta = R_in+r*np.sin(f*theta)
        mask = (radius <= R_out_at_theta) & (radius >= R_in_at_theta)
        
        xv = xv[mask][:,None]
        yv = yv[mask][:,None]
        X = np.concatenate([xv,yv], axis=1)
        
        theta = theta[mask][:,None]
        radius = radius[mask][:,None]
        
        labelsMat = np.concatenate([radius, theta], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    
    def curvedtorus3d(self, n=10000, noise=0, noise_type='uniform',
                       Rmax=0.25, seed=42, freq=4, density='uniform', rmax=None):
        if rmax is None:
            rmax=1/(4*(np.pi**2)*Rmax)
        X = []
        thetav = []
        phiv = []
        np.random.seed(seed)
        k = 0
        sigma = 0.75*np.pi
        while k < n:
            rU = np.random.uniform(0,1,3)
            if density != 'uniform':
                #theta = np.mod(sigma*np.random.normal(0, 1), 2*np.pi)
                theta = truncnorm.rvs(a=-np.pi/sigma,b=np.pi/sigma,scale=sigma)
            else:
                theta = 2*np.pi*rU[0]
            phi = 2*np.pi*rU[1]
            
            if rU[2] <= (Rmax + rmax*np.cos(theta))/(Rmax + rmax):
                thetav.append(theta)
                phiv.append(phi)
                k = k + 1
        
        thetav = np.array(thetav)[:,None]
        phiv = np.array(phiv)[:,None]

        if density != 'uniform':
            #q_true = np.exp(-(np.minimum(thetav, 2*np.pi-thetav)**2)/(2*sigma**2))
            q_true = truncnorm.pdf(thetav, a=-np.pi/sigma,b=np.pi/sigma,scale=sigma)
            #q_true = q_true/(Rmax+rmax*np.cos(thetav))
        else:
            q_true = 1
        dX = None

        np.random.seed(42)
        if noise_type == 'uniform':
            noise = noise*np.random.uniform(-1,1,(phiv.shape[0],1))
        elif noise_type == 'gaussian':
            noise = noise*np.random.normal(0,1,(phiv.shape[0],1))
        else:
            noise_u = 0.01 + 0.3*(1+np.cos(freq*phiv))/2
            noise_u = np.random.uniform(-noise_u,noise_u)
            noise = noise*noise_u
        X = np.concatenate([(Rmax+(1+noise)*rmax*np.cos(thetav))*np.cos(phiv),
                             (Rmax+(1+noise)*rmax*np.cos(thetav))*np.sin(phiv),
                             (1+noise)*rmax*np.sin(thetav)], axis=1)
        labelsMat = np.concatenate([np.mod(thetav, 2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, dX, q_true
    
    def curvedtorus3d_with_normal_dir(self, n=10000, density='uniform', seed=42):
        X_noisy, X, labelsMat, dX, (X_theta, X_phi, normal_dir) = self.wave_on_curvedtorus3d(
            n=n, noise=0, noise_type='ortho',
            seed=seed, density=density,
            wave_amp_r=0, wave_freq_r=0, wave_amp_R=0,
            wave_freq_R=0, rmax=None)
        return X, labelsMat, normal_dir


    def wave_on_curvedtorus3d(self, n=10000, noise=0, noise_type='ortho',
                       Rmax=0.25, seed=42, freq=4, density='uniform',
                       wave_amp_r=0.2, wave_freq_r=5, wave_amp_R=0.1, wave_freq_R=3, rmax=None):
        if rmax is None:
            rmax=1/(4*(np.pi**2)*Rmax)

        theta = np.pi
        phi = (3*np.pi)/(2*wave_freq_r + 1e-12)
        r_ = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phi)
        R_ = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * theta)

        r_prime_ = wave_freq_r * wave_amp_r * np.cos(wave_freq_r * phi)
        R_prime_ = wave_freq_R * wave_amp_R * np.cos(wave_freq_R * theta)
        
        X_theta_ = np.array([(R_prime_ - r_*np.sin(theta))*np.cos(phi),
                                (R_prime_ - r_*np.sin(theta))*np.sin(phi),
                                r_*np.cos(theta)])
        X_phi_ = np.array([r_prime_*np.cos(theta)*np.cos(phi) - (R_+r_*np.cos(theta))*np.sin(phi),
                            r_prime_*np.cos(theta)*np.sin(phi) + (R_+r_*np.cos(theta))*np.cos(phi),
                            r_prime_*np.sin(theta)])
        X_theta_cross_X_phi_max = np.linalg.norm(np.cross(X_theta_, X_phi_))

        X = []
        thetav = []
        phiv = []
        np.random.seed(seed)
        k = 0
        sigma = 0.75*np.pi
        while k < n:
            rU = np.random.uniform(0,1,3)
            if density != 'uniform':
                #theta = np.mod(sigma*np.random.normal(0, 1), 2*np.pi)
                theta = truncnorm.rvs(a=-np.pi/sigma,b=np.pi/sigma,scale=sigma)
            else:
                theta = 2*np.pi*rU[0]
            phi = 2*np.pi*rU[1]

            r_ = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phi)
            R_ = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * theta)

            r_prime_ = wave_freq_r * wave_amp_r * np.cos(wave_freq_r * phi)
            R_prime_ = wave_freq_R * wave_amp_R * np.cos(wave_freq_R * theta)
            
            X_theta_ = np.array([(R_prime_ - r_*np.sin(theta))*np.cos(phi),
                                 (R_prime_ - r_*np.sin(theta))*np.sin(phi),
                                  r_*np.cos(theta)])
            X_phi_ = np.array([r_prime_*np.cos(theta)*np.cos(phi) - (R_+r_*np.cos(theta))*np.sin(phi),
                               r_prime_*np.cos(theta)*np.sin(phi) + (R_+r_*np.cos(theta))*np.cos(phi),
                               r_prime_*np.sin(theta)])
            X_theta_cross_X_phi = np.linalg.norm(np.cross(X_theta_, X_phi_))
            X_theta_cross_X_phi = X_theta_cross_X_phi/X_theta_cross_X_phi_max

            #if rU[2] <= (Rmax + rmax*np.cos(theta))/(Rmax + rmax):
            #if rU[2] <= (R_ + r_*np.cos(theta))/(R_ + r_):
            if rU[2] <= X_theta_cross_X_phi:
                thetav.append(theta)
                phiv.append(phi)
                k = k + 1
        
        thetav = np.array(thetav)[:,None]
        phiv = np.array(phiv)[:,None]
        dX = None

        r = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phiv)
        R = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * thetav)
        X = np.concatenate([(R+r*np.cos(thetav))*np.cos(phiv),
                             (R+r*np.cos(thetav))*np.sin(phiv),
                              r*np.sin(thetav)], axis=1)
        
        np.random.seed(42)
        if 'uniform' in noise_type:
            noise = noise*np.random.uniform(-1,1,(X.shape[0],1))
        elif 'gaussian' in noise_type:
            noise = noise*np.random.normal(0,1,(X.shape[0],1))
        else:
            #noise_u = 0.01 + 0.3*(1+np.cos(freq*phiv))/2
            noise_u = np.cos(freq*phiv)**2
            noise_u = np.random.uniform(-noise_u,noise_u)
            noise = noise*noise_u

        r_prime = wave_freq_r * wave_amp_r * rmax * np.cos(wave_freq_r * phiv)
        R_prime = wave_freq_R * wave_amp_R * Rmax * np.cos(wave_freq_R * thetav)
        X_theta = np.concatenate([(R_prime - r*np.sin(thetav))*np.cos(phiv),
                                (R_prime - r*np.sin(thetav))*np.sin(phiv),
                                   r*np.cos(thetav)], axis=1)
        X_phi = np.concatenate([r_prime*np.cos(thetav)*np.cos(phiv) - (R+r*np.cos(thetav))*np.sin(phiv),
                                r_prime*np.cos(thetav)*np.sin(phiv) + (R+r*np.cos(thetav))*np.cos(phiv),
                                r_prime*np.sin(thetav)], axis=1)
        normal_dir = np.cross(X_theta, X_phi)
        normal_dir = normal_dir/np.linalg.norm(normal_dir, axis=1)[:,None]
        X_noisy = X + noise * normal_dir

        X_theta = X_theta/np.linalg.norm(X_theta,axis=1)[:,None]
        X_phi = X_phi/np.linalg.norm(X_phi,axis=1)[:,None]

        labelsMat = np.concatenate([np.mod(thetav, 2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X_noisy, X, labelsMat, dX, (X_theta, X_phi, normal_dir)
    
    def mnist(self, digits, n, n_pca=25, scale=True):
        X0, y0 = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = []
        y = []
        for digit in digits:
            X_ = X0[y0 == str(digit),:]
            X_= X_[:n,:]
            X.append(X_)
            y.append(np.zeros(n)+digit)
            
        X = np.concatenate(X, axis=0)
        if scale:
            X = X/np.max(np.abs(X))
        y = np.concatenate(y, axis=0)
        labelsMat = y[:,None]
        
        if n_pca:
            X_new = do_pca(X,n_pca)
        else:
            X_new = X
        
        print('X_new.shape = ', X_new.shape)
        return X_new, labelsMat, X, [28,28] 
    