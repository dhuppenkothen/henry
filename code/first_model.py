# Fit simple model with fixed hypers and basis functions, so we're just learning
# weights.

from __future__ import print_function

import numpy as np
import scipy as sp
import scipy.integrate
import utils
from simple_slice import slice_sample
import matplotlib.pyplot as plt

import collections
CHypers = collections.namedtuple('CHypers',
        ['pi_mu', 'pi_std', 'K', 'cc', 'ell'])

def log_intensity(tt, ww, hypers):
    Phi = np.zeros((hypers.K, np.array(tt).size))
    for kk in range(hypers.K):
        Phi[kk] = np.exp(-0.5*(tt - hypers.cc[kk])**2/hypers.ell**2)
    return hypers.pi_mu + np.dot(ww, Phi)

def log_prior(ww, hypers):
    """
    Log prior probability of basis weights up to a constant

    hypers has members:
        hypers.pi_std standard deviation of params

    WARNING: currently missing off constant wrt ww that depends on hypers
    """
    return -0.5*np.dot(ww, ww)/(hypers.pi_std**2)

def log_likelihood(ww, hypers, tt, t_obs, bin_c=None, bin_w=None):
    """
    Log likelihood of basis weights given Poisson time data

        ww     K, coefficients of basis functions
        hypers has members:
            hypers.pi_mu  mean intensity
            hypers.K      number of basis functions
            hypers.cc     basis function centers
            hypers.ell    basis function widths
        tt     N, array of event times
        t_obs L,2 start and end times of L intervals with observations

        Optional: bin_c, bin_w, bin centers and widths with which to approx integral
    """
    # L = - \int \lamba(t) dt + \sum_{n=1}^N \log \lambda(t_n)
    if bin_c is not None:
        integral = np.sum(np.exp(log_intensity(bin_c, ww, hypers))*bin_w)
        #print(integral)
    else:
        func = lambda t: np.exp(log_intensity(t, ww, hypers))
        integral = 0.0
        for ll in range(len(t_obs)):
            integral += scipy.integrate.quad(
                    func, t_obs[ll,0], t_obs[ll,1], epsabs=1e-2, epsrel=1e-2)[0]
        #print(integral)
        #print('---')
    return -integral + np.sum(log_intensity(tt, ww, hypers))

# Grab the data:
_, tt, _, t_obs = utils.load_gbm_bursts('../data/')

# Time range:
T0 = t_obs[0,0]
Tend = t_obs[-1,1]
T = Tend - T0
#ell = T / 10.0 # lengthscale
ell = T / 40.0 # lengthscale
# Basis function centres:
c0 = T0 - ell*5
cc = np.arange(T0 - ell*5, Tend + ell*5, ell)
K = len(cc) # Number of basis functions

hypers = CHypers(pi_mu=-5, pi_std=4.0, K=K, cc=cc, ell=ell)

# obs_bins for integral hack
bin_w = t_obs[:,1] - t_obs[:,0]
bin_c = t_obs[:,0] + bin_w/2
bps = 10
bin_c = np.zeros(bps * len(t_obs))
bin_w = np.zeros_like(bin_c)
for i in range(len(t_obs)):
    bw = (t_obs[i,1] - t_obs[i,0]) / bps
    bin_w[i*bps:((i+1)*bps)] = bw
    bin_c[i*bps:((i+1)*bps)] = np.linspace(
            t_obs[i,0] + bw/2, t_obs[i,1]-bw/2, bps)
#logdist = lambda w: log_prior(w, hypers) + \
#        log_likelihood(w, hypers, tt, t_obs)
logdist = lambda w: log_prior(w, hypers) + \
        log_likelihood(w, hypers, tt, t_obs, bin_c, bin_w)

ww = np.zeros(K)
S = 100
samples = slice_sample(
        ww, logdist, widths=hypers.pi_std,
        N=S, burn=0, step_out=True, verbose=2) # S,K

ww_end = samples[-1]

# to plot things, bin things up:
Nbins = 300
tbins = np.linspace(T0, Tend, Nbins)
intensity = np.exp(log_intensity(tbins, ww_end, hypers))
plt.clf()
h, bins, patches = plt.hist(tt, bins=500)
bin_size = bins[1]-bins[0]
plt.plot(tbins, intensity*bin_size, 'r', linewidth=3)
plt.show()

