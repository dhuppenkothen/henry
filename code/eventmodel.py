import numpy as np
import scipy.integrate

import utils
from simple_slice import slice_sample
import matplotlib.pyplot as plt

import collections
CHypers = collections.namedtuple('CHypers',
        ['pi_mu', 'pi_std', 'K', 'cc', 'ell'])



class EventModel(object):

    def __init__(self, tt, t_obs, hypers, bin_integral=False, bin_c=None, bin_w=None):
        """
        Initialize EventModel object.
        :param tt: event times
        :param t_obs: start and end times of the observed segments
        :param hypers: named tuple with attributes ['pi_mu', 'pi_std', 'K', 'cc', 'ell']
            hypers has members:
                hypers.pi_mu  mean intensity
                hypers.pi_std standard deviation of params
                hypers.K      number of basis functions
                hypers.cc     basis function centers
                hypers.ell    basis function widths
        :param bin_integral: Boolean, decide whether to integrate by quadrature (False) or by binning (True)
        :param bin_c: if bin_integral is True, this will set the centers of the bins to do the integral
        :param bin_w: if bin_integral is True, this will set the widths of the bins to use for integration

        :return:
        """

        self.tt = tt
        self.t_obs = t_obs

        self.K = hypers.K
        self.cc = hypers.cc
        self.ell = hypers.ell
        self.pi_mu = hypers.pi_mu
        self.pi_std = hypers.pi_std

        self.bin_integral = bin_integral
        self.bin_c = bin_c
        self.bin_w = bin_w

        return

    def log_intensity(self, times, ww):
        """
        Compute the log-intensity for the Gaussian basis functions
        :param ww:
        :return:
        """

        Phi = np.zeros((self.K, np.array(times).size))
        for kk in range(self.K):
            Phi[kk] = np.exp(-0.5*(times - self.cc[kk])**2/self.ell**2)
        return self.pi_mu + np.dot(ww, Phi)


    def log_prior(self, ww):
        """
        Log prior probability of basis weights up to a constant

        WARNING: currently missing off constant wrt ww that depends on hypers
        """
        return -0.5*np.dot(ww, ww)/(self.pi_std**2)

    def log_likelihood(self,ww):
        """
        Log likelihood of basis weights given Poisson time data

            ww     K, coefficients of basis functions
            t_obs L,2 start and end times of L intervals with observations

            Optional: bin_c, bin_w, bin centers and widths with which to approx integral
        """
        # L = - \int \lamba(t) dt + \sum_{n=1}^N \log \lambda(t_n)
        if self.bin_integral:
            assert(self.bin_c is not None)
            assert(self.bin_w is not None)
            integral = np.sum(np.exp(self.log_intensity(self.bin_c, ww))*self.bin_w)
            #print(integral)
        else:
            func = lambda t: np.exp(self.log_intensity(t, ww))
            integral = 0.0
            for ll in range(len(self.t_obs)):
                integral += scipy.integrate.quad(
                        func, self.t_obs[ll,0], self.t_obs[ll,1], epsabs=1e-2, epsrel=1e-2)[0]
            #print(integral)
            #print('---')
        return -integral + np.sum(self.log_intensity(self.tt, ww))


    def posterior(self, ww):

        logdist = self.log_prior(ww) + self.log_likelihood(ww)
        assert(not np.isnan(logdist))
        return logdist

    def __call__(self, ww):
        return self.posterior(ww)

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

## make the object
print("Making the object")
lpost = EventModel(tt,t_obs, hypers, bin_integral=True, bin_c=bin_c, bin_w=bin_w)

ww = np.zeros(K)
S = 100
print("sampling")
samples = slice_sample(
        ww, lpost, widths=hypers.pi_std,
        N=S, burn=0, step_out=True, verbose=2) # S,K

ww_end = samples[-1]

# to plot things, bin things up:
Nbins = 300
tbins = np.linspace(T0, Tend, Nbins)
intensity = np.exp(lpost.log_intensity(tbins, ww_end))
plt.clf()
h, bins, patches = plt.hist(tt, bins=500)
bin_size = bins[1]-bins[0]
plt.plot(tbins, intensity*bin_size, 'r', linewidth=3)
plt.show()

