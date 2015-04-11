import numpy as np
import scipy.integrate

import utils
from simple_slice import slice_sample
import matplotlib.pyplot as plt

import collections
CHypers = collections.namedtuple('CHypers',
        ['pi_mu', 'pi_std', 'cc', 'ell'])



class EventModel(object):

    def __init__(self, tt, t_obs, hypers, bin_integral=False, bin_c=None, bin_w=None, bin_counts=None):
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
        self.tt = np.array(tt)

        self.t_obs = np.array(t_obs)

        self.K = len(hypers.cc)
        self.cc = hypers.cc

        if np.size(hypers.ell) == 1:
            self.ell = np.ones_like(self.cc)*hypers.ell
        else:
            self.ell = hypers.ell

        self.pi_mu = hypers.pi_mu
        self.pi_std = hypers.pi_std

        self.bin_integral = bin_integral
        self.bin_c = bin_c
        self.bin_w = bin_w
        self.log_bin_w = np.log(bin_w)
        self.bin_counts = bin_counts

        return

    def log_intensity(self, times, ww):
        """
        Compute the log-intensity for the Gaussian basis functions
        :param ww:
        :return:
        """

        Phi = np.zeros((self.K, np.array(times).size))
        for kk in range(self.K):
            Phi[kk] = np.exp(-0.5*(times - self.cc[kk])**2/self.ell[kk]**2)


        return self.pi_mu + np.dot(ww, Phi)


    def log_prior(self, ww):
        """
        Log prior probability of basis weights up to a constant

        WARNING: currently missing off -0.5*ww.size*log(2pi)
        """
        #print(self.pi_std)
        return -0.5*np.dot(ww, ww)/(self.pi_std**2) - ww.size*np.log(self.pi_std)

    def draw_from_prior(self, times):
        ww = np.random.randn(self.K)*self.pi_std
        return self.log_intensity(times, ww)


    def log_likelihood(self,ww):
        """
        Log likelihood of basis weights given Poisson time data

            ww     K, coefficients of basis functions
            t_obs L,2 start and end times of L intervals with observations

            Optional: bin_c, bin_w, bin centers and widths with which to approx integral
        """
        # L = - \int \lamba(t) dt + \sum_{n=1}^N \log \lambda(t_n)
        if self.bin_counts is not None:
            #print("I am in binned likelihood!")
            lograte = self.log_intensity(self.bin_c, ww) + self.log_bin_w
            p = np.sum(self.bin_counts*lograte - np.exp(lograte))
            return p


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
        print("We are in the wrong posterior")
        logdist = self.log_prior(ww) + self.log_likelihood(ww)
        assert(not np.isnan(logdist))
        return logdist

    def __call__(self, ww):
        return self.posterior(ww)



class InferHypers(EventModel,object):

    def __init__(self, tt, t_obs, cc, bin_integral=False, bin_c=None, bin_w=None, bin_counts=None):
        """
        :param tt:
        :param t_obs:
        :param cc:
        :param bin_integral:
        :param bin_c:
        :param bin_w:
        :return:
        """
        self.tt = tt
        self.t_obs = t_obs
        self.T = t_obs[-1,1] - t_obs[0,0]

        self.K = len(cc)
        self.cc = cc
        self.sep = self.cc[1] - self.cc[0]

        #if np.size(ell) == 1:
        #    self.ell = np.ones_like(self.cc)*hypers.ell
        #else:
        #    self.ell = hypers.ell

        #self.pi_mu = hypers.pi_mu
        #self.pi_std = hypers.pi_std

        self.bin_integral = bin_integral
        self.bin_c = bin_c
        self.bin_w = bin_w
        self.log_bin_w = np.log(bin_w)
        self.bin_counts = bin_counts

        return

    def log_prior(self, pars):
        assert(np.size(pars)==self.K+3)

        self.pi_std = pars[1]

        if not (-9 < pars[0] < -1):
            return -np.inf

        if not (0 < pars[1] < 50.):
            return -np.inf

        #assert(pars[1] < 5.)

        if not (self.sep < pars[2] < self.T):
            return -np.inf

        return EventModel.log_prior(self, pars[3:])


    def log_likelihood(self,pars):
        assert(np.size(pars)==self.K+3)

        self.pi_mu = pars[0]
        self.ell = np.ones_like(self.cc)*pars[2]

        return EventModel.log_likelihood(self, pars[3:])


    def posterior(self, pars):
        #print("We are in the right posterior")
        pr = self.log_prior(pars)
        #print(pars[:3])
        #print(pr)
        if np.isinf(pr):
            #print("returning inf")
            return pr
        else:
            #print("returning " + str(pr + self.log_likelihood(pars)))
            return pr + self.log_likelihood(pars)

# Grab the data:
_, tt, t_no_obs, t_obs = utils.load_gbm_bursts('../data/')


# Time range:
T0 = t_obs[0,0]
Tend = t_obs[-1,1]
T = Tend - T0
#ell = T / 10.0 # lengthscale

def make_cc(T0, Tend, ell, spacing=1.0):
    """
    Compute the centres of the Gaussian components.

    :param T0: start time
    :param Tend: end time
    :param ell: width of the Gaussian
    :return: array with the component centres
    """
    T = Tend - T0
    #ell = T / 40.0 # lengthscale

    # Basis function centres:
    c0 = T0 - ell*5
    cc = np.arange(T0 - ell*5, Tend + ell*5, ell*spacing)
    return cc

cc = []
ell = []
for e in [T/120]:
    c = make_cc(T0, Tend, e, spacing=1.0)
    cc.append(c)
    ell.append(np.tile(e,len(c)))

cc = np.hstack(cc)
ell = np.hstack(ell)[0]*1.5



#K = len(cc)

hypers = CHypers(pi_mu=-5, pi_std=0.5, cc=cc, ell=ell)

# obs_bins for integral hack
bin_w = t_obs[:,1] - t_obs[:,0]
bin_c = t_obs[:,0] + bin_w/2
bps = 10
bin_c = np.zeros(bps * len(t_obs))
bin_w = np.zeros_like(bin_c)


#### FIX ME!
#tt -= 3.0

for i in range(len(t_obs)):
    bw = (t_obs[i,1] - t_obs[i,0]) / bps
    bin_w[i*bps:((i+1)*bps)] = bw
    bin_c[i*bps:((i+1)*bps)] = np.linspace(
            t_obs[i,0] + bw/2, t_obs[i,1]-bw/2, bps)

bin_counts = np.zeros_like(bin_c)
for i in range(len(bin_c)):
    bin_counts[i] = np.sum(((bin_c[i] - bin_w[i]/2) <= tt) & (tt <= (bin_c[i] + bin_w[i]/2)))


print("We are being really, really evil, but it's okay, because Alexander and Yuki gave us faulty data.")

#assert(np.sum(bin_counts) == len(tt))
#logdist = lambda w: log_prior(w, hypers) + \
#        log_likelihood(w, hypers, tt, t_obs)

## make the object
print("Making the object")
lpost = InferHypers(tt,t_obs, cc, bin_integral=True, bin_c=bin_c, bin_w=bin_w, bin_counts=bin_counts)

ww = np.zeros(lpost.K)
S = 200

pars = np.hstack([-5, 0.5, ell, ww])

print("initial prior: " + str(lpost.log_prior(pars)))
print("initial likelihood: " + str(lpost.log_likelihood(pars)))

print("sampling")

widths = np.hstack([0.5, 0.5, 500.0, np.tile(0.5, lpost.K)])

samples = slice_sample(
        pars, lpost, widths=widths*50.,
        N=100, burn=0, step_out=True, verbose=2) # S,K

#samples = slice_sample(
#        samples[-1], lpost, widths=widths,
#        N=S, burn=0, step_out=True, verbose=2) # S,K


ww_end = samples[-1]

plt.figure(1)

for ww_end in samples:
# to plot things, bin things up:
    Nbins = 300
    tbins = np.linspace(T0, Tend, Nbins)
    intensity = np.exp(lpost.log_intensity(tbins, ww_end[3:]))
    #plt.clf()
    h, bins, patches = plt.hist(tt, bins=500)
    bin_size = bins[1]-bins[0]
    plt.plot(tbins, intensity*bin_size, linewidth=3, alpha=0.5)

print(lpost.log_prior(samples[-1]))
print(lpost.log_likelihood(samples[-1]))
#plt.figure(2)
#plt.clf()
#for i in range(10):
#    plt.plot(tbins, np.exp(lpost.draw_from_prior(tbins)))

plt.show()

