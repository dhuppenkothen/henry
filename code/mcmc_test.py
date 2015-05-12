

import numpy as np

import utils
from simple_slice import slice_sample
import matplotlib.pyplot as plt

import collections
CHypers = collections.namedtuple('CHypers',
        ['pi_mu', 'pi_std', 'cc', 'ell'])


import binned_likelihood


class PriorTest(binned_likelihood.FuseData):



    def log_prior(self, pars):

        self._set_hypers(pars)

        assert(np.size(pars)==self.K+3)

        #mean_lograte = pars[0]

        if not (-2 < self.pi_mu < -1):
            return -np.inf


        #self.pi_std = pars[1]

        if not (0 < self.pi_std < 2.):
            return -np.inf

        #assert(pars[1] < 5.)

        #ell = pars[2]

        if not (self.sep < self.ell[0] < 2.*self.sep):
            return -np.inf

        return binned_likelihood.EventModel.log_prior(self, pars[3:])


    #def log_likelihood(self,pars):
    #    return 1.

    def sample_counts(self, pars):

        self._set_hypers(pars)


        mean_fermi = np.exp(lpost.log_intensity(self.bin_c, pars[3:]))*self.bin_w
        mean_int =  np.exp(lpost.log_intensity(self.bin_c_int, pars[3:]))*self.bin_w_int

        counts_fermi = np.random.poisson(mean_fermi)
        counts_int = np.random.poisson(mean_int)

        return counts_fermi, counts_int


    def sample_prior(self):

        pi_mu = np.random.uniform(-2 , -1)
        pi_std = np.random.uniform(0, 2.)
        ell  = np.random.uniform(self.sep, 2.*self.sep)

        ww = np.random.normal(0, pi_std, self.K)

        pars = np.hstack([[pi_mu, pi_std, ell], ww])

        return pars


    def mcmc_test(self, niter = 1000):

        widths = np.hstack([0.5, 0.5, 500.0, np.tile(0.5, self.K)])
        pars = self.sample_prior()

        samples_all = np.zeros((niter, len(pars)))

        for i in range(niter):
            self.bin_counts, self.bin_counts_int = self.sample_counts(pars)
            samples = slice_sample(
                        pars, self.posterior, widths=widths,
                        N=1, burn=0, step_out=True, verbose=2) # S,K
            pars = samples[-1]
            samples_all[i,:] = pars


        return samples_all
#Fermi
_, tt, t_no_obs, t_obs = utils.load_gbm_bursts('../data/')
#Integral
tt_int = np.loadtxt("../data/sgr1550_integral_bursts.dat")

## Integral observes before Fermi, so put unobserved interval into the first
## part
t_no_obs = np.vstack([[0.0, t_obs[0,0]], t_no_obs])


t_obs = t_obs[:1]
t_no_obs = t_no_obs[:1]

# Time range:
T0 = t_no_obs[0,0]
Tend = t_no_obs[-1,1]
T = Tend - T0



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


## Model could have several sets of basis functions with different widths
## InferHypers class currently assumes there is only one group with a shared lengthscale
cc = []
ell = []
for e in [T/200]:
    c = make_cc(T0, Tend, e, spacing=1.0)
    cc.append(c)
    ell.append(np.tile(e,len(c)))

cc = np.hstack(cc)
cc = cc[:2]
ell = np.hstack(ell)[0]*1.5



#K = len(cc)

#hypers = CHypers(pi_mu=-5, pi_std=0.5, cc=cc, ell=ell)

# obs_bins for integral hack

bps = 1 ## bins per observation window
bin_c = np.zeros(bps * len(t_obs))
bin_w = np.zeros_like(bin_c)

for i in range(len(t_obs)):
    bw = (t_obs[i,1] - t_obs[i,0]) / bps
    bin_w[i*bps:((i+1)*bps)] = bw
    bin_c[i*bps:((i+1)*bps)] = np.linspace(
            t_obs[i,0] + bw/2, t_obs[i,1]-bw/2, bps)

bin_c_int = np.zeros(bps*len(t_no_obs))
bin_w_int = np.zeros_like(bin_c_int)

for i in range(len(t_no_obs)):
    bw = (t_no_obs[i,1] - t_no_obs[i,0]) / bps
    bin_w_int[i*bps:((i+1)*bps)] = bw
    bin_c_int[i*bps:((i+1)*bps)] = np.linspace(
            t_no_obs[i,0] + bw/2, t_no_obs[i,1]-bw/2, bps)



bin_counts = np.zeros_like(bin_c)
for i in range(len(bin_c)):
    bin_counts[i] = np.sum(((bin_c[i] - bin_w[i]/2) <= tt) & (tt <= (bin_c[i] + bin_w[i]/2)))


bin_counts_int = np.zeros_like(bin_c_int)
for i in range(len(bin_c_int)):
    bin_counts_int[i] = np.sum(((bin_c_int[i] - bin_w_int[i]/2) <= tt_int) &
                               (tt_int <= (bin_c_int[i] + bin_w_int[i]/2)))


print("We are being really, really evil, but it's okay, because Alexander and Yuki gave us faulty data.")

#assert(np.sum(bin_counts) == len(tt))
#logdist = lambda w: log_prior(w, hypers) + \
#        log_likelihood(w, hypers, tt, t_obs)

## make the object
print("Making the object")
lpost = PriorTest(bin_c, bin_counts, bin_w, bin_c_int, bin_counts_int, bin_w_int, cc)

ww = np.ones(lpost.K)

## ell needs to be a scalar!
ell = 1.5*np.diff(cc)[0]

pars = np.hstack([-1.5, 0.5, ell, ww])

#print("parameters: " + str(pars))
print("initial prior: " + str(lpost.log_prior(pars)))
#print("initial likelihood: " + str(lpost.log_likelihood(pars)))

print("sampling")

### slice sampler initial step sizes
widths = np.hstack([0.5, 0.5, 500.0, np.tile(0.5, lpost.K)])


samples = lpost.mcmc_test(niter=10000)
#samples = slice_sample(
#        pars, lpost, widths=widths,
#        N=S, burn=0, step_out=True, verbose=2) # S,K


#ww_end = samples[-1]
