#Testing whether our MCMC Test actually works!
import numpy as np

from simple_slice import slice_sample


def log_prior(p):
    if p[0] < 5.0 or p[0] > 15:
        return -np.inf
    else:
        return np.log(1/10.)

def sample_prior():
    return np.array([np.random.uniform(5,15)])

def sample_counts(xx,p):
    return np.random.poisson(p[0], size=len(xx))

def log_likelihood(yy, p):
    lograte = np.log(p)
    ll = np.sum(yy*lograte - p)

    return ll

def log_posterior(yy, p):
    return log_prior(p) + log_likelihood(yy,p)


def mcmc_test(xx, niter=1000):
    p = sample_prior()
    samples_all = np.zeros((niter,1))
    width = 1.
    for i in range(niter):
        yy = sample_counts(xx, p)
        samples = slice_sample(
                    p, lambda p: log_posterior(yy,p), widths=width,
                    N=1, burn=0, step_out=True, verbose=2) # S,K
        p = samples[-1]
        samples_all[i,:] = p

    return samples_all

times = np.linspace(0,10,100)
bin_width = np.diff(times)

samples_all = mcmc_test(times, niter=50000)


