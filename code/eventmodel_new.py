import numpy as np

from simple_slice import slice_sample

import matplotlib.pyplot as plt


class EventModel(object):

    def __init__(self, x,y, K, cc):

        ## x-coordinate of data
        self.x = x
        ## y-coordinate of data
        self.y = y

        ## bin width, assume evenly sampled data
        self.width = np.diff(self.x)[0]

        ## number of components
        self.K = K

        ## cluster centers
        self.cc = cc

        ## distance between centers, assume centers evenly spaced
        if len(self.cc) == 1:
            self.sep = self.cc[0]
        else:
            self.sep = np.diff(self.cc)[0]

        self.max_l = self.x[-1] - self.x[0]


        self.pi_mu_prior = [-5., -1.]
        self.ell_prior = [self.sep, self.max_l]
        self.pi_std_prior = [0., 30.]

        return


    def log_intensity(self,pars):

        pi_mu = pars[0]
        ell = np.ones_like(self.cc)*pars[1]
        pi_std = pars[2]
        ww = pars[3:]

        Phi = np.zeros((self.K, np.array(self.x).size))
        for kk in range(self.K):
            Phi[kk] = np.exp(-0.5*(self.x - self.cc[kk])**2/ell[kk]**2)

        return pi_mu + np.dot(ww, Phi)


    def log_bin_rates(self, pars):

        rate = self.log_intensity(pars)
        return rate + np.log(self.width)

    def log_prior(self, pars):
        pi_mu = pars[0]
        ell = pars[1]
        pi_std = pars[2]
        ww = pars[3:]

        assert(np.size(pars)==self.K+3)

        if not (self.pi_mu_prior[0] <= pi_mu <= self.pi_mu_prior[1]):
            return -np.inf

        #print("pi_mu, %.3f, in right range for prior"%pi_mu)

        if not (self.ell_prior[0] <= ell <= self.ell_prior[1]):
            return -np.inf

        #print("ell, %.3f, in right range for prior"%ell)

        if not (self.pi_std_prior[0] <= pi_std <= self.pi_std_prior[1]):
            return -np.inf

        #print("pi_std, %.3f, in right range for prior"%pi_std)

        logpr = -0.5*np.dot(ww, ww)/(pi_std**2) - ww.size*np.log(pi_std)

        #print("log-prior: %.3f"%logpr)

        return logpr


    def sample_from_prior(self):
        pi_mu = np.random.uniform(self.pi_mu_prior[0], self.pi_mu_prior[1])
        ell = np.random.uniform(self.ell_prior[0], self.ell_prior[1])
        pi_std = np.random.uniform(self.pi_std_prior[0], self.pi_std_prior[1])
        ww = np.random.normal(0.0, pi_std, size=self.K)

        sample_pars = np.hstack(([pi_mu, ell, pi_std], ww))

        return sample_pars


    def log_likelihood(self, pars):

        lograte = self.log_bin_rates(pars)
        return np.sum(self.y*lograte - np.exp(lograte))

    def log_posterior(self, pars):
        logprior = self.log_prior(pars)
        if np.isinf(logprior):
            return logprior
        else:
            return logprior + self.log_likelihood(pars)


    def draw_from_prior(self, pars, log_intensity=True):
        ww = np.random.randn(self.K)*pars[2]
        pars[-1] = ww
        if log_intensity:
            return self.log_intensity(pars)
        else:
            return ww


    def sample_counts(self, pars):
        mean_fermi = np.exp(self.log_intensity(pars))*self.width
        counts_fermi = np.random.poisson(mean_fermi)
        return counts_fermi


    def mcmc_test(self, widths, niter = 1000):

        """
        DO NOT USE! TEST VERSION CURRENTLY IN eventmodel_tests.py!!!

        :param widths:
        :param niter:
        :return:
"""
        #widths = np.hstack([0.5, 500.0, 0.5, 0.5])
        pars = self.sample_from_prior()

        samples_all = np.zeros((niter, len(pars)))

        for i in range(niter):
            print("I am in simulation %i"%i)
            self.y = self.sample_counts(pars)
            samples = slice_sample(
                        pars, self.log_posterior, widths=widths,
                        N=1, burn=0, step_out=True, verbose=2) # S,K
            pars = samples[-1]
            samples_all[i,:] = pars


        return samples_all
#
    def __call__(self, pars):
        return self.log_posterior(pars)

