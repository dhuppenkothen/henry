import numpy as np
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
        return


    def model(self,pars):

        pi_mu = pars[0]
        ell = np.ones_like(self.cc)*pars[1]
        pi_std = pars[2]
        ww = pars[3:]

        Phi = np.zeros((self.K, np.array(self.x).size))
        for kk in range(self.K):
            Phi[kk] = np.exp(-0.5*(self.x - self.cc[kk])**2/ell[kk]**2)

        return pi_mu + np.dot(ww, Phi)


    def log_counts(self, pars):

        rate = self.model(pars)
        return rate + np.log(self.width)

    def log_prior(self, pars):
        pi_mu = pars[0]
        ell = pars[1]
        pi_std = pars[2]
        ww = pars[3:]

        assert(np.size(pars)==self.K+3)


        if not (-5 < pi_mu < -1):
            return -np.inf

        print("pi_mu, %.3f, in right range for prior"%pi_mu)

        if not (self.sep < ell < self.max_l):
            return -np.inf

        print("ell, %.3f, in right range for prior"%ell)


        if not (0 < pi_std < 30.):
            return -np.inf

        print("pi_std, %.3f, in right range for prior"%pi_std)


        logpr = -0.5*np.dot(ww, ww)/(pi_std**2) - ww.size*np.log(pi_std)

        #print("log-prior: %.3f"%logpr)

        return logpr


    def sample_from_prior(self):
        pi_mu = np.random.uniform(-5,-1)
        ell = np.random.uniform(self.sep, self.max_l)
        pi_std = np.random.uniform(0, 30)
        ww = np.random.normal(0.0, pi_std, size=self.K)

        sample_pars = np.hstack(([pi_mu, ell, pi_std], ww))

        return sample_pars


    def log_likelihood(self, pars):

        lograte = self.log_counts(pars)
        return np.sum(self.y*lograte - np.exp(lograte))

    def log_posterior(self, pars):
        logprior = self.log_prior(pars)
        if np.isinf(logprior):
            return logprior
        else:
            return logprior + self.log_likelihood(pars)


    def draw_from_prior(self, pars, model=True):
        ww = np.random.randn(self.K)*pars[2]
        pars[-1] = ww
        if model:
            return self.model(pars)
        else:
            return ww


    def sample_counts(self, pars):
        mean_fermi = np.exp(self.model(pars))*self.width
        counts_fermi = np.random.poisson(mean_fermi)
        return counts_fermi

