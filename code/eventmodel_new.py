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

    def logprior(self, pars):
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

    def log_likelihood(self, pars):

        lograte = self.log_counts(pars)
        return np.sum(self.y*lograte - np.exp(lograte))






def test_model():

    ## number of seconds
    nseconds = 60*60*24

    ## timescale is 0.1 second
    x = np.arange(0,nseconds,0.1)

    ## data just needs to be a placeholder
    y = np.zeros_like(x)

    ## three components
    K = 3
    sep = nseconds/(K+1)
    cc = [sep,2.*sep, 3.*sep]

    ## width is going to be quite small
    ell = sep/20.

    ## log-background count rate is 20
    pi_mu = 0.0

    ## don't need this parameter yet; hyperparameter prior on weights of components
    pi_std = 0.0

    ## set the log-weights:
    ww = [-5., 1., 2.0]

    pars = np.hstack(([pi_mu, ell, pi_std], ww))
    ## did the parameters get combined correctly?
    print(pars)

    ## initialize class
    lpost = EventModel(x,y, K, cc)

    ## confirm stuff is stored correctly
    print("x: " + str(lpost.x))
    print("y: " + str(lpost.y))
    print("K: " + str(lpost.K))
    print("cc: " + str(lpost.cc))
    print("sep: " + str(lpost.sep))

    ## make the test light curve
    yhat = np.exp(lpost.model(pars))

    plt.figure()
    plt.plot(lpost.x, yhat)
    plt.yscale("log")


    ## ... does this plot look correct for the values I put in? Yes, it does!

    ## test the log-rate

    print("binsize is %.3f"%lpost.width)
    logyhat = lpost.log_counts(pars)
    plt.figure()
    plt.plot(lpost.x, logyhat)
    plt.show()


    return



def test_prior():

    ## number of seconds
    nseconds = 60*60*24

    ## timescale is 1 second
    x = np.arange(0,nseconds,0.1)

    ## data just needs to be a placeholder
    y = np.zeros_like(x)

    ## three components
    K = 3
    sep = nseconds/(K+1)
    cc = [sep,2.*sep, 3.*sep]

    ## width is going large to pass the prior
    ell = sep*1.5

    ## log-background count rate is 20
    pi_mu = -2.0

    ## don't need this parameter yet; hyperparameter prior on weights of components
    pi_std = 5.0
    ## set the weights:
    ww = [-5., 1., 2.0]

    pars = np.hstack(([pi_mu, ell, pi_std], ww))
    ## did the parameters get combined correctly?
    print(pars)

    ## initialize class
    lpost = EventModel(x,y, K, cc)
    print("log-prior: %.3f"%lpost.logprior(pars))
    print("log-prior should be real-valued \n")
    ## put pi_mu outside range
    pars[0] = 1.0
    print("log-prior: %.3f"%lpost.logprior(pars))
    print("prior should be -inf \n")

    ## make only ell outside range
    pars[0] = -2.0
    pars[1] = ell/2.0

    print("log-prior: %.3f"%lpost.logprior(pars))
    print("prior should be -inf \n")

    ## make only pi_std outside prior
    pars[1] = ell*1.5
    pars[2] = -2.0

    print("log-prior: %.3f"%lpost.logprior(pars))
    print("prior should be -inf \n")


    ## make one of the weights crazy
    pars[2] = 2.0
    pars[4] = 20.0

    print("log-prior: %.3f"%lpost.logprior(pars))
    print("prior should be small compared to first run \n")


    return


def test_likelihood():

    ## number of seconds
    nseconds = 60*60*24

    ## timescale is 1 second
    x = np.arange(0,nseconds,0.1)

    ## data just needs to be a placeholder for now
    y = np.zeros_like(x)

    ## one component
    K = 1
    sep = nseconds/(2.)
    cc = [sep]

    ## width is going large to pass the prior
    ell = sep*1.5

    ## log-background count rate is 20
    pi_mu = -2.0

    ## don't need this parameter yet; hyperparameter prior on weights of components
    pi_std = 5.0
    ## set the weights:
    ww = [2.0]

    pars = np.hstack(([pi_mu, ell, pi_std], ww))
    ## did the parameters get combined correctly?
    print(pars)

    ## initialize class
    lpost = EventModel(x,y, K, cc)

    ## now make the actual simulated data:
    yhat = lpost.model(pars)

    ## Poissonify
    ycounts = np.random.poisson(yhat)

    lpost.y = ycounts

    ## compute likelihoods for various values of the parameters:
    sep_test = np.arange(1.01, 1.99, 0.1)
    pi_mu_test = np.arange(-5.0, -1.0, 0.5)
    ww_test = np.arange(-1, 5, 0.4)

    likelihoods = np.zeros((len(pi_mu_test), len(sep_test), len(ww_test)))
    print(likelihoods.shape)



    return


test_model()
test_prior()

