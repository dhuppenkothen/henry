
import numpy as np
import matplotlib.pyplot as plt

from eventmodel_new import EventModel

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
    print("log-prior: %.3f"%lpost.log_prior(pars))
    print("log-prior should be real-valued \n")
    ## put pi_mu outside range
    pars[0] = 1.0
    print("log-prior: %.3f"%lpost.log_prior(pars))
    print("prior should be -inf \n")

    ## make only ell outside range
    pars[0] = -2.0
    pars[1] = ell/2.0

    print("log-prior: %.3f"%lpost.log_prior(pars))
    print("prior should be -inf \n")

    ## make only pi_std outside prior
    pars[1] = ell*1.5
    pars[2] = -2.0

    print("log-prior: %.3f"%lpost.log_prior(pars))
    print("prior should be -inf \n")


    ## make one of the weights crazy
    pars[2] = 2.0
    pars[4] = 20.0

    print("log-prior: %.3f"%lpost.log_prior(pars))
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
    cc = np.array([sep])

    ## width is going large to pass the prior
    ell = sep*1.5

    ## log-background count rate is 20
    pi_mu = -2.0

    ## don't need this parameter yet; hyperparameter prior on weights of components
    pi_std = 5.0
    ## set the weights:
    ww = np.array([2.0])

    pars = np.hstack(([pi_mu, ell, pi_std], ww))
    ## did the parameters get combined correctly?
    print(pars)

    ## initialize class
    lpost = EventModel(x,y, K, cc)

    ## now make the actual simulated data:
    yhat = lpost.model(pars)

    ## Poissonify
    ycounts = np.random.poisson(np.exp(yhat)*lpost.width)

    lpost.y = ycounts

    ## compute likelihoods for various values of the parameters:
    sep_test = np.arange(1.01, 1.99, 0.1)
    pi_mu_test = np.arange(-5.0, -1.0, 0.5)
    ww_test = np.arange(-1, 5, 0.4)


    sep_ll = np.zeros_like(sep_test)
    pi_mu_ll = np.zeros_like(pi_mu_test)
    ww_ll = np.zeros_like(ww_test)

    for i,p in enumerate(pi_mu_test):
        pars[0] = p
        pars[1] = ell
        pars[-1] = ww[0]
        pi_mu_ll[i] = lpost.log_likelihood(pars)

    for j,s in enumerate(sep_test):
        pars[0] = pi_mu
        pars[1] = s*sep
        pars[-1] = ww[0]
        sep_ll[j] = lpost.log_likelihood(pars)


    for k,w in enumerate(ww_test):
        pars[0] = pi_mu
        pars[1] = ell
        pars[-1] = w
        ww_ll[k] = lpost.log_likelihood(pars)


    print("finished computing likelihoods")

    plt.figure()
    plt.plot(pi_mu_test, pi_mu_ll)
    plt.vlines(pi_mu, np.min(pi_mu_ll), np.max(pi_mu_ll))
    plt.title("Likelihood for pi_mu")

    plt.figure()
    plt.plot(sep_test*sep, sep_ll)
    plt.vlines(ell, np.min(sep_ll), np.max(sep_ll))

    plt.title("Likelihood for ell")

    plt.figure()
    plt.plot(ww_test, ww_ll)
    plt.vlines(ww[0], np.min(ww_ll), np.max(ww_ll))

    plt.title("Likelihood for ww")

    plt.show()
    return

def test_prior_draw():

    ## number of seconds
    nseconds = 60*60*24

    ## timescale is 1 second
    x = np.arange(0,nseconds,0.1)

    ## data just needs to be a placeholder for now
    y = np.zeros_like(x)

    ## one component
    K = 1
    sep = nseconds/(2.)
    cc = np.array([sep])

    ## width is going large to pass the prior
    ell = sep*1.5

    ## log-background count rate is 20
    pi_mu = -2.0

    ## don't need this parameter yet; hyperparameter prior on weights of components
    pi_std = 2.0
    ## set the weights:
    ww = np.array([2.0])

    pars = np.hstack(([pi_mu, ell, pi_std], ww))
    ## did the parameters get combined correctly?
    print(pars)

    ## initialize class
    lpost = EventModel(x,y, K, cc)

    ndraws = 10000
    ww_draws = np.array([lpost.draw_from_prior(pars, model=False) for n in xrange(ndraws)])

    print(ww_draws.shape)
    plt.figure()
    plt.hist(ww_draws, bins=50)
    #plt.show()

    ndraws = 20
    model_draws = np.array([lpost.draw_from_prior(pars, model=True) for n in xrange(ndraws)])

    plt.figure()
    for m in model_draws:
        plt.plot(lpost.x, m, lw=2)
    plt.show()

    return

def test_prior_sampling():

    ## number of seconds
    nseconds = 60*60*24

    ## timescale is 1 second
    x = np.arange(0,nseconds,0.1)

    ## data just needs to be a placeholder for now
    y = np.zeros_like(x)

    ## one component
    K = 1
    sep = nseconds/(2.)
    cc = np.array([sep])

    ## initialize class
    lpost = EventModel(x,y, K, cc)

    ndraws = 100000

    sample_pars = np.array([lpost.sample_from_prior() for n in xrange(ndraws)])

    plt.figure()
    plt.hist(sample_pars[:,0], bins=30)
    plt.title("Prior draws from pi_mu")

    plt.figure()
    plt.hist(sample_pars[:,1], bins=30)
    plt.title("Prior draws from ell")

    plt.figure()
    plt.hist(sample_pars[:,2], bins=30)
    plt.title("Prior draws from pi_std")

    plt.figure()
    plt.hist(sample_pars[:,3], bins=30)
    plt.title("Prior draws from ww")

    plt.show()

    return

def test_sample_counts():

    ## number of seconds
    nseconds = 60*60*24

    ## timescale is 1 second
    x = np.arange(0,nseconds,0.1)

    ## data just needs to be a placeholder for now
    y = np.zeros_like(x)

    ## one component
    K = 1
    sep = nseconds/(2.)
    cc = np.array([sep])

    ## width is going large to pass the prior
    ell = sep/10.0

    ## log-background count rate is 20
    pi_mu = 1.0

    ## don't need this parameter yet; hyperparameter prior on weights of components
    pi_std = 2.0
    ## set the weights:
    ww = np.array([5.0])

    pars = np.hstack(([pi_mu, ell, pi_std], ww))
    ## did the parameters get combined correctly?
    print(pars)

    ## initialize class
    lpost = EventModel(x,y, K, cc)

    ndraws = 20
    sample_counts = np.array([lpost.sample_counts(pars) for n in xrange(ndraws)])

    plt.figure()
    for s in sample_counts:
        plt.plot(lpost.x, s, lw=2)
    plt.show()

    return

#test_model()
#test_prior()
#test_likelihood()
#test_prior_draw()
#test_prior_sampling()
test_sample_counts()