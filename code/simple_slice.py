from __future__ import print_function

import numpy as np

def ensemble_slice_step(xx, logdist, verbose=0):
    """simple param-free axis-aligned slice sampling for ensemble of vectors

         ensemble_slice_step(xx) # update N,D xx in place

     Inputs:
                xx N,D  N states to update, each with D elements
           logdist  fn  function logprobstar = logdist(xx[i])

    xx is updated in place (dangerous?), but *also* returned if you want.
    """
    N, D = xx.shape
    assert(N >= 4)
    N1 = N//2
    N2 = N - N1

    # Stuff to set up slice sampling rungs
    N = 1
    burn = 0
    step_out = False
    def slice_widths(xs):
        # The ensemble range, inflated in hacky way:
        ww = 2*(xs.max(0) - xs.min(0))
        ww = np.maximum(ww, ww[ww>0].min())
        return ww

    # Update population 1
    x1 = xx[:N1]
    x2 = xx[N1:]
    widths = slice_widths(x2)
    for i,x in enumerate(x1):
        x = slice_sample(x, logdist, widths, N, burn, step_out, verbose)
        x1[i] = x

    # Update population 2
    widths = slice_widths(x1)
    for i,x in enumerate(x2):
        x = slice_sample(x, logdist, widths, N, burn, step_out, verbose)
        x2[i] = x

    return xx
 

def slice_sample(xx, logdist,
        widths=1.0, N=1, burn=0, step_out=True, verbose=2):
    """simple axis-aligned implementation of slice sampling for vectors

         xx_next = slice_sample(xx, logdist)
         samples = slice_sample(xx, logdist, N=200, burn=20)

     Inputs:
                xx  D,  initial state (or array with D elements)
           logdist  fn  function logprobstar = logdist(xx)
                 N  1,  Number of samples to save (default 1)
              burn  1,  after burning period of this length (default 0)
            widths  D,  or 1x1, step sizes for slice sampling (default 1.0)
          step_out bool set to True (default) if widths sometimes far too small
           verbose  1,  Non-zero report iterations. >1 report bracketing steps

     Outputs:
          samples  N,D  stored samples, or N,xx.shape
    """
    # Iain Murray 2004, 2009, 2010, 2013
    # Algorithm orginally by Radford Neal, e.g., Annals of Statistic (2003)
    # See also pseudo-code in David MacKay's text book p375

    # startup stuff
    D = xx.size
    samples = np.zeros([N] + list(xx.shape))
    widths = np.array(widths)
    if widths.size == 1:
        widths = np.tile(widths, D)
    log_Px = logdist(xx)
    perm = np.array(range(D))

    # Force xx in to vector for ease of use:
    xx_shape = xx.shape
    logdist_vec = lambda x: logdist(np.reshape(x, xx_shape))
    xx = xx.ravel().copy()
    x_l = xx.copy()
    x_r = xx.copy()
    xprime = xx.copy()

    # Main loop
    for ii in range(-burn,N):
        if verbose:
            print('Iteration %d      ' % (ii+1), end='\r')
        log_uprime = np.log(np.random.rand()) + log_Px

        # Random scan through axes
        np.random.shuffle(perm)
        for dd in perm:
            # Create a horizontal interval (x_l, x_r) enclosing xx
            rr = np.random.rand()
            x_l[dd] = xx[dd] - rr*widths[dd]
            x_r[dd] = xx[dd] + (1-rr)*widths[dd]
            if step_out:
                so = 0
                # Typo in early book editions: said compare to u, should be u'
                while logdist_vec(x_l) > log_uprime:
                    so += 1
                    x_l[dd] = x_l[dd] - widths[dd]
                while logdist_vec(x_r) > log_uprime:
                    so +=1
                    x_r[dd] = x_r[dd] + widths[dd]

                if so > 100:
                    print("%d steps out on dim %d, left %g, center %g, right %g, pi_std = %g"
                          % (so, dd, x_l[dd], xx[dd], x_r[dd], xx[1]))
                    #print("parameters: " + str(xx))
                    #print("prior x_l: %g"%logdist.log_prior(x_l))
                    #print("prior x_r: %g"%logdist.log_prior(x_r))
                    #print("prior xx: %g"%logdist.log_prior(xx))

                    #print("log-likelihood x_l: %g"%logdist.log_likelihood(x_l))
                    #print("log-likelihood x_r: %g"%logdist.log_likelihood(x_r))
                    #print("log-likelihood xx: %g"%logdist.log_likelihood(xx))


            # Inner loop:
            # Propose xprimes and shrink interval until good one found
            zz = 0
            while True:
                zz = zz + 1
                if verbose > 1:
                    print('Iteration %d   Step %d' % (ii+1, zz), end='\r')
                xprime[dd] = np.random.rand()*(x_r[dd] - x_l[dd]) + x_l[dd]
                log_Px = logdist_vec(xprime)
                if log_Px > log_uprime:
                    break # this is the only way to leave the while loop
                else:
                    # Shrink in
                    if xprime[dd] > xx[dd]:
                        x_r[dd] = xprime[dd]
                    elif xprime[dd] < xx[dd]:
                        x_l[dd] = xprime[dd]
                    else:
                        raise Exception('BUG DETECTED: Shrunk to current '
                            + 'position and still not acceptable.')
            xx[dd] = xprime[dd]
            x_l[dd] = xprime[dd]
            x_r[dd] = xprime[dd]

        # Record samples
        if ii >= 0:
            samples[ii, ...] = np.reshape(xx, xx_shape)
    if verbose:
        print('')
    return samples

