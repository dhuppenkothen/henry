
#import seaborn as sns
#sns.set_context("notebook", font_scale=2.5, rc={"axes.labelsize": 26})
import matplotlib.pyplot as plt
from pylab import *

import numpy as np
import utils

def plot_gbm_histogram(datadir="./"):

    tte_bursts, ctime_bursts, no_obs, obs = utils.load_gbm_bursts(datadir)
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    h, bins, patches = ax.hist(ctime_bursts, bins=500, histtype="stepfilled", color="black", edgecolor="none")
    binsize = bins[1] - bins[0]
    ymax = np.max(h)*1.1
    for c in no_obs:
        ax.fill_between(c, 0.0, ymax, facecolor="grey", alpha=0.2, edgecolor="none")
    ax.scatter(ctime_bursts, np.ones(len(ctime_bursts))*ymax*0.95, color="red")
    ax.set_xlabel("Time since Jan 22, 2009 00:00 [seconds]")
    ax.set_ylabel("Number of bursts in a %i-second interval."%int(binsize))
    ax.set_xlim([np.min([np.min(obs), np.min(no_obs)]), np.max([np.max(obs), np.max(no_obs)])])
    ax.set_ylim([0, ymax])
    plt.savefig("sgr1550_gbm_hist.pdf", format="pdf")
    plt.close()
 
    return

