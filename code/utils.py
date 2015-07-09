

from __future__ import with_statement
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


#### READ ASCII DATA FROM FILE #############
#
# This is a useful little function that reads
# data from file and stores it in a dictionary
# with as many lists as the file had columns.
# The dictionary has the following architecture
# (e.g. for a file with three columns):
#
# {'0':[1st column data], '1':[2nd column data], '2':[3rd column data]}
#
#
# NOTE: Each element of the lists is still a *STRING*, because
# the function doesn't make an assumption about what type of data
# you're trying to read! Numbers need to be converted before using them!
#
def conversion(filename):
    f=open(filename, 'r')
    output_lists=defaultdict(list)
    for line in f:
        if not line.startswith('#'):
             line=[value for value in line.split()]
             for col, data in enumerate(line):
                 output_lists[col].append(data)
    return output_lists




def load_data(datadir="./"):

    gbm_bursts, no_obs, obs = load_gbm_bursts(datadir=datadir)

    integral_bursts = np.loadtxt(datadir+"sgr1550_integral_bursts.dat")[:,2]

    return gbm_bursts, integral_bursts, no_obs, obs

#### LOAD FERMI/GBM DATA #######################
#
#
#
def load_gbm_bursts(datadir="./"):
    ## TTE Bursts are in um, MJD? Possible MET - MJDREF? Who knows?
    tte_bursts = conversion(datadir+"SGR1550jan22burstpeaks.txt")
    tte_bursts = [float(t) for t in tte_bursts[0] if not "--" in t]

    ### CTIME seems to be in MET seconds?
    ctime_bursts = conversion(datadir+"SGR1550jan22untrig.txt")
    ctime_met = np.array([np.float(t) for t in ctime_bursts[1]])
    ctime_utc = ctime_bursts[2]


    ## Time conversion from UTC hh:mm:ss format to seconds since 00:00:00
    ctime_utc_split = [c.split(":") for c in ctime_utc]
    ctime_utc_seconds = np.array([np.float(c[2]) + np.float(c[1])*60. +
                         np.float(c[0])*3600.0 for c in ctime_utc_split])

    ## time conversions from CTIME MET to seconds since start of Jan 22, 2009
    mjdrefi = 51910.0
    mjdreff = 7.428703703703703e-4

    ## days since MET reference time
    ctime_days = ctime_met/(24.*60.*60)
    ctime_mjd = ctime_days + mjdrefi #+ mjdreff

    ## MJD of 2009-01-22
    mjd = 54853

    ctime_frac = ctime_mjd - mjd
    ctime_seconds = ctime_frac*(60.*60.*24.)-3.

    bursts_all = combine_tte_ctime(tte_bursts, ctime_seconds, datadir=datadir)

    ### load start and end times of periods with no observations
    no_obs = np.loadtxt(datadir+"SGR1550jan22nonobs.txt")
    bursts_all = remove_occulted(bursts_all, no_obs)

    tte_data = np.loadtxt(datadir+"SGR1550jan22ttedata.txt")

    ## FIXME: Ask Yuki about when the actual start time of the burst search was

    tte_data_start = tte_data[0,0]
    #data_start = 2000.0

    obs = obs_times(no_obs, tte_data_start)
    return bursts_all, no_obs, obs


def remove_occulted(tte_bursts, no_obs):
    """
    Remove bursts during occulted intervals from GBM data.

    """
    for n in no_obs:
        occ_ind = np.where((n[0] < tte_bursts) & (tte_bursts < n[1]))[0]
        if len(occ_ind) > 0:
            #print(occ_ind)
            tte_bursts = np.delete(tte_bursts, occ_ind)

    return tte_bursts

def combine_tte_ctime(tte_bursts, ctime_seconds, datadir="./"):
    """
    Combine bursts from TTE and CTIME data in such a way that TTE
    bursts will be used when there is TTE data available, and CTIME
    bursts when there is not.

    """

    tte_data = np.loadtxt(datadir+"SGR1550jan22ttedata.txt")
    bursts_all = []

    tte_bursts = np.array(tte_bursts)

    no_tte_start= 0.0
    for t in tte_data:

        no_tte_end = t[0]
        tte_start = t[0]
        tte_end = t[1]

        bursts_ind = np.where((no_tte_start <= ctime_seconds) & (ctime_seconds <= no_tte_end))[0]
        bursts_all.extend(ctime_seconds[bursts_ind])

        bursts_ind = np.where((tte_start < tte_bursts) & (tte_bursts < tte_end))[0]
        bursts_all.extend(tte_bursts[bursts_ind])

        no_tte_start = t[1]

    return np.array(bursts_all)



def obs_times(no_obs, tstart=0.0):
    """
    Turn un-observed periods into observed periods
        no_obs: (L,2) array of start/end of unobserved time periods
    """

    ### for now, assume observations start at first time interval

    obs_start  = [tstart]

    ## add *end* times of unobserved periods
    ## leave out last one because there's no data after that
    obs_start.extend(no_obs[:-1,1])

    ## end times of observed periods are start times of unobserved periods
    obs_end = no_obs[:,0]

    obs = np.array([obs_start, obs_end]).T
    return obs

def loghist(data, bins=10, normed=False, weights=None):
    mindata = np.log10(np.min(data))
    maxdata = np.log10(np.max(data))
    bin_range = np.logspace(mindata, maxdata, bins)

    h, bin_edges = np.histogram(data, bins=bin_range, normed=normed, weights=weights)
    bin_diff = np.diff(bin_edges)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(bin_edges[:-1], h, width=bin_diff, color="grey", edgecolor="none", linewidth=0, alpha=0.4)
    ax.set_xscale("log")
    return fig, ax
