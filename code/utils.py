import numpy as np
import generaltools as gt


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
    ctime_bursts = np.array([np.float(t) for t in ctime_bursts[1]])

    ## time conversions from CTIME MET to seconds since start of Jan 22, 2009 
    mjdrefi = 51910.0
    #mjdreff = 7.428703703703703e-4

    ctime_days = ctime_bursts/(24.*60.*60.)
    ctime_mjd = ctime_days + mjdrefi#+mjdreff

    jd = 2454853.500000
    mjd = jd - 2400000.5

    ctime_frac = ctime_mjd - mjd
    ctime_seconds = ctime_frac*24.*60.*60.


    ### load start and end times of periods with no observations
    noobs = np.loadtxt(datadir+"SGR1550jan22nonobs.txt")
    return tte_bursts, ctime_seconds, noobs



#for t in tte_bursts:
#    f = np.where((ctime_bursts > t-0.1) & (ctime_bursts < t + 0.1))
#    print(f[0])
#    if len(f) > 0:
#        common_bursts.append([t, ctime_bursts[f[0]]])


