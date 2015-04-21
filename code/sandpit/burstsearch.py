### SEARCHING FOR BURSTS IN DATA
#
# Follows Lin+, 2013, ApJ 778:105, 11pp
#
# Uses Bayesian Blocks
import numpy as np
import glob
import pandas as pd
from astroML.density_estimation import bayesian_blocks
import argparse

def read_data(f):
    """
    Read data from a file name into a Pandas data frame.
    Assumes that the data comes in a very specific format, using the script
    on http://isdc.unige.ch/~savchenk/spiacs-online/spiacs.pl.
    :param f: filename
    :return: pandas DataFrame object
    """
    fl = open(f, "r")
    header = fl.readlines()[:2]
    fl.close()
    h = header[0].split()
    date = h[2][1:-1]
    start_time = np.float(h[3])
    df = pd.read_table(f, skip_blank_lines=True, skiprows=2,
                        header=0, names=("Time", "Counts"), delim_whitespace=True)
    df.date = date
    df.start_time = start_time
    df["Time"] = df["Time"] + df.start_time
    df.length = df.shape[0]
    print("start time for file %s: %.3f"%(f, start_time))
    return df

def combine_integral(datadir="./", fileroot="sgr1550_integral"):
    """
    Combines all integral data files downloaded from the website into a single file.
    Searches for all files with string fileroot in its name in directory datadir.

    :param datadir: directory with data
    :param fileroot: string to match the file names again
    :return: concatenated pandas DataFrame
    """
    files = glob.glob(datadir+fileroot+"*.dat")
    #print(files)
    df_all = []
    for f in files[1:3]:
         df_all.append(read_data(f))

    concatenated = pd.concat(df_all, ignore_index=True)
    concatenated.length = concatenated.shape[0]
    return concatenated



####################################################################################

def run_bayesianblocks(times, counts, p0):
    edges = bayesian_blocks(times, counts, fitness="events", p0=p0)
    #print("edges: " +str(edges))
    binned_counts = []
    for i in range(len(edges[:-1])):
        #print("first edge: " + str(edges[i]))
        #print("second edge: " +str(edges[i+1]))
        #print("times: " + str(times[:10]))
        e_inds = np.where((edges[i] <= times) & (times <= edges[i+1]))
        #print(e_inds)
        binned_counts.append(np.sum(counts[e_inds]))
    return edges, binned_counts


def initial_search(df, seg=20, timestep=10, p0=0.001):
    """
    Initial blind search for bursts.
    :param df: pandas Data Frame with data.
    :param seg: length of time segment to search
    :param timestep: step between consecutive segments
    :param p0: false alarm probability for Bayesian Blocks algorithm
    :return:
    """


    start_time = df.loc[df.index[0]]["Time"]
    end_time = df.loc[df.length-1]["Time"]


    t_start = start_time
    t_end = t_start + seg

    e_all, c_all = [], []
    while t_end < end_time:
        #print("start time: " + str(t_start))
        start_ind = df["Time"].searchsorted(t_start)[0]
        #print("start_ind: " + str(start_ind))
        end_ind =  df["Time"].searchsorted(t_end)[0]
        #print("end_ind: " + str(end_ind))

        df_temp = df[start_ind:end_ind]
        #df_temp.length = df_temp.shape[0]
        #df_temp.index = df_temp.index - df_temp.index[0]
        if len(df_temp) == 0:
            t_start += timestep
            t_end += timestep
            continue

        #print("seg: " + str(df_temp.loc[df_temp.index[-1]-1]["Time"] - df_temp.loc[df_temp.index[0]]["Time"]))

        ## throw out segments that are shorter than seg:
        if (df_temp.loc[df_temp.index[-1]-1]["Time"] - df_temp.loc[df_temp.index[0]]["Time"]) < seg-1.0:
            #print("Throwing out short segment")
            t_start += timestep
            t_end += timestep
            continue

        ## run Bayesian Blocks
        edges, binned_counts = run_bayesianblocks(np.array(df_temp["Time"]), np.array(df_temp["Counts"]), p0)
        e_all.append(edges)
        c_all.append(binned_counts)

        t_start += timestep
        t_end += timestep

    return e_all, c_all


def find_candidates(df, e_all, period=2.7):

    candidates = []
    for e in e_all:
        e = e[1:-1]
        #print("edges: " + str(e))
        c_ind = np.where(np.diff(e) < period)[0]
        #print("c_ind: " + str(c_ind))
        if len(c_ind) > 0:
            #print(candidates)
            for c in c_ind:
                #print("c: " + str(c))
                if len(candidates) == 0:
                    #print("appending first candidate: " + str(e[c]) + "\t" + str(e[c+1]))
                    candidates.append([e[c], e[c+1]])
                    continue
                else:
                    print(candidates[-1][0])
                cstart = [can[0] for can in candidates]
                cend = [can[1] for can in candidates]
                #if not (any(cstart) <= e[c] <= any(cend)) and not\
                #        (any(cstart) <= e[c+1] <= any(cend)):
                if not any([can[0] <= e[c] <= can[1] for can in candidates]) or not\
                    any([can[0] <= e[c+1] <= can[1] for can in candidates]):
                        candidates.append([e[c], e[c+1]])


    return np.sort(np.array(candidates))

def candidate_segments(df, candidates, add_on = 10):
    """
    Extract candidate segments: group together all blocks that are less than
    add_on seconds apart:
    :param df: pandas.DataFrame with Times and Counts data
    :param candidates: Nx2 array of N candidate block start/end times
    :param add_on: float number that sets the separation between two blocks to be considered in a new segment
    :return: Mx2 numpy array of segment start and end times
    """
    csorted = np.sort(candidates, axis=0)
    cdiff = csorted[1:,0] - csorted[:-1,1]
    cind = np.where(cdiff >  add_on)[0]

    seg_all = []
    cstart = csorted[0,0]
    for i,c in enumerate(csorted[:-1]):
        if csorted[i+1,0]- c[1] > 10.0:
            cend = c[1] + add_on
            seg_all.append([cstart, cend])
            cstart = csorted[i+1,0] - add_on
        else:
            continue

    return np.array(seg_all)

def second_run(df, seg_all, p0=0.001):

    e_sec, c_sec = [], []
    for s in seg_all:
        t_start = s[0]
        t_end = s[1]
        #print("start time: " + str(t_start))
        start_ind = df["Time"].searchsorted(t_start)[0]
        #print("start_ind: " + str(start_ind))
        end_ind =  df["Time"].searchsorted(t_end)[0]
        #print("end_ind: " + str(end_ind))

        df_temp = df[start_ind:end_ind]

        #print("seg: " + str(df_temp.loc[df_temp.index[-1]-1]["Time"] - df_temp.loc[df_temp.index[0]]["Time"]))

        ## throw out segments that are shorter than seg:

        ## run Bayesian Blocks
        edges, binned_counts = run_bayesianblocks(np.array(df_temp["Time"]), np.array(df_temp["Counts"]), p0)
        e_sec.append(edges[1:-1])
        c_sec.append(binned_counts[1:-1])

    return e_sec, c_sec


def find_bursts(e_sec, c_sec, period=2.7):

    bursts = []
    for edges,counts in zip(e_sec, c_sec):
        #if len(edges) == 2:
        #    continue
        ## compute duration of each block
        dt = np.diff(edges)
        ## compute count rate
        countrate = counts/dt
        bkg_counts = [c for t,c in zip(dt, countrate) if t >= 6.0]
        bkg_dur = [t for t,c in zip(dt, countrate) if t >= 6.0]

        #print(len(bkg_counts))
        #print(len(bkg_dur))
        bkg = np.average(bkg_counts, weights=bkg_dur)
        std = np.sqrt(np.average((bkg_counts-bkg)**2., weights=bkg_dur))
        #print("bkg: " + str(bkg))
        #print("std: " + str(std))
        #print("The background rate is: " + str(bkg))
        cind = np.where((countrate > 3.*std+bkg) & (dt < period/4.))[0]
        #print("dt: " + str(dt))
        #print("cind: " + str(cind))
        if len(cind) == 0:
            cind = [0]
        if len(cind) == 1:
           block_ind = [[cind[0]]]
        else:
        #edges_burst = edges[cind]
            block_ind = []
            ctemp = []
            for c,d in zip(cind[:-1], np.diff(cind)):
                #print("c: " + str(c))
                #print("d: " + str(d))
                ctemp.append(c)
                if d > 1:
                    block_ind.append(ctemp)
                    ctemp = []
                else:
                    continue
            if len(ctemp) > 0:
                block_ind.append(ctemp)

        #print("block_ind: " + str(block_ind))
        if np.size(block_ind) == 0:
            continue
        for b in block_ind:
            bstart = edges[b[0]]
            bend = edges[b[-1]]
            bursts.append((bend-bstart)/2.0+bstart)


        #for i,(e,c) in enumerate(zip(edges[cind[0]:-1], countrate[cind[0]:])):
        #    print("bstart: " + str(bstart))
        #    print("distance: " + str(edges[cind[0]+i+1]-e))
        #    if edges[cind[0]+i+1]-e < period/4.0 and c > bkg + 3.*std:
        #        continue
        #    else:
        #        bend = edges[i+cind[0]]
        #        print("Saving burst: " + str(bstart) + "\t" + str(bend))
        #        bursts.append((bend-bstart)/2.+bstart)
        #        bstart = edges[i+cind[0]]

    return bursts



def main():

    ## 1) Load Data
    df = read_data(filename)

    ## 2) Initial run
    e_all, c_all = initial_search(df, seg=seg, timestep=timestep, p0=p0)

    ## 3) Extract all candidate blocks with duration < neutron star spin period
    candidates = find_candidates(df, e_all, period=period)

    ## 4) make segments for second run
    seg_all = candidate_segments(df, candidates, add_on=add_on)

    e_sec, c_sec, = second_run(df, seg_all, p0=p0)

    bursts = find_bursts(e_sec, c_sec, period=period)
    np.savetxt(output, bursts)

    return

    ## parameters:
    ## seg
    ## timestep
    ## p0
    ## period
    ## add_on

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian Blocks magnetar burst search.")

    parser.add_argument("-f", "--filename", action="store", required=True, dest="filename",
                        help="Set filename for input data.")

    parser.add_argument("-p", "--p0", action="store", required=False, dest="p0", default=0.001, type=float,
                        help="False alarm probability for Bayesian Blocks algorithm. Default: 0.001")
    parser.add_argument("-s", "--segment", action="store", required=False, dest="seg", default=20., type=float,
                        help = "Length of segments to search. Default: 20 seconds "
                               "(or whatever time units the data is in.")
    parser.add_argument("-t", "--timestep", action="store", required=False, dest="timestep", default=10., type=float,
                        help = "Timestep between starting points of segments to search."
                               "Note: if timestep < seg, then segments overlap.")
    parser.add_argument("--period", action="store", required=False, dest="period", type=float, default=10.,
                        help = "This quantity sets the *distance* between two bins required to be separate bursts."
                               "Actual distance used in the code is period/4. Set to a really small value to"
                               "disable.")
    parser.add_argument("-a", "--addon", action="store", required=False, dest="add_on", type=float, default=10.0,
                        help = "Sets the minimum distance between two candidate bin groups to be used. The larger this"
                               "value, the more background is used for computing the background noise level.")

    parser.add_argument("-o", "--output", action="store", required=False, dest="output", default="bursts.dat",
                        help = "Set filename for output file with burst mid times. Default: 'bursts.dat'")

    args = parser.parse_args()
    filename=args.filename
    p0 = args.p0
    seg = args.seg
    timestep = args.timestep
    period = args.period
    add_on = args.add_on
    output = args.output

    main()