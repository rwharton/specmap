import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import multiprocessing as mp
import os
import your 
from numba import jit
from argparse import ArgumentParser
from subprocess import call

matplotlib.use('GTK3Agg')

def run_bp(infile, bpfile):
    bp_cmd = "bandpass %s > %s" %(infile, bpfile)
    print(bp_cmd)
    call(bp_cmd, shell=True)
    return 

def calc_bandpass(infile, workdir):
    """
    Calculate the bandpass for one input file
    using the SIGPROC taksk bandpass

    This will calculate the bandpass by averaging over all 
    the full observation.  We may want to change this later 
    to do it in segments or something to avoid having to 
    flag a whole channel.

    Return bp_file
    """
     
    tstart = time.time()
    
    infname = infile.rsplit('/')[-1] 
    inbase  = infname.rsplit('.', 1)[0]
    bpfile = "%s/%s.bpass" %(workdir, inbase)
    print(infile, bpfile)
    run_bp(infile, bpfile)
    
    tstop = time.time()
    print("Took %.1f minutes" %( (tstop-tstart)/60.))
   
    return bpfile


def get_chan_info(data_file):
    """
    Get channel info
    """
    yr = your.Your(data_file)
    foff = yr.your_header.foff
    fch1 = yr.your_header.fch1
    dt   = yr.your_header.tsamp
    nchans = yr.your_header.nchans

    return nchans, fch1, foff, dt


def calc_bp_stats(infile):
    """
    get channel means (bandpass) and standard 
    deviations using your
    """
    yr = your.Your(infile)
    nspec = yr.your_header.nspectra 
    
    dat = yr.get_data(0, nspec)

    bp_avg = np.mean(dat, axis=0)
    bp_std = np.std(dat, axis=0)

    nchans, fch1, foff, dt = get_chan_info(infile)
    freqs = np.arange(nchans) * foff + fch1

    return freqs, bp_avg, bp_std


def write_bpass(freqs, bp, outfile):
    """
    Write a bandpass file
    """
    with open(outfile, 'w') as fout:
        for ii in range(len(freqs)):
            ffi = freqs[ii] 
            bpi = bp[ii]
            outstr = "{:<12.4f}{:^12.4f}\n".format(ffi, bpi)
            fout.write(outstr)
    return


def your_calc_bandpass(infile, workdir):
    """
    Calculate the bandpass (mean and sig)
    using your 

    Return bp_file
    """
     
    tstart = time.time()
    
    infname = infile.rsplit('/')[-1] 
    inbase  = infname.rsplit('.', 1)[0]
    avg_file= "%s/%s_avg.bpass" %(workdir, inbase)
    std_file= "%s/%s_std.bpass" %(workdir, inbase)

    freqs, bp_avg, bp_std = calc_bp_stats(infile)

    write_bpass(freqs, bp_avg, avg_file)
    write_bpass(freqs, bp_std, std_file)
    
    tstop = time.time()
    print("Took %.1f minutes" %( (tstop-tstart)/60.))
    return avg_file, std_file




def moving_median(data, window):
    """
    Calculate running median and stdev
    """
    startIdxOffset = np.floor(np.divide(window, 2.0))
    endIdxOffset = np.ceil(np.divide(window, 2.0))
    
    startIndex = startIdxOffset
    endIndex = len(data) - 1 - endIdxOffset
    
    halfWindow = 0.0
    
    if (np.mod(window, 2.0) == 0):
        halfWindow = int(np.divide(window, 2.0))
    else:
        halfWindow = int(np.divide(window - 1.0, 2.0))
    
    mov_median = np.zeros(len(data))
    mov_std = np.zeros(len(data))

    startIndex = int(startIndex)
    endIndex = int(endIndex)

    # Calculate the moving median and std. dev. associated 
    # with each interval.
    for i in np.arange(startIndex, endIndex + 1, 1):
        istart = int(i - halfWindow)
        istop  = int(i + halfWindow + 1)
        medianValue = np.median(data[istart : istop])
        stdValue = np.std(data[istart : istop], ddof=1)

        mov_median[i] = medianValue
        mov_std[i] = stdValue
    
    # Set the values at the end points.
    for i in np.arange(0, startIndex, 1):
        mov_median[i] = mov_median[startIndex]
        mov_std[i] = mov_std[startIndex]
    
    for i in np.arange(endIndex + 1, len(data), 1):
        mov_median[i] = mov_median[endIndex]
        mov_std[i] = mov_std[endIndex]

    return mov_median, mov_std;

@jit(nopython=True)
def movingAverage(data, window_size):
    """
    Compute the moving average using brute force numpy cumsum.
    jit to make it faster
    """
    movingAvg = np.cumsum(data.astype(np.float64))
    movingAvg[window_size:] = movingAvg[window_size:] - movingAvg[:-window_size]
    movingAvg = movingAvg[window_size - 1:] / window_size

    # Prepend/append elements to the front/end of the list.
    appendLength = (len(data) - len(movingAvg)) / 2.0
    if (np.mod(appendLength, 1.0) != 0.0):
        prependArray = np.array([movingAvg[0]] * int(appendLength))
        appendArray = np.array([movingAvg[-1]] * int(appendLength + 1))

        movingAvg = np.concatenate((prependArray, movingAvg))
        movingAvg = np.concatenate((movingAvg, appendArray))
    else:
        prependArray = np.array([movingAvg[0]] * int(appendLength))
        appendArray = np.array([movingAvg[-1]] * int(appendLength))

        movingAvg = np.concatenate((prependArray, movingAvg))
        movingAvg = np.concatenate((movingAvg, appendArray))

    return movingAvg
    

def read_bp(bp_filename):
    """
    Read the bandpass file created by SIGPROC bandpass
    """
    # Read data from the file.
    freqs = []
    bp = []
    with open(bp_filename, 'r') as fin:
        for line in fin:
            if line[0] in [" ", "\n"]:
                continue
            else: pass
            cols = line.split()
            freq_val = float(cols[0])
            bp_val = float(cols[1])
            
            freqs.append(freq_val)
            bp.append(bp_val)

    freqs = np.array(freqs)
    bp = np.array(bp)

    return freqs, bp


def plot_bp(freqs, bp, mask_chans, diff_thresh=None, 
            val_thresh=None, outfile=None):
    """
    Plot bandpass with masked chans indicated
    """
    chans = np.arange(0, len(freqs), 1)
    good_chans = np.setdiff1d(chans, mask_chans)

    # if outputting file, turn off interactive mode
    if outfile is not None:
        plt.ioff()
    else:
        plt.ion()
 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #ax.plot(freqs[good_chans], bp[good_chans], 'k.')
    ax.plot(freqs, bp, c='k')
    ax.plot(freqs[mask_chans], bp[mask_chans], ls='', 
            marker='o', mec='r', mfc='none')

    if val_thresh is not None:
        ax.axhline(y=val_thresh, ls='--', c='g', alpha=0.5)
    else: pass

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("BP Coeff")    

    title_str = ""
    if diff_thresh is not None:
        diff_str = "diff_thresh = %.2f" %(diff_thresh)
        title_str += diff_str 
        if val_thresh is not None:
            title_str += ", "
        else: pass
    if val_thresh is not None:
        val_str = "val_thresh = %.2f" %(val_thresh)
        title_str += val_str
    else: pass

    if len(title_str):
        ax.set_title(title_str)
    else: pass

    ax.set_yscale('log')
   
    # If outfile, then save file, close window, and 
    # turn interactive mode back on
    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()
    else: 
        plt.show()

    return

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


def del_chans_to_string(nums):
    """
    Take list of channel numbers to remove and convert 
    them to a string that can be input to PRESTO
    """
    # Get list of ranges 
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    ranges = list(zip(edges, edges))

    # Shorten string using ":" when necessary
    out_str = ""
    for i in np.arange(0, len(ranges), 1):
        if (ranges[i][0] == ranges[i][1]):
            out_str = out_str + str(ranges[i][0]) + ","
        else:
            out_str = out_str + str(ranges[i][0]) +\
                      ":" + str(ranges[i][1]) + ","

    # Remove trailing comma if nec
    if out_str[-1] == ',':
        out_str = out_str.rstrip(',')
    else: pass
    
    return out_str


def bp_filter(bp_file, diff_thresh=0.10, val_thresh=0.1, 
              nchan_win=32, outfile=None):
    """
    Run the filter on a single bandpass file and find 
    what channels need to be zapped

    diff_thresh = fractional diff threshold to mask chans 
                  Mask if abs((bp-med)/med) > diff_thresh
    
    val_thresh  = min value threshold to mask chans 
                  Mask if bp < val_thresh

    nchan_win = number of channels in moving window

    if outfile is specified, then save a plot showing 
    the bandpass and masked channels
    """
    # Read in bp data 
    freqs, bp = read_bp(bp_file)

    # Calculate running median and stdev
    mov_median, mov_std = moving_median(bp, nchan_win)

    # Calc fractional difference from median
    # Fix in case there are any zeros
    abs_med = np.abs(mov_median)
    if np.any(abs_med):
        eps = 1e-3 * np.min( abs_med[ abs_med > 0 ] )
    else:
        eps = 1e-3  
    bp_diff = np.abs(bp - mov_median) / (abs_med + eps)

    # Find mask chans from diff
    diff_mask = np.where( bp_diff >= diff_thresh )[0]

    # Find mask chans from val
    val_mask = np.where( bp < val_thresh )[0]

    # Get unique, sorted list of all bad chans
    mask_chans = np.unique( np.hstack( (diff_mask, val_mask) ) )
    
    # Get list of good chans (might need)
    all_chans = np.arange(0, len(freqs))
    good_chans = np.setdiff1d(all_chans, mask_chans)

    # Make a plot if outfile is specified
    if outfile is not None:
        plot_bp(freqs, bp, mask_chans, diff_thresh=diff_thresh,
                val_thresh=val_thresh, outfile=outfile)
    else:
        pass 
   
    return mask_chans


def bp_bad_chans(infile, workdir, mode='std', diff_thresh=0.10,
                 val_thresh=-1, nchan_win=32, ret_str=True):
    """
    Make bandpass and find bad chans
    """ 
    # Get average and standard deviation of each channel
    avg_file, std_file = your_calc_bandpass(infile, workdir)

    # Do you want to use mean or std for flagging
    if mode=='std':
        bp_file = std_file
        outfile = std_file.split('.bpass', 1)[0] + '.png'
    elif mode=='avg':
        bp_file = avg_file
        outfile = avg_file.split('.bpass', 1)[0] + '.png'
    else:
        print("mode must be one of: avg, std")
        return
    
    # Find bad chans from bp file
    mask_chans = bp_filter(bp_file, diff_thresh=diff_thresh, 
                           val_thresh=val_thresh, 
                           nchan_win=nchan_win, outfile=outfile)

    if ret_str:
        outstr = ",".join(["%d" %mm for mm in mask_chans])
        return outstr
    else:
        return mask_chans


def calc_bp_stats_chunk(infile, tchunk):
    """
    get channel means (bandpass) and standard
    deviations over chunks of duration tchunk
    using the your package
    """
    yr = your.Your(infile)
    nspec = yr.your_header.nspectra
    nchans, fch1, foff, dt = get_chan_info(infile)
    freqs = np.arange(nchans) * foff + fch1

    if tchunk > 0:
        nsteps = int(nspec * dt / tchunk)
        nchunk = min( int(tchunk / dt), nspec )
    else:
        nsteps = 1
        nchunk = nspec

    dt_chunk = dt * nchunk
    tt = np.arange(nsteps) * dt_chunk

    avg_bps = np.zeros( (nsteps, nchans) )
    std_bps = np.zeros( (nsteps, nchans) )

    for ii in range(nsteps):
        print("%d/%d" %(ii+1, nsteps))
        dd = yr.get_data(ii * nchunk, nchunk)
        avg_ii = np.mean(dd, axis=0)
        std_ii = np.std(dd, axis=0)
        avg_bps[ii] = avg_ii
        std_bps[ii] = std_ii

    return tt, freqs, avg_bps, std_bps


def make_plot(tt, ff, bp, outfile=None):
    """
    Make plot
    """ 
    if outfile is not None:
        plt.ioff()
    else: pass

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    
    ext = [ff[0], ff[-1], tt[0], tt[-1]]

    # get median bp
    bpm = np.median(bp, axis=0)
    bpm_inv = np.zeros(len(bpm))
    bpm_inv[np.abs(bpm) > 0] = 1 / bpm[np.abs(bpm) > 0]

    bp_corr = bp * bpm_inv

    im = ax.imshow(bp_corr, aspect='auto', interpolation='nearest', 
                   origin='lower', vmin=0.75, vmax=1.25, extent=ext)

    cbar = plt.colorbar(im)

    ax.set_xlabel("Frequency (MHz)", fontsize=16)
    ax.set_ylabel("Time (s)", fontsize=16)

    if outfile is not None:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        plt.ion()
    else:
        plt.show()

    return
     

def rfi_plot(infile, tchunk, outbase):
    """
    Using the averaged file, make a plot showing the 
    mean and std of bandpass over time chunk tchunk 
    seconds
    """
    # get data
    tt, freqs, bpa, bps = calc_bp_stats_chunk(infile, tchunk)

    # outfiles 
    avg_outfile = "%s_avg_%ds.png" %(outbase, int(tchunk))
    std_outfile = "%s_std_%ds.png" %(outbase, int(tchunk))

    # Make avg plot
    make_plot(tt, freqs, bpa, outfile=avg_outfile)
    
    # Make std plot
    make_plot(tt, freqs, bps, outfile=std_outfile)

    return

   
def parse_input():
    """
    Use argparse to parse input
    """
    prog_desc = "Make RFI Diagonstic Plots"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('infile', 
                        help='Input *.fil file')
    parser.add_argument('tstat', 
              help='Time (in sec) to calc stats (default is 60.0)',
                        type=float, default=60.0)
    parser.add_argument('-o', '--outbase',
              help='Output file basename',
                        required=False)
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    tstart = time.time()
    args = parse_input()
    infile = args.infile
    tstat = args.tstat
    
    if args.outbase is not None:
        outbase = args.outbase
    else:
        outbase = "bp"

    rfi_plot(infile, tstat, outbase)
    tstop = time.time()

    dt = tstop - tstart
    print("Took %.2f minutes" %(dt/60.0))


 
