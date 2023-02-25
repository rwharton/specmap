import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from argparse import ArgumentParser
import your 
import time

def rfi_spec(infile, tchunk=10.0):
    """
    Get spec
    """
    yr    = your.Your(infile)
    fch1  = yr.fch1
    foff  = yr.foff
    dt    = yr.tsamp
    nspec = yr.your_header.nspectra
    nchan = yr.nchans

    nt_chunk = int(tchunk / dt)
    nchunks = int( nspec / nt_chunk )

    # Initialize output
    nf = int(nt_chunk / 2) + 1
    spec = np.zeros( (nchunks, nf ) )
    ffk = np.arange(nf) / (nt_chunk * dt)

    nstart = 0
    for ii in range(nchunks):
        dat = yr.get_data(nstart=nstart, nsamp=nt_chunk)
            
        # De-disperse at zero DM
        dd = np.sum(dat, axis=1)

        # Make spectrum
        ak = np.fft.rfft(dd)
        ppk = np.abs(ak)**2.0

        # Add to array
        spec[ii] = ppk

    return ffk, spec


def avg_spec(ffk, spec, df_out=1.0):
    """
    Average spec such that output spectrum 
    has approx output of df_out
    """
    df_in = np.diff(ffk)[0]
    if df_out <= df_in:
        return ffk, spec
    else:
        pass
    avg_fac = int( df_out / df_in )
    nf_out = int( spec.shape[1] / avg_fac )
    
    spec_avg = spec[:, 0:avg_fac * nf_out]
    spec_avg = np.reshape(spec_avg, (-1, nf_out, avg_fac))
    spec_avg = np.mean(spec_avg, axis=-1)

    ffk_out = (np.arange( nf_out ) + 0.5) * avg_fac * df_in 
    
    return ffk_out, spec_avg


def middle_stats(dd, frac=0.90):
    """
    Get stats by sorting and only using 
    the inner frac% of data
    """
    dds = np.sort(dd.ravel())
    frac_lo = (1 - frac) / 2.
    frac_hi = 1 - frac_lo
    idx_lo = int(frac_lo * len(dds))
    idx_hi = int(frac_hi * len(dds))
    ddm = dds[ idx_lo : idx_hi ]

    m_med = np.median(ddm)
    m_std = np.std(ddm)

    return m_med, m_std


def get_spec_peaks(ffk, spec_avg, sig=20, fmin=1.0):
    """
    get peaks in avg spec 
    """
    mma, ssa = middle_stats(spec_avg)
    spec_avg /= mma
    ssa /= mma

    xx = np.where( spec_avg > 1 + sig * ssa )[0]
    ffpks = ffk[xx]

    ffsigs = (spec_avg[xx] - 1) / ssa

    yy = np.where( ffpks > fmin )[0]
    ffpks  = ffpks[yy]
    ffsigs = ffsigs[yy] 
    
    return ffpks, ffsigs


def write_peaks(outbase, ffpks, ffsigs):
    """
    Write freq and sigs to file
    """
    hdr_fmt = "# {:^10} {:^10}\n"  
    dat_fmt = "{:<10.2f} {:>10.1f}\n"
    outfile = "%s_freqs.txt" %outbase

    with open(outfile, 'w') as fout:
        hdr1 = hdr_fmt.format("Freq", "SNR")
        hdr2 = hdr_fmt.format("(Hz)", "")
        hdr3 = "#" + "=" * len(hdr1) + "\n"
        fout.write( hdr1 + hdr2 + hdr3 )

        for ii in range(len(ffpks)):
            ff_ii  = ffpks[ii]
            snr_ii = ffsigs[ii]
            fout.write( dat_fmt.format(ff_ii, snr_ii) )
    return 
        


def make_rfi_plot(ffk, spec, mm, ss, tchunk, sig=20, 
                  xlim=None, outname=None, ret_pks=False):
    """
    make one rfi plot for xlim 
    """
    if outname is not None:
        plt.ioff()

    # Parse limits
    if xlim is None:
        xmin = ffk[0]
        xmax = ffk[-1]
    else:
        xmin, xmax = xlim 

    # Make avg spec
    spec_avg = np.mean(spec, axis=0)
    mma, ssa = middle_stats(spec_avg)
    spec_avg /= mma 
    ssa /= mma
    sa_max = 1.1 * np.max( spec_avg[ ffk > 2.0 ] )
    smax = np.max( (1 + 20 * ssa, sa_max) )

    # Set up figure
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(3, 2, figure=fig) 
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1:, :])

    # Make spectrum vs time
    ext = [ ffk[0], ffk[-1], 0, len(spec) * tchunk ]
    ax2.imshow(spec, aspect='auto', origin='lower', 
               extent=ext, vmax= mm + 20 * ss )

    # average spec
    ax1.plot(ffk, spec_avg, zorder=2)
    
    # Get peaks and plot
    ffpks, ffsigs = get_spec_peaks(ffk, spec_avg, sig=sig, fmin=1.0)
    #print(ffpks[ffpks > 2 ])
    for ffi in ffpks:
        ax1.axvline(x=ffi, color='r', ls='--', alpha=0.5, zorder=1)

    ax1.set_xlim(ffk[0], ffk[-1])
    #ax1.set_yscale('log') 
    ax1.set_ylim(0, smax)

    # Set xlims
    ax1.set_xlim( (xmin, xmax) )
    ax2.set_xlim( (xmin, xmax) )

    ax1.set_ylabel("Fourier Power", fontsize=14)
    ax2.set_xlabel("Fourier Frequency (Hz)", fontsize=14)
    ax2.set_ylabel("Time (s)", fontsize=14)

    if outname is not None:
        plt.savefig("%s.png" %outname, dpi=150, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()

    if ret_pks:
        return ffpks, ffsigs
    else:
        return 



def make_rfi_plots(infile, outbase, tchunk=60.0, df_out=None, sig=20):
    """ 
    Make rfi plots
    """
    ffk, spec = rfi_spec(infile, tchunk=tchunk)

    if df_out is not None:
        ffk1, spec1 = avg_spec(ffk, spec, df_out=df_out)
        ffk = ffk1
        spec = spec1 
    else:
        pass

    mm, ss = middle_stats(spec)

    # Make full plot
    outname = "%s_full" %outbase
    ffpks, ffsigs = make_rfi_plot(ffk, spec, mm, ss, tchunk, xlim=None, 
                          outname=outname, ret_pks=True, sig=sig)
    
    # Make plot to 300 Hz
    outname = "%s_300Hz" %outbase
    make_rfi_plot(ffk, spec, mm, ss, tchunk, xlim=(0,300), 
                  sig=sig, outname=outname)
    
    # Make plot to 100 Hz
    outname = "%s_100Hz" %outbase
    make_rfi_plot(ffk, spec, mm, ss, tchunk, xlim=(0,100), 
                  sig=sig, outname=outname)
    
    # Make plot to 30 Hz
    outname = "%s_030Hz" %outbase
    make_rfi_plot(ffk, spec, mm, ss, tchunk, xlim=(0,30), 
                  sig=sig, outname=outname)

    # Write peaks
    write_peaks(outbase, ffpks, ffsigs)  
    
    return

def parse_input():
    """
    Use argparse to parse input
    """
    prog_desc = "Make diagnostic plots to find RFI periodicities in fil file"
    parser = ArgumentParser(description=prog_desc)

    parser.add_argument('infile', help='Input *.fil file')
    parser.add_argument('-t', '--tspec',
                        help='Time chunk for making FFTs in sec (default=60)',
                        required=False, type=float, default=60.0)
    parser.add_argument('-df', '--df',
                        help='Output freq resolution in Hz (default=-1, native)',
                        required=False, type=float, default=-1.0)
    parser.add_argument('-fsig', '--fsig',
                        help='Freq peak SNR (default=20)',
                        required=False, type=float, default=20)
    parser.add_argument('-o', '--outbase',
                        help='Output file basename for plots (no suffix)',
                        required=True)

    args = parser.parse_args()
    
    infile = args.infile
    tspec  = args.tspec
    df_out = args.df
    fsig   = args.fsig
    outbase = args.outbase

    return infile, tspec, df_out, fsig, outbase

debug = 0

if __name__ == "__main__":
    if debug:
        pass
    else:
        tstart = time.time()
        infile, tchunk, df_out, fsig, outbase = parse_input()
        if df_out <= 0:
            df_out = None 
        make_rfi_plots(infile, outbase, tchunk=tchunk, df_out=df_out, sig=fsig)
        tstop = time.time()

        dt = tstop - tstart
        print("Took %.2f minutes" %(dt/60.0))

