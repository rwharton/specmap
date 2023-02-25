# specmap
Script to make plots of periodic RFI. 

De-disperses data to DM=0 pc/cc, and calculates the power 
spectrum for short time chunks (def: 60 sec) of data. Will 
produce four plots showing the spectum vs time: the full range 
out to Nyquist, 0-300 Hz, 0-100 Hz, and 0-30 Hz.  Will also 
produce a list of frequency peaks above a given threshold.

## Requirements 

Assumes you have `your` installed already.

## Usage

    usage: specmap.py [-h] [-t TSPEC] [-df DF] [-fsig FSIG] -o OUTBASE infile

    Make diagnostic plots to find RFI periodicities in fil file

    positional arguments:
        infile                Input *.fil file

    optional arguments:
        -h, --help              show this help message and exit
        -t TSPEC, --tspec TSPEC
                                Time chunk for making FFTs in sec (default=60)
        -df DF, --df DF         Output freq resolution in Hz (default=-1, native)
        -fsig FSIG, --fsig FSIG
                                Freq peak SNR (default=20)
        -o OUTBASE, --outbase OUTBASE
                                Output file basename for plots (no suffix)

## Output

Will produce four plots and one text file.  All will start 
with `outbase`. 
