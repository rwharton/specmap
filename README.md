# specmap
Script to make plots of periodic RFI

usage: specmap.py [-h] [-t TSPEC] [-df DF] [-fsig FSIG] -o OUTBASE infile

Make diagnostic plots to find RFI periodicities in fil file

positional arguments:
  infile                Input *.fil file

optional arguments:
  -h, --help            show this help message and exit
  -t TSPEC, --tspec TSPEC
                        Time chunk for making FFTs in sec (default=60)
  -df DF, --df DF       Output freq resolution in Hz (default=-1, native)
  -fsig FSIG, --fsig FSIG
                        Freq peak SNR (default=20)
  -o OUTBASE, --outbase OUTBASE
                        Output file basename for plots (no suffix)
