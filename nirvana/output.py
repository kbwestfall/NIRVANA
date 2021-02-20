import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
import multiprocessing as mp

from astropy.io import fits
from astropy.table import Table,Column

from .plotting import fileprep
from .fitting import bisym_model
from .models.axisym import AxisymmetricDisk
from .models.geometry import projected_polar

def extractfile(f):
    try: 
        #get info out of each file and make bisym model
        args, resdict, chains, meds = fileprep(f)
        velmodel,sigmodel = bisym_model(args, resdict, plot=True)

        #axisym fit holding xc, yc, pa, and inc constant
        fit = AxisymmetricDisk()
        fit.lsq_fit(args,p0=[resdict['xc'], resdict['yc'], resdict['pa'], resdict['inc'], 0, 100, 10], fix = [1, 1, 1, 1, 0, 0, 0])
        symmodel = fit.model()

        #fractional difference between bisym and axisym
        asym = np.abs((velmodel-symmodel)/velmodel)
        median = np.ma.median(asym)

    #failure if bad file
    except:
        median, args, asym, resdict = (None, None, None, None)

    return median, args, asym, resdict

def extractdir(cores=10, directory='/data/manga/digiorgio/nirvana/'):
    '''
    Scan an entire directory for nirvana output files and extract useful data from them.
    '''

    #find nirvana files
    fs = glob(directory + '*.nirv')
    with mp.Pool(cores) as p:
        out = p.map(extractfile, fs)

    medians = np.zeros(len(fs))
    galaxies = np.zeros(len(fs), dtype=object)
    asyms = np.zeros(len(fs), dtype=object)
    dicts = np.zeros(len(fs), dtype=object)
    for i in range(len(out)):
        medians[i], galaxies[i], asyms[i], dicts[i] = out[i]

    return medians, galaxies, asyms, dicts

def dictformatting(d, drp=None, dap=None):
    #load dapall and drpall
    if drp is None:
        drp = fits.open('/data/manga/spectro/redux/MPL-10/drpall-v3_0_1.fits')[1].data
    if dap is None:
        dap = fits.open('/data/manga/spectro/analysis/MPL-10/dapall-v3_0_1-3.0.1.fits')[1].data
    try:
        data = list(d.values())
        for i in range(len(data)):
            #put arrays into longer array to make them the same length
            if type(data[i]) is np.ndarray:
                dnew = np.zeros(20)
                dnew[:len(data[i])] = data[i]
                data[i] = dnew

        #make mask to get rid of extra padding in arrays
        velmask = np.ones(20,dtype=bool)
        velmask[:len(d['vt'])] = False
        sigmask = np.ones(20,dtype=bool)
        sigmask[:len(d['sig'])] = False

        #corresponding indicies in dapall and drpall
        drpindex = np.where(drp['plateifu'] == f"{d['plate']}-{d['ifu']}")[0][0]
        dapindex = np.where(dap['plateifu'] == f"{d['plate']}-{d['ifu']}")[0][0]
        data += [velmask, sigmask, drpindex, dapindex]

    #failure for empty dict
    except:
        data = None

    return data

def makealltable(dicts, outfile=None):
    '''
    Take a list of dictionaries from extractdir and turn them into an astropy table (and a fits file if a filename is given).
    '''


    #load dapall and drpall
    drp = fits.open('/data/manga/spectro/redux/MPL-10/drpall-v3_0_1.fits')[1].data
    dap = fits.open('/data/manga/spectro/analysis/MPL-10/dapall-v3_0_1-3.0.1.fits')[1].data

    #make names and dtypes for columns
    names = list(dicts[0].keys()) + ['velmask','sigmask','drpindex','dapindex']
    dtypes = ['D','D','D','D','D','D','20D','20D','20D','20D',
              'D','D','D','D','D','D','D','D','D','D','D','D',
              '20D','20D','20D','20D','20D','20D','20D','20D',
              'I','I','S','20L','20L','I','I']

    data = []
    for d in dicts: data += [dictformatting(d, drp, dap)]

    t = Table(names=names,dtype=dtypes)
    for d in data: t.add_row(d)

    #correct bad data types
    for n in names:
        if t[n].dtype in [np.complex64, np.complex128]: t[n].dtype = np.float64

    #rearrange columns
    t = t['plate','ifu','type','drpindex','dapindex',
          'inc','pa','pab','vsys','vt','v2t','v2r','sig','velmask','sigmask',
          'incl','pal','pabl','vsysl','vtl','v2tl','v2rl','sigl',
          'incu','pau','pabu','vsysu','vtu','v2tu','v2ru','sigu']
    
    #write if desired
    if outfile is not None: t.write(outfile, format='fits', overwrite=True)
    return t

def imagefits(f, outfile=None):
    '''
    Make a fits file for an individual galaxy with its fit parameters and relevant data.
    '''

    #get relevant data
    args, resdict, chains, meds = fileprep(f)

    #dapall and drpall indices
    drp = fits.open('/data/manga/spectro/redux/MPL-10/drpall-v3_0_1.fits')[1].data
    dap = fits.open('/data/manga/spectro/analysis/MPL-10/dapall-v3_0_1-3.0.1.fits')[1].data
    drpindex = np.where(drp['plateifu'] == f"{resdict['plate']}-{resdict['ifu']}")[0][0]
    dapindex = np.where(dap['plateifu'] == f"{resdict['plate']}-{resdict['ifu']}")[0][0]

    velmask = np.ones(20,dtype=bool)
    velmask[:len(resdict['vt'])] = False
    sigmask = np.ones(20,dtype=bool)
    sigmask[:len(resdict['sig'])] = False

    names = list(dicts[0].keys()) + ['velmask','sigmask','drpindex','dapindex']
    dtypes = ['D','D','D','D','D','D','20D','20D','20D','20D',
              'D','D','D','D','D','D','D','D','D','D','D','D',
              '20D','20D','20D','20D','20D','20D','20D','20D',
              'I','I','S','20L','20L','I','I']

    t = Table(names=names,dtype=dtypes)
    hdus = [fits.PrimaryHDU(),fits.BinTableHDU(t)]
    for n in ['vel','sig_phys2','sb','vel_ivar','sig_ivar','sb_ivar','vel_mask']:
        hdus += [fits.ImageHDU(args.remap(n))]
    hdul = fits.HDUList(hdus)
    if outfile is None: 
        hdul.writeto(f"nirvana_{resdict['plate']}-{resdict['ifu']}_{resdict['type']}.fits")
    else: hdul.writeto(outfile)