"""

.. include:: ../include/links.rst
"""

import multiprocessing as mp

import numpy as np
from scipy import stats, optimize

import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except:
    tqdm = None

import dynesty

from .beam import smear, ConvolveFFTW
from .geometry import projected_polar
from ..data.manga import MaNGAGasKinematics, MaNGAStellarKinematics
from ..data.util import trim_shape, unpack
from ..data.fitargs import FitArgs
from ..models.higher_order import bisym_model


def smoothing(array, weight=1):
    """
    A penalty function for encouraging smooth arrays. 
    
    For each bin, it computes the average of the bins to the left and right and
    computes the chi squared of the bin with that average. It repeats the
    values at the left and right edges, so they are effectively smoothed with
    themselves.

    Args:
        array (`numpy.ndarray`_):
            Array to be analyzed for smoothness.
        weight (:obj:`float`, optional):
            Normalization factor for resulting chi squared value

    Returns:
        :obj:`float`: Chi squared value that serves as a measurement for how
        smooth the array is, normalized by the weight.
    """

    edgearray = np.array([array[0], *array, array[-1]]) #bin edges
    avgs = (edgearray[:-2] + edgearray[2:])/2 #average of surrounding bins
    chisq = (avgs - array)**2 / np.abs(array) #chi sq of each bin to averages
    chisq[~np.isfinite(chisq)] = 0 #catching nans
    return chisq.sum() * weight

def unifprior(key, params, bounds, indx=0, func=lambda x:x):
    '''
    Uniform prior transform for a given key in the params and bounds dictionaries.

    Args:
        key (:obj:`str`):
            Key in params and bounds dictionaries.
        params (:obj:`dict`):
            Dictionary of untransformed fit parameters. Assumes the format
            produced :func:`nirvana.fitting.unpack`.
        params (:obj:`dict`):
            Dictionary of uniform prior bounds on fit parameters. Assumes the
            format produced :func:`nirvana.fitting.unpack`.
        indx (:obj:`int`, optional):
            If the parameter is an array, what index of the array to start at.
    
    Returns:
        :obj:`float` or `numpy.ndarray`_ of transformed fit parameters.

    '''
    if bounds[key].ndim > 1:
        return (func(bounds[key][:,1]) - func(bounds[key][:,0])) * params[key][indx:] + func(bounds[key][:,0])
    else:
        return (func(bounds[key][1]) - func(bounds[key][0])) * params[key] + func(bounds[key][0])

def ptform(params, args, gaussprior=False):
    '''
    Prior transform for :class:`dynesty.NestedSampler` fit. 
    
    Defines the prior volume for the supplied set of parameters. Uses uniform
    priors by default but can switch to truncated normal if specified.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        gaussprior (:obj:`bool`, optional):
            Flag to use the (experimental) truncated normal priors.

    Returns:
        :obj:`tuple`: Tuple of parameter values transformed into the prior
        volume.
    '''

    #unpack params and bounds into dicts
    paramdict = unpack(params, args)
    bounddict = unpack(args.bounds, args, bound=True)

    #attempt at smarter posteriors, currently super slow though
    #truncated gaussian prior around guess values
    if gaussprior and args.guess is not None:
        guessdict = unpack(args.guess,args)
        incp  = trunc(paramdict['inc'],guessdict['incg'],2,guessdict['incg']-5,guessdict['incg']+5)
        pap   = trunc(paramdict['pa'],guessdict['pag'],10,0,360)
        pabp  = 180 * paramdict['pab']
        vsysp = trunc(paramdict['vsys'],guessdict['vsysg'],1,guessdict['vsysg']-5,guessdict['vsysg']+5)
        vtp  = trunc(paramdict['vt'],guessdict['vtg'],50,0,400)
        v2tp = trunc(paramdict['v2t'],guessdict['v2tg'],50,0,200)
        v2rp = trunc(paramdict['v2r'],guessdict['v2rg'],50,0,200)

    #uniform priors defined by bounds
    else:
        #uniform prior on sin(inc)
        #incfunc = lambda i: np.cos(np.radians(i))
        #incp = np.degrees(np.arccos(unifprior('inc', paramdict, bounddict,func=incfunc)))
        pap = unifprior('pa', paramdict, bounddict)
        incp = stats.norm.ppf(paramdict['inc'], *bounddict['inc'])
        #pap = stats.norm.ppf(paramdict['pa'], *bounddict['pa'])
        pabp = unifprior('pab', paramdict, bounddict)
        vsysp = unifprior('vsys', paramdict, bounddict)

        #continuous prior to correlate bins
        if args.weight == -1:
            vtp  = np.array(paramdict['vt'])
            v2tp = np.array(paramdict['v2t'])
            v2rp = np.array(paramdict['v2r'])
            vs = [vtp, v2tp, v2rp]
            if args.disp:
                sigp = np.array(paramdict['sig'])
                vs += [sigp]

            #step outwards from center bin to make priors correlated
            for vi in vs:
                mid = len(vi)//2
                vi[mid] = 400 * vi[mid]
                for i in range(mid-1, -1+args.fixcent, -1):
                    vi[i] = stats.norm.ppf(vi[i], vi[i+1], 50)
                for i in range(mid+1, len(vi)):
                    vi[i] = stats.norm.ppf(vi[i], vi[i-1], 50)

        #uncorrelated bins with unif priors
        else:
            vtp  = unifprior('vt',  paramdict, bounddict, int(args.fixcent))
            v2tp = unifprior('v2t', paramdict, bounddict, int(args.fixcent))
            v2rp = unifprior('v2r', paramdict, bounddict, int(args.fixcent))
            if args.disp: 
                sigp = unifprior('sig', paramdict, bounddict)

    #reassemble params array
    repack = [incp, pap, pabp, vsysp]

    #do centers if desired
    if args.nglobs == 6: 
        if gaussprior:
            xcp = stats.norm.ppf(paramdict['xc'], guessdict['xc'], 5)
            ycp = stats.norm.ppf(paramdict['yc'], guessdict['yc'], 5)
        else:
            xcp = unifprior('xc', paramdict, bounddict)
            ycp = unifprior('yc', paramdict, bounddict)
        repack += [xcp,ycp]

    #repack all the velocities
    repack += [*vtp, *v2tp, *v2rp]
    if args.disp: repack += [*sigp]
    return repack

def loglike(params, args, squared=False):
    '''
    Log likelihood for :class:`dynesty.NestedSampler` fit. 
    
    Makes a model based on current parameters and computes a chi squared with
    tht
    original data.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        squared (:obj:`bool`, optional):
            Whether to compute the chi squared against the square of the
            dispersion profile or not. 

    Returns:
        :obj:`float`: Log likelihood value associated with parameters.
    '''

    #unpack params into dict
    paramdict = unpack(params, args)

    #make velocity and dispersion models
    velmodel, sigmodel = bisym_model(args, paramdict)

    #compute chi squared value with error if possible
    llike = (velmodel - args.kin.vel)**2

    #inflate ivar with noise floor
    if args.kin.vel_ivar is not None: 
        vel_ivar = 1/(1/args.kin.vel_ivar + args.noise_floor**2)
        llike = llike * vel_ivar - .5 * np.log(2*np.pi * vel_ivar)
    llike = -.5 * np.ma.sum(llike)

    #add in penalty for non smooth rotation curves
    if args.weight != -1:
        llike = llike - smoothing(paramdict['vt'],  args.weight) \
                      - smoothing(paramdict['v2t'], args.weight) \
                      - smoothing(paramdict['v2r'], args.weight)

    #add in sigma model if applicable
    if sigmodel is not None:
        #compute chisq with squared sigma or not
        if squared:
            sigdata = args.kin.sig_phys2
            sigdataivar = args.kin.sig_phys2_ivar if args.kin.sig_phys2_ivar is not None else np.ones_like(sigdata)
            siglike = (sigmodel**2 - sigdata)**2

        #calculate chisq with unsquared data
        else:
            sigdata = np.sqrt(args.kin.sig_phys2)
            sigdataivar = np.sqrt(args.kin.sig_phys2_ivar) if args.kin.sig_phys2_ivar is not None else np.ones_like(sigdata)
            siglike = (sigmodel - sigdata)**2

        #inflate ivar with noisefloor
        if sigdataivar is not None: 
            sigdataivar = 1/(1/sigdataivar + args.noise_floor**2)
            siglike = siglike * sigdataivar - .5 * np.log(2*np.pi * sigdataivar)
        llike -= .5*np.ma.sum(siglike)

        #smooth profile
        if args.weight != -1:
            llike -= smoothing(paramdict['sig'], args.weight*.1)

    #apply a penalty to llike if 2nd order terms are too large
    if hasattr(args, 'penalty') and args.penalty:
        vtm  = paramdict['vt' ].mean()
        v2tm = paramdict['v2t'].mean()
        v2rm = paramdict['v2r'].mean()

        #scaling penalty if 2nd order profs are big
        llike -= args.penalty * (v2tm - vtm)/vtm
        llike -= args.penalty * (v2rm - vtm)/vtm

    return llike

def fit(plate, ifu, galmeta = None, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-11', nbins=None,
        cores=10, maxr=None, cen=True, weight=10, smearing=True, points=500,
        stellar=False, root=None, verbose=False, disp=True, 
        fixcent=True, method='dynesty', remotedir=None, floor=5, penalty=100,
        mock=None):
    '''
    Main function for fitting a MaNGA galaxy with a nonaxisymmetric model.

    Gets velocity data for the MaNGA galaxy with the given plateifu and fits it
    according to the supplied arguments. Will fit a nonaxisymmetric model based
    on models from Leung (2018) and Spekkens & Sellwood (2007) to describe
    bisymmetric features as well as possible. Uses `dynesty` to explore
    parameter space to find best fit values.

    Args:
        plate (:obj:`int`):
            MaNGA plate number for desired galaxy.
        ifu (:obj:`int`):
            MaNGA IFU design number for desired galaxy.
        daptype (:obj:`str`, optional):
            DAP type included in filenames.
        dr (:obj:`str`, optional):
            Name of MaNGA data release in file paths.
        nbins (:obj:`int`, optional):
            Number of radial bins to use. Will be calculated automatically if
            not specified.
        cores (:obj:`int`, optional):
            Number of threads to use for parallel fitting.
        maxr (:obj:`float`, optional):
            Maximum radius to make bin edges extend to. Will be calculated
            automatically if not specified.
        cen (:obj:`bool`, optional):
            Flag for whether or not to fit the position of the center.
        weight (:obj:`float`, optional):
            How much weight to assign to the smoothness penalty of the rotation
            curves. 
        smearing (:obj:`bool`, optional):
            Flag for whether or not to apply beam smearing to fits.
        points (:obj:`int`, optional):
            Number of live points for :class:`dynesty.NestedSampler` to use.
        stellar (:obj:`bool`, optional):
            Flag to fit stellar velocity information instead of gas.
        root (:obj:`str`, optional):
            Direct path to maps and cube files, circumventing `dr`.
        verbose (:obj:`bool`, optional):
            Flag to give verbose output from :class:`dynesty.NestedSampler`.
        disp (:obj:`bool`, optional):
            Flag for whether to fit the velocity dispersion profile as well.
            2010. Not currently functional
        fixcent (:obj:`bool`, optional):
            Flag for whether to fix the center velocity bin at 0.
        method (:obj:`str`, optional):
            Which fitting method to use. Defaults to `'dynesty'` but can also
            be 'lsq'`.
        remotedir (:obj:`str`, optional):
            If a directory is given, it will download data from sas into that
            base directory rather than looking for it locally
        floor (:obj:`float`, optional):
            Intrinsic scatter to add to velocity and dispersion errors in
            quadrature in order to inflate errors to a more realistic level.
        penalty (:obj:`float`, optional):
            Penalty to impose in log likelihood if 2nd order velocity profiles
            have too high of a mean value. Forces model to fit dominant
            rotation with 1st order profile
        mock (:obj:`tuple`, optional):
            A tuple of the `params` and `args` objects output by
            :func:`nirvana.plotting.fileprep` to fit instead of real data. Can
            be used to fit a galaxy with known parameters for testing purposes.

    Returns:
        :class:`dynesty.NestedSampler`: Sampler from `dynesty` containing
        information from the fit.    
        :class:`~nirvana.data.fitargs.FitArgs`: Object with all of the relevant
        data for the galaxy as well as the parameters used for the fit.
    '''
    nglobs = 6 if cen else 4
    if mock is not None:
        args, params, residnum = mock
        args.kin.vel, args.kin.sig = bisym_model(args, params)
        if residnum:
            try:
                residlib = np.load('residlib.dict', allow_pickle=True)
                vel2d = args.kin.remap('vel')
                resid = trim_shape(residlib[residnum], vel2d)
                newvel = vel2d + resid
                args.kin.vel = args.kin.bin(newvel)
                args.kin.remask(resid.mask)
            except:
                raise ValueError('Could not apply residual correctly. Check that residlib.dict is in the appropriate place')


    #get info on galaxy and define bins and starting guess
    else:
        if stellar:
            kin = MaNGAStellarKinematics.from_plateifu(plate, ifu, daptype=daptype, dr=dr,
                                                        cube_path=root,
                                                        image_path=root, maps_path=root, 
                                                        remotedir=remotedir)
        else:
            kin = MaNGAGasKinematics.from_plateifu(plate, ifu, line='Ha-6564', daptype=daptype,
                                                    dr=dr,  cube_path=root,
                                                    image_path=root, maps_path=root, 
                                                    remotedir=remotedir)

        #set basic fit parameters for galaxy
        args = FitArgs(kin, nglobs, weight, disp, fixcent, floor, penalty, points, smearing, maxr)

    #set bin edges
    if galmeta is not None: 
        if mock is None: args.kin.phot_inc = galmeta.guess_inclination()
        args.kin.reff = galmeta.reff

    inc = args.getguess(galmeta=galmeta)[1] if args.kin.phot_inc is None else args.kin.phot_inc
    if nbins is not None: args.setedges(nbins, nbin=True, maxr=maxr)
    else: args.setedges(inc, maxr=maxr)

    #discard if number of bins is too small
    if len(args.edges) - fixcent < 3:
        raise ValueError('Galaxy unsuitable: too few radial bins')

    #define a variable for speeding up convolutions
    #has to be a global because multiprocessing can't pickle cython
    global conv
    conv = ConvolveFFTW(args.kin.spatial_shape)

    #starting positions for all parameters based on a quick fit
    #not used in dynesty
    args.clip()
    theta0 = args.getguess(galmeta=galmeta)
    ndim = len(theta0)

    #adjust dimensions according to fit params
    nbin = len(args.edges) - args.fixcent
    if disp: ndim += nbin + args.fixcent
    args.setnbins(nbin)
    print(f'{nbin + args.fixcent} radial bins, {ndim} parameters')
    
    #prior bounds and asymmetry defined based off of guess
    #args.setbounds()
    args.setbounds(incpad=3, incgauss=True)
    args.getasym()

    #open up multiprocessing pool if needed
    if cores > 1 and method == 'dynesty':
        pool = mp.Pool(cores)
        pool.size = cores
    else: pool = None

    if method == 'lsq':
        #minfunc = lambda x: loglike(x, args)
        def minfunc(params):
            velmodel, sigmodel = bisym_model(args, unpack(params, args))
            velchisq = (velmodel - args.kin.vel)**2 * args.kin.vel_ivar
            sigchisq = (sigmodel - args.kin.sig)**2 * args.kin.sig_ivar
            return velchisq + sigchisq

        lsqguess = np.append(args.guess, [np.median(args.sig)] * (args.nbins + args.fixcent))
        sampler = optimize.least_squares(minfunc, x0=lsqguess, method='trf',
                  bounds=(args.bounds[:,0], args.bounds[:,1]), verbose=2, diff_step=[.01] * len(lsqguess))
        args.guess = lsqguess

    elif method == 'dynesty':
        #dynesty sampler with periodic pa and pab
        sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=points,
                periodic=[1,2], pool=pool,
                ptform_args = [args], logl_args = [args], verbose=verbose)
        sampler.run_nested()

        if pool is not None: pool.close()

    else:
        raise ValueError('Choose a valid fitting method: dynesty or lsq')

    return sampler, args