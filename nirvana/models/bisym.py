"""
Module with classes and functions used to fit an bisymmetric disk to a set of kinematics.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

import warnings

import multiprocessing as mp

from IPython import embed

import numpy as np
from scipy import stats, optimize

try:
    from tqdm import tqdm
except:
    tqdm = None

try:
    import pyfftw
except:
    pyfftw = None

import dynesty

from .geometry import projected_polar, deriv_projected_polar
from .beam import ConvolveFFTW, smear, deriv_smear
from ..data.manga import MaNGAGasKinematics, MaNGAStellarKinematics
from ..data.util import trim_shape, unpack, cinv
from ..data.fitargs import FitArgs
from ..models.higher_order import bisym_model

from . import oned
from .util import cov_err
from ..data.scatter import IntrinsicScatter
from ..data.util import impose_positive_definite, inverse

from .thindisk import ThinDisk

#warnings.simplefilter('ignore', RuntimeWarning)

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
        func (:obj:`function`, optional):
            Function used to map input values to bounds. Defaults to uniform.
    
    Returns:
        :obj:`float` or `numpy.ndarray`_ of transformed fit parameters.

    '''
    if bounds[key].ndim > 1:
        return (func(bounds[key][:,1]) - func(bounds[key][:,0])) * params[key][indx:] + func(bounds[key][:,0])
    else:
        return (func(bounds[key][1]) - func(bounds[key][0])) * params[key] + func(bounds[key][0])

def ptform(params, args):
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

    Returns:
        :obj:`tuple`: Tuple of parameter values transformed into the prior
        volume.
    '''

    #unpack params and bounds into dicts
    paramdict = unpack(params, args)
    bounddict = unpack(args.bounds, args, bound=True)

    #uniform priors defined by bounds
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
        xcp = unifprior('xc', paramdict, bounddict)
        ycp = unifprior('yc', paramdict, bounddict)
        repack += [xcp,ycp]

    #do scatter terms with logunif
    if args.scatter:
        velscp = unifprior('vel_scatter', paramdict, bounddict, func=lambda x:10**x)
        sigscp = unifprior('sig_scatter', paramdict, bounddict, func=lambda x:10**x)

    #repack all the velocities
    repack += [*vtp, *v2tp, *v2rp]
    if args.disp: repack += [*sigp]
    if args.scatter: repack += [velscp, sigscp]
    return repack

def loglike(params, args):
    '''
    Log likelihood for :class:`dynesty.NestedSampler` fit. 
    
    Makes a model based on current parameters and computes a chi squared with
    original data.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  

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
        if args.scatter: 
            vel_ivar = 1/(1/args.kin.vel_ivar + paramdict['vel_scatter']**2)
        else:
            vel_ivar = 1/(1/args.kin.vel_ivar + args.noise_floor**2)
        llike = llike * vel_ivar 
    llike = -.5 * np.ma.sum(llike + np.log(2*np.pi * vel_ivar))

    #add in penalty for non smooth rotation curves
    if args.weight != -1:
        if args.scatter: velweight = args.weight / paramdict['vel_scatter']
        else: velweight = args.weight
        llike = llike - smoothing(paramdict['vt'],  velweight) \
                      - smoothing(paramdict['v2t'], velweight) \
                      - smoothing(paramdict['v2r'], velweight)

    #add in sigma model if applicable
    if sigmodel is not None:
        #compute chisq
        sigdata = np.sqrt(args.kin.sig_phys2)
        sigdataivar = np.sqrt(args.kin.sig_phys2_ivar) if args.kin.sig_phys2_ivar is not None else np.ones_like(sigdata)
        siglike = (sigmodel - sigdata)**2

        #inflate ivar with noisefloor
        if sigdataivar is not None: 
            if args.scatter: 
                sigdataivar = 1/(1/args.kin.sig_ivar + paramdict['sig_scatter']**2)
            else:
                sigdataivar = 1/(1/sigdataivar + args.noise_floor**2)
            siglike = siglike * sigdataivar - .5 * np.log(2*np.pi * sigdataivar)

        llike -= .5*np.ma.sum(siglike)

        #smooth profile
        if args.weight != -1:
            if args.scatter: sigweight = args.weight / paramdict['sig_scatter']
            else: sigweight = args.weight
            llike -= smoothing(paramdict['sig'], sigweight*.1)

    #apply a penalty to llike if 2nd order terms are too large
    if hasattr(args, 'penalty') and args.penalty:
        if args.scatter: penalty = args.penalty / paramdict['vel_scatter']
        else: penalty = args.penalty
        vtm  = paramdict['vt' ].mean()
        v2tm = paramdict['v2t'].mean()
        v2rm = paramdict['v2r'].mean()

        #scaling penalty if 2nd order profs are big
        llike -= penalty * (v2tm - vtm)/vtm
        llike -= penalty * (v2rm - vtm)/vtm

    return llike

def covarlike(params, args):
    '''
    Log likelihood function utilizing the full covariance matrix of the data.

    Performs the same function as :func:`loglike` but uses the covariance
    matrix for all of the spaxels rather than just the errors for each
    individual spaxel. It takes the exact same arguments and outputs the same
    things too, so it should be able to be switched in and out.

    Args:
        params (:obj:`tuple`):
            Tuple of parameters that are being fit. Assumes the standard order
            of parameters constructed in :func:`nirvana.fitting.fit`.
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  

    Returns:
        :obj:`float`: Log likelihood value associated with parameters.
    '''
    #unpack, generate models and resids
    paramdict = unpack(params, args)
    velmodel, sigmodel = bisym_model(args, paramdict)
    velresid = (velmodel - args.kin.vel)[~args.kin.vel_mask]
    sigresid = (sigmodel - args.kin.sig)[~args.kin.sig_mask]

    #calculate loglikes for velocity and dispersion
    vellike = -.5 * velresid.T.dot(args.velcovinv.dot(velresid)) + args.velcoeff
    if sigmodel is not None:
        siglike = -.5 * sigresid.T.dot(args.sigcovinv.dot(sigresid)) + args.sigcoeff
    else: siglike = 0

    #smoothing penalties
    if args.weight and args.weight != -1:
        weightlike = - smoothing(paramdict['vt'],  args.weight) \
                     - smoothing(paramdict['v2t'], args.weight) \
                     - smoothing(paramdict['v2r'], args.weight)
        if siglike: 
            weightlike -= smoothing(paramdict['sig'], args.weight*.1)
    else: weightlike = 0

    #second order penalties
    if hasattr(args, 'penalty') and args.penalty:
        vtm  = paramdict['vt' ].mean()
        v2tm = paramdict['v2t'].mean()
        v2rm = paramdict['v2r'].mean()

        #scaling penalty if 2nd order profs are big
        penlike = - (args.penalty * (v2tm - vtm)/vtm) \
                  - (args.penalty * (v2rm - vtm)/vtm)
    else: penlike = 0 

    return vellike + siglike + weightlike + penlike 

def fit(plate, ifu, galmeta = None, daptype='HYB10-MILESHC-MASTARHC2', dr='MPL-11', nbins=None,
        cores=10, maxr=None, cen=True, weight=10, smearing=True, points=500,
        stellar=False, root=None, verbose=False, disp=True, 
        fixcent=True, remotedir=None, floor=5, penalty=100,
        mock=None, covar=False, scatter=False, maxbins=10):
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
        covar (:obj:`bool`, optional):
            Whether to use the (currently nonfunctional) covariance likelihood
            rather than the normal one
        scatter (:obj:`bool`, optional):
            Whether to include intrinsic scatter as a fit parameter. Currently
            not working well.
        maxbins (:obj:`int`, optional):
            Maximum number of radial bins to allow. Overridden by ``nbins`` if
            it's larger.

    Returns:
        :class:`dynesty.NestedSampler`: Sampler from `dynesty` containing
        information from the fit.    
        :class:`~nirvana.data.fitargs.FitArgs`: Object with all of the relevant
        data for the galaxy as well as the parameters used for the fit.
    '''

    #set number of global parameters 
    #inc, pa, pab, vsys by default, xc and yc optionally
    nglobs = 6 if cen else 4

    #set up mock galaxy data with real residuals if desired
    if mock is not None:
        args, params, residnum = mock
        args.kin.vel, args.kin.sig = bisym_model(args, params)
        args.penalty = penalty

        #add in real residuals from fit
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
            kin = MaNGAStellarKinematics.from_plateifu(plate, ifu,
                    daptype=daptype, dr=dr, cube_path=root, image_path=root,
                    maps_path=root, remotedir=remotedir, covar=covar,
                    positive_definite=True)
        else:
            kin = MaNGAGasKinematics.from_plateifu(plate, ifu, line='Ha-6564',
                    daptype=daptype, dr=dr,  cube_path=root, image_path=root,
                    maps_path=root, remotedir=remotedir, covar=covar,
                    positive_definite=True)

        #set basic fit parameters for galaxy
        veltype = 'Stars' if stellar else 'Gas'
        args = FitArgs(kin, veltype, nglobs, weight, disp, fixcent, floor, penalty,
                points, smearing, maxr, scatter)

    #get galaxy metadata
    if galmeta is not None: 
        if mock is None: args.kin.phot_inc = galmeta.guess_inclination()
        args.kin.reff = galmeta.reff

    #clip bad regions of the data
    args.clip()

    #set bins manually if nbins is specified
    if nbins is not None: 
        if nbins > maxbins: maxbins = nbins
        args.setedges(nbins, nbin=True, maxr=maxr)

    #set bins automatically based off of FWHM and photometric inc
    else: 
        inc = args.getguess(galmeta=galmeta)[1] if args.kin.phot_inc is None else args.kin.phot_inc
        args.setedges(inc, maxr=maxr)

        #keep number of bins under specified limit
        if len(args.edges) > maxbins + 1 + args.fixcent:
            args.setedges(maxbins, nbin=True, maxr=maxr)

    #discard if number of bins is too small
    if len(args.edges) - fixcent < 3:
        raise ValueError('Galaxy unsuitable: too few radial bins')

    #set up fftw for speeding up convolutions
    if pyfftw is not None: args.conv = ConvolveFFTW(args.kin.spatial_shape)
    else: args.conv = None

    #starting positions for all parameters based on a quick fit
    #not used in dynesty
    theta0 = args.getguess(galmeta=galmeta)
    ndim = len(theta0)

    #clip and invert covariance matrices
    if args.kin.vel_covar is not None and covar: 
        #goodvelcovar = args.kin.vel_covar[np.ix_(goodvel, goodvel)]
        goodvelcovar = np.diag(1/args.kin.vel_ivar)[np.ix_(goodvel, goodvel)]# + 1e-10
        args.velcovinv = cinv(goodvelcovar)
        sign, logdet = np.linalg.slogdet(goodvelcovar)#.todense())
        if sign != 1:
            raise ValueError('Determinant of velocity covariance is not positive')
        args.velcoeff = -.5 * (np.log(2 * np.pi) * goodvel.sum() + logdet)

        if args.kin.sig_phys2_covar is not None:
            goodsig = ~args.kin.sig_mask
            #goodsigcovar = args.kin.sig_covar[np.ix_(goodsig, goodsig)]
            goodsigcovar = np.diag(1/args.kin.sig_ivar)[np.ix_(goodsig, goodsig)]# + 1e-10
            args.sigcovinv = cinv(goodsigcovar)
            sign, logdet = np.linalg.slogdet(goodsigcovar)#.todense())
            if sign != 1:
                raise ValueError('Determinant of dispersion covariance is not positive')
            args.sigcoeff = -.5 * (np.log(2 * np.pi) * goodsig.sum() + logdet)

        else: args.sigcovinv = None

        if not np.isfinite(args.velcovinv).all():
            raise Exception('nans in velcovinv')
        if not np.isfinite(args.sigcovinv).all():
            raise Exception('nans in sigcovinv')
        if not np.isfinite(args.velcoeff):
            raise Exception('nans in velcoeff')
        if not np.isfinite(args.sigcoeff):
            raise Exception('nans in sigcoeff')

    else: args.velcovinv, args.sigcovinv = (None, None)


    #adjust dimensions according to fit params
    nbin = len(args.edges) - args.fixcent
    if disp: ndim += nbin + args.fixcent
    if scatter: ndim += 2
    args.setnbins(nbin)
    print(f'{nbin + args.fixcent} radial bins, {ndim} parameters')
    
    #prior bounds and asymmetry defined based off of guess
    if galmeta is not None: 
        args.setphotpa(galmeta)
        args.setbounds(incpad=3, incgauss=True)#, papad=10, pagauss=True)
    else: args.setbounds(incpad=3, incgauss=True)
    args.getasym()

    #open up multiprocessing pool if needed
    if cores > 1:
        pool = mp.Pool(cores)
        pool.size = cores
    else: pool = None

    #dynesty sampler with periodic pa and pab
    if not covar: sampler = dynesty.NestedSampler(loglike, ptform, ndim, nlive=points,
            periodic=[1,2], pool=pool,
            ptform_args = [args], logl_args = [args], verbose=verbose)
    else: sampler = dynesty.NestedSampler(covarlike, ptform, ndim, nlive=points,
            periodic=[1,2], pool=pool,
            ptform_args = [args], logl_args = [args], verbose=verbose)
    sampler.run_nested()

    if pool is not None: pool.close()

    return sampler, args


class BisymmetricDisk(ThinDisk):
    r"""
    Model for a rotating thin disk with a bisymmetric flow; cf. Spekkens &
    Sellwood (2007, ApJ, 664, 204).

    The model assumes the disk is infinitely thin and has a single set of
    geometric parameters:

        - :math:`x_c, y_c`: The coordinates of the galaxy dynamical center.
        - :math:`\phi`: The on-sky position angle of the major axis of the
          galaxy (the angle from N through E)
        - :math:`i`: The inclination of the disk; the angle of the disk
          normal relative to the line-of-sight such that :math:`i=0` is a
          face-on disk.
        - :math:`V_{\rm sys}`: The systemic (bulk) velocity of the galaxy
          taken as the line-of-sight velocity at the dynamical center.
        - :math:`\phi_b`: The on-sky position angle of the primary axis of
          the bisymmetric flow (the angle from N through E)

    In addition to these parameters, the model instantiation requires class
    instances that define the rotation curve, the radial bisymmetric flow
    amplitude, the tangential bisymmetric flow amplitude, and velocity
    dispersion profile. These classes must have:

        - an ``np`` attribute that provides the number of parameters in the
          model
        - a ``guess_par`` method that provide initial guess parameters for
          the model, and
        - ``lb`` and ``ub`` attributes that provide the lower and upper
          bounds for the model parameters.

    Importantly, note that the model fits the parameters for the *projected*
    rotation curve. I.e., that amplitude of the fitted function is actually,
    e.g., :math:`V_{\rm rot} \sin i`.

    .. todo::
        Describe the attributes

    Args:
        vt (:class:`~nirvana.models.oned.Func1D`, optional):
            The parameterization to use for the disk rotation curve (first-order
            tangential term).  If None, defaults to
            :class:`~nirvana.models.oned.HyperbolicTangent`.
        v2t (:class:`~nirvana.models.oned.Func1D`, optional):
            The parameterization to use for the second-order tangential term.
            If None, defaults to :class:`~nirvana.models.oned.PowerExp`.
        v2r (:class:`~nirvana.models.oned.Func1D`, optional):
            The parameterization to use for the second-order radial term.  If
            None, defaults to :class:`~nirvana.models.oned.PowerExp`.
        dc (:class:`~nirvana.models.oned.Func1D`, optional):
            The parameterization to use for the disk dispersion profile.  If
            None, the dispersion profile is not included in the fit!

    """
    def __init__(self, vt=None, v2t=None, v2r=None, dc=None):
        # Velocity components
        self.vt = oned.HyperbolicTangent() if vt is None else vt
        self.v2t = oned.PowerExp() if v2t is None else v2t
        self.v2r = oned.PowerExp() if v2r is None else v2r
        # Velocity dispersion curve (can be None)
        self.dc = dc

        # Instantiate the base class, which basically keeps all of the geometric
        # parameters.  NOTE: The parametric curves above need to be defined
        # first because instatiation of the base class calls reinit(), which
        # sets default guess parameters.  TODO: Consider instantiating without
        # guess parameters...
        super().__init__()

        # Total number of parameters:  Basic geometric parameters, the position
        # angle of the bisymmetry, and the radials profiles of the velocity
        # terms.
        self.np = self.nbp + 1 + self.vt.np + self.v2t.np + self.v2r.np
        if self.dc is not None:
            # Add in the velocity dispersion terms.
            self.np += self.dc.np
        # Flag which parameters are freely fit
        self.free = np.ones(self.np, dtype=bool)
        self.nfree = np.sum(self.free)
        # This call to reinit adds the workspace attributes
        self.reinit()

    def guess_par(self):
        """
        Return a list of generic guess parameters.

        .. todo::
            Could enable this to base the guess on the data to be fit, but at
            the moment these are hard-coded numbers.

        Returns:
            `numpy.ndarray`_: Vector of guess parameters
        """
        gp = np.concatenate((super().guess_par(), [0.], self.vt.guess_par(),
                             self.v2t.guess_par(), self.v2r.guess_par()))
        return gp if self.dc is None else np.append(gp, self.dc.guess_par())

    def par_names(self, short=False):
        """
        Return a list of strings with the parameter names.

        Args:
            short (:obj:`bool`, optional):
                Return truncated nomenclature for the parameter names.

        Returns:
            :obj:`list`: List of parameter name strings.
        """
        base = super().par_names(short=short)
        if short:
            base += ['pab']
            vt = [f'vt_{p}' for p in self.vt.par_names(short=True)]
            v2t = [f'v2t_{p}' for p in self.v2t.par_names(short=True)]
            v2r = [f'v2r_{p}' for p in self.v2r.par_names(short=True)]
            dc = [] if self.dc is None else [f's_{p}' for p in self.dc.par_names(short=True)]
        else:
            base += ['Bisymmetry PA']
            vt = [f'VT: {p}' for p in self.vt.par_names()]
            v2t = [f'V2T: {p}' for p in self.v2t.par_names()]
            v2r = [f'V2R: {p}' for p in self.v2r.par_names()]
            dc = [] if self.dc is None else [f'Disp: {p}' for p in self.dc.par_names()]
        return base + vt + v2t + v2r + dc

    # These "private" functions yield slices of the parameter vector for the
    # desired set of parameters
    def _vt_slice(self):
        s = self.nbp + 1
        return slice(s, s + self.vt.np)
    def _v2t_slice(self):
        s = self.nbp + 1 + self.vt.np
        return slice(s, s + self.v2t.np)
    def _v2r_slice(self):
        s = self.nbp + 1 + self.vt.np + self.v2t.np
        return slice(s, s + self.v2r.np)
    def _dc_slice(self):
        s = self.nbp + 1 + self.vt.np + self.v2t.np + self.v2r.np
        return slice(s, s + self.dc.np)

    def vt_par(self, err=False):
        """
        Return the tangential velocity perameters. Returns None if parameters
        are not defined yet.

        Args:
            err (:obj:`bool`, optional):
                Return the parameter errors instead of the parameter values.

        Returns:
            `numpy.ndarray`_: Vector with parameters or parameter errors for the
            rotation curve (first-order rotation).
        """
        p = self.par_err if err else self.par
        return None if p is None else p[self._vt_slice()]

    def v2t_par(self, err=False):
        """
        Return the 2nd-order tangential velocity perameters. Returns None if
        parameters are not defined yet.

        Args:
            err (:obj:`bool`, optional):
                Return the parameter errors instead of the parameter values.

        Returns:
            `numpy.ndarray`_: Vector with parameters or parameter errors for the
            second-order tangential term.
        """
        p = self.par_err if err else self.par
        return None if p is None else p[self._v2t_slice()]

    def v2r_par(self, err=False):
        """
        Return the 2nd-order radial velocity perameters. Returns None if
        parameters are not defined yet.

        Args:
            err (:obj:`bool`, optional):
                Return the parameter errors instead of the parameter values.

        Returns:
            `numpy.ndarray`_: Vector with parameters or parameter errors for the
            second-order radial term.
        """
        p = self.par_err if err else self.par
        return None if p is None else p[self._v2r_slice()]

    def dc_par(self, err=False):
        """
        Return the dispersion profile parameters. Returns None if parameters
        are not defined yet or if no dispersion profile has been defined.

        Args:
            err (:obj:`bool`, optional):
                Return the parameter errors instead of the parameter values.

        Returns:
            `numpy.ndarray`_: Vector with parameters or parameter errors for the
            dispersion profile.
        """
        p = self.par_err if err else self.par
        return None if p is None or self.dc is None else p[self._dc_slice()]

    def par_bounds(self, base_lb=None, base_ub=None, pab_lb=None, pab_ub=None):
        """
        Return the lower and upper boundaries on the model parameters.

        The default geometric bounds (see ``base_lb``, ``base_ub``) are set by
        the minimum and maximum available x and y coordinates, -350 to 350 for
        the position angle, 1 to 89 for the inclination, -300 to 300 for the
        systemic velocity, and -100 to 100 degrees for the position angle of the
        bisymmetric feature.

        .. todo::
            Could enable this to base the bounds on the data to be fit, but
            at the moment these are hard-coded numbers.

        Args:
            base_lb (`numpy.ndarray`_, optional):
                The lower bounds for the "base" parameters. If None, the
                defaults are used (see above).
            base_ub (`numpy.ndarray`_, optional):
                The upper bounds for the "base" parameters. If None, the
                defaults are used (see above).

        Returns:
            :obj:`tuple`: A two-tuple providing, respectively, the lower and
            upper boundaries for all model parameters.
        """
        _base_lb, _base_ub = super().par_bounds(base_lb=base_lb, base_ub=base_ub)
        _base_lb = np.append(_base_lb, [-100. if pab_lb is None else pab_lb])
        _base_ub = np.append(_base_ub, [100. if pab_ub is None else pab_ub])
        lb = np.concatenate((_base_lb, self.vt.lb, self.v2t.lb, self.v2r.lb))
        ub = np.concatenate((_base_ub, self.vt.ub, self.v2t.ub, self.v2r.ub))
        return (lb, ub) if self.dc is None \
                    else (np.append(lb, self.dc.lb), np.append(ub, self.dc.ub))

    def in_disk_bar_angle(self, par=None):
        # Reset the parameter values
        if par is not None:
            self._set_par(par)
        inc = np.radians(self.par[3])
        pab = np.radians(self.par[5])
        #   - Calculate the in-plane angle relative to the bisymmetric flow axis
        _pab = (pab + np.pi/2) % np.pi - np.pi/2        # Impose a range of [-pi/2, pi/2]
        return np.degrees(np.arctan(np.tan(_pab)/np.cos(inc)))

    def model(self, par=None, x=None, y=None, sb=None, beam=None, is_fft=False, cnvfftw=None,
              ignore_beam=False):
        """
        Evaluate the model.

        Note that arguments passed to this function overwrite any existing
        attributes of the object, and subsequent calls to this function will
        continue to use existing attributes, unless they are overwritten.  For
        example, if ``beam`` is provided here, it overwrites any existing
        :attr:`beam_fft` and any subsequent calls to ``model`` **that do not
        provide a new** ``beam`` will use the existing :attr:`beam_fft`.  To
        remove all internal attributes to get a "clean" instantiation, either
        define a new :class:`BisymmetricDisk` instance or use :func:`reinit`.

        .. warning::
            
            Input coordinates and surface-brightness data types are all
            converted to `numpy.float64`_.  This is always true, even though
            this is only needed when using
            :class:`~nirvana.models.beam.ConvolveFFTW`.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. If None, the internal
                :attr:`par` is used. Length should be either :attr:`np` or
                :attr:`nfree`. If the latter, the values of the fixed
                parameters in :attr:`par` are used.
            x (`numpy.ndarray`_, optional):
                The 2D x-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`x` is used.
            y (`numpy.ndarray`_, optional):
                The 2D y-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`y` is used.
            sb (`numpy.ndarray`_, optional):
                2D array with the surface brightness of the object. This is used
                to weight the convolution of the kinematic fields according to
                the luminosity distribution of the object.  Must have the same
                shape as ``x``. If None, the convolution is unweighted.  If a
                convolution is not performed (either ``beam`` or
                :attr:`beam_fft` are not available, or ``ignore_beam`` is True),
                this array is ignored.
            beam (`numpy.ndarray`_, optional):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT). If not provided, the internal
                :attr:`beam_fft` is used.
            is_fft (:obj:`bool`, optional):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.  Ignored if ``beam`` is not provided.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
                An object that expedites the convolutions using
                FFTW/pyFFTW. If None, the convolution is done using numpy
                FFT routines.
            ignore_beam (:obj:`bool`, optional):
                Ignore the beam-smearing when constructing the model. I.e.,
                construct the *intrinsic* model.

        Returns:
            `numpy.ndarray`_, :obj:`tuple`: The velocity field model, and the
            velocity dispersion field model, if the latter is included
        """
        # Initialize the thin disk model
        self._init_model(par, x, y, sb, beam, is_fft, cnvfftw, ignore_beam)

        # Get the coordinate data
        #   - Convert the relevant angles to radians
        pa, inc = np.radians(self.par[2:4])
        pab = np.radians(self.par[5])
        #   - Compute the in-plane radius and azimuth
        r, theta = projected_polar(self.x - self.par[0], self.y - self.par[1], pa, inc)
        #   - Calculate the in-plane angle relative to the bisymmetric flow axis
        _pab = (pab + np.pi/2) % np.pi - np.pi/2        # Impose a range of [-pi/2, pi/2]
        theta_b = theta - np.arctan(np.tan(_pab)/np.cos(inc))

        # Construct the line-of-sight velocities. NOTE: All velocity component
        # amplitudes are projected; i.e., the sin(inclination) terms are
        # absorbed into the velocity amplitudes.
        cost = np.cos(theta)
        vel = self.par[4] + self.vt.sample(r, par=self.par[self._vt_slice()]) * cost \
                - self.v2t.sample(r, par=self.par[self._v2t_slice()]) * cost * np.cos(2*theta_b) \
                - self.v2r.sample(r, par=self.par[self._v2r_slice()]) * np.sin(theta) \
                    * np.sin(2*theta_b)

        if self.dc is None:
            # Only modeling the velocity field
            return vel if self.beam_fft is None or ignore_beam \
                        else smear(vel, self.beam_fft, beam_fft=True, sb=self.sb,
                                   cnvfftw=cnvfftw)[1]

        # Modeling both the velocity and velocity-dispersion field
        sig = self.dc.sample(r, par=self.par[self._dc_slice()])
        return (vel, sig) if self.beam_fft is None or ignore_beam \
                        else smear(vel, self.beam_fft, beam_fft=True, sb=self.sb, sig=sig,
                                   cnvfftw=cnvfftw)[1:]

    def deriv_model(self, par=None, x=None, y=None, sb=None, beam=None, is_fft=False, cnvfftw=None,
                    ignore_beam=False):
        """
        Evaluate the derivative of the model w.r.t all input parameters.

        Note that arguments passed to this function overwrite any existing
        attributes of the object, and subsequent calls to this function will
        continue to use existing attributes, unless they are overwritten.  For
        example, if ``beam`` is provided here, it overwrites any existing
        :attr:`beam_fft` and any subsequent calls to ``model`` **that do not
        provide a new** ``beam`` will use the existing :attr:`beam_fft`.  To
        remove all internal attributes to get a "clean" instantiation, either
        define a new :class:`AxisymmetricDisk` instance or use :func:`reinit`.

        .. warning::
            
            Input coordinates and surface-brightness data types are all
            converted to `numpy.float64`_.  This is always true, even though
            this is only needed when using
            :class:`~nirvana.models.beam.ConvolveFFTW`.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. If None, the internal
                :attr:`par` is used. Length should be either :attr:`np` or
                :attr:`nfree`. If the latter, the values of the fixed
                parameters in :attr:`par` are used.
            x (`numpy.ndarray`_, optional):
                The 2D x-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`x` is used.
            y (`numpy.ndarray`_, optional):
                The 2D y-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`y` is used.
            sb (`numpy.ndarray`_, optional):
                2D array with the surface brightness of the object. This is used
                to weight the convolution of the kinematic fields according to
                the luminosity distribution of the object.  Must have the same
                shape as ``x``. If None, the convolution is unweighted.  If a
                convolution is not performed (either ``beam`` or
                :attr:`beam_fft` are not available, or ``ignore_beam`` is True),
                this array is ignored.
            beam (`numpy.ndarray`_, optional):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT). If not provided, the internal
                :attr:`beam_fft` is used.
            is_fft (:obj:`bool`, optional):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.  Ignored if ``beam`` is not provided.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
                An object that expedites the convolutions using
                FFTW/pyFFTW. If None, the convolution is done using numpy
                FFT routines.
            ignore_beam (:obj:`bool`, optional):
                Ignore the beam-smearing when constructing the model. I.e.,
                construct the *intrinsic* model.

        Returns:
            `numpy.ndarray`_, :obj:`tuple`: The velocity field model, and the
            velocity dispersion field model, if the latter is included
        """
        # Initialize the thin disk model
        self._init_model(par, x, y, sb, beam, is_fft, cnvfftw, ignore_beam)

        # Initialize the derivative arrays needed for the coordinate calculation
        dx = np.zeros(self.x.shape+(self.np,), dtype=float)
        dy = np.zeros(self.x.shape+(self.np,), dtype=float)
        dpa = np.zeros(self.np, dtype=float)
        dinc = np.zeros(self.np, dtype=float)

        dx[...,0] = -1.
        dy[...,1] = -1.
        dpa[2] = np.radians(1.)
        dinc[3] = np.radians(1.)

        pa, inc = np.radians(self.par[2:4])
        r, theta, dr, dtheta = deriv_projected_polar(self.x - self.par[0], self.y - self.par[1],
                                                     pa, inc, dxdp=dx, dydp=dy, dpadp=dpa,
                                                     dincdp=dinc)

        # Calculate the in-plane angle relative to the bisymmetric flow axis and its derivative
        dpab = np.zeros(self.np, dtype=float)
        dpab[5] = np.radians(1.)
        pab = np.radians(self.par[5])
        # - Intermediate calculation
        _pab = (pab + np.pi/2) % np.pi - np.pi/2        # Impose a range of [-pi/2, pi/2]
        c = np.tan(_pab)/np.cos(inc)
        dc = dpab / np.cos(inc) / np.cos(_pab)**2 \
                + np.tan(_pab) * np.sin(inc) / np.cos(inc)**2 * dinc
        # - Finish the calculation
        theta_b = theta - np.arctan(c)
        dtheta_b = dtheta - (dc / (1 + c**2))[None,...]

        # Calculate the rotation speed and its parameter derivatives
        dvt = np.zeros(self.x.shape+(self.np,), dtype=float)
        slc = self._vt_slice()
        vt, dvt[...,slc] = self.vt.deriv_sample(r, par=self.par[slc])
        dvt += self.vt.ddx(r, par=self.par[slc])[...,None]*dr

        # Calculate the 2nd-order tangential speed and its parameter derivatives
        dv2t = np.zeros(self.x.shape+(self.np,), dtype=float)
        slc = self._v2t_slice()
        v2t, dv2t[...,slc] = self.v2t.deriv_sample(r, par=self.par[slc])
        dv2t += self.v2t.ddx(r, par=self.par[slc])[...,None]*dr

        # Calculate the 2nd-order radial speed and its parameter derivatives
        dv2r = np.zeros(self.x.shape+(self.np,), dtype=float)
        slc = self._v2r_slice()
        v2r, dv2r[...,slc] = self.v2r.deriv_sample(r, par=self.par[slc])
        dv2r += self.v2r.ddx(r, par=self.par[slc])[...,None]*dr

        # Construct the line-of-sight velocities and parameter derivatives.
        # NOTE: All velocity component amplitudes are projected; i.e., the
        # sin(inclination) terms are absorbed into the velocity amplitudes.
        cost = np.cos(theta)
        sint = np.sin(theta)
        cos2tb = np.cos(2*theta_b)
        sin2tb = np.sin(2*theta_b)

        v = self.par[4] \
                + vt * cost \
                - v2t * cos2tb * cost \
                - v2r * sin2tb * sint

        dv = dvt * cost[...,None] - (vt*sint)[...,None]*dtheta \
                - dv2t * (cos2tb*cost)[...,None] + 2 * (v2t*sin2tb*cost)[...,None] * dtheta_b \
                    + (v2t*cos2tb*sint)[...,None] * dtheta \
                - dv2r * (sin2tb*sint)[...,None] - 2 * (v2r*cos2tb*sint)[...,None] * dtheta_b \
                    - (v2r*sin2tb*cost)[...,None] * dtheta
        dv[...,4] = 1.

        if self.dc is None:
            # Only modeling the velocity field
            if self.beam_fft is None or ignore_beam:
                # Not smearing
                return v, dv
            # Smear and propagate through the derivatives
            _, v, _, _, dv, _ = deriv_smear(v, dv, self.beam_fft, beam_fft=True, sb=self.sb,
                                            cnvfftw=self.cnvfftw)
            return v, dv

        # Modeling both the velocity and velocity-dispersion field. Calculate
        # the dispersion profile and its parameter derivatives
        slc = self._dc_slice()
        dsig = np.zeros(self.x.shape+(self.np,), dtype=float)
        sig, dsig[...,slc] = self.dc.deriv_sample(r, par=self.par[slc])
        dsig += self.dc.ddx(r, par=self.par[slc])[...,None]*dr
        if self.beam_fft is None or ignore_beam:
            # Not smearing
            return v, sig, dv, dsig

        # Smear and propagate through the derivatives
        _, v, sig, _, dv, dsig = deriv_smear(v, dv, self.beam_fft, beam_fft=True, sb=self.sb,
                                             sig=sig, dsig=dsig, cnvfftw=self.cnvfftw)
        return v, sig, dv, dsig

    def report(self, fit_message=None):
        """
        Report the current parameters of the model.
        """
        if self.par is None:
            print('No parameters to report.')
            return

        vfom, sfom = self._get_fom()(self.par, sep=True)
        parn = self.par_names()
        max_parn_len = max([len(n) for n in parn])+4

        print('-'*70)
        print(f'{"Fit Result":^70}')
        print('-'*70)
        if fit_message is not None:
            print(f'Fit status message: {fit_message}')
        if self.fit_status is not None:
            print(f'Fit status: {self.fit_status}')
        print(f'Fit success: {"True" if self.fit_status else "False"}')
        print('-'*10)
        print(f'Base parameters:')
        slc = self._base_slice()
        ps = 0 if slc.start is None else slc.start
        pe = slc.stop
        for i in range(ps,pe):
            print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[i]:.1f}').format(parn[i])
                    + (f'' if self.par_err is None else f' +/- {self.par_err[i]:.1f}'))
        # Bisymmetry angle
        print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[pe]:.1f}').format(parn[pe])
                + (f'' if self.par_err is None else f' +/- {self.par_err[pe]:.1f}'))
        print('-'*10)

        print(f'First-order tangential speed parameters:')
        slc = self._vt_slice()
        ps = slc.start
        pe = slc.stop
        for i in range(ps,pe):
            print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[i]:.1f}').format(parn[i])
                    + (f'' if self.par_err is None else f' +/- {self.par_err[i]:.1f}'))

        print(f'Second-order tangential speed parameters:')
        slc = self._v2t_slice()
        ps = slc.start
        pe = slc.stop
        for i in range(ps,pe):
            print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[i]:.1f}').format(parn[i])
                    + (f'' if self.par_err is None else f' +/- {self.par_err[i]:.1f}'))

        print(f'Second-order radial speed parameters:')
        slc = self._v2r_slice()
        ps = slc.start
        pe = slc.stop
        for i in range(ps,pe):
            print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[i]:.1f}').format(parn[i])
                    + (f'' if self.par_err is None else f' +/- {self.par_err[i]:.1f}'))

        if self.dc is None:
            print('-'*10)
            if self.scatter is not None:
                print(f'Intrinsic Velocity Scatter: {self.scatter[0]:.1f}')
            vchisqr = np.sum(vfom**2)
            print(f'Velocity measurements: {len(vfom)}')
            print(f'Velocity chi-square: {vchisqr}')
            print(f'Reduced chi-square: {vchisqr/(len(vfom)-self.nfree)}')
            print('-'*70)
            return

        print('-'*10)
        print(f'Dispersion profile parameters:')
        slc = self._dc_slice()
        ps = slc.start
        pe = slc.stop
        for i in range(ps,pe):
            print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[i]:.1f}').format(parn[i])
                    + (f'' if self.par_err is None else f' +/- {self.par_err[i]:.1f}'))
        print('-'*10)
        if self.scatter is not None:
            print(f'Intrinsic Velocity Scatter: {self.scatter[0]:.1f}')
        vchisqr = np.sum(vfom**2)
        print(f'Velocity measurements: {len(vfom)}')
        print(f'Velocity chi-square: {vchisqr}')
        if self.scatter is not None:
            print(f'Intrinsic Dispersion**2 Scatter: {self.scatter[1]:.1f}')
        schisqr = np.sum(sfom**2)
        print(f'Dispersion measurements: {len(sfom)}')
        print(f'Dispersion chi-square: {schisqr}')
        print(f'Reduced chi-square: {(vchisqr + schisqr)/(len(vfom) + len(sfom) - self.nfree)}')
        print('-'*70)


