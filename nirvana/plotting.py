"""
Plotting for nirvana outputs.

.. include:: ../include/links.rst
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable as mal
import re
import os
import traceback

import dynesty
import dynesty.plotting
import corner
import pickle
from glob import glob
from tqdm import tqdm
from astropy.io import fits

from .fitting import bisym_model, unpack
from .data.manga import MaNGAStellarKinematics, MaNGAGasKinematics
from .data.kinematics import Kinematics
from .models.beam import smear, ConvolveFFTW
from .models.geometry import projected_polar


def get_logformatter():
    """
    Return a matplotlib label formatter that uses decimals instead of exponentials.

    This is a hack that I pulled off of stackoverflow (since lost the url).
    It breaks some imshow things, I think. Probably better to get this
    functionality by subclassing from ticker.LogFormatter...
    """
    # TODO: Introduce an upper/lower limit to switch back to exponential form ;
    # i.e., make label go something like 10^4, 1000, ..., 0.01, 0.001, 10^{-4}
    return ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(
                                        int(np.maximum(-np.log10(y),0)))).format(y))


def dynmeds(samp, stds=False, fixcent=True):
    """
    Get median values for each variable's posterior in a
    `dynesty.NestedSampler`_ sampler.

    Args:
        samp (:obj:`str`, `dynesty.NestedSampler`_, `dynesty.results.Results`_):
            Sampler, results, or file of dumped results from `dynesty`_ fit.
        stds (:obj:`bool`, optional):
            Flag for whether or not to return standard deviations of the
            posteriors as well.

    Returns:
        `numpy.ndarray`_: Median values of all of the parameters in the
        `dynesty`_ sampler. If ``stds == True``, it will instead return a
        :obj:`tuple` of three `numpy.ndarray`_ objects. The first is the
        median values, the second is the lower 1 sigma bound for all of the
        posteriors, and the third is the upper 1 sigma bound.
    """

    #get samples and weights
    if type(samp) == str: res = pickle.load(open(samp,'rb'))
    elif type(samp)==dynesty.results.Results: res = samp
    else: res = samp.results
    samps = res.samples
    weights = np.exp(res.logwt - res.logz[-1])

    #iterate through and get 50th percentile of values
    meds = np.zeros(samps.shape[1])
    for i in range(samps.shape[1]):
        meds[i] = dynesty.utils.quantile(samps[:,i],[.5],weights)[0]

    #pull out 1 sigma values on either side of the mean as well if desired
    if stds:
        lstd = np.zeros(samps.shape[1])
        ustd = np.zeros(samps.shape[1])
        for i in range(samps.shape[1]):
            lstd[i] = dynesty.utils.quantile(samps[:,i],[.5-.6826/2],weights)[0]
            ustd[i] = dynesty.utils.quantile(samps[:,i],[.5+.6826/2],weights)[0]
        return meds, lstd, ustd

    return meds

def profs(samp, args, plot=None, stds=False, jump=None, **kwargs):
    '''
    Turn a sampler output by `nirvana` into a set of rotation curves.
    
    Args:
        samp (:obj:`str`, `dynesty.NestedSampler`_, `dynesty.results.Results`_):
            Sampler, results, or file of dumped results from `dynesty` fit.
        args (:class:`nirvana.data.fitargs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        plot (:class:`matplotlib.axes._subplots.Axes`, optional):
            Axis to plot the rotation curves on. If not specified, it will not
            try to plot anything.
        stds (:obj:`bool`, optional):
            Flag for whether to fetch the standard deviations as well.
        jump (:obj:`int`, optional):
            Number of radial bins in the sampler. Will be calculated
            automatically if not specified.
        **kwargs:
            args for :func:`plt.plot`.

    Returns:
        :obj:`dict`: Dictionary with all of the median values of the
        posteriors in the sampler. Has keys for inclination `inc`, first
        order position angle `pa`, second order position angle `pab`,
        systemic velocity `vsys`, x and y center coordinates `xc` and `yc`,
        `numpy.ndarray`_ of first order tangential velocities `vt`,
        `numpy.ndarray`_ objects of second order tangential and radial
        velocities `v2t` and `v2r`, and `numpy.ndarray`_ of velocity
        dispersions `sig`. If `stds == True` it will also contain keys for
        the 1 sigma lower bounds of the velocity parameters `vtl`, `v2tl`,
        `v2rl`, and `sigl` as well as their 1 sigma upper bounds `vtu`,
        `v2tu`, `v2ru`, and `sigu`. Arrays have lengths that are the same as
        the number of bins (determined automatically or from `jump`). All
        angles are in degrees and all velocities must be in consistent units.

        If `plot == True`, it will also display a plot of the profiles.
    '''

    #get and unpack median values for params
    meds = dynmeds(samp, stds=stds, fixcent=args.fixcent)

    #get standard deviations and put them into the dictionary
    if stds:
        meds, lstd, ustd = meds
        paramdict = unpack(meds, args, jump=jump)
        paramdict['incl'], paramdict['pal'], paramdict['pabl'], paramdict['vsysl'] = lstd[:4]
        paramdict['incu'], paramdict['pau'], paramdict['pabu'], paramdict['vsysu'] = ustd[:4]
        if args.nglobs == 6:
            paramdict['xcl'], paramdict['ycl'] = lstd[4:6]
            paramdict['xcu'], paramdict['ycu'] = ustd[4:6]

        start = args.nglobs
        jump = len(args.edges) - args.fixcent
        vs = ['vt', 'v2t', 'v2r']
        for i,v in enumerate(vs):
            for b in ['l','u']:
                exec(f'paramdict["{v}{b}"] = {b}std[start + {i}*jump : start + {i+1}*jump]')
                if args.fixcent:
                    exec(f'paramdict["{v}{b}"] = np.insert(paramdict["{v}{b}"], 0, 0)')

        #dispersion stds
        if args.disp: 
            sigjump = jump + 1
            paramdict['sigl'] = lstd[start + 3*jump:start + 3*jump + sigjump]
            paramdict['sigu'] = ustd[start + 3*jump:start + 3*jump + sigjump]

    else: paramdict = unpack(meds, args, jump=jump)

    #plot profiles if desired
    if plot is not None: 
        if not isinstance(plot, matplotlib.axes._subplots.Axes): f,plot = plt.subplots()
        ls = [r'$V_t$',r'$V_{2t}$',r'$V_{2r}$']
        [plot.plot(args.edges, p, label=ls[i], **kwargs) 
                for i,p in enumerate([paramdict['vt'], paramdict['v2t'], paramdict['v2r']])]

        #add in lower and upper bounds
        if stds: 
            errors = [[paramdict['vtl'], paramdict['vtu']], [paramdict['v2tl'], paramdict['v2tu']], [paramdict['v2rl'], paramdict['v2ru']]]
            for i,p in enumerate(errors):
                plot.fill_between(args.edges, p[0], p[1], alpha=.5) 

        plt.xlabel(r'$R_e$')
        plt.ylabel(r'$v$ (km/s)')
        plt.legend(loc=2)

    return paramdict

def fileprep(f, plate=None, ifu=None, smearing=True, stellar=False, maxr=None, mix=False, cen=True, fixcent=True, clip=True, remotedir=None, gal=None):

    #unpack fits file
    if type(f) == str and '.fits' in f:
        isfits = True
        with fits.open(f) as fitsfile:
            table = fitsfile[1].data
            maxr = fitsfile[0].header['maxr']
            smearing = fitsfile[0].header['smearing']

        keys = table.columns.names
        vals = [table[k][0] for k in keys]
        resdict = dict(zip(keys, vals))
        for v in ['vt','v2t','v2r','vtl','vtu','v2tl','v2tu','v2rl','v2ru']:
            resdict[v] = resdict[v][resdict['velmask'] == 0]
        for s in ['sig','sigl','sigu']:
            resdict[s] = resdict[s][resdict['sigmask'] == 0]

        if gal is None:
            if resdict['type'] == 'Stars':
                args = MaNGAStellarKinematics.from_plateifu(resdict['plate'],resdict['ifu'], ignore_psf=not smearing, remotedir=remotedir)
            else:
                args = MaNGAGasKinematics.from_plateifu(resdict['plate'],resdict['ifu'], ignore_psf=not smearing, remotedir=remotedir)
        else:
            args = gal

        chains = None
        fill = len(resdict['velmask'])
        fixcent = resdict['vt'][0] == 0
        lenmeds = 6 + 3*(fill - resdict['velmask'].sum() - fixcent) + (fill - resdict['sigmask'].sum())
        meds = np.zeros(lenmeds)

    else:
        isfits = False
        #get sampler in right format
        if type(f) == str: chains = pickle.load(open(f,'rb'))
        elif type(f) == np.ndarray: chains = f
        elif type(f) == dynesty.nestedsamplers.MultiEllipsoidSampler: chains = f.results

        if plate is None or ifu is None:
            fname = re.split('/', f[:-5])[-1]
            info = re.split('/|-|_', fname)
            plate = int(info[0]) if plate is None else plate
            ifu = int(info[1]) if ifu is None else ifu
            stellar = True if 'stel' in info else False
            cen = True if 'nocen' not in info else False
            smearing = True if 'nosmear' not in info else False
            #maxr = float([i for i in info if '.' in i and 'r' in i][0][:-1])
            maxr = float([i for i in info if 'r' in i][0][:-1])

            if 'fixcent' in info: fixcent = True
            elif 'freecent' in info: fixcent = False

        #mock galaxy using stored values
        if plate == 0:
            mock = np.load('mockparams.npy', allow_pickle=True)[ifu]
            print('Using mock:', mock['name'])
            params = [mock['inc'], mock['pa'], mock['pab'], mock['vsys'], mock['vts'], mock['v2ts'], mock['v2rs'], mock['sig']]
            args = Kinematics.mock(56,*params)
            cnvfftw = ConvolveFFTW(args.spatial_shape)
            smeared = smear(args.remap('vel'), args.beam_fft, beam_fft=True, sig=args.remap('sig'), sb=args.remap('sb'), cnvfftw=cnvfftw)
            args.sb  = args.bin(smeared[0])
            args.vel = args.bin(smeared[1])
            args.sig = args.bin(smeared[2])
            args.fwhm  = 2.44

        #load in MaNGA data
        else:
            if stellar:
                args = MaNGAStellarKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing, remotedir=remotedir)
            else:
                args = MaNGAGasKinematics.from_plateifu(plate,ifu, ignore_psf=not smearing, remotedir=remotedir)

    #set relevant parameters for galaxy
    args.setdisp(True)
    args.setnglobs(4) if not cen else args.setnglobs(6)
    args.setmix(mix)
    args.setfixcent(fixcent)
    if clip: args.clip()

    vel_r = args.remap('vel')
    sig_r = args.remap('sig') if args.sig_phys2 is None else np.sqrt(np.abs(args.remap('sig_phys2')))

    if not isfits: meds = dynmeds(chains)

    #get appropriate number of edges  by looking at length of meds
    nbins = (len(meds) - args.nglobs - 3*args.mix - fixcent)/4
    if not nbins.is_integer(): 
        raise ValueError('Dynesty output array has a bad shape.')
    else: nbins = int(nbins)

    if not isfits:
        args.setedges(nbins - 1 + args.fixcent, nbin=True, maxr=maxr)
        resdict = profs(chains, args, stds=True)
        resdict['plate'] = plate
        resdict['ifu'] = ifu
        resdict['type'] = 'Stars' if stellar else 'Gas'
    else:
        args.edges = resdict['bin_edges'][~resdict['velmask']]

    return args, resdict, chains, meds

def summaryplot(f, plate=None, ifu=None, smearing=True, stellar=False, maxr=None, mix=False, cen=True, save=False, clobber=False, fixcent=True, remotedir=None):
    '''
    Make a summary plot for a `nirvana` output file with MaNGA velocity field.

    Shows the values for the global parameters of the galaxy, the rotation
    curves (with 1 sigma lower and upper bounds) for the different velocity
    components, then comparisons of the MaNGA data, the model, and the residuals
    for the rotational velocity and the velocity dispersion. 

    Args:
        f (:class:`dynesty.NestedSampler` or :obj:`str` or
        :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        plate (:obj:`int`), optional):
            MaNGA plate number for desired galaxy. Must be specified if
            `auto=False`.
        ifu (:obj:`int`), optional):
            MaNGA IFU design number for desired galaxy. Must be specified if
            `auto=False`.
        smearing (:obj:`bool`, optional):
            Flag for whether or not to apply beam smearing to models.
        stellar (:obj:`bool`, optional):
            Flag for whether or not to use stellar velocity data instead of gas.
        maxr(:obj:`float`, optional:
            Maximum radius to make edges go out to in units of effective radii
        mix (:obj:`bool`, optional):
            Flag for whether or not the fit is a Bayesian mixture. [NOT WORKING]
        cen (:obj:`bool`, optional):
            Flag for whether the position of the center was fit.
        save (:obj:`bool`, optional):
            Flag for whether to save the plot. Will save as a pdf in the same
            directory as `f` is in but inside a folder called `plots`.
        clobber (:obj:`bool`, optional):
            Flag to overwrite plot file if it already exists. Only matters if
            `save=True`
        
    Returns:
        :obj:`dict`: Dictionary with all of the median values of the
        posteriors in the sampler. Has keys for inclination `inc`, first
        order position angle `pa`, second order position angle `pab`,
        systemic velocity `vsys`, x and y center coordinates `xc` and `yc`,
        `numpy.ndarray`_ of first order tangential velocities `vt`,
        `numpy.ndarray`_ objects of second order tangential and radial
        velocities `v2t` and `v2r`, and `numpy.ndarray`_ of velocity
        dispersions `sig`. If `stds == True` it will also contain keys for
        the 1 sigma lower bounds of the velocity parameters `vtl`, `v2tl`,
        `v2rl`, and `sigl` as well as their 1 sigma upper bounds `vtu`,
        `v2tu`, `v2ru`, and `sigu`. Arrays have lengths that are the same as
        the number of bins (determined automatically or from `jump`). All
        angles are in degrees and all velocities must be in consistent units.

        Plot: The values for the global parameters of the galaxy, the rotation
        curves (with 1 sigma lower and upper bounds) for the different velocity
        components, then comparisons of the MaNGA data, the model, the
        residuals, and chisq for the rotational velocity and the velocity dispersion. 
    '''

    if save and not clobber:
        path = f[:f.rfind('/')+1]
        fname = f[f.rfind('/')+1:-5]
        if os.path.isfile(f'{path}/plots/{fname}.pdf'):
            raise ValueError('Plot file already exists')

    args, resdict, chains, meds = fileprep(f, plate, ifu, smearing, stellar, maxr, mix, cen, fixcent, remotedir=remotedir)
    velmodel, sigmodel = bisym_model(args,resdict,plot=True)
    vel_r = args.remap('vel')
    sig_r = np.sqrt(args.remap('sig_phys2')) if hasattr(args, 'sig_phys2') else args.remap('sig')

    #mask border if necessary
    if args.bordermask is not None:
        velmodel = np.ma.array(velmodel, mask=args.bordermask)
        vel_r = np.ma.array(vel_r, mask=args.bordermask)
        if sigmodel is not None:
            sigmodel = np.ma.array(sigmodel, mask=args.bordermask)
            sig_r = np.ma.array(sig_r, mask=args.bordermask)

    if args.vel_ivar is None: args.vel_ivar = np.ones_like(args.vel)
    if args.sig_ivar is None: args.sig_ivar = np.ones_like(args.sig)

    nvar = len(args.vel) + len(args.sig) - len(meds)
    rchisq = np.sum((vel_r - velmodel)**2 * args.remap('vel_ivar')) / nvar

    #print global parameters on figure
    fig = plt.figure(figsize = (12,9))
    plt.subplot(3,4,1)
    ax = plt.gca()
    plt.axis('off')
    plt.title(f"{resdict['plate']}-{resdict['ifu']} {resdict['type']}",size=20)
    plt.text(.1, .82, r'$i$: %0.1f$^{+%0.1f}_{-%0.1f}$ deg.'
            %(resdict['inc'], resdict['incu'] - resdict['inc'], 
            resdict['inc'] - resdict['incl']), transform=ax.transAxes, size=20)
    plt.text(.1, .64, r'$\phi$: %0.1f$^{+%0.1f}_{-%0.1f}$ deg.'
            %(resdict['pa'], resdict['pau'] - resdict['pa'], 
            resdict['pa'] - resdict['pal']), transform=ax.transAxes, size=20)
    plt.text(.1, .46, r'$\phi_b$: %0.1f$^{+%0.1f}_{-%0.1f}$ deg.'
            %(resdict['pab'], resdict['pabu'] - resdict['pab'], 
            resdict['pab'] - resdict['pabl']), transform=ax.transAxes, size=20)
    plt.text(.1, .28, r'$v_{{sys}}$: %0.1f$^{+%0.1f}_{-%0.1f}$ km/s'
            %(resdict['vsys'], resdict['vsysu'] - resdict['vsys'], 
            resdict['vsys'] - resdict['vsysl']),transform=ax.transAxes, size=20)
    plt.text(.1, .1, r'$\chi_r^2$: %0.1f' % rchisq, 
            transform=ax.transAxes, size=20)
    if cen: plt.text(.1, -.08, r'$(x_c, y_c)$: (%0.1f, %0.1f)' %  
            (resdict['xc'], resdict['yc']), transform=ax.transAxes, size=20)

    #image
    plt.subplot(3,4,2)
    if args.image is not None: plt.imshow(args.image)
    else: plt.text(.5,.5, 'No image found', horizontalalignment='center',
            transform=plt.gca().transAxes, size=14)

    plt.axis('off')

    #Radial velocity profiles
    plt.subplot(3,4,3)
    #if chains is not None: profs(chains, args, plt.gca(), stds=True)
    #else: 
    ls = [r'$V_t$',r'$V_{2t}$',r'$V_{2r}$']
    for i,v in enumerate(['vt', 'v2t', 'v2r']):
        plt.plot(args.edges, resdict[v], label=ls[i]) 

    errors = [[resdict['vtl'], resdict['vtu']], [resdict['v2tl'], resdict['v2tu']], [resdict['v2rl'], resdict['v2ru']]]
    for i,p in enumerate(errors):
        plt.fill_between(args.edges, p[0], p[1], alpha=.5) 

    plt.ylim(bottom=0)
    plt.title('Velocity Profiles')

    #dispersion profile
    plt.subplot(3,4,4)
    plt.plot(args.edges, resdict['sig'])
    plt.fill_between(args.edges, resdict['sigl'], resdict['sigu'], alpha=.5)
    plt.ylim(bottom=0)
    plt.title('Velocity Dispersion Profile')
    plt.xlabel(r'$R_e$')
    plt.ylabel(r'$v$ (km/s)')

    #MaNGA Ha velocity field
    plt.subplot(3,4,5)
    plt.title(f"{resdict['type']} Velocity Data")
    vmax = min(np.max(np.abs(vel_r)), 300)
    plt.imshow(vel_r, cmap='jet', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    #Vel model from dynesty fit
    plt.subplot(3,4,6)
    plt.title('Velocity Model')
    plt.imshow(velmodel,'jet', origin='lower', vmin=-vmax, vmax=vmax) 
    plt.tick_params(left=False, bottom=False,labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(label='km/s', cax=cax)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    #Residuals from vel fit
    plt.subplot(3,4,7)
    plt.title('Velocity Residuals')
    resid = vel_r - velmodel
    vmax = min(np.abs(vel_r-velmodel).max(), 50)
    plt.imshow(vel_r-velmodel, 'jet', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(label='km/s', cax=cax)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    #Chisq from vel fit
    plt.subplot(3,4,8)
    plt.title('Velocity Chi Squared')
    velchisq = (vel_r - velmodel)**2 * args.remap('vel_ivar')
    plt.imshow(velchisq, 'jet', origin='lower', vmin=0, vmax=50)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(cax=cax)

    #MaNGA Ha velocity disp
    plt.subplot(3,4,9)
    plt.title(f"{resdict['type']} Dispersion Data")
    vmax = min(np.max(sig_r), 200)
    plt.imshow(sig_r, cmap='jet', origin='lower', vmax=vmax, vmin=0)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=0)

    #disp model from dynesty fit
    plt.subplot(3,4,10)
    plt.title('Dispersion Model')
    plt.imshow(sigmodel, 'jet', origin='lower', vmin=0, vmax=vmax) 
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=0)

    #Residuals from disp fit
    plt.subplot(3,4,11)
    plt.title('Dispersion Residuals')
    resid = sig_r - sigmodel
    vmax = min(np.abs(sig_r - sigmodel).max(), 50)
    plt.imshow(sig_r-sigmodel, 'jet', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    #Chisq from sig fit
    plt.subplot(3,4,12)
    plt.title('Dispersion Chi Squared')
    sigchisq = (sig_r - sigmodel)**2 * args.remap('sig_ivar')
    plt.imshow(sigchisq, 'jet', origin='lower', vmin=0, vmax=50)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    plt.colorbar(cax=cax)

    plt.tight_layout()

    if args.mix:
        dynesty.plotting.cornerplot(chains, dims = [-3, -2, -1])

    if save:
        path = f[:f.rfind('/')+1]
        fname = f[f.rfind('/')+1:-5]
        plt.savefig(f'{path}plots/{fname}.pdf', format='pdf')
        plt.close()

    return fig

def separate_components(f, plate=None, ifu=None, smearing=True, stellar=False, maxr=None, mix=False, cen=True): 
    '''
    Make a plot `nirvana` output file with the different velocity components
    searated.

    Plot the first order velocity component and the two second order velocity
    components next to each other along with the full model, data, image, and
    global parameters

    Args:
        f (:class:`dynesty.NestedSampler` or :obj:`str` or
        :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        plate (:obj:`int`), optional):
            MaNGA plate number for desired galaxy. Must be specified if
            `auto=False`.
        ifu (:obj:`int`), optional):
            MaNGA IFU design number for desired galaxy. Must be specified if
            `auto=False`.
        smearing (:obj:`bool`, optional):
            Flag for whether or not to apply beam smearing to models.
        stellar (:obj:`bool`, optional):
            Flag for whether or not to use stellar velocity data instead of gas.
        mix (:obj:`bool`, optional):
            Flag for whether or not the fit is a Bayesian mixture. [NOT WORKING]
        cen (:obj:`bool`, optional):
            Flag for whether the position of the center was fit.
        
    Returns:
        Plot: The global parameters of the galaxy, the image of the galaxy,
        the data of the velocity field on the first row. The second row is the
        full model follwed by the different components broken out with + and =
        signs between.
    '''

    args, resdict, chains, meds = fileprep(f, plate, ifu, smearing, stellar, maxr, mix, cen, fixcent)
    z = np.zeros(len(resdict['vt']))
    vtdict, v2tdict, v2rdict = [resdict.copy(), resdict.copy(), resdict.copy()]
    vtdict['v2t'] = z
    vtdict['v2r'] = z
    v2tdict['vt'] = z
    v2tdict['v2r'] = z
    v2rdict['vt'] = z
    v2rdict['v2t'] = z
    if maxr is not None:
        r,th = projected_polar(args.x, args.y, resdict['pa'], resdict['inc'])
        r /= args.reff
        rmask = r > maxr
        args.vel_mask |= rmask
        args.sig_mask |= rmask

    velmodel, sigmodel = bisym_model(args, resdict, plot=True)
    vtmodel,  sigmodel = bisym_model(args, vtdict,  plot=True)
    v2tmodel, sigmodel = bisym_model(args, v2tdict, plot=True)
    v2rmodel, sigmodel = bisym_model(args, v2rdict, plot=True)
    vel_r = args.remap('vel')

    plt.figure(figsize = (12,6))
    vmax = min(max(np.max(np.abs(velmodel)), np.max(np.abs(vtmodel)), np.max(np.abs(v2tmodel)), np.max(np.abs(v2rmodel)), np.max(np.abs(vel_r))), 300)

    plt.subplot(241)
    ax = plt.gca()
    plt.axis('off')
    plt.title(f"{resdict['plate']}-{resdict['ifu']} {resdict['type']}",size=20)
    plt.text(.1, .8, r'$i$: %0.1f$^\circ$'%resdict['inc'], 
            transform=ax.transAxes, size=20)
    plt.text(.1, .6, r'$\phi$: %0.1f$^\circ$'%resdict['pa'], 
            transform=ax.transAxes, size=20)
    plt.text(.1, .4, r'$\phi_b$: %0.1f$^\circ$'%resdict['pab'], 
            transform=ax.transAxes, size=20)
    plt.text(.1, .2, r'$v_{{sys}}$: %0.1f km/s'%resdict['vsys'], 
            transform=ax.transAxes, size=20)

    #image
    plt.subplot(242)
    plt.imshow(args.image)
    plt.axis('off')

    #MaNGA Ha velocity field
    plt.subplot(243)
    plt.title(r'H$\alpha$ Velocity Data')
    vmax = min(np.max(np.abs(vel_r)), 300)
    plt.imshow(vel_r, cmap='RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cax = mal(plt.gca()).append_axes('right', size='5%', pad=.05)
    cb = plt.colorbar(cax=cax)
    cb.set_label('km/s', labelpad=-10)

    plt.subplot(245)
    plt.imshow(velmodel, cmap = 'RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.text(1.05,.5,'=', transform=plt.gca().transAxes, size=30)
    plt.xlabel(r'$V$', fontsize=16)

    plt.subplot(246)
    plt.imshow(vtmodel, cmap = 'RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.text(1.05,.5,'+', transform=plt.gca().transAxes, size=30)
    plt.xlabel(r'$V_t$', fontsize=16)

    plt.subplot(247)
    plt.imshow(v2tmodel, cmap = 'RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.text(1.05,.5,'+', transform=plt.gca().transAxes, size=30)
    plt.xlabel(r'$V_{2t}$', fontsize=16)

    plt.subplot(248)
    plt.imshow(v2rmodel, cmap = 'RdBu', origin='lower', vmin=-vmax, vmax=vmax)
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.xlabel(r'$V_{2r}$', fontsize=16)

    plt.tight_layout(h_pad=2)

def sinewave(f, plate=None, ifu=None, smearing=True, stellar=False, maxr=None, mix=False, cen=True): 
    '''
    Compare the `nirvana` fit to the data azimuthally in radial bins.

    Breaks down the data into radial bins and plots the velocity data points
    in each bin azimuthally, overplotting the fit for each bin. These are
    separated out into a Joy Division style plot.

    Args:
        f (:class:`dynesty.NestedSampler` or :obj:`str` or
        :class:`dynesty.results.Results`):
            Sampler, results, or file of dumped results from `dynesty` fit.
        plate (:obj:`int`), optional):
            MaNGA plate number for desired galaxy. Must be specified if
            `auto=False`.
        ifu (:obj:`int`), optional):
            MaNGA IFU design number for desired galaxy. Must be specified if
            `auto=False`.
        smearing (:obj:`bool`, optional):
            Flag for whether or not to apply beam smearing to models.
        stellar (:obj:`bool`, optional):
            Flag for whether or not to use stellar velocity data instead of gas.
        mix (:obj:`bool`, optional):
            Flag for whether or not the fit is a Bayesian mixture. [NOT WORKING]
        cen (:obj:`bool`, optional):
            Flag for whether the position of the center was fit.
        
    Returns:
        Plot: the data points in each radial bin plotted azimuthally, color
        coded and separated on an arbitrary y axis. The curve the fit generated
        is plotted on top. 
    '''

    #prep the data, parameters, and coordinates
    args, resdict, chains, meds = fileprep(f, plate, ifu, smearing, stellar, maxr, mix, cen, fixcent)
    inc, pa, pab = np.radians([resdict['inc'], resdict['pa'], resdict['pab']])
    r,th = projected_polar(args.x, args.y, pa, inc)
    r /= args.reff

    plt.figure(figsize=(4,len(args.edges)*.75))
    c = plt.cm.jet(np.linspace(0,1,len(args.edges)-1))
    plt.title(f"{resdict['plate']}-{resdict['ifu']} {resdict['type']}")

    #for each radial bin, plot data points and model
    for i in range(len(args.edges)-1):
        cut = (r > args.edges[i]) * (r < args.edges[i+1])
        sort = np.argsort(th[cut])
        thcs = th[cut][sort]
        plt.plot(np.degrees(thcs), args.vel[cut][sort]+100*i, '.', c=c[i])
        
        #generate model from fit parameters
        velmodel = resdict['vsys'] + np.sin(inc) * (resdict['vt'][i] * np.cos(thcs) \
                 - resdict['v2t'][i] * np.cos(2 * (thcs - pab)) * np.cos(thcs) \
                 - resdict['v2r'][i] * np.sin(2 * (thcs - pab)) * np.sin(thcs))
        plt.plot(np.degrees(thcs), velmodel+100*i, 'k--')
        plt.tick_params(left=False, labelleft=False)
        plt.xlabel('Azimuth (deg)')
        plt.tight_layout()

def plotdir(directory=None, fname=None, **kwargs):
    '''
    Make summaryplots of an entire directory of output files.

    Will try to look for automatically named nirvana output files unless told
    otherwise. 
    
    Args:
        directory (:obj:`str`, optional):
            Directory to look for files in
        fname (:obj:`str`, optional):
            Filename format for files you want plotted with appropriate
            wildcards. Defaults to standard nirvana output format
        kwargs (optional):
            Args for summaryplot
    '''
    if directory is None: directory = '/data/manga/digiorgio/nirvana/'
    if fname is None: fname = '*-*_*.nirv'
    fs = glob(directory+fname)
    if len(fs) == 0: raise FileNotFoundError('No files found')
    else: print(len(fs), 'files found')
    for i in tqdm(range(len(fs))):
        try:
            summaryplot(fs[i], save=True, **kwargs)
        except Exception:
            print(fs[i], 'failed')
            print(traceback.format_exc())


def init_ax(fig, pos, facecolor='0.85', tickdir='in', top=True, right=True, majlen=4, minlen=2,
            grid=True, gridcolor='0.75'):
    """
    Convenience method for initializing a `matplotlib.axes.Axes`_ object.

    Args:
        fig (`matplotlib.figure.Figure`_):
            Figure in which to place the Axes object.
        pos (:obj:`list`):
            The rectangle outlining the locating of the axes in the Figure
            object, specifying the left, bottom, width, and height
            dimensions. See the ``rect`` argument of
            `matplotlib.figure.Figure.add_axes`_.
        facecolor (:obj:`str`, optional):
            Color for the axis background.
        tickdir (:obj:`str`, optional):
            Direction for the axis tick marks.
        top (:obj:`bool`, optional):
            Add ticks to the top axis.
        right (:obj:`bool`, optional):
            Add ticks to the right axis.
        majlen (:obj:`int`, optional):
            Length for the major tick marks.
        minlen (:obj:`int`, optional):
            Length for the minor tick marks.
        grid (:obj:`bool`, optional):
            Include a major-axis grid.
        gridcolor (:obj:`str`, optional):
            Color for the grid lines.

    Returns:
        `matplotlib.axes.Axes`_: Axes object
    """
    ax = fig.add_axes(pos, facecolor=facecolor)
    ax.minorticks_on()
    ax.tick_params(which='major', length=majlen, direction=tickdir, top=top, right=right)
    ax.tick_params(which='minor', length=minlen, direction=tickdir, top=top, right=right)
    if grid:
        ax.grid(True, which='major', color=gridcolor, zorder=0, linestyle='-')
    return ax


def get_twin(ax, axis, tickdir='in', majlen=4, minlen=2):
    """
    Construct the "twin" axes of the provided Axes object.

    Args:
        ax (`matplotlib.axes.Axes`_):
            Original Axes object.
        axis (:obj:`str`):
            The axis to *mirror*. E.g., to get the right ordinate that
            mirrors the left ordinate, use ``axis='y'``.
        tickdir (:obj:`str`, optional):
            Direction for the axis tick marks.
        majlen (:obj:`int`, optional):
            Length for the major tick marks.
        minlen (:obj:`int`, optional):
            Length for the minor tick marks.

    Returns:
        `matplotlib.axes.Axes`_: Axes object selecting the requested axis
        twin.
    """
    axt = ax.twinx() if axis == 'y' else ax.twiny()
    axt.minorticks_on()
    axt.tick_params(which='major', length=majlen, direction=tickdir)
    axt.tick_params(which='minor', length=minlen, direction=tickdir)
    return axt


def rotate_y_ticks(ax, rotation, va):
    """
    Rotate all the existing y tick labels by the provided rotation angle
    (deg) and reset the vertical alignment.

    Args:
        ax (`matplotlib.axes.Axes`_):
            Rotate the tick labels for this Axes object. **The object is
            edited in place.**
        rotation (:obj:`float`):
            Rotation angle in degrees
        va (:obj:`str`):
            Vertical alignment for the tick labels.
    """
    for tick in ax.get_yticklabels():
        tick.set_rotation(rotation)
        tick.set_verticalalignment(va)


