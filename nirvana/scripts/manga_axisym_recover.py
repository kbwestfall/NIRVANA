"""
Script that runs the axisymmetric, least-squares fit for MaNGA data.
"""
import os
import argparse
import pathlib

from IPython import embed

import numpy as np
from scipy import sparse
from matplotlib import pyplot

from astropy.io import fits

from ..data import manga
from ..data.bin2d import Bin2D, VoronoiBinning
from ..models import axisym
from ..models import oned
from ..models import twod
from ..models import geometry
from ..models.beam import gauss2d_kernel, ConvolveFFTW
from ..util import fileio

import warnings
warnings.simplefilter('error', RuntimeWarning)

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ofile', type=str, help='Name for output file')
    parser.add_argument('nsim', type=int, help='Number of simulations to perform')

    parser.add_argument('--basep', default=[0., 0., 45., 30., 0.], nargs=5,
                        help='Base thin disk parameters: x0, y0, pa (deg), inc (deg), vsys')

    parser.add_argument('--rc', default='HyperbolicTangent', type=str,
                        help='Rotation curve parameterization to use: HyperbolicTangent or PolyEx')
    parser.add_argument('--rcp', default=[150., 10.], nargs='*',
                        help='*Deprojected* rotation curve parameters')
                        
    parser.add_argument('--dc', default=None, type=str,
                        help='Dispersion profile parameterization to use: Exponential, ExpBase, '
                             'or Const.  If None, velocity dispersion is not simulated/fit.')
    parser.add_argument('--dcp', default=None, nargs='*',
                        help='Dispersion profile parameters')

    parser.add_argument('--ignore_disp', dest='fit_disp', default=True, action='store_false',
                        help='Use the dispersion model to create the synthetic data but ignore '
                             'it when fitting.')

    # TODO: Allow for Sersic profile to have a different geometry than the kinematics?
    parser.add_argument('--sersic', default=[10., 1.], nargs=2,
                        help='Sersic photometry parameters for luminosity weighting: Reff '
                             'in arcsec and the Sersic index.')
    parser.add_argument('--intrinsic_sb', default=False, action='store_true',
                        help='By default, the intrinsic SB is used to generate the synthetic '
                             'data, but the *smoothed* SB is used for the luminosity weighting '
                             'during the fit to mimic the use in the data.  Setting this flag'
                             'performs the more idealized (and correct) simulation where both '
                             'synthetic data and the model are generated using the same SB.')
    parser.add_argument('--snr', type=float, default=30., help='S/N normalization')
    parser.add_argument('--binning_snr', type=float, default=None,
                        help='Minimum S/N used for binning the data.  If not provided, data are '
                             'not binned.')

    parser.add_argument('--ifusize', type=float, default=32.,
                        help='Size of the hexagonal patch in arcsec.')

    parser.add_argument('--covar_sim', default=False, action='store_true',
                        help='Include MaNGA-like covariance when generating the noise field for '
                             'each simulation.')
    parser.add_argument('--covar_fit', default=False, action='store_true',
                        help='Assume MaNGA-like covariance when *fitting* the synthetic data in '
                             'each simulation.')

    parser.add_argument('--psf_sim', type=float, default=2.5, 
                        help='FWHM in arcsec of a Gaussian PSF to use when generating the '
                             'synthetic model data to fit.  If negative, no PSF is included.')
    parser.add_argument('--psf_fit', type=float, default=2.5, 
                        help='FWHM of the Gaussian PSF to use when *fitting* the synthetic data '
                             'in each simulation.  If negative, no PSF is included in the fit.')

    parser.add_argument('--verbose', default=-1, type=int,
                        help='Verbosity level.  -1=surpress all terminal output; 0=only status '
                             'output written to terminal; 1=show fit result QA plot; 2=full '
                             'output.')

    parser.add_argument('--screen', default=False, action='store_true',
                        help='Indicate that the script is being run behind a screen (used to set '
                             'matplotlib backend).') 

    parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                        help='Overwrite any existing files.')

#    parser.add_argument('--scatter', default=0.0, help='Intrinsic scatter to add')
#    parser.add_argument('--vel_rej', nargs='*', default=[15,10,10,10],
#                        help='The velocity rejection sigma.  Provide 1 or 4 numbers.  If 1 '
#                             'number provided, the same sigma threshold is used for all fit '
#                             'iterations.')
#    parser.add_argument('--sig_rej', nargs='*', default=[15,10,10,10],
#                        help='The dispersion rejection sigma.  Provide 1 or 4 numbers.  If 1 '
#                             'number provided, the same sigma threshold is used for all fit '
#                             'iterations.')
# TODO: Bring these back when simulating a fit to a specific MaNGA target?
#    parser.add_argument('plate', type=int, help='MaNGA plate identifier (e.g., 8138)')
#    parser.add_argument('ifu', type=int, help='MaNGA ifu identifier (e.g., 12704)')
#    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
#                        help='DAP analysis key used to select the data files.  This is needed '
#                             'regardless of whether or not you specify the directory with the '
#                             'data files (using --root).')
#    parser.add_argument('--dr', default='MPL-11', type=str,
#                        help='The MaNGA data release.  This is only used to automatically '
#                             'construct the directory to the MaNGA galaxy data (see also '
#                             '--redux and --analysis), and it will be ignored if the root '
#                             'directory is set directly (using --root).')
#    parser.add_argument('--redux', default=None, type=str,
#                        help='Top-level directory with the MaNGA DRP output.  If not defined and '
#                             'the direct root to the files is also not defined (see --root), '
#                             'this is set by the environmental variable MANGA_SPECTRO_REDUX.')
#    parser.add_argument('--analysis', default=None, type=str,
#                        help='Top-level directory with the MaNGA DAP output.  If not defined and '
#                             'the direct root to the files is also not defined (see --root), '
#                             'this is set by the environmental variable MANGA_SPECTRO_ANALYSIS.')
#    parser.add_argument('--root', default=None, type=str,
#                        help='Path with *all* fits files required for the fit.  This includes ' \
#                             'the DRPall file, the DRP LOGCUBE file, and the DAP MAPS file.  ' \
#                             'The LOGCUBE file is only required if the beam-smearing is ' \
#                             'included in the fit.')

    return parser.parse_args() if options is None else parser.parse_args(options)


def _fit_meta_dtype(par_names):
    """
    Set the data type for a `numpy.recarray`_ used to hold metadata of the
    best-fit model.

    Args:
        par_names (array-like):
            Array of strings with the short names for the model parameters.

    Returns:
        :obj:`list`: The list of tuples providing the name, data type, and shape
        of each `numpy.recarray`_ column.
    """
    gp = [(f'G_{n}'.upper(), np.float) for n in par_names]
    bp = [(f'F_{n}'.upper(), np.float) for n in par_names]
    bpe = [(f'E_{n}'.upper(), np.float) for n in par_names]
    
    return [('MAPN', np.int),
            ('IFUSIZE', np.float),
            ('PIXSCALE', np.float),
            ('PSFINP', np.float),
            ('PSFOUT', np.float),
            ('SNR', np.float),
            ('BINSNR', np.float),
            ('REFF', np.float),
            ('SERSICN', np.float),
            ('PA', np.float),
            ('ELL', np.float),
            ('VNFIT', np.int),
            ('VCHI2', np.float),
            ('SNFIT', np.int),
            ('SCHI2', np.float),
            ('CHI2', np.float),
            ('RCHI2', np.float),
            ('STATUS', np.int),
            ('SUCCESS', np.int)] + gp + bp + bpe


def main(args):

    # Running the script behind a screen, so switch the matplotlib backend
    if args.screen:
        pyplot.switch_backend('agg')

    if args.nsim < 1:
        raise ValueError('Must run at least 1 simulation.')
    ofile = pathlib.Path(args.ofile)
    if ofile.exists() and not args.overwrite:
        raise FileExistsError(f'{str(ofile)} exists!')

    # Instantiate the disk
    rc_class = getattr(oned, args.rc)()
    dc_class = None if args.dc is None else getattr(oned, args.dc)()
    disk = axisym.AxisymmetricDisk(rc=rc_class, dc=dc_class)
    fit_disp = args.fit_disp and args.dc is not None

    # Set the input parameters
    if len(args.rcp) != disk.rc.np:
        raise ValueError(f'Incorrect number of rotation-curve parameters; expected {disk.rc.np}, '
                         f'provided {len(args.rcp)}.')
    p0 = np.append(args.basep, args.rcp)
    if args.dc is not None:
        if args.dcp is None:
            raise ValueError('Must provide parameters for dispersion parameterization.')
        if len(args.dcp) != disk.dc.np:
            raise ValueError(f'Incorrect number of dispersion parameters; expected {disk.dc.np},'
                             f' provided {len(args.dcp)}.')
        p0 = np.append(p0, args.dcp)

    # Get the IFU Mask
    pixelscale = 0.5
    width_buffer = 10
    n = int(np.floor(args.ifusize/pixelscale)) + width_buffer
    if n % 2 != 0:
        n += 1
    x = np.arange(n, dtype=float)[::-1] - n//2
    y = np.arange(n, dtype=float) - n//2
    x, y = np.meshgrid(pixelscale*x, pixelscale*y)
    ifu_mask = geometry.point_inside_polygon(geometry.hexagon_vertices(d=args.ifusize),
                                             np.column_stack((x.ravel(), y.ravel())))
    ifu_mask = np.logical_not(ifu_mask).reshape(x.shape)

    # Set the beam-smearing kernel(s)
    cnvfftw = ConvolveFFTW(x.shape)
    sig2fwhm = np.sqrt(8*np.log(2))
    beam_sim = gauss2d_kernel(n, args.psf_sim/pixelscale/sig2fwhm) if args.psf_sim > 0 else None
    beam_fit = gauss2d_kernel(n, args.psf_fit/pixelscale/sig2fwhm) if args.psf_fit > 0 else None

    # Set the surface-brightness profile
    # TODO: Oversample and then block average to mimic integration over the size
    # of the pixel?
    reff, sersic_n = args.sersic
    # NOTE: This forces the Sersic profile to have the same pa and inclination
    # as the kinematics...
    pa, inc = args.basep[2:4]
    ell = 1 - np.cos(np.radians(inc))
    sb = twod.Sersic2D(1., reff, sersic_n, ellipticity=ell, position_angle=pa)(x,y)
    smeared_sb = cnvfftw(sb, beam_sim)

    # Set the S/N using the *smeared* SB and renormalize to the provided value
    snr = np.sqrt(smeared_sb)
    scale_fac = args.snr/np.amax(snr)
    snr *= scale_fac
    smeared_sb *= scale_fac**2
    smeared_sb_err = smeared_sb/snr

    # Bin the data
    gpm = np.logical_not(ifu_mask)
    binid = np.full(ifu_mask.shape, -1, dtype=int)
    if args.binning_snr is None:
        # Make a fake binid based on the ifu_mask
        binid[gpm] = np.arange(np.sum(gpm))
        binner = Bin2D(binid=binid)
        binned_sb = binner.remap(binner.bin(smeared_sb), masked=False, fill_value=0.)
        binned_sb_err = binner.remap(binner.bin(smeared_sb_err), masked=False, fill_value=0.)
        binned_snr = binner.remap(binner.bin(snr), masked=False, fill_value=0.)
    else:
        binid[gpm] = VoronoiBinning.bin_index(x[gpm], y[gpm], smeared_sb[gpm],
                                              smeared_sb_covar[np.ix_(gpm,gpm)] if args.covar_sim
                                                else smeared_sb_err[gpm],
                                              args.binning_snr, show=False)
        binner = Bin2D(binid=binid)
        binned_sb = binner.remap(binner.bin(smeared_sb), masked=False, fill_value=0.)

        if args.covar_sim:
            print('Generating Flux Covariance')
            _, smeared_sb_covar = manga.manga_map_covar(1./smeared_sb_err**2,
                                                        positive_definite=False, fill=True)
            binned_sb_covar = binner.remap_covar(binner.bin_covar(smeared_sb_covar))
            binned_sb_err = np.sqrt(binned_sb_covar.diagonal())
        else:
            binned_sb_covar = None
            binned_sb_err = binner.bin_covar(sparse.diags(smeared_sb_err.ravel()**2, format='csr'))
            binned_sb_err = binner.remap(np.sqrt(binned_sb_err.diagonal()), masked=False,
                                         fill_value=0.)
        binned_snr = binned_sb/(binned_sb_err + (binned_sb_err == 0.))

    # Get the model velocities and dispersions
    if disk.dc is None:
        vel = disk.model(par=p0, x=x, y=y, sb=sb, beam=beam_sim)
        sig = None
    else:
        vel, sig = disk.model(par=p0, x=x, y=y, sb=sb, beam=beam_sim)
    _, vel, sig = binner.bin_moments(smeared_sb, vel, sig)
    vel = binner.remap(vel)
    sig = np.full(vel.shape, 30., dtype=float) if sig is None else binner.remap(sig)
    vel_ivar = np.ma.MaskedArray(np.ma.divide(binned_snr, sig)**2, mask=ifu_mask)
    sig_ivar = np.ma.MaskedArray(np.ma.divide(binned_snr, sig)**2, mask=ifu_mask)

    if args.covar_sim:
        print('Generating Velocity Covariance')
        _, vel_covar = manga.manga_map_covar(vel_ivar, positive_definite=False, fill=True)
        if sig is not None:
            print('Generating Sigma Covariance')
            _, sig_covar = manga.manga_map_covar(sig_ivar, positive_definite=False, fill=True)
    else:
        vel_covar = None
        sig_covar = None

    # Initialize the table output
    disk_par_names = disk.par_names(short=True)
    metadata = fileio.init_record_array(args.nsim, _fit_meta_dtype(disk_par_names))
    # NOTE: Instead of adding single values to the header of the output file,
    # these quantities are kept and repeated in the table to facilitate
    # concatenation of tables from multiple simulations.
    metadata['MAPN'] = n
    metadata['IFUSIZE'] = args.ifusize
    metadata['PIXSCALE'] = pixelscale
    metadata['PSFINP'] = args.psf_sim
    metadata['PSFOUT'] = args.psf_fit
    metadata['SNR'] = args.snr
    if args.binning_snr is not None:
        metadata['BINSNR'] = args.binning_snr
    metadata['REFF'] = reff
    metadata['SERSICN'] = sersic_n
    metadata['PA'] = pa
    metadata['ELL'] = ell

    # Get the noise-free mock
    noisefree_mock = disk.mock_observation(p0, x=x, y=y,
                                           sb=sb if args.intrinsic_sb else smeared_sb,
                                           binid=binid, vel_ivar=vel_ivar, vel_covar=vel_covar,
                                           vel_mask=ifu_mask, sig_ivar=sig_ivar,
                                           sig_covar=sig_covar, sig_mask=ifu_mask, beam=beam_sim,
                                           cnvfftw=cnvfftw, positive_definite=True)

    # Generate *all* the deviates.  All the deviates are drawn here to speed up
    # the multivariate deviates.
    vgpm, dv, sgpm, ds2 = noisefree_mock.deviate(size=args.nsim,
                                                 sigma='ignore' if disk.dc is None else 'drawsqr')
    if args.nsim == 1:
        dv = np.expand_dims(dv, 0)
        if ds2 is not None:
            ds2 = np.expand_dims(ds2, 0)

    _vel = noisefree_mock.vel.copy()
    _sig2 = None if disk.dc is None else noisefree_mock.sig_phys2.copy()

    noisy_mock = noisefree_mock.copy()
    noisy_mock._set_beam(beam_fit, None)
    disk_fom = disk._get_fom()
    for i in range(args.nsim):
        print(f'Sim {i+1}/{args.nsim}', end='\r')
        noisy_mock.vel[vgpm] = _vel[vgpm] + dv[i]
        if disk.dc is not None:
            _sig2[sgpm] += ds2[i]
            noisy_mock.update_sigma(sig=_sig2, sqr=True)
            _sig2[sgpm] -= ds2[i]

        # TODO: Change this to use the same fit function as used by the
        # manga_axisym.py script.
        disk.lsq_fit(noisy_mock, sb_wgt=True, p0=p0, scatter=None, verbose=args.verbose,
                     assume_posdef_covar=True, ignore_covar=not args.covar_fit, cnvfftw=cnvfftw)

        vfom, sfom = disk_fom(disk.par, sep=True)
        metadata['VNFIT'][i] = np.sum(disk.vel_gpm)
        metadata['VCHI2'][i] = np.sum(vfom**2)
        if disk.dc is not None:
            metadata['SNFIT'][i] = np.sum(disk.sig_gpm)
            metadata['SCHI2'][i] = np.sum(sfom**2)
        metadata['CHI2'][i] = metadata['VCHI2'][i] + metadata['SCHI2'][i]
        metadata['RCHI2'][i] = metadata['CHI2'][i] \
                                / (metadata['VNFIT'][i] + metadata['SNFIT'][i] - disk.np)

        metadata['STATUS'][i] = disk.fit_status
        metadata['SUCCESS'][i] = int(disk.fit_success)

        for n, gp, p, pe in zip(disk_par_names, p0, disk.par, disk.par_err):
            metadata[f'G_{n}'.upper()][i] = gp
            metadata[f'F_{n}'.upper()][i] = p
            metadata[f'E_{n}'.upper()][i] = pe
    print(f'Sim {args.nsim}/{args.nsim}')

    # Build the output fits extension (base) headers
    #   - Primary header
    prihdr = fileio.initialize_primary_header()
    #   - Add the model types to the primary header
    prihdr['MODELTYP'] = ('AxisymmetricDisk', 'nirvana class used to fit the data')
    prihdr['RCMODEL'] = (disk.rc.__class__.__name__, 'Rotation curve parameterization')
    if disk.dc is not None:
        prihdr['DCMODEL'] = (disk.dc.__class__.__name__, 'Dispersion profile parameterization')
    hdus = [fits.PrimaryHDU(header=prihdr),
            fits.BinTableHDU.from_columns([fits.Column(name=n,
                                                       format=fileio.rec_to_fits_type(metadata[n]),
                                                       array=metadata[n])
                                             for n in metadata.dtype.names],
                                           name='FITMETA')]

    if args.ofile.split('.')[-1] == 'gz':
        _ofile = args.ofile[:ofile.rfind('.')]
        compress = True
    else:
        compress = False
        _ofile = args.ofile

    fits.HDUList(hdus).writeto(_ofile, overwrite=True, checksum=True)
    if compress:
        fileio.compress_file(_ofile, overwrite=True)
        os.remove(_ofile)

