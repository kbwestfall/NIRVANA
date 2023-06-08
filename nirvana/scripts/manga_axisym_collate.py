"""
Script to collate results from AxisymmetricDisk fits to MaNGA data into a single
fits file.
"""
import warnings
from pathlib import Path
import argparse

from IPython import embed

import numpy as np
from astropy.io import fits

import nirvana
from nirvana.data.manga import manga_paths, manga_file_names, MaNGAGlobalPar
from nirvana.models.axisym import AxisymmetricDisk, _fit_meta_dtype
from nirvana.models.oned import Func1D
from nirvana.models.multitrace import _ad_meta_dtype
from nirvana.util import fileio
from nirvana.util.inspect import all_subclasses

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir', type=str,
                        help='Top-level directory with the output from nirvana_manga_axisym or '
                             'nirvana_manga_asymdrift.  This script recursively combs through '
                             'the directory structure, looking for files produced by the fitting '
                             'scripts.  The file names are expected to be '
                             '"nirvana-manga-axisym-[plate]-[ifu]-[tracer].fits.gz" or '
                             '"nirvana-manga-asymdrift-[plate]-[ifu].fits.gz", respectively.  Each '
                             'combination of ([plate], [ifu], [tracer]) *must* be unique.  '
                             'All files must also be the result of fitting the same '
                             'parameterization of the axisymmetric model.')
    parser.add_argument('ofile', type=str, help='Name for the output file.')
    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
                        help='DAP analysis key used to select the data files.  This is used '
                             'to select the relevant extension of the DAPall file for '
                             'cross-matching.')
    parser.add_argument('--dr', default='MPL-11', type=str,
                        help='The MaNGA data release.  This is only used to automatically '
                             'construct the MaNGA DRPall and DAPall file names.')
    parser.add_argument('--redux', default=None, type=str,
                        help='Top-level directory with the MaNGA DRP output.  If not defined and '
                             'the direct root to the files is also not defined (see --root), '
                             'this is set by the environmental variable MANGA_SPECTRO_REDUX.')
    parser.add_argument('--analysis', default=None, type=str,
                        help='Top-level directory with the MaNGA DAP output.  If not defined, '
                             'this is set by the environmental variable MANGA_SPECTRO_ANALYSIS.  '
                             'This is only used to automatically construct the MaNGA DRPall and '
                             'DAPall file names.')
    parser.add_argument('--drpall', default=None, type=str,
                        help='The full path to the MaNGA DRPall file.  If None, the file name '
                             'is constructed assuming the default paths.')
    parser.add_argument('--dapall', default=None, type=str,
                        help='The full path to the MaNGA DAPall file.  If None, the file name '
                             'is constructed assuming the default paths.')
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help='Instead of trolling the directory for output files, attempt to '
                             'find output for all plateifus in the DAPall file, for both Gas '
                             'and Stars.')
    parser.add_argument('-o', '--overwrite', default=False, action='store_true',
                        help='Overwrite existing an existing output file.')
    parser.add_argument('-a', '--asymdrift', default=False, action='store_true',
                        help='Default is to find files resulting from nirvana_manga_axisym.  '
                             'This option changes to finding output from nirvana_manga_asymdrift.')
    parser.add_argument('-r', '--max_nr', default=None, type=int,
                        help='Maximum number of samples for radial profiles.  If not provided, '
                             'set by the maximum number of samples in the files to collate.  If '
                             'provided, code still searches files, but it will use either this '
                             'number or the maximum number of samples in the files to collate, '
                             'whichever is smaller.')

    return parser.parse_args() if options is None else parser.parse_args(options)

def main(args):

    # Set the file name roots
    nirvana_root = 'nirvana-manga-asymdrift' if args.asymdrift else 'nirvana-manga-axisym'

    # Set the top-level directory
    idir = Path(args.dir).resolve()

    # Check input directory exists and attempt to find files to collate
    if not idir.exists():
        raise NotADirectoryError(f'{idir} does not exist!')
    files = list(idir.rglob(f'{nirvana_root}*.fits.gz'))
    if len(files) == 0:
        raise ValueError(f'No files found with the expected naming convention in {idir}!')

    oroot = np.unique([f.parent.parent for f in files])
    if len(oroot) > 1:
        raise ValueError('Currently cannot handle more than one root directory.')
    oroot = oroot[0]

    # Get the list of plates, ifus, and tracers
    if args.full:
        plateifu = None
    else:
        if args.asymdrift:
            plateifu = np.array(['-'.join(f.name.split('.')[0].split('-')[-2:]) for f in files])
            if plateifu.size != np.unique(plateifu).size:
                raise ValueError('Plate-IFUs must be unique for all asymdrift files found in '
                                 f'{args.dir}!')
        else:
            pltifutrc = np.array(['-'.join(f.name.split('.')[0].split('-')[-3:])
                                for f in files])
            if pltifutrc.size != np.unique(pltifutrc).size:
                raise ValueError('Plate-IFUs must be unique for all axisym files for each tracer '
                                f'found in {args.dir}!')
            plateifu = np.unique(['-'.join(p.split('-')[:2]) for p in pltifutrc])

    # Check the output file and determine if it is expected to be compressed
    ofile = Path(args.ofile).resolve()
    if ofile.exists() and not args.overwrite:
        raise FileExistsError(f'{ofile} already exists!')
    if ofile.name.split('.')[-1] == 'gz':
        _ofile = ofile.with_suffix('')
        compress = True
    else:
        _ofile = ofile
        compress = False

    # Attempt to find the DRPall and DAPall files
    if args.drpall is None or args.dapall is None:
        _drpall_path, _, _, _dapall_path, _ \
                = manga_paths(0, 0, dr=args.dr, redux_path=args.redux, analysis_path=args.analysis)
        _drpall_file, _, _, _dapall_file, _ \
                = manga_file_names(0, 0, dr=args.dr)
    _drpall_file = (Path(_drpall_path).resolve() / _drpall_file) \
                        if args.drpall is None else Path(args.drpall).resolve()
    if not _drpall_file.exists():
        raise FileNotFoundError(f'DRPall file {_drpall_file} does not exist!')
    _dapall_file = (Path(_dapall_path).resolve() / _dapall_file) \
                        if args.dapall is None else Path(args.dapall).resolve()
    if not _dapall_file.exists():
        raise FileNotFoundError(f'DAPall file {_dapall_file} does not exist!')

    # Find the first pair of Gas and Stars files, if files are not asymdrift
    # fits
    if args.asymdrift:
        ex_files = [files[0], files[0]]
        md_keys = ['GMODTYP', 'SMODTYP']
        rc_keys = ['GRCMOD', 'SRCMOD']
        dc_keys = ['GDCMOD', 'SDCMOD']
        meta_ext = ['GAS_FITMETA', 'STR_FITMETA']
    else:
        # NOTE: Can't loop through plate-ifu numbers in case "full" was
        # selected.  Need to iterate through files, pull out the plate ifu and
        # then check if both exist.
        for f in files:
            plate, ifu = f.name.split('-')[3:5]
            gas_file = oroot / plate / f'{nirvana_root}-{plate}-{ifu}-Gas.fits.gz'
            str_file = oroot / plate / f'{nirvana_root}-{plate}-{ifu}-Stars.fits.gz'
            if gas_file.exists() and str_file.exists():
                ex_files = [gas_file, str_file]
                break
        md_keys = ['MODELTYP', 'MODELTYP']
        rc_keys = ['RCMODEL', 'RCMODEL']
        dc_keys = ['DCMODEL', 'DCMODEL']
        meta_ext = ['FITMETA', 'FITMETA']

    # Use the example files to get the rc and dc class names for each tracer, and check
    # that both models are both AxisymmetricDisk models
    rc_class = [None, None]
    dc_class = [None, None]
    for i in range(2):
        with fits.open(ex_files[i]) as hdu:
            if hdu[0].header[md_keys[i]] != 'AxisymmetricDisk':
                raise ValueError('All results must use the same thin-disk model type.')
            rc_class[i] = hdu[0].header[rc_keys[i]]
            dc_class[i] = hdu[0].header[dc_keys[i]] if dc_keys[i] in hdu[0].header else None

    # Check the input rotation curve and dispersion profile parameterization
    func1d_classes = all_subclasses(Func1D)
    func1d_class_names = [c.__name__ for c in func1d_classes]
    if not all([rc in func1d_class_names for rc in rc_class]):
        raise ValueError(f'Rotation curves must be a known 1D parameterization.')
    if not all([dc is None or dc in func1d_class_names for dc in dc_class]):
        raise ValueError(f'Dispersion profiles must be a known 1D parameterization.')
    gas_disk = AxisymmetricDisk(rc=func1d_classes[func1d_class_names.index(rc_class[0])](),
                                dc=None if dc_class[0] is None 
                                else func1d_classes[func1d_class_names.index(dc_class[0])]())
    str_disk = AxisymmetricDisk(rc=func1d_classes[func1d_class_names.index(rc_class[1])](),
                                dc=None if dc_class[1] is None 
                                else func1d_classes[func1d_class_names.index(dc_class[1])]())

    # Get the maximum number of radii for the binned data
    maxnr = 0
    max_adnr = 0
    print('Finding maximum number of radial bins')
    nf = len(files)
    for i, f in enumerate(files):
        print(f'{i+1}/{nf}', end='\r')
        with fits.open(f) as hdu:
            if args.asymdrift:
                try:
                    maxnr = max(maxnr, hdu['GAS_FITMETA'].data['BINR'].shape[1])
                except:
                    maxnr = max(maxnr, 1)
                try:
                    maxnr = max(maxnr, hdu['STR_FITMETA'].data['BINR'].shape[1])
                except:
                    maxnr = max(maxnr, 1)
                try:
                    max_adnr = max(max_adnr, hdu['ADPROF'].data['BINR'].shape[1])
                except:
                    max_adnr = max(max_adnr, 1)
            else:
                try:
                    maxnr = max(maxnr, hdu['FITMETA'].data['BINR'].shape[1])
                except:
                    maxnr = max(maxnr, 1)
        if args.max_nr is not None:
            maxnr = min(args.max_nr, maxnr)
            max_adnr = min(args.max_nr, max_adnr)
        if (args.asymdrift and maxnr == args.max_nr and max_adnr == args.max_nr) \
                or (not args.asymdrift and maxnr == args.max_nr):
            break
    print(f'{nf}/{nf}')

    # Get the data type for the output tables
    _gas_dtype = _fit_meta_dtype(gas_disk.par_names(short=True), maxnr, gas_disk.mbm)
    _str_dtype = _fit_meta_dtype(str_disk.par_names(short=True), maxnr, str_disk.mbm)
    gas_meta_keys = [d[0] for d in _gas_dtype]
    str_meta_keys = [d[0] for d in _str_dtype]
    # Added columns are:
    #   FINISHED:   Flag that the observation has the correct fit type and that
    #               the output files exist
    #   QUAL:       A bitmask quality flag pulled from the primary header of the
    #               primary fit output file; see nirvana.models.thindisk.ThinDiskGlobalBitMask
    _gas_dtype += [('FINISHED', np.int), ('QUAL', np.int)]
    _str_dtype += [('FINISHED', np.int), ('QUAL', np.int)]
    if args.asymdrift:
        _ad_dtype = _ad_meta_dtype(max_adnr)
        ad_meta_keys = [d[0] for d in _ad_dtype]
        _ad_dtype += [('FINISHED', np.int), ('QUAL', np.int)]

    # Read the DRPall file
    with fits.open(_drpall_file) as hdu:
        drpall = hdu['MANGA'].data
    # Read the DAPall file
    with fits.open(_dapall_file) as hdu:
        dapall = hdu[args.daptype].data

    # Generate the list of results to collate
    indx = np.where(dapall['DAPDONE'])[0] if args.full \
                else np.array([np.where(dapall['PLATEIFU'] == p)[0][0] for p in plateifu])

    # Collate the data
    gas_metadata = fileio.init_record_array(indx.size, _gas_dtype)
    str_metadata = fileio.init_record_array(indx.size, _str_dtype)
    if args.asymdrift:
        ad_metadata = fileio.init_record_array(indx.size, _ad_dtype)
    for i, j in enumerate(indx):
        print(f'Collating {i+1}/{indx.size}', end='\r')

        galmeta = MaNGAGlobalPar(dapall['PLATE'][j], dapall['IFUDESIGN'][j], drpall=drpall,
                                 dapall=dapall)

#        assert galmeta.dapindx == j, 'Bad match'

        # Save all the metadata regardless of whether or not the galaxy was fit
        gas_metadata['MANGAID'][i] = str_metadata['MANGAID'][i] = galmeta.mangaid
        gas_metadata['PLATEIFU'][i] = str_metadata['PLATEIFU'][i] = galmeta.plateifu
        if args.asymdrift:
            ad_metadata['MANGAID'][i] = galmeta.mangaid
            ad_metadata['PLATEIFU'][i] = galmeta.plateifu
        gas_metadata['PLATE'][i] = str_metadata['PLATE'][i] = galmeta.plate
        gas_metadata['IFU'][i] = str_metadata['IFU'][i] = galmeta.ifu
        gas_metadata['DRPALLINDX'][i] = str_metadata['DRPALLINDX'][i] = galmeta.drpindx
        gas_metadata['DAPALLINDX'][i] = str_metadata['DAPALLINDX'][i] = galmeta.dapindx
        gas_metadata['MNGTARG1'][i] = str_metadata['MNGTARG1'][i] = galmeta.mngtarg1
        gas_metadata['MNGTARG3'][i] = str_metadata['MNGTARG3'][i] = galmeta.mngtarg3
        gas_metadata['DRP3QUAL'][i] = str_metadata['DRP3QUAL'][i] = galmeta.drp3qual
        gas_metadata['DAPQUAL'][i] = str_metadata['DAPQUAL'][i] = galmeta.dapqual
        gas_metadata['OBJRA'][i] = str_metadata['OBJRA'][i] = galmeta.ra
        gas_metadata['OBJDEC'][i] = str_metadata['OBJDEC'][i] = galmeta.dec
        gas_metadata['Z'][i] = str_metadata['Z'][i] = galmeta.z
        gas_metadata['ASEC2KPC'][i] = str_metadata['ASEC2KPC'][i] = galmeta.kpc_per_arcsec()
        gas_metadata['PHOTKEY'][i] = str_metadata['PHOTKEY'][i] = galmeta.phot_key
        gas_metadata['REFF'][i] = str_metadata['REFF'][i] = galmeta.reff
        gas_metadata['SERSICN'][i] = str_metadata['SERSICN'][i] = galmeta.sersic_n
        gas_metadata['PA'][i] = str_metadata['PA'][i] = galmeta.pa
        gas_metadata['ELL'][i] = str_metadata['ELL'][i] = galmeta.ell
        gas_metadata['Q0'][i] = str_metadata['Q0'][i] = galmeta.q0

        # Add the gas data
        gas_file = f'{nirvana_root}-{galmeta.plateifu}.fits.gz' if args.asymdrift \
                        else f'{nirvana_root}-{galmeta.plateifu}-Gas.fits.gz'
        gas_file = oroot / str(galmeta.plate) / gas_file
        if gas_file.exists():
            with fits.open(gas_file) as hdu:
                # Confirm the output has the expected model parameterization
                if hdu[0].header[md_keys[0]] != 'AxisymmetricDisk' \
                        or hdu[0].header[rc_keys[0]] != rc_class[0] \
                        or dc_class[0] is not None and hdu[0].header[dc_keys[0]] != dc_class[0] \
                        or dc_class[0] is None and dc_keys[0] in hdu[0].header:
                    warnings.warn(f'{gas_file} is not an AxisymmetricDisk fit with the same model '
                                  f'parameterization as determined by the template file, '
                                  f'{ex_files[0]}.  Skipping this file.')
                    # Toggle flag
                    gas_metadata['QUAL'][i] \
                            = gas_disk.gbm.turn_on(hdu[0].header['QUAL'], 'DIFFMOD')
                else:
                    gas_metadata['FINISHED'][i] = 1
                    gas_metadata['QUAL'][i] = hdu[0].header['QUAL']
                    for k in gas_meta_keys:
                        if k not in hdu[meta_ext[0]].columns.names:
                            continue
                        if hdu[meta_ext[0]].data[k].size > 1 \
                                and hdu[meta_ext[0]].data[k][0].shape != gas_metadata[k][i].shape:
                            nr = hdu[meta_ext[0]].data[k][0].size
                            gas_metadata[k][i,:nr] = hdu[meta_ext[0]].data[k][0]
                        else:
                            gas_metadata[k][i] = hdu[meta_ext[0]].data[k][0]
        else:
            # Toggle flag
            gas_metadata['QUAL'][i] \
                    = gas_disk.gbm.turn_on(gas_disk.gbm.minimum_dtype()(0), 'NOMODEL')

        # Add the stellar data
        str_file = f'{nirvana_root}-{galmeta.plateifu}.fits.gz' if args.asymdrift \
                        else f'{nirvana_root}-{galmeta.plateifu}-Stars.fits.gz'
        str_file = oroot / str(galmeta.plate) / str_file
        if str_file.exists():
            with fits.open(str_file) as hdu:
                # Confirm the output has the expected model parameterization
                if hdu[0].header[md_keys[1]] != 'AxisymmetricDisk' \
                        or hdu[0].header[rc_keys[1]] != rc_class[1] \
                        or dc_class[1] is not None and hdu[0].header[dc_keys[1]] != dc_class[1] \
                        or dc_class[1] is None and dc_keys[1] in hdu[0].header:
                    warnings.warn(f'{str_file} is not an AxisymmetricDisk fit with the same model '
                                  f'parameterization as determined by the template file, '
                                  f'{ex_files[1]}.  Skipping this file.')
                    # Toggle flag
                    str_metadata['QUAL'][i] \
                            = str_disk.gbm.turn_on(hdu[0].header['QUAL'], 'DIFFMOD')
                else:
                    str_metadata['FINISHED'][i] = 1
                    str_metadata['QUAL'][i] = hdu[0].header['QUAL']
                    for k in str_meta_keys:
                        if k not in hdu[meta_ext[0]].columns.names:
                            continue
                        if hdu[meta_ext[1]].data[k].size > 1 \
                                and hdu[meta_ext[1]].data[k][0].shape != str_metadata[k][i].shape:
                            nr = hdu[meta_ext[1]].data[k][0].size
                            str_metadata[k][i,:nr] = hdu[meta_ext[1]].data[k][0]
                        else:
                            str_metadata[k][i] = hdu[meta_ext[1]].data[k][0]
        else:
            # Toggle flag
            str_metadata['QUAL'][i] \
                    = str_disk.gbm.turn_on(str_disk.gbm.minimum_dtype()(0), 'NOMODEL')

        if not args.asymdrift:
            continue

        # Add the AD data
        ad_file = oroot / str(galmeta.plate) / f'{nirvana_root}-{galmeta.plateifu}.fits.gz' 
        if ad_file.exists():
            with fits.open(ad_file) as hdu:
                # TODO: Confirm the output has the expected model parameterization?
                ad_metadata['FINISHED'][i] = 1
                ad_metadata['QUAL'][i] = hdu[0].header['QUAL']
                for k in ad_meta_keys:
                    if k not in hdu['ADPROF'].columns.names:
                        continue
                    if hdu['ADPROF'].data[k].size > 1 \
                            and hdu['ADPROF'].data[k][0].shape != ad_metadata[k][i].shape:
                        nr = hdu['ADPROF'].data[k][0].size
                        ad_metadata[k][i,:nr] = hdu['ADPROF'].data[k][0]
                    else:
                        ad_metadata[k][i] = hdu['ADPROF'].data[k][0]
        else:
            # Toggle flag
            # WARNING: Create multidisk for this?  For now, these masks are the
            # same, so it doesn't matter.
            ad_metadata['QUAL'][i] \
                    = str_disk.gbm.turn_on(str_disk.gbm.minimum_dtype()(0), 'NOMODEL')

    print(f'Collating {indx.size}/{indx.size}')

    # TODO: Add more to the headers?
    hdr = fits.Header()
    hdr['MODELTYP'] = 'AxisymmetricDisk'
    gas_hdr = hdr.copy()
    gas_hdr[rc_keys[0]] = rc_class[0]
    if dc_class[0] is not None:
        gas_hdr[dc_keys[0]] = dc_class[0]
    str_hdr = hdr.copy()
    str_hdr[rc_keys[1]] = rc_class[1]
    if dc_class[1] is not None:
        str_hdr[dc_keys[1]] = dc_class[1]

    hdus = [fits.PrimaryHDU(header=hdr),
            fits.BinTableHDU.from_columns(
                        [fits.Column(name=n, format=fileio.rec_to_fits_type(gas_metadata[n]),
                                     dim=fileio.rec_to_fits_col_dim(gas_metadata[n]),
                                     array=gas_metadata[n]) for n in gas_metadata.dtype.names],
                            name='GAS', header=gas_hdr),
            fits.BinTableHDU.from_columns(
                        [fits.Column(name=n, format=fileio.rec_to_fits_type(str_metadata[n]),
                                     dim=fileio.rec_to_fits_col_dim(str_metadata[n]),
                                     array=str_metadata[n]) for n in str_metadata.dtype.names],
                            name='STARS', header=str_hdr)]
    if args.asymdrift:
        hdus += [fits.BinTableHDU.from_columns(
                            [fits.Column(name=n, format=fileio.rec_to_fits_type(ad_metadata[n]),
                                     dim=fileio.rec_to_fits_col_dim(ad_metadata[n]),
                                     array=ad_metadata[n]) for n in ad_metadata.dtype.names],
                                name='AD')]
    fits.HDUList(hdus).writeto(_ofile, overwrite=True, checksum=True)
    if compress:
        # Compress the file
        fileio.compress_file(str(_ofile), overwrite=True, rm_original=True)



