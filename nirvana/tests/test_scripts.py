
import os
import shutil

from IPython import embed

from astropy.io import fits

from nirvana.scripts import manga_axisym, manga_asymdrift
from nirvana.tests.util import remote_data_file, requires_remote


@requires_remote
def test_manga_axisym():
    odir = remote_data_file('tests/8138')
    if os.path.isdir(odir):
        shutil.rmtree(odir)

    args = manga_axisym.parse_args(['8138', '12704', '--root', remote_data_file(), '--odir', odir,
                                    '-t', 'Gas', '--min_vel_snr', '5', '--min_sig_snr', '5',
                                    '--max_vel_err', '100', '--max_sig_err', '100',
                                    '--min_unmasked', '10', '--coherent', '--skip_plots'])
    manga_axisym.main(args)

#    args = manga_axisym.parse_args(['8138', '12704', '--root', remote_data_file(), '--odir', odir,
#                                    '-t', 'Gas', '--min_vel_snr', '5', '--min_sig_snr', '5',
#                                    '--max_vel_err', '100', '--max_sig_err', '100',
#                                    '--min_unmasked', '10', '--coherent', '--covar', '--skip_plots'])
#    manga_axisym.main(args)
#    embed()
#    exit()

    main_output_file = os.path.join(odir, 'nirvana-manga-axisym-8138-12704-Gas.fits.gz')
    assert os.path.isfile(main_output_file), 'Output file not created.'

    with fits.open(main_output_file) as hdu:
        assert len(hdu) == 24, 'Data model changed'
        assert hdu['FITMETA'].data['RCHI2'] < 1.1, 'Fit dramatically changed'

    shutil.rmtree(odir)


@requires_remote
def test_manga_asymdrift():
    odir = remote_data_file('tests/8138')
    if os.path.isdir(odir):
        shutil.rmtree(odir)

    args = manga_asymdrift.parse_args(['8138', '12704', '--root', remote_data_file(),
                                       '--odir', odir,
                                       '--gas_min_vel_snr', '5', '--gas_min_sig_snr', '5',
                                       '--gas_max_vel_err', '100', '--gas_max_sig_err', '100',
                                       '--str_min_vel_snr', '5', '--str_min_sig_snr', '10',
                                       '--str_max_vel_err', '100', '--str_max_sig_err', '100',
                                       '--min_unmasked', '10', '--coherent', '--skip_plots'])
                                       #]) #, '--verbose', '2'])
    manga_asymdrift.main(args)

    main_output_file = os.path.join(odir, 'nirvana-manga-asymdrift-8138-12704.fits.gz')
    assert os.path.isfile(main_output_file), 'Output file not created.'

    with fits.open(main_output_file) as hdu:
        assert len(hdu) == 66, 'Data model changed'
        assert hdu['GAS_FITMETA'].data['RCHI2'] < 1.1, 'Fit dramatically changed'
        assert hdu['STR_FITMETA'].data['RCHI2'] < 1.1, 'Fit dramatically changed'
        assert hdu[0].header['RCHI2'] < 1.1, 'Fit dramatically changed'

    shutil.rmtree(odir)


