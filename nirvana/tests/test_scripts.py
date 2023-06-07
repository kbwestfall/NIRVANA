
import os
import shutil

from IPython import embed

from astropy.io import fits

from nirvana.scripts import manga_axisym
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

    main_output_file = os.path.join(odir, 'nirvana-manga-axisym-8138-12704-Gas.fits.gz')
    assert os.path.isfile(main_output_file), 'Output file not created.'

    with fits.open(main_output_file) as hdu:
        assert len(hdu) == 24, 'Data model changed'
        assert hdu['FITMETA'].data['RCHI2'] < 1.1, 'Fit dramatically changed'

    shutil.rmtree(odir)

