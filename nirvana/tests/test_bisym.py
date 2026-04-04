
from IPython import embed

import numpy as np

from nirvana.data import manga
from nirvana.tests.util import remote_data_file, requires_remote
from nirvana.models.bisym import BisymmetricDisk
from nirvana.models.beam import gauss2d_kernel


def test_disk():
    disk = BisymmetricDisk()
    disk.par[:2] = 0.           # Ensure that the center is at 0,0

    disk.par[2] = 45.       # Position angle
    disk.par[3] = 60.       # Inclination
    disk.par[4] = 0.        # Systemic velocity
    disk.par[5] = 75.       # Position angle of the bisymmetric feature
                            # relative to the major axis position angle

    disk.par[6] = 150.      # *Projected* asymptotic rotation speed
    disk.par[7] = 5.        # Rotation scale
    disk.par[8] = 50.       # *Projected* 2nd order tangential amplitude
    disk.par[9] = 5.       # 2nd order tangential scale
    disk.par[10] = 2.       # 2nd order tangential power
    disk.par[11] =  0.      # *Projected* 2nd order tangential amplitude
    disk.par[12] = 5.      # 2nd order tangential scale
    disk.par[13] = 2.       # 2nd order tangential power

    n = 71
    x = (np.arange(n) - n//2).astype(float)[::-1]
    y = (np.arange(n) - n//2).astype(float)
    x, y = np.meshgrid(x, y)

    vel = disk.model(x=x, y=y)
    beam = gauss2d_kernel(n, 3.)
    _vel = disk.model(x=x, y=y, beam=beam)

    assert np.isclose(vel[n//2,n//2], _vel[n//2,n//2]), 'Smearing moved the center.'


#@requires_remote
#def test_lsq_nopsf():
#
#    # Read the data to fit
#    data_root = remote_data_file()
#    kin = manga.MaNGAGasKinematics.from_plateifu(8078, 12703, cube_path=data_root,
#                                                 maps_path=data_root, ignore_psf=True)
#
#    # Set the disk velocity field
#    bisym_disk = BisymmetricDisk()
#    # Fit it with a non-linear least-squares optimizer
#    bisym_disk.lsq_fit(kin, verbose=2, analytic_jac=False)
#
#test_lsq_nopsf()
#


