
from IPython import embed

import numpy

from matplotlib import pyplot

from scipy import stats, special
from nirvana.data import manga
from nirvana.data import util
from nirvana.data import scatter
from nirvana.tests.util import remote_data_file, requires_remote
from nirvana.models.oned import HyperbolicTangent, Exponential
from nirvana.models.axisym import AxisymmetricDisk
from nirvana.models.bisym import BisymmetricDisk
from nirvana.models.beam import gauss2d_kernel, ConvolveFFTW
from nirvana.models.multitrace import MultiTracerDisk


@requires_remote
def test_lsq_nopsf():

    stellar_disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
    gas_disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())

    # This ties the center coordinates and the inclination, but leaves the
    # position angle and the systemic velocity to be free
    disk = MultiTracerDisk([gas_disk, stellar_disk], tie_base=[True, True, False, True, False])
                           #, tie_disk=[True, False, False, False])

    p0 = disk.guess_par()
    fix = numpy.zeros(p0.size, dtype=bool)
    # Fix the center
    fix[0:2] = True

    # Read the data to fit
    data_root = remote_data_file()
    stellar_kin = manga.MaNGAStellarKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root)

    gas_kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                     maps_path=data_root)

    disk.lsq_fit([gas_kin, stellar_kin], sb_wgt=True, p0=p0, fix=fix, verbose=2)


