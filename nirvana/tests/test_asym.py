"""
Module for testing the geometry module.
"""
import os

from IPython import embed

import numpy

from nirvana.models import geometry
from nirvana.models import asymmetry
from nirvana.models import axisym
from nirvana.models import bisym
from nirvana.models import oned
from nirvana.data.util import select_major_axis, inverse
from nirvana.data import manga
from nirvana.tests.util import remote_data_file, requires_remote

def test_no_reflection():
    x, y = numpy.meshgrid(numpy.arange(10), numpy.arange(10))
    d, i = asymmetry.symmetric_reflection_map(x.ravel(), y.ravel(), reflect=None)
    assert numpy.array_equal(i, numpy.arange(x.size)), \
            'No reflection should just return the input indices.'


def test_symmetric_reflection_map():

    x, y = numpy.meshgrid(numpy.arange(10), numpy.arange(10))
    d, i = asymmetry.symmetric_reflection_map(x.ravel(), y.ravel())

    assert d.shape[0] == i.shape[0] and d.shape[0] == 4, 'Should produce 4 reflections'

    assert numpy.array_equal(d[0], numpy.zeros(x.size, dtype=float)), \
            'First set should not be reflected, so pixel map should be direct.  Bad distances.'
    assert numpy.array_equal(i[0], numpy.arange(x.size)), \
            'First set should not be reflected, so pixel map should be direct.  Bad indices.'


def test_axisymmetric_symmetry():

    ifusize = 22
    pixelscale = 0.5
    width_buffer = 10
    n = int(numpy.floor(ifusize/pixelscale)) + width_buffer
    if n % 2 != 0:
        n += 1
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(pixelscale*x, pixelscale*y)
    ifu_mask = geometry.point_inside_polygon(geometry.hexagon_vertices(d=ifusize),
                                             numpy.column_stack((x.ravel(), y.ravel())))
    ifu_mask = numpy.logical_not(ifu_mask).reshape(x.shape)    

    disk = axisym.AxisymmetricDisk(rc=oned.HyperbolicTangent(), dc=oned.Exponential())
    
    disk.par[:2] = 0.0
    disk.par[2] = 0.
    disk.par[3] = 70.
    disk.par[-3] = 2.

    v, sig = disk.model(x=x, y=y)
    d, i = asymmetry.symmetric_reflection_map(x.ravel()-disk.par[0], y.ravel()-disk.par[1],
                                              rotation=-disk.par[2])
    # Reflection order is None, x, y, xy
    maxd = 0.4
    v_x = numpy.ma.MaskedArray(v.ravel()[i[1]], mask=d[1] > maxd).reshape(v.shape)
    v_y = numpy.ma.MaskedArray(-v.ravel()[i[2]], mask=d[2] > maxd).reshape(v.shape)
    v_xy = numpy.ma.MaskedArray(-v.ravel()[i[3]], mask=d[3] > maxd).reshape(v.shape)

    # All reflections should be symmetric
    assert numpy.ma.allclose(v, v_x), 'Bad x (minor axis) symmetry.'
    assert numpy.ma.allclose(v, v_y), 'Bad y (major axis) symmetry.'
    assert numpy.ma.allclose(v, v_xy), 'Bad 4-point symmetry'


def test_bisymmetric_symmetry():

    ifusize = 22
    pixelscale = 0.5
    width_buffer = 10
    n = int(numpy.floor(ifusize/pixelscale)) + width_buffer
    if n % 2 != 0:
        n += 1
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(pixelscale*x, pixelscale*y)
    ifu_mask = geometry.point_inside_polygon(geometry.hexagon_vertices(d=ifusize),
                                             numpy.column_stack((x.ravel(), y.ravel())))
    ifu_mask = numpy.logical_not(ifu_mask).reshape(x.shape)    

    disk = bisym.BisymmetricDisk(vt=oned.HyperbolicTangent(), dc=oned.Exponential())
    disk.par[:2] = 0.           # Ensure that the center is at 0,0

    disk.par[2] = 20.       # Position angle
    disk.par[3] = 40.       # Inclination
    disk.par[4] = 0.        # Systemic velocity
    disk.par[5] = 45.       # Position angle of the bisymmetric feature
                            # relative to the major axis position angle

    disk.par[6] = 150.      # *Projected* asymptotic rotation speed
    disk.par[7] = 2.        # Rotation scale
    disk.par[8] = 50.       # *Projected* 2nd-order tangential amplitude
    disk.par[9] = 2.        # 2nd-order tangential scale
    disk.par[10] = 1.       # 2nd-order tangential power
    disk.par[11] = 50.      # *Projected* 2nd-order radial amplitude
    disk.par[12] = 2.       # 2nd-order radial scale
    disk.par[13] = 1.       # 2nd-order radial power

    v, sig = disk.model(x=x, y=y)
    d, i = asymmetry.symmetric_reflection_map(x.ravel()-disk.par[0], y.ravel()-disk.par[1],
                                              rotation=-disk.par[2])
    # Reflection order is None, x, y, xy
    maxd = 0.4
    v_x = numpy.ma.MaskedArray(v.ravel()[i[1]], mask=d[1] > maxd).reshape(v.shape)
    v_y = numpy.ma.MaskedArray(-v.ravel()[i[2]], mask=d[2] > maxd).reshape(v.shape)
    v_xy = numpy.ma.MaskedArray(-v.ravel()[i[3]], mask=d[3] > maxd).reshape(v.shape)

    # Axisymmetric reflections should not be symmetric
    assert not numpy.ma.allclose(v, v_x), 'Bad x (minor axis) symmetry.'
    assert not numpy.ma.allclose(v, v_y), 'Bad y (major axis) symmetry.'
    # But 180 degree rotation *should* be.
    assert numpy.ma.allclose(v, v_xy), 'Bad 4-point symmetry'


def test_bisymmetric_asymmetry_maps():

    ifusize = 22
    pixelscale = 0.5
    width_buffer = 10
    n = int(numpy.floor(ifusize/pixelscale)) + width_buffer
    if n % 2 != 0:
        n += 1
    x = numpy.arange(n, dtype=float)[::-1] - n//2
    y = numpy.arange(n, dtype=float) - n//2
    x, y = numpy.meshgrid(pixelscale*x, pixelscale*y)
    ifu_mask = geometry.point_inside_polygon(geometry.hexagon_vertices(d=ifusize),
                                             numpy.column_stack((x.ravel(), y.ravel())))
    ifu_mask = numpy.logical_not(ifu_mask).reshape(x.shape)    

    disk = bisym.BisymmetricDisk(vt=oned.HyperbolicTangent(), dc=oned.Exponential())
    disk.par[:2] = 0.           # Ensure that the center is at 0,0

    disk.par[2] = 20.       # Position angle
    disk.par[3] = 40.       # Inclination
    disk.par[4] = 0.        # Systemic velocity
    disk.par[5] = 45.       # Position angle of the bisymmetric feature
                            # relative to the major axis position angle

    disk.par[6] = 150.      # *Projected* asymptotic rotation speed
    disk.par[7] = 2.        # Rotation scale
    disk.par[8] = 50.       # *Projected* 2nd-order tangential amplitude
    disk.par[9] = 2.        # 2nd-order tangential scale
    disk.par[10] = 1.       # 2nd-order tangential power
    disk.par[11] = 50.      # *Projected* 2nd-order radial amplitude
    disk.par[12] = 2.       # 2nd-order radial scale
    disk.par[13] = 1.       # 2nd-order radial power

    v, sig = disk.model(x=x, y=y)
#    pa, inc = numpy.radians(disk.par[2:4])
#    r, th = geometry.projected_polar(x - disk.par[0], y - disk.par[1], pa, inc)
#    wedge = 30.
#    major_gpm = select_major_axis(r, th, r_range='all', wedge=wedge)
#    maxr = numpy.max(r[major_gpm])
#
#    _r = numpy.linspace(0, maxr, num=100)
#    _vt_r = disk.vt.sample(_r, par=disk.par[disk._vt_slice()])
#    _v2t_r = disk.v2t.sample(_r, par=disk.par[disk._v2t_slice()])
#    _v2r_r = disk.v2r.sample(_r, par=disk.par[disk._v2r_slice()])
#
#    _th = numpy.linspace(-numpy.pi, numpy.pi, num=100)
#    _pab = (numpy.radians(disk.par[5]) + numpy.pi/2) % numpy.pi - numpy.pi/2
#    _th_b = _th - numpy.arctan(numpy.tan(_pab)/numpy.cos(inc))
#    # Use the radius at which the 2nd-order terms are maximized.
#    _amp_r = max(disk.par[9]*disk.par[10], disk.par[12]*disk.par[13])
#    _amp_vt = disk.vt.sample(_amp_r, par=disk.par[disk._vt_slice()])
#    _amp_v2t = disk.v2t.sample(_amp_r, par=disk.par[disk._v2t_slice()])
#    _amp_v2r = disk.v2r.sample(_amp_r, par=disk.par[disk._v2r_slice()])
#    _vt_t = _amp_vt * numpy.cos(_th)
#    _v2t_t = _amp_v2t * numpy.cos(_th) * numpy.cos(2*_th_b)
#    _v2r_t = _amp_v2r * numpy.sin(_th) * numpy.sin(2*_th_b)
#
#    pyplot.plot(_th, _vt_t, linestyle='--')
#    pyplot.plot(_th, -_v2t_t, linestyle='--')
#    pyplot.plot(_th, -_v2r_t, linestyle='--')
#    pyplot.plot(_th, -_v2t_t-_v2r_t, linestyle='--')
#    pyplot.plot(_th, _vt_t - _v2t_t - _v2r_t, linestyle='-', color='k')
#    pyplot.show()

    v_x, v_y, v_xy = asymmetry.onsky_asymmetry_maps(x-disk.par[0], y-disk.par[1], v,
                                                    pa=disk.par[2], odd=True, maxd=0.4)

    # 180 deg symmetry should be 0 for unmasked pixels.
    assert numpy.isclose(numpy.ma.std(v_xy), 0.), 'Bad 180 deg symmetry value'

#    from matplotlib import pyplot
#    w,h = pyplot.figaspect(1)
#    fig = pyplot.figure(figsize=(2.0*w,2.0*h))
#    ax = fig.add_axes([0.05, 0.5, 0.2, 0.2])
#    ax.imshow(v, origin='lower', interpolation='nearest')
#    ax = fig.add_axes([0.27, 0.5, 0.2, 0.2])
#    ax.imshow(v_x, origin='lower', interpolation='nearest')
#    ax = fig.add_axes([0.49, 0.5, 0.2, 0.2])
#    ax.imshow(v_y, origin='lower', interpolation='nearest')
#    ax = fig.add_axes([0.71, 0.5, 0.2, 0.2])
#    ax.imshow(v_xy, origin='lower', interpolation='nearest')
#    pyplot.show()
#    fig.clear()
#    pyplot.close(fig)


@requires_remote
def test_asymmetry_data():

    # Read the data to fit
    data_root = remote_data_file()
    #galmeta = manga.MaNGAGlobalPar(8078, 12703, drpall_path=data_root)
    kin = manga.MaNGAGasKinematics.from_plateifu(8078, 12703, cube_path=data_root,
                                                 maps_path=data_root) #, covar=True)

    v = kin.remap('vel')
    v_x, v_y, v_xy = asymmetry.onsky_asymmetry_maps(kin.grid_x-0.17, kin.grid_y-0.11,
                                                    v-2.35, pa=5.9, #galmeta.pa,
                                                    mask=numpy.ma.getmaskarray(v).copy(),
                                                    odd=True, maxd=0.4)

    assert numpy.ma.mean(numpy.absolute(v_x)) > numpy.ma.mean(numpy.absolute(v_xy)), \
            'Asymmetry comparison changed; v_x should be > than v_xy'
    assert numpy.ma.mean(numpy.absolute(v_y)) > numpy.ma.mean(numpy.absolute(v_xy)), \
            'Asymmetry comparison changed; v_y should be > than v_xy'


@requires_remote
def test_asymmetry_data_with_err():

    # Read the data to fit
    data_root = remote_data_file()
    #galmeta = manga.MaNGAGlobalPar(8138, 12704, drpall_path=data_root)
    kin = manga.MaNGAGasKinematics.from_plateifu(8138, 12704, cube_path=data_root,
                                                 maps_path=data_root, covar=True)

    v = kin.remap('vel')
    ivar = kin.remap('vel_ivar')
    covar = kin.remap_covar('vel_covar')
    mask = numpy.ma.getmaskarray(v) | numpy.ma.getmaskarray(ivar)

    v_x, v_y, v_xy, v_x_var, v_y_var, v_xy_var \
            = asymmetry.onsky_asymmetry_maps(kin.grid_x, kin.grid_y, v.data-27.1, pa=165.9, #galmeta.pa,
                                             ivar=ivar, mask=mask, odd=True, maxd=0.4)

    _v_x, _v_y, _v_xy, _v_x_covar, _v_y_covar, _v_xy_covar \
            = asymmetry.onsky_asymmetry_maps(kin.grid_x, kin.grid_y, v.data-27.1, pa=165.9, #galmeta.pa,
                                             covar=covar, mask=mask, odd=True, maxd=0.4)

    assert numpy.array_equal(v_x, _v_x), 'Difference in error affected asymmetry map!'
    assert numpy.sum(v_x.compressed() == 0.) == 33, 'Number of 0 pixels changed'

    _v_x_err = numpy.ma.MaskedArray(numpy.ma.sqrt(_v_x_covar.diagonal()).reshape(v.shape),
                                    mask=v_x.mask.copy())
    v_x_err = numpy.ma.sqrt(v_x_var)
    assert not numpy.array_equal(v_x_err, _v_x_err), \
            'Calculations with and without covariance should not be identical'

@requires_remote
def test_asymmetry_plot():

    # Read the data to fit
    data_root = remote_data_file()
    ofile = remote_data_file('test.png')
    if os.path.isfile(ofile):
        os.remove(ofile)
    
    plate = 8078
    ifu = 12703
    xc = 0.
    yc = 0.
    pa = 5.7
    vsys = 3.4
#    plate = 8138
#    ifu = 12704
#    xc = 0.
#    yc = 0.
#    pa = 165.9
#    vsys = 27.1

    maps_file, cube_file, image_file \
            = manga.manga_files_from_plateifu(plate, ifu, cube_path=data_root,
                                              image_path=data_root, maps_path=data_root)

    galmeta = manga.MaNGAGlobalPar(plate, ifu, drpall_path=data_root)
    kin = manga.MaNGAGasKinematics(maps_file, cube_file=cube_file, image_file=image_file)
    kin.asymmetry_plot(galmeta=galmeta, xc=xc, yc=yc, pa=pa, vsys=vsys, fwhm=galmeta.psf_fwhm[1],
                       ofile=ofile)

    os.remove(ofile)


