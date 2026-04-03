"""
Module for testing the geometry module.
"""

from IPython import embed

import numpy as np

from nirvana.models import geometry


def test_disk_ellipse():
    pa = np.radians(90.)
    inc = np.radians(90.)
    x, y = geometry.disk_ellipse(10., pa, inc)

    assert np.all((x >=-10) & (x <= 10.)), 'Bad radius'
    assert np.allclose(y, np.zeros(y.size)), 'Should be a line.'

    pa = np.radians(0.)
    x, y = geometry.disk_ellipse(10., pa, inc)
    assert np.allclose(x, np.zeros(x.size)), 'Should be a line.'

    pa = np.radians(45.)
    inc = np.radians(45.)
    x, y = geometry.disk_ellipse(10., pa, inc, xc=2, yc=-3, num=10000)
    assert np.isclose(np.mean(x), 2., rtol=1e-3), 'Bad center'
    assert np.isclose(np.mean(y), -3., rtol=1e-3), 'Bad center'


def test_polar():

    n = 51
    x = np.arange(n, dtype=float)[::-1] - n//2
    y = np.arange(n, dtype=float) - n//2
    x, y = np.meshgrid(x, y)

    # Parameters are xc, yc, rotation, pa, inclination
    par = [10., 10., 10., 45., 30.]
    if par[2] > 0:
        xf, yf = map(lambda x,y : x - y,
                     geometry.rotate(x, y, np.radians(par[2]), clockwise=True), par[:2])
    else:
        xf, yf = x - par[0], y - par[1]

    r, th = geometry.projected_polar(xf, yf, *np.radians(par[3:]))

    indx = np.unravel_index(np.argmin(r), r.shape)
    assert indx[0] > n//2 and indx[1] < n//2, 'Offset in wrong direction.'

    assert np.amax(r) > n/2, 'Inclination should result in larger radius'
    assert not np.any(th < 0) and not np.any(th > 2*np.pi), 'Theta has wrong range'


def test_deriv_rotate():

    x = np.array([1., 0., -1., 0.])
    y = np.array([0., 1., 0., -1.])

    rot = np.pi/3.
    drot = 1e-3
    xrp, yrp = geometry.rotate(x, y, rot+drot)
    xrn, yrn = geometry.rotate(x, y, rot)
    xr, yr, dxr, dyr = geometry.deriv_rotate(x, y, rot)
    assert np.allclose(dxr[:,0], (xrp-xrn)/drot, rtol=1e-3, atol=0.), \
            'Deriv does not match finite difference'
    assert np.allclose(dyr[:,0], (yrp-yrn)/drot, rtol=1e-3, atol=0.), \
            'Deriv does not match finite difference'

    # Check clockwise rotation
    rot = np.pi/3.
    drot = 1e-3
    xrp, yrp = geometry.rotate(x, y, rot+drot, clockwise=True)
    xrn, yrn = geometry.rotate(x, y, rot, clockwise=True)
    xr, yr, dxr, dyr = geometry.deriv_rotate(x, y, rot, clockwise=True)
    assert np.allclose(dxr[:,0], (xrp-xrn)/drot, rtol=1e-3, atol=0.), \
            'Deriv does not match finite difference'
    assert np.allclose(dyr[:,0], (yrp-yrn)/drot, rtol=1e-3, atol=0.), \
            'Deriv does not match finite difference'

    # Check defining rotation in degrees
    rot = 30.
    drot = 0.1
    dxdp = np.zeros(x.shape+(1,), dtype=float)
    dydp = np.zeros(y.shape+(1,), dtype=float)
    drotdp = np.atleast_1d(np.radians(1))
    xr, yr, dxr, dyr \
            = geometry.deriv_rotate(x, y, np.radians(rot), dxdp=dxdp, dydp=dydp, drotdp=drotdp)
    xrp, yrp = geometry.rotate(x, y, np.radians(rot+drot))
    xrn, yrn = geometry.rotate(x, y, np.radians(rot))
    assert np.allclose(dxr[:,0], (xrp-xrn)/drot, rtol=1e-2, atol=0.), \
            'X derivative does not match finite difference'
    assert np.allclose(dyr[:,0], (yrp-yrn)/drot, rtol=1e-2, atol=0.), \
            'Y derivative does not match finite difference'


def test_deriv_projected_polar():

    # NOTE: A point at x = 1. and y = 0. will cause this to fail because the
    # brute force calculation of the derivative of theta results in a wrap
    # degeneracy.
#    x = np.array([2., 0., -2., 0., 1., 0., -1., 0.])
#    y = np.array([0., 2., 0., -2., 0., 1., 0., -1.])

    x = np.array([2., 0., -2., 0., 0., -1., 0.])
    y = np.array([0., 2., 0., -2., 1., 0., -1.])

    # Parameter vector is x0, y0, position angle, inclination.  x0 and y0 have
    # the same units as x and y; pa and inclination are in degrees.
    p = np.array([0.5, -0.5, 45., 30.])
    dp = np.array([0.0001, 0.0001, 0.001, 0.001])

    _x = x - p[0]
    _y = y - p[1]
    _pa = np.radians(p[2])
    _inc = np.radians(p[3])

    _dx = np.tile(np.array([-1., 0., 0., 0.]), (x.size, 1))
    _dy = np.tile(np.array([0., -1., 0., 0.]), (x.size, 1))
    _dpa = np.array([0., 0., np.radians(1.), 0.])
    _dinc = np.array([0., 0., 0., np.radians(1.)])

    r, t, dr, dt = geometry.deriv_projected_polar(_x, _y, _pa, _inc, dxdp=_dx, dydp=_dy,
                                                  dpadp=_dpa, dincdp=_dinc)

    # Finite difference approach
    rp = np.zeros((x.size, p.size), dtype=float)
    tp = np.zeros((x.size, p.size), dtype=float)
    for i in range(p.size):
        _p = p.copy()
        _p[i] += dp[i]
        rp[:,i], tp[:,i] = geometry.projected_polar(x - _p[0], y - _p[1], np.radians(_p[2]),
                                                    np.radians(_p[3]))

    assert np.allclose(dr, (rp-r[:,None])/dp[None,:], rtol=0., atol=1e-4), \
        'Radius derivative does not match finite difference'
    assert np.allclose(dt, (tp-t[:,None])/dp[None,:], rtol=0., atol=1e-3), \
        'Azimuth derivative does not match finite difference'


def test_hexagon():
    v = geometry.hexagon_vertices()

    # NOTE: This requires too much numerical precision.  Think of a better way
    # to test this...
#    _v = geometry.hexagon_vertices(incircle=True, orientation='vertical')
#    assert not np.any(geometry.point_inside_polygon(_v, v)), \
#                'Rotation should lead to all vertices of v being on the _v polygon.'

    _v = geometry.hexagon_vertices(incircle=True)

    assert np.all(geometry.point_inside_polygon(_v, v)), \
            'Second hexagon should be larger than the first.'


