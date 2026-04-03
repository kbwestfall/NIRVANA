from IPython import embed

import numpy as np

from nirvana.models import oned


#TODO: Should abstract these to a generic code that tests the derivatives.


def test_step():
    n = 10
    edges = np.arange(n, dtype=float)+1
    rng = np.random.default_rng()
    steps = rng.uniform(low=0., high=n+2., size=n)

    f = oned.StepFunction(edges, par=steps)

    # Sample
    x = np.arange(n+2, dtype=float)+0.5
    y = f.sample(x)

    assert np.allclose(np.concatenate(([steps[0]], steps, [steps[-1]])), y), \
            'Evaluation is bad'

    # Check that the sampling coordinate doesn't need to be sorted on input
    _x = x.copy()
    rng.shuffle(_x)
    _y = f.sample(_x)
    srt = np.argsort(_x)

    assert np.array_equal(_y[srt], y), 'sorting of input coordinates should not matter'


def test_step_ddx():
    n = 10
    edges = np.arange(n, dtype=float)+1
    rng = np.random.default_rng()
    steps = rng.uniform(low=0., high=n+2., size=n)

    f = oned.StepFunction(edges, par=steps)

    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 0.0001, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_step_deriv():
    n = 10
    edges = np.arange(n, dtype=float)+1
    rng = np.random.default_rng()
    steps = rng.uniform(low=0., high=n+2., size=n)

    f = oned.StepFunction(edges, par=steps)

    x = np.arange(n+2, dtype=float)+0.5
    y = f.sample(x)
    _y, dy = f.deriv_sample(x)

    assert np.array_equal(y, _y), 'sample and deriv_sample give different function values'
    _dy = np.zeros_like(dy)
    _dy[0,0] = 1.
    _dy[-1,-1] = 1.
    _dy[1:-1] = np.identity(n)
    assert np.array_equal(dy, _dy), 'bad derivative'

    _x = x.copy()
    rng.shuffle(_x)
    _y, _dy = f.deriv_sample(_x)
    srt = np.argsort(_x)

    assert np.array_equal(y, _y[srt]), 'sorting of the input coordinates should not matter'
    assert np.array_equal(dy, _dy[srt]), 'sorting of the input coordinates should not matter'


def test_step_2d():
    n = 10
    edges = np.arange(n, dtype=float)

    vrot = 200.
    hrot = 3.
    steps = vrot*np.tanh(edges/hrot)

    f = oned.StepFunction(edges, par=steps)

    shape = (31,31)
    x = np.arange(shape[1]).astype(float) - shape[1]//2
    y = np.arange(shape[0]).astype(float) - shape[0]//2
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    v = f.sample(r)
    assert v.shape == r.shape, 'shape changed'

    _v, _dv = f.deriv_sample(r)
    assert np.array_equal(v, _v), 'sample and deriv_sample sample differently'
    assert _dv.ndim == 3, 'deriv array has wrong dimensionality'

    ddr = f.ddx(r)
    assert ddr.shape == r.shape, 'ddr shape changed'

    d2dr2 = f.d2dx2(r)
    assert d2dr2.shape == r.shape, 'd2dr2 shape changed'


def test_lin():
    n = 10
    edges = np.arange(n, dtype=float)+1
    rng = np.random.default_rng()
    anchors = rng.uniform(low=0., high=n+2., size=n)

    f = oned.PiecewiseLinear(edges, par=anchors)

    x = np.arange(n+1, dtype=float)+0.5
    y = f.sample(x)

    assert np.allclose(np.concatenate(([anchors[0]], (anchors[1:] + anchors[:-1])/2,
                                             [anchors[-1]])), y), 'Evaluation is bad'

    # Check that the sampling coordinate doesn't need to be sorted on input
    _x = x.copy()
    rng.shuffle(_x)
    _y = f.sample(_x)
    srt = np.argsort(_x)

    assert np.array_equal(_y[srt], y), 'sorting of input coordinates should not matter'


def test_lin_2d():
    n = 10
    edges = np.arange(n, dtype=float)

    vrot = 200.
    hrot = 3.
    anchors = vrot*np.tanh(edges/hrot)

    f = oned.PiecewiseLinear(edges, par=anchors)

    shape = (31,31)
    x = np.arange(shape[1]).astype(float) - shape[1]//2
    y = np.arange(shape[0]).astype(float) - shape[0]//2
    X, Y = np.meshgrid(x, y)

    r = np.sqrt(X**2 + Y**2)
    v = f.sample(r)
    assert v.shape == r.shape, 'shape changed'

    _v, _dv = f.deriv_sample(r)
    assert np.array_equal(v, _v), 'sample and deriv_sample sample differently'
    assert _dv.ndim == 3, 'deriv array has wrong dimensionality'

    ddr = f.ddx(r)
    assert ddr.shape == r.shape, 'ddr shape changed'

    d2dr2 = f.d2dx2(r)
    assert d2dr2.shape == r.shape, 'd2dr2 shape changed'


def test_lin_ddx():
    n = 10
    edges = np.arange(n, dtype=float)+1
    rng = np.random.default_rng()
    anchors = rng.uniform(low=0., high=n+2., size=n)

    f = oned.PiecewiseLinear(edges, par=anchors)

    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 0.0001, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_lin_deriv():
    n = 10
    edges = np.arange(n, dtype=float)+1
    rng = np.random.default_rng()
    anchors = rng.uniform(low=0., high=n+2., size=n)

    f = oned.PiecewiseLinear(edges, par=anchors)

    x = np.arange(n+1, dtype=float)+0.5
    y = f.sample(x)
    _y, dy = f.deriv_sample(x)

    assert np.array_equal(y, _y), 'sample and deriv_sample give different function values'
    _dy = np.zeros_like(dy)
    _dy[0,0] = 1.
    _dy[1:] = np.diag(np.full(n-1, 0.5), 1) + np.diag(np.full(n, 0.5))
    _dy[-1,-1] = 1.
    assert np.array_equal(dy, _dy), 'bad derivative'
    assert np.array_equal(y, _y), 'sample and deriv_sample give different function values'

    _x = x.copy()
    rng.shuffle(_x)
    _y, _dy = f.deriv_sample(_x)
    srt = np.argsort(_x)

    assert np.array_equal(y, _y[srt]), 'sorting of the input coordinates should not matter'
    assert np.array_equal(dy, _dy[srt]), 'sorting of the input coordinates should not matter'


def test_tanh():
    f = oned.HyperbolicTangent(par=[1.,1.])
    y = f.sample([1.])

    assert np.isclose(y[0], np.tanh(1.)), 'Function changed.'


def test_tanh_ddx():
    f = oned.HyperbolicTangent(par=[1.,1.])
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 0.0001, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_tanh_deriv():
    par = np.array([1., 1.])
    dp = np.array([0.0001, 0.0001])
    f = oned.HyperbolicTangent(par=par)
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = np.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_plex():
    f = oned.PolyEx(par=[1.,1.,0.1])
    y = f.sample([1.])

    assert np.isclose(y[0], 1.1*(1-np.exp(-1.))), 'Function changed.'


def test_plex_ddx():
    f = oned.PolyEx(par=[1.,1.,0.1])
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 0.0001, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_plex_deriv():
    par = np.array([1., 1., 0.1])
    dp = np.array([0.0001, 0.0001, 0.0001])
    f = oned.PolyEx(par=par)
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = np.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_exp():
    f = oned.Exponential(par=[1.,1.])
    y = f.sample([1.])

    assert np.isclose(y[0], np.exp(-1.)), 'Function changed.'


def test_exp_ddx():
    f = oned.Exponential(par=[1.,1.])
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 0.0001, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_exp_deriv():
    par = np.array([1., 1.])
    dp = np.array([0.0001, 0.0001])
    f = oned.Exponential(par=par)
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = np.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_expbase():
    f = oned.ExpBase(par=[1.,1.,1.])
    y = f.sample([1.])

    assert np.isclose(y[0], np.exp(-1.)+1), 'Function changed.'


def test_expbase_ddx():
    f = oned.ExpBase(par=[1.,1.,1.])
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 0.0001, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_expbase_deriv():
    par = np.array([1., 1., 1.])
    dp = np.array([0.0001, 0.0001, 0.001])
    f = oned.ExpBase(par=par)
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = np.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_powexp():
    f = oned.PowerExp(par=[1.,1.,1.])
    y = f.sample([1.])
    assert np.isclose(y[0], 1.), 'Function changed.'


def test_powexp_ddx():
    f = oned.PowerExp(par=[1.,1.,1.])
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 0.00001, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_powexp_deriv():
    par = np.array([1., 1., 1.])
    dp = np.array([0.0001, 0.0001, 0.0001])
    f = oned.PowerExp(par=par)
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = np.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_powlaw():
    f = oned.PowerLaw(par=[1.,2.])
    y = f.sample([1.])
    assert np.isclose(y[0], 1.), 'Function changed.'


def test_powlaw_ddx():
    f = oned.PowerLaw(par=[1.,2.])
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 1e-10, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-5), 'Derivatives are wrong'


def test_powlaw_deriv():
    par = np.array([1., 2.])
    dp = np.array([0.0001, 0.0001])
    f = oned.PowerLaw(par=par)
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = np.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_crc():
    f = oned.ConcentratedRotationCurve(par=[1.,1.,2.,0.1])
    y = f.sample([1.])
    assert np.isclose(y[0], 2.**0.1 * 2**(-1/2.)), 'Function changed.'


def test_crc_ddx():
    f = oned.ConcentratedRotationCurve(par=[1.,1.,2.,0.1])
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    dx = np.full(n, 0.0001, dtype=float)
    y = f.sample(x)
    dy = f.ddx(x)
    fd_dy = (f.sample(x+dx) - y)/dx
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


def test_crc_deriv():
    par = np.array([1.,1.,2.,0.1])
    dp = np.array([0.0001, 0.0001, 0.0001, 0.0001])
    f = oned.ConcentratedRotationCurve(par=par)
    rng = np.random.default_rng()
    n = 10
    x = rng.uniform(low=0., high=2., size=n)
    y, dy = f.deriv_sample(x)
    yp = np.zeros((x.size, par.size), dtype=float)
    for i in range(par.size):
        _p = par.copy()
        _p[i] += dp[i]
        yp[...,i] = f.sample(x, par=_p)

    fd_dy = (yp - y[...,None])/dp[None,:]
    assert np.allclose(dy, fd_dy, rtol=0., atol=1e-4), 'Derivatives are wrong'


