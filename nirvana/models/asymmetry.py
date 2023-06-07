"""
Functions for parameterizing asymmetry.

.. include:: ../include/links.rst
"""
import warnings

from IPython import embed

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree

from .geometry import rotate
from ..data import util

import warnings
warnings.simplefilter('error', sparse.SparseEfficiencyWarning)


def asymmetry(args, pa, vsys, xc=0, yc=0, maxd=.5):
    '''
    Calculate global asymmetry parameter and map of asymmetry.

    Using Equation 7 from Andersen & Bershady (2007), the symmetry of a galaxy
    is calculated. This is done by reflecting it across the supplied position
    angle (assumed to be the major axis) and the angle perpendicular to that
    (assumed to be the minor axis). The value calculated is essentially a
    normalized error weighted difference between the velocity measurements on
    either side of a reflecting line. 

    The function tries to find the nearest neighbor to a spaxel's reflected
    coordinates. If the distance is too far, the spaxel is masked.

    The returned value is the mean of the sums of all of the normalized
    asymmetries over the whole velocity field for the major and minor axes and
    should scale between 0 and 1, wih 0 being totally symmetrical and 1 being
    totally asymmetrical. The map is the average of the asymmetry maps for the
    major and minor axes.

    Args:
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and fit parameters for the galaxy
        pa (:obj:`float`):
            Position angle of galaxy (in degrees)
        vsys (:obj:`float`):
            Systemic velocity of galaxy in the same units as the velocity
        xc (:obj:`float`, optional):
            x position of the center of the velocity field. Must be in the same
            units as `args.x`.
        yc (:obj:`float`, optional):
            y position of the center of the velocity field. Must be in the same
            units as `args.y`.
        maxd (:obj:`float`, optional):
            Maximum distance to allow the nearest neighbor finding to look for
            a reflexted spaxel. Any neighbor pairs at a radius larger than this
            will be masked. Must be in the same units as the spatial
            coordinates `args.x` and `args.y`.

    Returns:
        :obj:`float`: Global asymmetry value for the entire galaxy. Scales
        between 0 and 1. 
        `numpy.ndarray`_: Array of spatially resolved
        asymmetry values for every possible point in the velocity field. Should
        be the same shape as the input velocity fields.

    '''

    #construct KDTree of spaxels for matching
    x = args.x - xc
    y = args.y - yc
    tree = KDTree(list(zip(x,y)))
    
    #compute major and minor axis asymmetry 
    arc2d = []
    for axis in [0,90]:
        #match spaxels to their reflections, mask out ones without matches
        d,i = tree.query(reflect(pa - axis, x, y).T)
        mask = np.ma.array(np.ones(len(args.vel)), mask = (d>maxd) | args.vel_mask)

        #compute Andersen & Bershady (2013) A_RC parameter 2D maps
        vel = args.remap(args.vel * mask) - vsys
        ivar = args.remap(args.vel_ivar * mask)
        velr = args.remap(args.vel[i] * mask - vsys)
        ivarr = args.remap(args.vel_ivar[i] * mask)
        arc2d += [A_RC(vel, velr, ivar, ivarr)]
    
    #mean of maps to get global asym
    arc = np.mean([np.sum(a) for a in arc2d])
    asymmap = np.ma.array(arc2d).mean(axis=0)
    return arc, asymmap

def A_RC(vel, velr, ivar, ivarr):
    '''
    Compute velocity field asymmetry for a velocity field and its reflection.

    From Andersen & Bershady (2013) equation 7 but doesn't sum over whole
    galaxy so asymmmetry is spatially resolved. 

    Using Equation 7 from Andersen & Bershady (2007), the symmetry of a galaxy
    is calculated. This is done by reflecting it across the supplied position
    angle (assumed to be the major axis) and the angle perpendicular to that
    (assumed to be the minor axis). The value calculated is essentially a
    normalized error weighted difference between the velocity measurements on
    either side of a reflecting line. 

    The returned array is  of all of the normalized asymmetries over the whole
    velocity field for the major and minor axes and should scale between 0 and
    1, wih 0 being totally symmetrical and 1 being totally asymmetrical. The
    map is the average of the asymmetry maps for the major and minor axes.

    Args:
        vel (`numpy.ndarray`_):
            Array of velocity measurements for a galaxy.
        velr (`numpy.ndarray`_):
            Array of velocity measurements for a galaxy reflected across the desired axis.
        ivar (`numpy.ndarray`_):
            Array of velocity inverse variances for a galaxy.
        ivarr (`numpy.ndarray`_):
            Array of velocity inverse variances for a galaxy reflected across
            the desired axis.

    Returns:
        `numpy.ndarray`_: Array of rotational asymmetries for all of the
        velocity measurements supplied. Should be the same shape as the input
        arrays.
    '''
    return (np.abs(np.abs(vel) - np.abs(velr))/np.sqrt(1/ivar + 1/ivarr) 
         / (.5*np.sum(np.abs(vel) + np.abs(velr))/np.sqrt(1/ivar + 1/ivarr)))

def reflect(pa, x, y):
    '''
    Reflect arrays coordinates across a given position angle.

    Args:
        pa (:obj:`float`):
            Position angle to reflect across (in degrees)
        x (`numpy.ndarray`_):
            Array of x coordinate positions. Must be the same shape as `y`.
        y (`numpy.ndarray`_):
            Array of y coordinate positions. Must be the same shape as `x`.

    Returns:
        :obj:`tuple`: Two `numpy.ndarray`_ objects of the new reflected x and y
        coordinates
    '''

    th = np.radians(90 - pa) #turn position angle into a regular angle

    #reflection matrix across arbitrary angle
    ux = np.cos(th) 
    uy = np.sin(th)
    return np.dot([[ux**2 - uy**2, 2*ux*uy], [2*ux*uy, uy**2 - ux**2]], [x, y])


def symmetric_reflection_map(x, y, rotation=None, reflect='all'):
    r"""
    For each input coordinate, find the distance to the nearest pixel and its
    index after reflecting about an axis.

    This is primarily used to construct asymmetry measurements.  Instead of
    interpolating new data, this returns the pixel indices so that they can be
    used to propagate covariance between pixels.  The axis for the reflection
    crosses through :math:`(x,y) = (0,0)` with a counter-clockwise rotation
    provided by ``rotation``.

    The allowed reflections are:

        #. ``None``: No reflections are performed.
        #. ``'x'``: Reflects the data about the (rotated) :math:`y=0` axis.
        #. ``'y'``: Reflects the data about the (rotated) :math:`x=0` axis.
        #. ``'xy'``: Reflects the data about both the (rotated) :math:`x=0` and
           :math:`y=0` axes.  Note this is the same as rotating by 180 deg and
           applying no reflection.
        #. ``'all'``: Perform all four of the above.  The order is same as 1-4
           above.

    The returned objects provide the distance between the mapped pixels and the
    index of the coordinate nearest each input coordinate after being reflected
    about the provided axis.  Note that ``reflect=None`` should produce the
    trivial result that:

    .. code-block:: python

        d, i = symmetric_reflection_map(x, y, reflect=None)
        assert np.array_equal(i, np.arange(x.size))

    To compare an original set of data and its reflected, e.g.:

    .. code-block:: python

        d, i = symmetric_reflection_map(x, y)
        data_reflected_about_x = data[i[1]]

    where ``x`` and ``y`` are the Cartesian coordinates for the measurements
    provided by ``data``.
    
    Inspired by ``symmetrize_velfield`` in M. Cappellari's `plotbin`_ package.

    Args:
        x (`numpy.ndarray`_):
            The vector of Cartesian :math:`x` coordinates.  Must be 1D.
        y (`numpy.ndarray`_):
            The vector of Cartesian :math:`y` coordinates.  Must have the same
            shape as ``x``.
        rotation (:obj:`float`, optional):
            In the frame of the provided coordinate system, this is the
            counter-clockwise rotation of the new Cartesian :math:`x` axis to
            use for the reflections.
        reflect (:obj:`str`, :obj:`list`, optional):
            The type of reflections to apply.  See descriptions above.  Multiple
            reflections can be applied using a list of strings, however
            ``'all'`` cannot be an element of the list.  If a list is provided,
            the order of the reflections is maintained in the returned array(s).

    Returns:
        :obj:`tuple`: Returns two `numpy.ndarray`_ objects, the distance between
        the input pixels and the pixel nearest to its reflected location and the
        index of the nearest pixel.  See the function description.  If multiple
        reflections are requested, they are ordered along the first axis of the
        return arrays.  For example, if ``reflect=[None, 'x']``, the shape of
        the arrays are ``(2,x.size)``, and the matching indices when reflected
        about the (rotated) :math:`x` axis is the 2nd entry in the first axis.
    """
    # Check the input
    if x.ndim != 1:
        raise ValueError('Must provide 1D vectors.')
    if y.shape != x.shape:
        raise ValueError('x and y arrays must have the same shape.')

    if reflect is None:
        # Not reflecting, so the matches are exact
        return np.zeros_like(x), np.arange(x.size)

    # Setup the reflections
    reflect_options = [None, 'x', 'y', 'xy']
    if reflect == 'all':
        _reflect = reflect_options
    elif isinstance(reflect, str):
        _reflect = [reflect]
    if any([r not in reflect_options for r in _reflect]):
        raise ValueError(f'Unknown reflection.  Options are {reflect_options}.')
    nrefl = len(_reflect)

    # Get the rotated coordinates.  After rotation the major axis should be
    # oriented along the y-axis.
    if rotation is not None:
        # Rotate the *coordinate system*, not the points within the existing
        # coordinate system (just means we use a clockwise rotation instead of a
        # counter-clockwise one, which is the same as a negative
        # counter-clockwise rotation)
        x_rot, y_rot = rotate(x, y, -np.radians(rotation))
    else:
        x_rot, y_rot = x, y

    # Stack the coordinates for all reflections
    _x = np.concatenate([x_rot if r in [None, 'y'] else -x_rot for r in _reflect])
    _y = np.concatenate([y_rot if r in [None, 'x'] else -y_rot for r in _reflect])

    # Construct KDTree of spaxels for matching
    tree = KDTree(np.column_stack((x_rot, y_rot)))
    # Do the matching and return
    return tuple([o.reshape(nrefl, -1) if nrefl > 1 else o \
                    for o in tree.query(np.column_stack((_x, _y)))]) 


def onsky_asymmetry_maps(x, y, data, pa=0., ivar=None, mask=None, covar=None, maxd=None,
                         odd=False):
    r"""
    Compute the difference between the data and axial reflections of itself to
    measure non-axisymmetries.

    The axis of reflection is given by the provided position angle (from N
    through E in degrees) defined as being along the Cartesian :math:`y` axis of
    the reflected frame.  The reflections are performed by
    :func:`symmetric_reflection_map`.  Typically the PA is along the major axis
    of a galaxy, such that the reflections are, respectively, about the major
    axis, about the minor axis, and about both (the same as a 180 degree
    rotation).

    The data can have an "odd" or "even" symmetry; see the ``odd`` argument.

    .. warning::

        The function performs the difference computations using matrix
        multiplication.  This simplifies propagation of errors, but it also
        means that data subtracted from themselves are *ignored*.  I.e., any
        pixel matched to itself in the reflection symmetry mapping is
        effectively masked in the calculation; the difference values, *as well
        as any propagated error*, will be 0.
        
    Args:
        x (`numpy.ndarray`_):
            The 2D map of the Cartesian :math:`x` coordinates.
        y (`numpy.ndarray`_):
            The 2D map of the Cartesian :math:`y` coordinates.  Must have the same
            shape as ``x``.
        data (`numpy.ndarray`_):
            The on-sky 2D data.  Must have the same shape as ``x``.
        pa (:obj:`float`, optional):
            The on-sky position angle of the axis --- from N (+y) through E (+x)
            in degrees --- used to define the :math:`y` axis of the symmetry
            coordinate system.
        ivar (`numpy.ndarray`_, optional):
            The inverse variance in the provided data.  If provided, must have
            the same shape as ``x``.
        mask (`numpy.ndarray`_, optional):
            The bad-value mask for the provided data (True is bad).  If
            provided, must have the same shape as ``x``.
        covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
            Covariance matrix for the provided data.  If both this and ``ivar``
            are provided, the provided covariance takes precedence.  The shape
            of the matrix must be :math:`(N_{\rm pix}, N_{\rm pix})` where
            :math:`N_{\rm pix}` is the number of data values.
        maxd (:obj:`float`, optional):
            Maximum distance between input and reflected coordinates to use in
            the asymmetry calculation.  Must have the same units as the input
            ``x`` and ``y``.
        odd (:obj:`bool`, optional):
            Flag that the map has odd symmetry, where we define data with odd
            symmetry such that :math:`f(x,-y) = -f(x,y)`.  For example, velocity
            fields have odd symmetry, but velocity dispersion fields have even
            symmetry.

    Returns:
        :obj:`tuple`: Return the three maps with the reflected differences.  If
        errors are provided (``ivar`` or ``covar``), the errors in the
        difference are also returned.  I.e., there are either 3 or 6 objects
        returned, depending on if errors are provided.
    """
    # Check the input
    if y.shape != x.shape:
        raise ValueError('x and y arrays must have the same shape.')
    if data.shape != x.shape:
        raise ValueError('x and data arrays must have the same shape.')
    if ivar is not None and ivar.shape != x.shape:
        raise ValueError('x and ivar arrays must have the same shape.')
    if mask is not None and mask.shape != x.shape:
        raise ValueError('x and mask arrays must have the same shape.')
    if covar is not None and covar.shape != (x.size, x.size):
        raise ValueError('covar array must have shape (x.size, x.size).')
    if ivar is not None and covar is not None:
        warnings.warn('Both ivar and covar were provided, ignoring ivar.')

    # Perform the pixel matching
    d, i = symmetric_reflection_map(x.ravel(), y.ravel(), rotation=-pa)

    # Setup the mask
    bpm = np.zeros((4,x.size), dtype=bool) if maxd is None else np.array([_d > maxd for _d in d])
    if mask is not None:
        bpm |= np.array([mask.ravel()[i[j]] for j in range(4)])
    gpm = np.logical_not(bpm)

    # Workspace arrays
    ii = np.arange(x.size)
    ones = np.ones(x.size, dtype=float)
    sign = np.full(x.size, -1. if odd else 1., dtype=float)

    # Create the 3 symmetry maps and their errors (if possible)
    map_diff = []
    if ivar is None and covar is None:
        _covar = None
        diff_covar = None
    else:
        diff_covar = []
        _covar = sparse.diags(util.inverse(ivar.ravel())) if covar is None else covar
    for j,fac in zip([1,2,3], [ones, sign, sign]):
        _gpm = gpm[j] & gpm[0]
        tform = sparse.csr_matrix((fac[_gpm], (ii[_gpm], i[j,_gpm])), shape=(x.size,x.size)) \
                    - sparse.csr_matrix((ones[_gpm], (ii[_gpm], ii[_gpm])), shape=(x.size,x.size))
        map_diff += [np.ma.MaskedArray(tform.dot(data.ravel()),
                                       mask=np.logical_not(_gpm)).reshape(data.shape)]
        if _covar is not None:
            diff_covar += [tform.dot(_covar.dot(tform.transpose()))]
    if ivar is not None and covar is None:
        diff_covar = [np.ma.MaskedArray(c.diagonal(),
                                        mask=np.ma.getmaskarray(m).copy()).reshape(data.shape)
                        for m,c in zip(map_diff, diff_covar)]

    return tuple(map_diff) if _covar is None else tuple(map_diff + diff_covar)


