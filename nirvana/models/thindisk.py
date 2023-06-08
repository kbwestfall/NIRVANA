"""
Base class for thin disk models.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

import warnings

from IPython import embed

import numpy as np
from scipy import optimize

from .beam import ConvolveFFTW
from .util import cov_err
from ..data.kinematics import Kinematics
from ..data.scatter import IntrinsicScatter
from ..data.util import impose_positive_definite, cinv, inverse
from ..util.bitmask import BitMask

#warnings.simplefilter('error', RuntimeWarning)

class ThinDiskFitBitMask(BitMask):
    """
    Bin-by-bin mask used to track fit rejections.
    """
    prefix = 'MBIT'
    def __init__(self):
        # TODO: np.array just used for slicing convenience
        mask_def = np.array([['DIDNOTUSE', 'Datum flagged on input'],
                             ['REJ_ERR', 'Datum has large measurement error'],
                             ['REJ_SNR', 'Datum has low signal-to-noise'],
                             ['REJ_UNR', 'Datum unreasonably discrepant'],
                             ['REJ_RESID', 'Datum rejected by iterative fit'],
                             ['DISJOINT', 'Datum is spatially disjoint']])
        super().__init__(mask_def[:,0], descr=mask_def[:,1])

    @staticmethod
    def base_flags():
        """
        Return the list of "base-level" flags that are *always* ignored,
        regardless of the fit iteration.
        """
        return ['DIDNOTUSE', 'REJ_ERR', 'REJ_SNR', 'REJ_UNR', 'DISJOINT']

    def reset_to_base_flags(self, kin, vel_mask, sig_mask):
        """
        Reset the masks to only include the "base" flags.
        
        As the best-fit parameters change over the course of a set of rejection
        iterations, the residuals with respect to the model change. This method
        resets the flags back to the base-level rejection (i.e., independent of
        the model), allowing the rejection to be based on the most recent set of
        parameters and potentially recovering data that was previously rejected
        because of a poor model fit.

        .. warning::

            The argument objects are *all* modified in place.

        Args:
            kin (:class:`~nirvana.data.kinematics.Kinematics`):
                Object with the data being fit.
            vel_mask (`numpy.ndarray`_):
                Bitmask used to track velocity rejections.
            sig_mask (`numpy.ndarray`_):
                Bitmask used to track dispersion rejections. Can be None.
        """
        # Turn off the relevant rejection for all pixels
        vel_mask = self.turn_off(vel_mask, flag='REJ_RESID')
        # Reset the data mask held by the Kinematics object
        kin.vel_mask = self.flagged(vel_mask, flag=self.base_flags())
        if sig_mask is None:
            return
        # Turn off the relevant rejection for all pixels
        sig_mask = self.turn_off(sig_mask, flag='REJ_RESID')
        # Reset the data mask held by the Kinematics object
        kin.sig_mask = self.flagged(sig_mask, flag=self.base_flags())


class ThinDiskParBitMask(BitMask):
    """
    Mask used to track parameter issues.
    """
    prefix = 'PBIT'
    def __init__(self):
        # TODO: np.array just used for slicing convenience
        mask_def = np.array([['FIXED', 'Fixed parameter'],
                             ['TIED', 'Tied parameter'],
                             ['LOWERBOUND', 'Lower trust boundary active'],
                             ['UPPERBOUND', 'Upper trust boundary active'],
                             ['LOWERERR', 'lt 1 sigma above lower boundary'],
                             ['UPPERERR', 'lt 1 sigma below upper boundary']])
        super().__init__(mask_def[:,0], descr=mask_def[:,1])


class ThinDiskGlobalBitMask(BitMask):
    """
    Fit-wide quality flag.
    """
    prefix = 'GBIT'
    def __init__(self):
        # NOTE: np.array just used for slicing convenience
        mask_def = np.array([['LOWINC', 'Best fit inclination unreasonably low'],
                             ['NOMODEL', 'Model not fit or failed'],
                             ['DIFFMOD', 'Model discrepancy']])
        super().__init__(mask_def[:,0], descr=mask_def[:,1])


class ThinDisk:
    r"""
    Provides a base class for razor-thin disk models.

    The model assumes the disk is infinitely thin and has a single set of
    geometric and bulk-flow parameters:

        - :math:`x_c, y_c`: The coordinates of the galaxy dynamical center.
        - :math:`\phi`: The position angle of the galaxy (the angle from N
          through E)
        - :math:`i`: The inclination of the disk; the angle of the disk
          normal relative to the line-of-sight such that :math:`i=0` is a
          face-on disk.
        - :math:`V_{\rm sys}`: The systemic (bulk) velocity of the galaxy
          taken as the line-of-sight velocity at the dynamical center.

    Other than that, this is an abstract base class.

    .. todo::
        Describe the attributes

    """

    gbm = ThinDiskGlobalBitMask()
    """
    Global bitmask.
    """

    mbm = ThinDiskFitBitMask()
    """
    Measurement-specific bitmask.
    """

    pbm = ThinDiskParBitMask()
    """
    Parameter-specific bitmask.
    """

    def __init__(self, **kwargs):
        # Number of "base" parameters
        self.nbp = 5
        # Initialize common attributes
        self.reinit()

    def __repr__(self):
        """
        Provide the representation of the object when written to the screen.
        """
        # Collect the attributes relevant to construction of a model
        attr = [n for n in ['par', 'x', 'y', 'sb', 'beam_fft'] if getattr(self, n) is not None]
        return f'<{self.__class__.__name__}: Defined attr - {",".join(attr)}>'

    def reinit(self):
        """
        Reinitialize the object.

        This resets the model parameters to the guess parameters and erases any
        existing data used to construct the models.  Note that, just like when
        instantiating a new object, any calls to :func:`model` after
        reinitialization will require at least the coordinates (``x`` and ``y``)
        to be provided to successfully calculate the model.
        """
        self.par = self.guess_par() # Model parameters
        self.par_err = None         # Model parameter errors
        self.par_mask = None        # Model parameter masks
        self.x = None               # On-sky coordinates at which to evaluate the model
        self.y = None
        self.beam_fft = None        # FFT of the beam kernel
        self.kin = None             # Kinematics object with data to fit
        self.sb = None              # Surface-brightness weighting for each x,y position
        self.vel_gpm = None         # "Good-pixel mask" for velocity measurements
        self.sig_gpm = None         # "Good-pixel mask" for dispersion measurements
        self.cnvfftw = None         # ConvolveFFTW object used to perform convolutions
        # TODO: Set this to the bitmask dtype
        self.global_mask = 0        # Global bitmask value
        self.fit_status = None      # Status integer for a fit
        self.fit_success = None     # Flag that a fit was successful

    def guess_par(self):
        """
        Return a list of generic guess parameters.

        .. todo::
            Could enable this to base the guess on the data to be fit, but at
            the moment these are hard-coded numbers.

        Returns:
            `numpy.ndarray`_: Vector of guess parameters
        """
        # Return the guess parameters for the geometric parameters
        return np.array([0., 0., 45., 30., 0.])

    def par_names(self, short=False):
        """
        Return a list of strings with the parameter names.

        Args:
            short (:obj:`bool`, optional):
                Return truncated nomenclature for the parameter names.

        Returns:
            :obj:`list`: List of parameter name strings.
        """
        if short:
            return ['x0', 'y0', 'pa', 'inc', 'vsys']
        return ['X center', 'Y center', 'Position Angle', 'Inclination', 'Systemic Velocity']

    def _base_slice(self):
        """
        Return a slice object used to select the base parameters from the
        parameter vector.
        """
        return slice(self.nbp)

    def base_par(self, err=False):
        """
        Return the base (largely geometric) parameters. Returns None if
        parameters are not defined yet.

        Args:
            err (:obj:`bool`, optional):
                Return the parameter errors instead of the parameter values.

        Returns:
            `numpy.ndarray`_: Vector with parameters or parameter errors for the
            "base" parameters.
        """
        p = self.par_err if err else self.par
        return None if p is None else p[self._base_slice()]

    def par_bounds(self, base_lb=None, base_ub=None):
        """
        Return the lower and upper boundaries on the model parameters.

        The default geometric bounds (see ``base_lb``, ``base_ub``) are set
        by the minimum and maximum available x and y coordinates, -350 to 350
        for the position angle, 1 to 89 for the inclination, and -300 to 300
        for the systemic velocity.

        .. todo::

            Could enable this to base the bounds on the data to be fit, but
            at the moment these are hard-coded numbers.

        Args:
            base_lb (`numpy.ndarray`_, optional):
                The lower bounds for the "base" parameters: x0, y0, pa, inc,
                vsys. If None, the defaults are used (see above).
            base_ub (`numpy.ndarray`_, optional):
                The upper bounds for the "base" parameters: x0, y0, pa, inc,
                vsys. If None, the defaults are used (see above).

        Returns:
            :obj:`tuple`: A two-tuple providing, respectively, the lower and
            upper boundaries for all model parameters.
        """
        if base_lb is not None and len(base_lb) != self.nbp:
            raise ValueError('Incorrect number of lower bounds for the base '
                             f'parameters; found {len(base_lb)}, expected {self.nbp}.')
        if base_ub is not None and len(base_ub) != self.nbp:
            raise ValueError('Incorrect number of upper bounds for the base '
                             f'parameters; found {len(base_ub)}, expected {self.nbp}.')

        if (base_lb is None or base_ub is None) and (self.x is None or self.y is None):
            raise ValueError('Cannot define limits on center.  Provide base_lb,base_ub or set '
                             'the evaluation grid coordinates (attributes x and y).')

        if base_lb is None:
            minx = np.amin(self.x)
            miny = np.amin(self.y)
            base_lb = np.array([minx, miny, -350., 1., -300.])
        if base_ub is None:
            maxx = np.amax(self.x)
            maxy = np.amax(self.y)
            base_ub = np.array([maxx, maxy, 350., 89., 300.])
        return (base_lb, base_ub)

    def _set_par(self, par):
        """
        Set the full parameter vector, accounting for any fixed parameters.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.
        """
        # WARNING: The nfree and free *must* be part of the derived class.
        if par.ndim != 1:
            raise ValueError('Parameter array must be a 1D vector.')
        if par.size == self.np:
            self.par = par.copy()
            return
        if par.size != self.nfree:
            raise ValueError('Must provide {0} or {1} parameters.'.format(self.np, self.nfree))
        self.par[self.free] = par.copy()

    def _init_coo(self, x, y):
        """
        Initialize the coordinate arrays.

        .. warning::
            
            Input coordinate data types are all converted to `numpy.float64`_.
            This is always true, even though this is only needed when using
            :class:`~nirvana.models.beam.ConvolveFFTW`.

        Args:
            x (`numpy.ndarray`_):
                The 2D x-coordinates at which to evaluate the model.  If not
                None, replace the existing :attr:`x` with this array.
            y (`numpy.ndarray`_):
                The 2D y-coordinates at which to evaluate the model.  If not
                None, replace the existing :attr:`y` with this array.

        Raises:
            ValueError:
                Raised if the shapes of :attr:`x` and :attr:`y` are not the same.
        """
        if x is None and y is None:
            # Nothing to do
            return

        # Define it and check it
        if x is not None:
            self.x = x.astype(float)
        if y is not None:
            self.y = y.astype(float)
        if self.x.shape != self.y.shape:
            raise ValueError('Input coordinates must have the same shape.')

    def _init_sb(self, sb):
        """
        Initialize the surface brightness array.

        .. warning::
            
            Input surface-brightness data types are all converted to
            `numpy.float64`_.  This is always true, even though this is only
            needed when using :class:`~nirvana.models.beam.ConvolveFFTW`.

        Args:
            sb (`numpy.ndarray`_):
                2D array with the surface brightness of the object.  If not
                None, replace the existing :attr:`sb` with this array.

        Raises:
            ValueError:
                Raised if the shapes of :attr:`sb` and :attr:`x` are not the same.
        """
        if sb is None:
            # Nothing to do
            return

        # Check it makes sense to define the surface brightness
        if self.x is None:
            raise ValueError('Input coordinates must be instantiated first!')

        # Define it and check it
        self.sb = sb.astype(float)
        if self.sb.shape != self.x.shape:
            raise ValueError('Input coordinates must have the same shape.')

    def _init_beam(self, beam, is_fft, cnvfftw):
        """
        Initialize the beam-smearing kernel and the convolution method.

        Args:
            beam (`numpy.ndarray`_):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT).  If not None, replace existing
                :attr:`beam_fft` with this array (or its FFT, depending on the
                provided ``is_fft``).
            is_fft (:obj:`bool`):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`):
                An object that expedites the convolutions using FFTW/pyFFTW.  If
                provided, the shape *must* match :attr:``beam_fft`` (after this
                is potentially updated by the provided ``beam``).  If None, a
                new :class:`~nirvana.models.beam.ConvolveFFTW` instance is
                constructed to perform the convolutions.  If the class cannot be
                constructed because the user doesn't have pyfftw installed, then
                the convolutions fall back to the numpy routines.
        """
        if beam is None:
            # Nothing to do
            return

        # Check it makes sense to define the beam
        if self.x is None:
            raise ValueError('Input coordinates must be instantiated first!')
        if self.x.ndim != 2:
            raise ValueError('To perform convolution, must provide 2d coordinate arrays.')

        # Assign the beam and check it
        self.beam_fft = beam if is_fft else np.fft.fftn(np.fft.ifftshift(beam))
        if self.beam_fft.shape != self.x.shape:
            raise ValueError('Currently, convolution requires the beam map to have the same '
                                'shape as the coordinate maps.')

        # Convolutions will be performed, try to setup the ConvolveFFTW
        # object (self.cnvfftw).
        if cnvfftw is None:
            if self.cnvfftw is not None and self.cnvfftw.shape == self.beam_fft.shape:
                # ConvolveFFTW is ready to go
                return

            try:
                self.cnvfftw = ConvolveFFTW(self.beam_fft.shape)
            except:
                warnings.warn('Could not instantiate ConvolveFFTW; proceeding with numpy '
                              'FFT/convolution routines.')
                self.cnvfftw = None
        else:
            # A cnvfftw was provided, check it
            if not isinstance(cnvfftw, ConvolveFFTW):
                raise TypeError('Provided cnvfftw must be a ConvolveFFTW instance.')
            if cnvfftw.shape != self.beam_fft.shape:
                raise ValueError('cnvfftw shape does not match beam shape.')
            self.cnvfftw = cnvfftw

    def _init_par(self, p0, fix):
        """
        Initialize the relevant parameter vectors that tracks the full set of
        model parameters and which of those are freely fit by the model.

        Args:
            p0 (`numpy.ndarray`_):
                The initial parameters for the model.  Can be None.  Length must
                be :attr:`np`, if not None.
            fix (`numpy.ndarray`_):
                A boolean array selecting the parameters that should be fixed
                during the model fit.  Can be None.  Length must be :attr:`np`,
                if not None.
        """
        if p0 is None:
            p0 = self.guess_par()
        _p0 = np.atleast_1d(p0)
        if _p0.size != self.np:
            raise ValueError('Incorrect number of model parameters.')
        self.par = _p0.copy()
        self.par_err = None
        _free = np.ones(self.np, dtype=bool) if fix is None else np.logical_not(fix)
        if _free.size != self.np:
            raise ValueError('Incorrect number of model parameter fitting flags.')
        self.free = _free.copy()
        self.nfree = np.sum(self.free)

    def _init_model(self, par, x, y, sb, beam, is_fft, cnvfftw, ignore_beam):
        """
        Initialize the attributes in preparation of a model calculation.

        Args:
            par (`numpy.ndarray`_):
                The list of parameters to use. If None, the internal
                :attr:`par` is used. Length should be either :attr:`np` or
                :attr:`nfree`. If the latter, the values of the fixed
                parameters in :attr:`par` are used.  See :func:`_set_par`.
            x (`numpy.ndarray`_):
                The 2D x-coordinates at which to evaluate the model. If None,
                the internal :attr:`x` is used. See :func:`_init_coo`.
            y (`numpy.ndarray`_):
                The 2D y-coordinates at which to evaluate the model. If None,
                the internal :attr:`y` is used. See :func:`_init_coo`.
            sb (`numpy.ndarray`_):
                2D array with the surface brightness of the object. This is used
                to weight the convolution of the kinematic fields according to
                the luminosity distribution of the object.  Must have the same
                shape as ``x``. If None, the internal :attr:`sb` is used or, if
                that is also None, the convolution is unweighted.  If a
                convolution is not performed (either ``beam`` or
                :attr:`beam_fft` are not available, or ``ignore_beam`` is True),
                this array is ignored.  See :func:`_init_sb`.
            beam (`numpy.ndarray`_):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT). If None, the internal :attr:`beam_fft`
                is used.  See :func:`_init_beam`.
            is_fft (:obj:`bool`):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.  See :func:`_init_beam`.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`):
                An object that expedites the convolutions using
                FFTW/pyFFTW.  See :func:`_init_beam`.
            ignore_beam (:obj:`bool`):
                Ignore the beam-smearing when constructing the model. I.e.,
                construct the *intrinsic* model.
        """
        # Initialize the coordinates (this does nothing if both x and y are None)
        self._init_coo(x, y)
        # Initialize the convolution kernel (this does nothing if beam is None)
        self._init_beam(beam, is_fft, cnvfftw)
        if self.beam_fft is not None and not ignore_beam:
            # Initialize the surface brightness, only if it would be used
            self._init_sb(sb)
        # Check that the model can be calculated
        if self.x is None or self.y is None:
            raise ValueError('No coordinate grid defined.')
        # Reset the parameter values
        if par is not None:
            self._set_par(par)

    def _init_data(self, kin, scatter, assume_posdef_covar, ignore_covar):
        """
        Initialize the data attributes needed for fitting the data with this model.

        Args:
            kin (:class:`~nirvana.data.kinematics.Kinematics`):
                The object providing the kinematic data to be fit.
            scatter (:obj:`float`, array-like):
                Introduce a fixed intrinsic-scatter term into the model. The 
                scatter is added in quadrature to all measurement errors in the
                calculation of the merit function. If no errors are available,
                this has the effect of renormalizing the unweighted merit
                function by 1/scatter.  Can be None, which means no intrinsic
                scatter is added.  If both velocity and velocity dispersion are
                being fit, this can be a single number applied to both datasets
                or a 2-element vector that provides different intrinsic scatter
                measurements for each kinematic moment (ordered velocity then
                velocity dispersion).
            assume_posdef_covar (:obj:`bool`):
                If the :class:`~nirvana.data.kinematics.Kinematics` includes
                covariance matrices, this forces the code to proceed assuming
                the matrices are positive definite.
            ignore_covar (:obj:`bool`):
                If the :class:`~nirvana.data.kinematics.Kinematics` includes
                covariance matrices, ignore them and just use the inverse
                variance.
        """
        # Initialize the data to fit
        self.kin = kin
        self.vel_gpm = np.logical_not(self.kin.vel_mask)
        self.sig_gpm = None if self.dc is None else np.logical_not(self.kin.sig_mask)

        # Determine which errors were provided
        self.has_err = self.kin.vel_ivar is not None if self.dc is None \
                        else self.kin.vel_ivar is not None and self.kin.sig_ivar is not None
        if not self.has_err and (self.kin.vel_err is not None or self.kin.sig_err is not None):
            warnings.warn('Some errors being ignored if both velocity and velocity dispersion '
                          'errors are not provided.')
        self.has_covar = self.kin.vel_covar is not None if self.dc is None \
                            else self.kin.vel_covar is not None and self.kin.sig_covar is not None
        if not self.has_covar \
                and (self.kin.vel_covar is not None or self.kin.sig_covar is not None):
            warnings.warn('Some covariance matrices being ignored if both velocity and velocity '
                          'dispersion covariances are not provided.')
        if ignore_covar:
            # Force ignoring the covariance
            # TODO: This requires that, e.g., kin.vel_ivar also be defined...
            self.has_covar = False

        # Check the intrinsic scatter input
        self.scatter = None
        if scatter is not None:
            self.scatter = np.atleast_1d(scatter)
            if self.scatter.size > 2:
                raise ValueError('Should provide, at most, one scatter term for each kinematic '
                                 'moment being fit.')
            if self.dc is not None and self.scatter.size == 1:
                warnings.warn('Using single scatter term for both velocity and velocity '
                              'dispersion.')
                self.scatter = np.array([scatter, scatter])

        # Set the internal error attributes
        if self.has_err:
            self._v_err = np.sqrt(inverse(self.kin.vel_ivar))
            self._s_err = None if self.dc is None \
                                else np.sqrt(inverse(self.kin.sig_phys2_ivar))
            if self.scatter is not None:
                self._v_err = np.sqrt(self._v_err**2 + self.scatter[0]**2)
                if self.dc is not None:
                    self._s_err = np.sqrt(self._s_err**2 + self.scatter[1]**2)
        elif not self.has_err and not self.has_covar and self.scatter is not None:
            self.has_err = True
            self._v_err = np.full(self.kin.vel.shape, self.scatter[0], dtype=float)
            self._s_err = None if self.dc is None \
                                else np.full(self.kin.sig.shape, self.scatter[1], dtype=float)
        else:
            self._v_err = None
            self._s_err = None

        # Set the internal covariance attributes
        if self.has_covar:
            # Construct the matrices used to calculate the merit function in
            # the presence of covariance.
            vel_pd_covar = self.kin.vel_covar[np.ix_(self.vel_gpm,self.vel_gpm)]
            sig_pd_covar = None if self.dc is None \
                            else self.kin.sig_phys2_covar[np.ix_(self.sig_gpm,self.sig_gpm)]
            if not assume_posdef_covar:
                # Force the matrices to be positive definite
                print('Forcing vel covar to be pos-def')
                vel_pd_covar = impose_positive_definite(vel_pd_covar)
                if self.dc is None:
                    sig_pd_covar = None 
                else:
                    print('Forcing sig covar to be pos-def')
                    sig_pd_covar = impose_positive_definite(sig_pd_covar)

            if self.scatter is not None:
                # A diagonal matrix with only positive values is, by definition,
                # positive definite; and the sum of two positive-definite
                # matrices is also positive definite.
                vel_pd_covar += np.diag(np.full(vel_pd_covar.shape[0], self.scatter[0]**2,
                                                dtype=float))
                if self.dc is not None:
                    sig_pd_covar += np.diag(np.full(sig_pd_covar.shape[0], self.scatter[1]**2,
                                                    dtype=float))

            self._v_ucov = cinv(vel_pd_covar, upper=True)
            self._s_ucov = None if sig_pd_covar is None else cinv(sig_pd_covar, upper=True)
        else:
            self._v_ucov = None
            self._s_ucov = None

    def model(self, par=None, x=None, y=None, sb=None, beam=None, is_fft=False, cnvfftw=None,
              ignore_beam=False):
        """
        Evaluate the model.

        Note that arguments passed to this function overwrite any existing
        attributes of the object, and subsequent calls to this function will
        continue to use existing attributes, unless they are overwritten.  For
        example, if ``beam`` is provided here, it overwrites any existing
        :attr:`beam_fft` and any subsequent calls to ``model`` **that do not
        provide a new** ``beam`` will use the existing :attr:`beam_fft`.  To
        remove all internal attributes to get a "clean" instantiation, either
        define a new :class:`AxisymmetricDisk` instance or use :func:`reinit`.

        .. warning::
            
            Input coordinates and surface-brightness data types are all
            converted to `numpy.float64`_.  This is always true, even though
            this is only needed when using
            :class:`~nirvana.models.beam.ConvolveFFTW`.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. If None, the internal
                :attr:`par` is used. Length should be either :attr:`np` or
                :attr:`nfree`. If the latter, the values of the fixed
                parameters in :attr:`par` are used.
            x (`numpy.ndarray`_, optional):
                The 2D x-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`x` is used.
            y (`numpy.ndarray`_, optional):
                The 2D y-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`y` is used.
            sb (`numpy.ndarray`_, optional):
                2D array with the surface brightness of the object. This is used
                to weight the convolution of the kinematic fields according to
                the luminosity distribution of the object.  Must have the same
                shape as ``x``. If None, the convolution is unweighted.  If a
                convolution is not performed (either ``beam`` or
                :attr:`beam_fft` are not available, or ``ignore_beam`` is True),
                this array is ignored.
            beam (`numpy.ndarray`_, optional):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT). If not provided, the internal
                :attr:`beam_fft` is used.
            is_fft (:obj:`bool`, optional):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.  Ignored if ``beam`` is not provided.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
                An object that expedites the convolutions using
                FFTW/pyFFTW. If None, the convolution is done using numpy
                FFT routines.
            ignore_beam (:obj:`bool`, optional):
                Ignore the beam-smearing when constructing the model. I.e.,
                construct the *intrinsic* model.

        Returns:
            `numpy.ndarray`_, :obj:`tuple`: The velocity field model, and the
            velocity dispersion field model, if the latter is included
        """
        raise NotImplementedError(f'Model not implemented for {self.__class__.__name__}')

    def deriv_model(self, par=None, x=None, y=None, sb=None, beam=None, is_fft=False, cnvfftw=None,
                    ignore_beam=False):
        """
        Evaluate the derivative of the model w.r.t all input parameters.

        Note that arguments passed to this function overwrite any existing
        attributes of the object, and subsequent calls to this function will
        continue to use existing attributes, unless they are overwritten.  For
        example, if ``beam`` is provided here, it overwrites any existing
        :attr:`beam_fft` and any subsequent calls to ``model`` **that do not
        provide a new** ``beam`` will use the existing :attr:`beam_fft`.  To
        remove all internal attributes to get a "clean" instantiation, either
        define a new :class:`AxisymmetricDisk` instance or use :func:`reinit`.

        .. warning::
            
            Input coordinates and surface-brightness data types are all
            converted to `numpy.float64`_.  This is always true, even though
            this is only needed when using
            :class:`~nirvana.models.beam.ConvolveFFTW`.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. If None, the internal
                :attr:`par` is used. Length should be either :attr:`np` or
                :attr:`nfree`. If the latter, the values of the fixed
                parameters in :attr:`par` are used.
            x (`numpy.ndarray`_, optional):
                The 2D x-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`x` is used.
            y (`numpy.ndarray`_, optional):
                The 2D y-coordinates at which to evaluate the model. If not
                provided, the internal :attr:`y` is used.
            sb (`numpy.ndarray`_, optional):
                2D array with the surface brightness of the object. This is used
                to weight the convolution of the kinematic fields according to
                the luminosity distribution of the object.  Must have the same
                shape as ``x``. If None, the convolution is unweighted.  If a
                convolution is not performed (either ``beam`` or
                :attr:`beam_fft` are not available, or ``ignore_beam`` is True),
                this array is ignored.
            beam (`numpy.ndarray`_, optional):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT). If not provided, the internal
                :attr:`beam_fft` is used.
            is_fft (:obj:`bool`, optional):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.  Ignored if ``beam`` is not provided.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
                An object that expedites the convolutions using
                FFTW/pyFFTW. If None, the convolution is done using numpy
                FFT routines.
            ignore_beam (:obj:`bool`, optional):
                Ignore the beam-smearing when constructing the model. I.e.,
                construct the *intrinsic* model.

        Returns:
            `numpy.ndarray`_, :obj:`tuple`: The velocity field model, and the
            velocity dispersion field model, if the latter is included
        """
        raise NotImplementedError(f'Model derivative not implemented for {self.__class__.__name__}')

    def mock_observation(self, par, kin=None, x=None, y=None, sb=None, binid=None,
                         vel_ivar=None, vel_covar=None, vel_mask=None, sig_ivar=None,
                         sig_covar=None, sig_mask=None, sig_corr=None, beam=None, is_fft=False,
                         cnvfftw=None, ignore_beam=False, add_err=False, positive_definite=False,
                         rng=None):
        r"""
        Construct a mock observation.

        The mock data can be defined in one of two ways:

            #. If a :class:`~nirvana.data.kinematics.Kinematics` object is
               provided using the ``kin`` keyword, all other keyword arguments
               are ignored and the mock data are meant to explicitly mimic the
               real observations.

            #. Otherwise, most of the other keywords are used to instantiate a
               mock :class:`~nirvana.data.kinematics.Kinematics` object.  Just
               like when instantiating a
               :class:`~nirvana.data.kinematics.Kinematics` object, the provided
               data arrays *must* be 2D and all have the same shape.

        The mock observation is constructed by sampling the model exactly as
        done when fitting real observations.  If errors are provided and
        ``add_err`` is True, Gaussian error is added to the model values.

        .. warning::

            - Unlike :func:`model`, this method does *not* use any pre-existing
              attributes in place of keyword arguments that are not directly
              provided.  For example, if ``beam`` is None on input,
              :attr:`beam_fft` will *not* be used even if available.  Instead,
              this method calls :func:`reinit` to reinitialize the object such
              that the mock observation is built from the provided arguments
              only.

            - Covariance matrices are expected to be positive-definite on input!

            - When adding error, masked elements are ignored (i.e., no error is
              added to them).

        Args:
            par (array-like):
                The list of model parameters to use. Length must be :attr:`np`.
            kin (:class:`~nirvana.data.kinematics.Kinematics`, optional):
                Object with the kinematic data to mimic.  If None, must provide
                at least ``x``, ``y``, and ``vel_ivar``.
            x (`numpy.ndarray`_, optional):
                The 2D x-coordinates at which to evaluate the model. Ignored if
                ``kin`` is provided.
            y (`numpy.ndarray`_, optional):
                The 2D y-coordinates at which to evaluate the model.  Shape must
                match ``x``.  Ignored if ``kin`` is provided.  
            sb (`numpy.ndarray`_, optional):
                2D array with the (nominally unbinned) surface brightness of the
                object. This is used to weight the convolution of the kinematic
                fields according to the luminosity distribution of the object.
                Must have the same shape as ``x``. If None, the convolution is
                unweighted.  If a convolution is not performed (``beam`` is None
                or ``ignore_beam`` is True) or if ``kin`` is provided, this
                array is ignored.
            binid (`numpy.ndarray`_, optional):
                2D integer array associating each measurement with a unique bin
                ID number. Measurements not associated with any bin should have
                a value of -1 in this array. If None, all unmasked measurements
                (see ``vel_mask`` and ``sig_mask``) are considered unique.
                Shape must match ``x``.  Ignored if ``kin`` is provided.
            vel_ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
                2D array with the inverse variance of the velocity measurements.
                If both this and ``vel_covar`` are None, no errors are added to
                the model data.  Shape must match ``x``, and all array elements
                with the same bin ID should have the same value.  If both
                ``vel_ivar`` and ``vel_covar`` are provided, the latter takes
                precedence.  Ignored if ``kin`` is provided.
            vel_covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance matrix for the velocity measurements.  If both this
                and ``vel_ivar`` are None, no errors are added to the model
                data.  The array must be square, where the length along one side
                matches the number elements in ``x``.  If both ``vel_ivar`` and
                ``vel_covar`` are provided, the latter takes precedence.
                Ignored if ``kin`` is provided.
            vel_mask (`numpy.ndarray`_, optional):
                2D boolean array with a bad-pixel mask for the velocity
                measurements (a mask value of True is ignored).  If None, all
                velocity values are considered valid.  Shape must match ``x``.
                If ``vel_ivar`` is a `numpy.ma.MaskedArray`_, this mask is
                *combined* with ``vel_ivar.mask``.  Ignored if ``kin`` is
                provided.
            sig_ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
                2D array with the inverse variance of the velocity dispersion
                measurements.  If ``kin`` is provided or no velocity dispersion
                parameterization is defined (i.e., :attr:`dc` is None), this is
                ignored.  If both this and ``sig_covar`` are None, no errors are
                added to the model data.  Shape must match ``x``, and all array
                elements with the same bin ID should have the same value.  If
                both ``sig_ivar`` and ``sig_covar`` are provided, the latter
                takes precedence.
            sig_covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance matrix for the velocity dispersion measurements.  If
                ``kin`` is provided or no velocity dispersion parameterization
                is defined (i.e., :attr:`dc` is None), this is ignored.  If both
                this and ``sig_ivar`` are None, no errors are added to the model
                data.  The array must be square, where the length along one side
                matches the number elements in ``x``.  If both ``sig_ivar`` and
                ``sig_covar`` are provided, the latter takes precedence.
            sig_mask (`numpy.ndarray`_, optional):
                2D boolean array with a bad-pixel mask for the velocity
                dispersion measurements (a mask value of True is ignored).  If
                ``kin`` is provided or no velocity dispersion parameterization
                is defined (i.e., :attr:`dc` is None), this is ignored.  If
                None, all velocity dispersion values are considered valid.
                Shape must match ``x``.  If ``sig_ivar`` is a
                `numpy.ma.MaskedArray`_, this mask is *combined* with
                ``sig_ivar.mask``.
            sig_corr (`numpy.ndarray`_, optional):
                A quadrature correction for the velocity dispersion
                measurements. If None, velocity dispersions are assumed to be
                the *astrophysical* Doppler broadening of the kinematic tracer.
                If provided, the observed velocity dispersion used to construct
                the mock :class:`~nirvana.data.kinematics.Kinematics` object is:

                .. math::
                    \sigma_{\rm obs}^2 = \sigma^2 + \sigma_{\rm corr}^2,

                where the model is used to calculate :math:`\sigma`.
            beam (`numpy.ndarray`_, optional):
                The 2D rendering of the beam-smearing kernel, or its Fast
                Fourier Transform (FFT).  If ``kin`` is provided, this is
                ignored.  If not provided, no beam convolution is performed when
                constructing the mock observations.
            is_fft (:obj:`bool`, optional):
                The provided ``beam`` object is already the FFT of the
                beam-smearing kernel.  Ignored if ``kin`` is provided or
                ``beam`` is not provided.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
                An object that expedites the convolutions using
                FFTW/pyFFTW. If None, the convolution is done using numpy
                FFT routines.
            ignore_beam (:obj:`bool`, optional):
                Regardless of the availability of the beam profile, ignore it
                and do not include it in the returned object.
            add_err (:obj:`bool`, optional):
                If provided by either ``kin`` or the method keyword arguments,
                use the error distributions to add error to the mock model data.
                Nominally, error should be added to mock observations, but it's
                more efficient to create a noiseless dataset and then create
                many noise realizations (see
                :func:`~nirvana.data.kinematics.Kinematics.deviate`).  This
                keyword only results in a single noise realization.
            rng (`numpy.random.Generator`, optional):
                Generator object to use.  If None, a new Generator is
                instantiated; see :func:`~nirvana.data.util.gaussian_deviates`.

        Returns:
            :class:`~nirvana.data.kinematics.Kinematics`:  Object providing the
            mock observations.  This object can be fit, e.g., with
            :func:`lsq_fit`, as would be done with real observations.
        """

        # Get the input errors.  This is done first for the checks that follow.
        if kin is None:
            _vel_ivar = vel_ivar
            _vel_covar = vel_covar
            _sig_ivar = sig_ivar
            _sig_covar = sig_covar
        else:
            _vel_ivar = None if kin.vel_ivar is None \
                        else kin.remap('vel_ivar', masked=False, fill_value=0.)
            _vel_covar = None if kin.vel_covar is None else kin.remap_covar('vel_covar')
            _sig_ivar = None if self.dc is None or kin.sig_ivar is None \
                        else kin.remap('sig_ivar', masked=False, fill_value=0.)
            _sig_covar = None if self.dc is None or kin.sig_covar is None \
                            else kin.remap_covar('sig_covar')

        if add_err:
            # Check: Dispersion model not defined but error and mask provided
            if self.dc is None and any([s is not None for s in [_sig_ivar, _sig_covar, _sig_mask]]):
                warnings.warn('AxisymmetricDisk instance does not include dispersion profile '
                                'model.  Ignoring provided dispersion errors and mask.')
            # Check: Dispersion model available and mismatch between availability of errors
            if self.dc is not None and (_vel_ivar is not None and _sig_ivar is None \
                                        or _vel_ivar is None and _sig_ivar is not None \
                                        or _vel_covar is not None and _sig_covar is None \
                                        or _vel_covar is None and _sig_covar is not None):
                raise ValueError('When adding error, must add error to both vel and sigma.')

        # Instantiate the mock Kinematics object
        if kin is None:
            # Revert the fft to get the spatial representation of the beam profile
            _beam = None if beam is None or ignore_beam \
                        else (np.fft.fftshift(np.fft.ifftn(beam).real) if is_fft else beam)

            vel = np.zeros(x.shape, dtype=float)
            sig = None if self.dc is None else np.zeros(x.shape, dtype=float)
            _kin = Kinematics(vel, vel_ivar=_vel_ivar, vel_mask=vel_mask, vel_covar=_vel_covar,
                              x=x, y=y, sb=sb, sig=sig, sig_ivar=_sig_ivar, sig_mask=sig_mask,
                              sig_covar=_sig_covar, sig_corr=sig_corr, psf=_beam, binid=binid,
                              grid_x=x, grid_y=y, grid_sb=sb, positive_definite=positive_definite)
        else:
            _kin = kin.copy()

        # Reinitialize
        self.reinit()

        # Initialize the fit parameters
        self._init_par(par, None)

        # Initialize the model
        #                                                                            is_fft
        self._init_model(par, _kin.grid_x, _kin.grid_y, _kin.grid_sb, _kin.beam_fft, True, cnvfftw,
        #                ignore_beam
                         False)

        # Initialize the data
        #                     scat, posdef, ignore_covar
        self._init_data(_kin, None, True, True)

        # Generate and bin the model data.  NOTE: Call above to _init_data sets
        # _kin to self.kin, which is used by binned_model...
        _kin.vel, _sig = self.binned_model(par)
        # If available, include the dispersion correction
        if _kin.sig_corr is not None:
            _sig = np.sqrt(_sig**2 + _kin.sig_corr**2)

        # Set whether or not to add Gaussian noise
        if not add_err or _vel_ivar is None and _vel_covar is None:
            _kin.update_sigma(sig=_sig)
            return _kin

        # Get the deviates
        sigma_method = 'ignore' if self.dc is None else 'draw' 
        vgpm, dv, sgpm, ds = _kin.deviate(sigma=sigma_method, rng=rng)
        # Add the velocity error
        _kin.vel[vgpm] += dv
        # Add the dispersion error and update the dispersion values
        if self.dc is not None:
            _sig[sgpm] += ds
            _kin.update_sigma(sig=_sig)

        return _kin

    def fisher_matrix(self, par, kin, fix=None, sb_wgt=False, scatter=None,
                      assume_posdef_covar=False, ignore_covar=True, cnvfftw=None,
                      inverse=False):
        r"""
        Construct the Fisher Information Matrix (FIM) at a specific position in
        parameter space for a set of observational constraints.

        The FIM is calculated using the derivatives of the fit metric with
        respect to the model parameters (the Jacobian), just as done during the
        least-squares minimization.  The shape of the Jacobian is :math:`(N,M)`,
        where :math:`N` is the number of data points and :math:`M` is the number
        of model parameters.  The FIM, :math:`\mathbf{I}(\theta)`, is the
        :math:`(M,M)` matrix defined as:

        .. math::

            \mathbf{I}(\theta) = \mathbf{J}^T \mathbf{J}

        Importantly, the Jacobian and the FIM are both *independent of the
        measurements*, but not independent of the statistical uncertainties in
        those measurements.

        .. warning::

            Currently, this class *does not construct a model of the
            surface-brightness distribution*.  Instead, any weighting of the
            model during convolution with the beam profile uses the as-observed
            surface-brightness distribution, instead of a model of the intrinsic
            surface brightness distribution.  See ``sb_wgt``.

        Args:
            par (array-like):
                The list of model parameters at which to calculate the FIM.
                I.e., the FIM is dependent on the position in parameter space.
                Length must be :attr:`np`.
            kin (:class:`~nirvana.data.kinematics.Kinematics`):
                Object with the kinematic data.  This provides the measurement
                uncertainties needed for the construction of the FIM, and it
                also provides the grid on which to compute the model, the
                surface-brightness weighting, the beam-smearing kernel, and the
                binning function.
            fix (`numpy.ndarray`_, optional):
                A boolean array selecting the parameters that are fixed.  These
                parameters are excluded from the FIM.
            sb_wgt (:obj:`bool`, optional):
                Flag to use the surface-brightness data provided by ``kin`` to
                weight the model when applying the beam-smearing.  **See the
                warning above**.
            scatter (:obj:`float`, array-like, optional):
                Introduce a fixed intrinsic-scatter term into the model. The 
                scatter is added in quadrature to all measurement errors in the
                calculation of the merit function. If no errors are available,
                this has the effect of renormalizing the unweighted merit
                function by 1/scatter.  Can be None, which means no intrinsic
                scatter is added.  If both velocity and velocity dispersion are
                being fit, this can be a single number applied to both datasets
                or a 2-element vector that provides different intrinsic scatter
                measurements for each kinematic moment (ordered velocity then
                velocity dispersion).
            assume_posdef_covar (:obj:`bool`, optional):
                If the :class:`~nirvana.data.kinematics.Kinematics` include
                covariance matrices, this forces the code to proceed assuming
                the matrices are positive definite.
            ignore_covar (:obj:`bool`, optional):
                If the :class:`~nirvana.data.kinematics.Kinematics` include
                covariance matrices, ignore them and just use the inverse
                variance.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
                An object that expedites the convolutions using FFTW/pyFFTW.  If
                provided, the shape *must* match ``kin.spatial_shape``.  If
                None, a new :class:`~nirvana.models.beam.ConvolveFFTW` instance
                is constructed to perform the convolutions.  If the class cannot
                be constructed because the user doesn't have pyfftw installed,
                then the convolutions fall back to numpy routines.
            inverse (:obj:`bool`, optional):
                Return the inverse of the FIM, which is equivalent to the
                covariance matrix in the model parameters.

        Returns:
            `numpy.ndarray`_: The :math:`(M,M)` Fisher Information Matrix, or
            its inverse, at the selected position in parameter space.
        """
        self._init_par(par, fix)
        self._init_model(par, kin.grid_x, kin.grid_y, kin.grid_sb if sb_wgt else None,
                         kin.beam_fft, True, cnvfftw, False)
        self._init_data(kin, scatter, assume_posdef_covar, ignore_covar)
#        self._fit_prep(kin, par, fix, scatter, sb_wgt, assume_posdef_covar, ignore_covar,
#                       cnvfftw)
        jac = self._get_jac()(self.par[self.free])
        return cov_err(jac) if inverse else np.dot(jac.T,jac)

    # This slew of "private" functions consolidate the velocity residual and
    # chi-square calculations
    def _v_resid(self, vel):
        return self.kin.vel[self.vel_gpm] - vel[self.vel_gpm]
    def _deriv_v_resid(self, dvel):
        return -dvel[np.ix_(self.vel_gpm, self.free)]
    def _v_chisqr(self, vel):
        return self._v_resid(vel) / self._v_err[self.vel_gpm]
    def _deriv_v_chisqr(self, dvel):
        return self._deriv_v_resid(dvel) / self._v_err[self.vel_gpm, None]
    def _v_chisqr_covar(self, vel):
        return np.dot(self._v_resid(vel), self._v_ucov)
    def _deriv_v_chisqr_covar(self, dvel):
        return np.dot(self._deriv_v_resid(dvel).T, self._v_ucov).T

    # This slew of "private" functions consolidate the velocity dispersion
    # residual and chi-square calculations
    def _s_resid(self, sig):
        return self.kin.sig_phys2[self.sig_gpm] - sig[self.sig_gpm]**2
    def _deriv_s_resid(self, sig, dsig):
        return -2 * sig[self.sig_gpm,None] * dsig[np.ix_(self.sig_gpm, self.free)]
    def _s_chisqr(self, sig):
        return self._s_resid(sig) / self._s_err[self.sig_gpm]
    def _deriv_s_chisqr(self, sig, dsig):
        return self._deriv_s_resid(sig, dsig) / self._s_err[self.sig_gpm, None]
    def _s_chisqr_covar(self, sig):
        return np.dot(self._s_resid(sig), self._s_ucov)
    def _deriv_s_chisqr_covar(self, sig, dsig):
        return np.dot(self._deriv_s_resid(sig, dsig).T, self._s_ucov).T

    def binned_model(self, par, ignore_beam=False):
        """
        Compute the binned model data.

        Args:
            par (`numpy.ndarray`_):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.
            ignore_beam (:obj:`bool`, optional):
                Ignore the beam-smearing when constructing the model. I.e.,
                construct the *intrinsic* model.

        Returns:
            :obj:`tuple`: Model velocity and velocity dispersion data.  The
            latter is None if the model has no dispersion parameterization.
        """
        self._set_par(par)
        if self.dc is None:
            vel = self.model(ignore_beam=ignore_beam)
            sig = None
        else:
            vel, sig = self.model(ignore_beam=ignore_beam)
        return self.kin.bin_moments(self.sb, vel, sig)[1:]

    def deriv_binned_model(self, par):
        """
        Compute the binned model data and its derivatives.

        Args:
            par (`numpy.ndarray`_):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.

        Returns:
            :obj:`tuple`: Model velocity data, velocity dispersion data,
            velocity derivative, and velocity dispersion derivative.  The
            velocity dispersion components (2nd and 4th objects) are None if the
            model has no dispersion parameterization.
        """
        self._set_par(par)
        if self.dc is None:
            vel, dvel = self.deriv_model()
            sig, dsig = None, None
        else:
            vel, sig, dvel, dsig = self.deriv_model()
        _, vel, sig, _, dvel, dsig \
                = self.kin.deriv_bin_moments(self.sb, vel, sig, None, dvel, dsig)
        return vel, sig, dvel, dsig

    def _resid(self, par, sep=False):
        """
        Calculate the residuals between the data and the current model.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.
            sep (:obj:`bool`, optional):
                Return separate vectors for the velocity and velocity
                dispersion residuals, instead of appending them.

        Returns:
            :obj:`tuple`, `numpy.ndarray`_: Difference between the data and the
            model for all measurements, either returned as a single vector for
            all data or as separate vectors for the velocity and velocity
            dispersion data (based on ``sep``).
        """
        vel, sig = self.binned_model(par)
        vfom = self._v_resid(vel)
        sfom = numpy.array([]) if self.dc is None else self._s_resid(sig)
        return (vfom, sfom) if sep else np.append(vfom, sfom)

    def _deriv_resid(self, par, sep=False):
        """
        Calculate the derivative of the fit residuals w.r.t. all the *free*
        model parameters.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.
            sep (:obj:`bool`, optional):
                Return separate vectors for the velocity and velocity dispersion
                residual derivatives, instead of appending them.

        Returns:
            :obj:`tuple`, `numpy.ndarray`_: Derivatives in the difference
            between the data and the model for all measurements, either returned
            as a single array for all data or as separate arrays for the
            velocity and velocity dispersion data (based on ``sep``).
        """
        vel, sig, dvel, dsig = self.deriv_binned_model(par)
        if self.dc is None:
            return (self._deriv_v_resid(dvel), numpy.array([])) \
                        if sep else self._deriv_v_resid(dvel)
        resid = (self._deriv_v_resid(vel), self._deriv_s_resid(sig, dsig))
        return resid if sep else np.vstack(resid)

    def _chisqr(self, par, sep=False):
        """
        Calculate the error-normalized residual (close to the signed
        chi-square metric) between the data and the current model.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.
            sep (:obj:`bool`, optional):
                Return separate vectors for the velocity and velocity
                dispersion residuals, instead of appending them.

        Returns:
            :obj:`tuple`, `numpy.ndarray`_: Difference between the data and the
            model for all measurements, normalized by their errors, either
            returned as a single vector for all data or as separate vectors for
            the velocity and velocity dispersion data (based on ``sep``).
        """
        vel, sig = self.binned_model(par)
        if self.has_covar:
            vfom = self._v_chisqr_covar(vel)
            sfom = np.array([]) if self.dc is None else self._s_chisqr_covar(sig)
        else:
            vfom = self._v_chisqr(vel)
            sfom = np.array([]) if self.dc is None else self._s_chisqr(sig)
        return (vfom, sfom) if sep else np.append(vfom, sfom)

    def _deriv_chisqr(self, par, sep=False):
        """
        Calculate the derivatives of the error-normalized residuals (close to
        the signed chi-square metric) w.r.t. the *free* model parameters.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length should be either
                :attr:`np` or :attr:`nfree`. If the latter, the values of the
                fixed parameters in :attr:`par` are used.
            sep (:obj:`bool`, optional):
                Return separate vectors for the velocity and velocity
                dispersion residuals, instead of appending them.

        Returns:
            :obj:`tuple`, `numpy.ndarray`_: Derivatives of the error-normalized
            difference between the data and the model for all measurements
            w.r.t. the *free* model parameters, either returned as a single
            array for all data or as separate arrays for the velocity and
            velocity dispersion data (based on ``sep``).
        """
        vel, sig, dvel, dsig = self.deriv_binned_model(par)
        vf = self._deriv_v_chisqr_covar if self.has_covar else self._deriv_v_chisqr
        if self.dc is None:
            return (vf(dvel), numpy.array([])) if sep else vf(dvel)

        sf = self._deriv_s_chisqr_covar if self.has_covar else self._deriv_s_chisqr
        dchisqr = (vf(dvel), sf(sig, dsig))
        if not np.all(np.isfinite(dchisqr[0])) or not np.all(np.isfinite(dchisqr[1])):
            raise ValueError('Error in derivative computation.')
        return dchisqr if sep else np.vstack(dchisqr)

    def _get_fom(self):
        """
        Return the figure-of-merit function to use given the availability of
        errors.
        """
        return self._chisqr if self.has_err or self.has_covar else self._resid

    def _get_jac(self):
        """
        Return the Jacobian function to use given the availability of errors.
        """
        return self._deriv_chisqr if self.has_err or self.has_covar else self._deriv_resid

    # TODO: Include an argument here that allows the PSF convolution to be
    # toggled, regardless of whether or not the `kin` object has the beam
    # defined.
    def lsq_fit(self, kin, sb_wgt=False, p0=None, fix=None, lb=None, ub=None, scatter=None,
                verbose=0, assume_posdef_covar=False, ignore_covar=True, cnvfftw=None,
                analytic_jac=True, maxiter=5):
        """
        Use `scipy.optimize.least_squares`_ to fit the model to the provided
        kinematics.

        It is possible that the call to `scipy.optimize.least_squares`_ returns
        fitted parameters that are identical to the input guess parameters.
        Because of this, the fit can be repeated multiple times (see
        ``maxiter``), where each attempt slightly peturbs the parameters.

        Once complete, the best-fitting parameters are saved to :attr:`par`
        and the parameter errors (estimated by the parameter covariance
        matrix constructed as a by-product of the least-squares fit) are
        saved to :attr:`par_err`.

        .. warning::

            Currently, this class *does not construct a model of the
            surface-brightness distribution*.  Instead, any weighting of the
            model during convolution with the beam profile uses the as-observed
            surface-brightness distribution, instead of a model of the intrinsic
            surface brightness distribution.  See ``sb_wgt``.

        Args:
            kin (:class:`~nirvana.data.kinematics.Kinematics`):
                Object with the kinematic data to fit.
            sb_wgt (:obj:`bool`, optional):
                Flag to use the surface-brightness data provided by ``kin`` to
                weight the model when applying the beam-smearing.  **See the
                warning above**.
            p0 (`numpy.ndarray`_, optional):
                The initial parameters for the model. Length must be
                :attr:`np`.
            fix (`numpy.ndarray`_, optional):
                A boolean array selecting the parameters that should be fixed
                during the model fit.
            lb (`numpy.ndarray`_, optional):
                The lower bounds for the parameters. If None, the defaults
                are used (see :func:`par_bounds`). The length of the vector
                must match the total number of parameters, even if some of
                the parameters are fixed.
            ub (`numpy.ndarray`_, optional):
                The upper bounds for the parameters. If None, the defaults
                are used (see :func:`par_bounds`). The length of the vector
                must match the total number of parameters, even if some of
                the parameters are fixed.
            scatter (:obj:`float`, array-like, optional):
                Introduce a fixed intrinsic-scatter term into the model. The 
                scatter is added in quadrature to all measurement errors in the
                calculation of the merit function. If no errors are available,
                this has the effect of renormalizing the unweighted merit
                function by 1/scatter.  Can be None, which means no intrinsic
                scatter is added.  If both velocity and velocity dispersion are
                being fit, this can be a single number applied to both datasets
                or a 2-element vector that provides different intrinsic scatter
                measurements for each kinematic moment (ordered velocity then
                velocity dispersion).
            verbose (:obj:`int`, optional):
                Verbosity level to pass to `scipy.optimize.least_squares`_.
            assume_posdef_covar (:obj:`bool`, optional):
                If the :class:`~nirvana.data.kinematics.Kinematics` includes
                covariance matrices, this forces the code to proceed assuming
                the matrices are positive definite.
            ignore_covar (:obj:`bool`, optional):
                If the :class:`~nirvana.data.kinematics.Kinematics` includes
                covariance matrices, ignore them and just use the inverse
                variance.
            cnvfftw (:class:`~nirvana.models.beam.ConvolveFFTW`, optional):
                An object that expedites the convolutions using FFTW/pyFFTW.  If
                provided, the shape *must* match ``kin.spatial_shape``.  If
                None, a new :class:`~nirvana.models.beam.ConvolveFFTW` instance
                is constructed to perform the convolutions.  If the class cannot
                be constructed because the user doesn't have pyfftw installed,
                then the convolutions fall back to the numpy routines.
            analytic_jac (:obj:`bool`, optional):
                Use the analytic calculation of the Jacobian matrix during the
                fit optimization.  If False, the Jacobian is calculated using
                finite-differencing methods provided by
                `scipy.optimize.least_squares`_.
            maxiter (:obj:`int`, optional):
                The call to `scipy.optimize.least_squares`_ is repeated when it
                returns best-fit parameters that are *identical* to the input
                parameters.  This parameter sets the maximum number of times the
                fit will be repeated.  Set this to 1 to ignore these occurences;
                ``maxiter`` cannot be None.
        """
        if maxiter is None:
            raise ValueError('Maximum number of iterations cannot be None.')

        # Prepare to fit the data.
        self._init_par(p0, fix)
        self._init_model(None, kin.grid_x, kin.grid_y, kin.grid_sb if sb_wgt else None,
                         kin.beam_fft, True, cnvfftw, False)
        self._init_data(kin, scatter, assume_posdef_covar, ignore_covar)
#        self._fit_prep(kin, p0, fix, scatter, sb_wgt, assume_posdef_covar, ignore_covar,
#                       cnvfftw)
        
        # Get the method used to generate the figure-of-merit and the Jacobian
        # matrix.
        fom = self._get_fom()
        # If the analytic Jacobian matrix is not used, the derivative of the
        # merit function wrt each parameter is determined by a 1% change in each
        # parameter.
        jac_kwargs = {'jac': self._get_jac()} if analytic_jac \
                        else {'diff_step': np.full(self.np, 0.01, dtype=float)[self.free]}

        # Parameter boundaries
        _lb, _ub = self.par_bounds()
        if lb is None:
            lb = _lb
        if ub is None:
            ub = _ub
        if len(lb) != self.np or len(ub) != self.np:
            raise ValueError('Length of one or both of the bound vectors is incorrect.')

        # Setup to iteratively fit, where the iterations are meant to ensure
        # that the least-squares fit actually optimizes the parameters.
        # - Set the random number generator with a fixed seed so that the
        #   result is deterministic.
        rng = np.random.default_rng(seed=909)
        # - Set the free parameters.  These are save to a new vector so that the
        #   initial parameters can be tracked for each iteration without losing
        #   the original input.
        _p0 = self.par[self.free]
        p = _p0.copy()
        # - Reset any parameter errors
        pe = None
        # - Start counting the iterations
        niter = 0
        while niter < maxiter:
            # Run the optimization
            result = optimize.least_squares(fom, p, # method='lm', #xtol=None,
                                            x_scale='jac', method='trf', xtol=1e-12,
                                            bounds=(lb[self.free], ub[self.free]), 
                                            verbose=max(verbose,0), **jac_kwargs)
            # Attempt to calculate the errors
            try:
                pe = np.sqrt(np.diag(cov_err(result.jac)))
            except:
                warnings.warn('Unable to compute parameter errors from precision matrix.')
                pe = None

            # The fit should change the input parameters.
            if np.all(np.absolute(p-result.x) > 1e-3):
                break

            # If it doesn't, something likely went wrong with the fit.  Perturb
            # the input guesses a bit and retry.
            p = _p0 + rng.normal(size=self.nfree)*(pe if pe is not None else 0.1*p0)
            p = np.clip(p, lb[self.free], ub[self.free])
            niter += 1

        # TODO: Add something to the fit status/success flags that tests if
        # niter == maxiter and/or if the input parameters are identical to the
        # final best-fit prameters?  Note that the input parameters, p0, may not
        # be identical to the output parameters because of the iterations mean
        # that p != p0 !

        # Save the fit status
        self.fit_status = result.status
        self.fit_success = result.success

        # Save the best-fitting parameters
        self._set_par(result.x)

        if pe is None:
            self.par_err = None
        else:
            self.par_err = np.zeros(self.np, dtype=float)
            self.par_err[self.free] = pe

        # Initialize the mask
        self.par_mask = self.pbm.init_mask_array(self.np)
        # Check if any parameters are "at" the boundary
        pm = self.par_mask[self.free]
        for v, flg in zip([-1, 1], ['LOWERBOUND', 'UPPERBOUND']):
            indx = result.active_mask == v
            if np.any(indx):
                pm[indx] = self.pbm.turn_on(pm[indx], flg)
        # Check if any parameters are within 1-sigma of the boundary
        indx = self.par[self.free] - self.par_err[self.free] < lb[self.free]
        if np.any(indx):
            pm[indx] = self.pbm.turn_on(pm[indx], 'LOWERERR')
        indx = self.par[self.free] + self.par_err[self.free] > ub[self.free]
        if np.any(indx):
            pm[indx] = self.pbm.turn_on(pm[indx], 'UPPERERR')
        # Flag the fixed parameters
        self.par_mask[self.free] = pm
        indx = np.logical_not(self.free)
        if np.any(indx):
            self.par_mask[indx] = self.pbm.turn_on(self.par_mask[indx], 'FIXED')

        # Print the report
        if verbose > -1:
            self.report(fit_message=result.message)

    def report(self, fit_message=None, intro=True):
        """
        Report the current parameters of the model to the screen.

        Args:
            fit_message (:obj:`str`, optional):
                The status message returned by the fit optimization.
            intro (:obj:`bool`, optional):
                Print the fit intro lines.
        """
        pass

    # TODO: This currently requires errors.  Allow for rejection without
    # errors...
    def reject(self, vel_sigma_rej=5, show_vel=False, vel_plot=None, sig_sigma_rej=5,
               show_sig=False, sig_plot=None, verbose=False, plots_only=False):
        """
        Reject kinematic data based on the error-weighted residuals with respect
        to the current disk.

        The method requires that the kinematic data already be ingested
        (:attr:`kin`) and that the model parameters have been set (:attr:`par`).

        The rejection iteration is done using
        :class:`~nirvana.data.scatter.IntrinsicScatter`, independently for the
        velocity and velocity dispersion measurements (if the latter is selected
        and/or available).

        Note that you can both show the QA plots and have them written to a file
        (e.g., ``show_vel`` can be True and ``vel_plot`` can provide a file).

        Args:
            vel_sigma_rej (:obj:`float`, optional):
                Rejection sigma for the velocity measurements.  If None, no data are
                rejected and the function basically just measures the intrinsic
                scatter.
            show_vel (:obj:`bool`, optional):
                Show the QA plot for the velocity rejection (see
                :func:`~nirvana.data.scatter.IntrinsicScatter.show`).
            vel_plot (:obj:`str`, optional):
                Write the QA plot for the velocity rejection to this file (see
                :func:`~nirvana.data.scatter.IntrinsicScatter.show`).
            sig_sigma_rej (:obj:`float`, optional):
                Rejection sigma for the dispersion measurements.  If None, no data
                are rejected and the function basically just measures the intrinsic
                scatter.
            show_sig (:obj:`bool`, optional):
                Show the QA plot for the velocity dispersion rejection (see
                :func:`~nirvana.data.scatter.IntrinsicScatter.show`).
            sig_plot (:obj:`str`, optional):
                Write the QA plot for the velocity dispersion rejection to this
                file (see :func:`~nirvana.data.scatter.IntrinsicScatter.show`).
            verbose (:obj:`bool`, optional):
                Verbose scatter fitting output.
            plots_only (:obj:`bool`, optional):
                Do not perform any additional rejection, only construct the
                plots.  The rejections are based on the existing rejections in
                :attr:`kin` and the scatter is set by :attr:`scatter`.

        Returns:
            :obj:`tuple`: Returns two pairs of objects, one for each kinematic
            moment. The first object is the vector flagging the data that should
            be rejected and the second is the estimated intrinsic scatter about
            the model. If the dispersion is not included in the rejection, the
            last two objects returned are both None.
        """

        # NOTE: BEWARE that this is repeating code in the _resid private
        # methods.  Make sure these stay consistent!

        # Get the models; this assumes the parameters are already set!
        models = self.binned_model(self.par)
        _verbose = 2 if verbose else 0

        # Reject based on error-weighted residuals, accounting for intrinsic
        # scatter
        vmod = models[0] if len(models) == 2 else models
        resid = self.kin.vel - vmod     # NOTE: This should be the same as _v_resid
        v_err_kwargs = {'covar': self.kin.vel_covar} if self.has_covar \
                            else {'err': np.sqrt(inverse(self.kin.vel_ivar))}
        scat = IntrinsicScatter(resid, gpm=self.vel_gpm, npar=self.nfree, **v_err_kwargs)
        if plots_only:
            scat.sig = 0. if self.scatter is None else self.scatter[0]
            scat.rej = np.zeros(resid.size, dtype=bool) \
                            if self.kin.vel_mask is None else self.kin.vel_mask.copy()
            vel_rej = None
            vel_sig = scat.sig
        else:
            vel_sig, vel_rej, vel_gpm \
                    = scat.iter_fit(sigma_rej=vel_sigma_rej, fititer=5, verbose=_verbose)
        # Show and/or plot the result, if requested
        if show_vel:
            scat.show()
        if vel_plot is not None:
            scat.show(ofile=vel_plot)

        if self.kin.sig is None or self.dc is None:
            # No dispersion data or model to use for rejection
            return vel_rej, vel_sig, None, None

        # Reject based on error-weighted residuals, accounting for intrinsic
        # scatter
        smod = models[1]
        resid = self.kin.sig_phys2 - smod**2    # NOTE: This should be the same as _s_resid
        sig_err_kwargs = {'covar': self.kin.sig_phys2_covar} if self.has_covar \
                            else {'err': np.sqrt(inverse(self.kin.sig_phys2_ivar))}
        scat = IntrinsicScatter(resid, gpm=self.sig_gpm, npar=self.nfree, **sig_err_kwargs)
        if plots_only:
            scat.sig = 0. if self.scatter is None else self.scatter[1]
            scat.rej = np.zeros(resid.size, dtype=bool) \
                            if self.kin.sig_mask is None else self.kin.sig_mask.copy()
            sig_rej = None
            sig_sig = scat.sig
        else:
            sig_sig, sig_rej, sig_gpm \
                    = scat.iter_fit(sigma_rej=sig_sigma_rej, fititer=5, verbose=_verbose)
        # Show and/or plot the result, if requested
        if show_sig:
            scat.show()
        if sig_plot is not None:
            scat.show(ofile=sig_plot)

        return vel_rej, vel_sig, sig_rej, sig_sig

