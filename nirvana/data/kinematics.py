"""
Implements base class to hold observational data fit by the kinematic
model.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""
import copy
from IPython import embed

import numpy as np
from scipy import sparse
from scipy import linalg
from astropy.stats import sigma_clip
from matplotlib import pyplot, rc, patches, ticker, colors
import warnings

from .bin2d import Bin2D
from .util import gaussian_deviates, growth_lim, impose_positive_definite
from .util import select_kinematic_axis, bin_stats, find_largest_coherent_region
from ..util import plot
from ..models import asymmetry
from ..models.beam import construct_beam, ConvolveFFTW, smear
from ..models.geometry import projected_polar, disk_ellipse


class Kinematics:
    r"""
    Base class to hold data fit by the kinematic model.

    All data to be fit by this package must be contained in a class
    that inherits from this one.

    On the coordinate grids: If the data are binned, the provided
    ``x`` and ``y`` arrays are assumed to be the coordinates of the
    unique bin measurements. I.e., all the array elements in the same
    bin should have the same ``x`` and ``y`` values. However, when
    modeling the data we need the coordinates of each grid cell, not
    the (irregular) binned grid coordinate. These are provided by the
    ``grid_x`` and ``grid_y`` arguments; these two arrays are
    *required* if ``binid`` is provided.

    Args:
        vel (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
            The velocity measurements of the kinematic tracer to be
            modeled.  Must be a square 2D array.
        vel_ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            Inverse variance of the velocity measurements. If None,
            all values are set to 1.
        vel_mask (`numpy.ndarray`_, optional):
            A boolean array with the bad-pixel mask (pixels to ignore
            have ``mask==True``) for the velocity measurements. If
            None, all pixels are considered valid. If ``vel`` is
            provided as a masked array, this mask is combined with
            ``vel.mask``.
        vel_covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_, optional):
            Covariance matrix for velocity measurements.  If the input map
            arrays have a total of :math:`N_{\rm spax}` spaxels, the shape of
            the covariance array must be :math:`(N_{\rm spax}, N_{\rm spax})`.
            If None, all spaxels are independent.
        x (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            The on-sky Cartesian :math:`x` coordinates of each
            velocity measurement. Units are irrelevant, but they
            should be consistent with any expectations of the fitted
            model. If None, ``x`` is just the array index, except
            that it is assumed to be sky-right (increasing from
            *large to small* array indices; aligned with right
            ascension coordinates). Also, the coordinates are offset
            such that ``x==0`` is at the center of the array and
            increase along the first axis of the velocity array.
        y (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            The on-sky Cartesian :math:`y` coordinates of each
            velocity measurement. Units are irrelevant, but they
            should be consistent with any expectations of the fitted
            model. If None, ``y`` is just the array index, offset
            such that ``y==0`` is at the center of the array and
            increase along the second axis of the velocity array.
        sb (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            The observed surface brightness of the kinematic tracer.
            Ignored if None.
        sb_ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            Inverse variance of the surface-brightness measurements.
            If None and ``sb`` is provided, all values are set to 1.
        sb_mask (`numpy.ndarray`_, optional):
            A boolean array with the bad-pixel mask (pixels to ignore
            have ``mask==True``) for the surface-brightness
            measurements. If None, all pixels are considered valid.
        sb_covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_, optional):
            Covariance matrix for surface-brightness measurements.  If the input
            map arrays have a total of :math:`N_{\rm spax}` spaxels, the shape
            of the covariance array must be :math:`(N_{\rm spax}, N_{\rm
            spax})`.  If None, all spaxels are independent.
        sb_anr (`numpy.ndarray`_, optional):
            For emission-line data, this is the amplitude-to-noise ratio.
            Ignored if not provided.
        sig (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            The velocity dispersion of the kinematic tracer. Ignored
            if None.
        sig_ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_, optional):
            Inverse variance of the velocity dispersion measurements.
            If None and ``sig`` is provided, all values are set to 1.
        sig_mask (`numpy.ndarray`_, optional):
            A boolean array with the bad-pixel mask (pixels to ignore
            have ``mask==True``) for the velocity-dispersion
            measurements. If None, all measurements are considered
            valid.
        sig_covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_, optional):
            Covariance matrix for velocity-dispersion measurements.  If the
            input map arrays have a total of :math:`N_{\rm spax}` spaxels, the
            shape of the covariance array must be :math:`(N_{\rm spax}, N_{\rm
            spax})`.  If None, all spaxels are independent.
        sig_corr (`numpy.ndarray`_, optional):
            A quadrature correction for the velocity dispersion
            measurements. If None, velocity dispersions are assumed
            to be the *astrophysical* Doppler broadening of the
            kinematic tracer. If provided, the corrected velocity
            dispersion is:

            .. math::

                \sigma^2 = \sigma_{\rm obs}^2 - \sigma_{\rm corr}^2

            where :math:`\sigma_{\rm obs}` is provided by ``sig``.
        psf_name (:obj:`str`, optional):
            Identifier for the psf used. For example, this can be the
            wavelength band where the PSF was measured. If provided, this
            identifier is only used for informational purposes in output
            files.
        psf (`numpy.ndarray`_, optional):
            An image of the point-spread function of the
            observations. If ``aperture`` is not provided, this
            should be the effective smoothing kernel for the
            kinematic fields. Otherwise, this is the on-sky seeing
            kernel and the effective smoothing kernel is constructed
            as the convolution of this image with ``aperture``. If
            None, any smearing of the kinematic data is ignored.
            Shape must match ``vel`` and the extent of the PSF map
            must identically match ``vel``.
        aperture (`numpy.ndarray`_, optional):
            Monochromatic image of the spectrograph aperture. See
            ``psf`` for how this is used.
        binid (`numpy.ndarray`_, optional):
            Integer array associating each measurement with a unique
            bin number. Measurements not associated with any bin
            should have a value of -1 in this array. If None, all
            (unmasked) measurements are considered unique.
        grid_x (`numpy.ndarray`_, optional):
            The on-sky Cartesian :math:`x` coordinates of *each*
            element in the data grid. If the data are unbinned, this
            array is identical to `x` (except that *every* value
            should be valid). This argument is *required* if
            ``binid`` is provided.
        grid_y (`numpy.ndarray`_, optional):
            The on-sky Cartesian :math:`y` coordinates of *each*
            element in the data grid. See the description of
            ``grid_x``.
        grid_sb (`numpy.ndarray`_, optional):
            The relative surface brightness of the kinematic tracer over the
            full coordinate grid.  If None, this is either assumed to be unity
            or set by the provided ``sb``.  When fitting the data with, e.g., 
            :class:`~nirvana.model.axisym.AxisymmetricDisk` via the ``sb_wgt``
            parameter in its fitting method, this will be the weighting used.
            The relevance of this array is to enable the weighting used in
            constructing the model velocity field to be *unbinned* for otherwise
            binned kinematic data.
        grid_wcs (`astropy.wcs.WCS`_, optional):
            World coordinate system for the on-sky grid. Currently, this is
            only used for output files.
        reff (:obj:`float`, optional):
            Effective radius in same units as :attr:`x` and :attr:`y`.
        fwhm (:obj:`float`, optional):
            The FWHM of the PSF of the galaxy in the same units as :attr:`x` and
            :attr:`y`.
        image (`numpy.ndarray`_, optional):
            Galaxy image, typically from a PNG, and only used for plotting.  For
            format, see matplotlib.images.imread.
        phot_inc (:obj:`float`, optional):
            Photometric inclination in degrees.
        phot_pa (:obj:`float`, optional):
            Photometric position angle in degrees.
        maxr (:obj:`float`, optional):
            Maximum radius of useful data in effective radii.
        positive_definite (:obj:`bool`, optional):
            Use :func:`~nirvana.data.util.impose_positive_definite` to force
            the provided covariance matrix to be positive definite.
        quiet (:obj:`bool`, optional):
            Suppress message output.

    Raises:
        ValueError:
            Raised if the input arrays are not 2D or square, if any
            of the arrays do not match the shape of ``vel``, if
            either ``x`` or ``y`` is provided but not both or
            neither, or if ``binid`` is provided but ``grid_x`` or
            ``grid_y`` is None.
    """

    tracer = None
    """
    Tracer designation.  Undefined by default, but can be set for derived classes.
    """

    # TODO: We should change sb_anr to be just a generic S/N ratio.  I.e.,
    # Kinematics shouldn't care about the type of tracer.
    # TODO: Remove arguments that are redundant with the GlobalPar class?
    def __init__(self, vel, vel_ivar=None, vel_mask=None, vel_covar=None, x=None, y=None, sb=None,
                 sb_ivar=None, sb_mask=None, sb_covar=None, sb_anr=None, sig=None, sig_ivar=None,
                 sig_mask=None, sig_covar=None, sig_corr=None, psf_name=None, psf=None,
                 aperture=None, binid=None, grid_x=None, grid_y=None, grid_sb=None, grid_wcs=None,
                 reff=None, fwhm=None, image=None, phot_inc=None, phot_pa=None, maxr=None,
                 positive_definite=False, quiet=False):

        # Check shape of input arrays
        self.nimg = vel.shape[0]
        if len(vel.shape) != 2:
            raise ValueError('Input arrays to Kinematics must be 2D.')
        # TODO: I don't remember why we have this restriction (maybe it was
        # just because I didn't want to have to worry about having to
        # accommodate anything but MaNGA kinematic fields yet), but we should
        # look to get rid of this constraint of a square map.
        if vel.shape[1] != self.nimg:
            raise ValueError('Input arrays to Kinematics must be square.')
        for a in [vel_ivar, vel_mask, x, y, sb, sb_ivar, sb_mask, sig, sig_ivar, sig_mask,
                  sig_corr, psf, aperture, binid, grid_x, grid_y, grid_sb]:
            if a is not None and a.shape != vel.shape:
                raise ValueError('All arrays provided to Kinematics must have the same shape.')
        if (x is None and y is not None) or (x is not None and y is None):
            raise ValueError('Must provide both x and y or neither.')
        if binid is not None and grid_x is None or grid_y is None:
            raise ValueError('If the data are binned, you must provide the pixel-by-pixel input '
                             'coordinate grids, grid_x and grid_y.')

        # Basic properties
        self.spatial_shape = vel.shape
        self.psf_name = 'unknown' if psf_name is None else psf_name
        self._set_beam(psf, aperture)
        self.reff = reff
        self.fwhm = fwhm
        self.image = image
        self.sb_anr = sb_anr
        self.phot_inc = phot_inc
        self.phot_pa = phot_pa
        self.maxr = maxr

        # Build coordinate arrays
        if x is None:
            # No coordinate arrays provided, so just assume a
            # coordinate system with 0 at the center. Ensure that
            # coordinates mimic being "sky-right" (i.e., x increases
            # toward lower pixel indices).
            self.x, self.y = map(lambda x : x - self.nimg//2,
                                 np.meshgrid(np.arange(self.nimg)[::-1], np.arange(self.nimg)))
        else:
            self.x, self.y = x, y

        # Build map data
        self.sb, self.sb_ivar, self.sb_mask = self._ingest(sb, sb_ivar, sb_mask)
        self.vel, self.vel_ivar, self.vel_mask = self._ingest(vel, vel_ivar, vel_mask)
        self.sig, self.sig_ivar, self.sig_mask = self._ingest(sig, sig_ivar, sig_mask)
        # Have to treat sig_corr separately
        if isinstance(sig_corr, np.ma.MaskedArray):
            self.sig_mask |= np.ma.getmaskarray(sig_corr)
            self.sig_corr = sig_corr.data
        else:
            self.sig_corr = sig_corr

        # The following arrays and Bin2D object are used to convert between
        # arrays holding the data for the unique bins to arrays with the full
        # data map.
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_wcs = grid_wcs
        self.binner = Bin2D(binid=binid)

        # Unravel and select the valid values for all arrays
        for attr in ['x', 'y', 'sb', 'sb_ivar', 'sb_mask', 'vel', 'vel_ivar', 'vel_mask', 'sig', 
                     'sig_ivar', 'sig_mask', 'sig_corr', 'sb_anr']:
            if getattr(self, attr) is not None:
                setattr(self, attr, self.binner.unique(getattr(self, attr)))

        # Set the surface-brightness grid.  This needs to be after the
        # unraveling of the attributes done in the lines above so that I can use
        # self.remap in the case that grid_sb is not provided directly.
        self.grid_sb = self.remap('sb').filled(0.0) if grid_sb is None else grid_sb

        # Ingest the covariance matrices, if they're provided
        self.vel_covar = self._ingest_covar(vel_covar, positive_definite=positive_definite)
        self.sb_covar = self._ingest_covar(sb_covar, positive_definite=False) #positive_definite)
        self.sig_covar = self._ingest_covar(sig_covar, positive_definite=positive_definite)

        # TODO: Should issue a some warning/error if the user has provided both
        # ivar and covar and they are not consistent
        self.update_sigma()

    # TODO: Adding these properties was a way of avoiding crashes in other parts
    # of the code, but we should figure out which of these we actually need?
    @property
    def nspax(self):
        return self.binner.nbin

    @property
    def binid(self):
        return self.binner.ubinid

    @property
    def nbin(self):
        return self.binner.nbin.size

    def _set_beam(self, psf, aperture):
        """
        Instantiate :attr:`beam` and :attr:`beam_fft`.

        If both ``psf`` and ``aperture`` are None, the convolution
        kernel for the data is assumed to be unknown.

        Args:
            psf (`numpy.ndarray`_):
                An image of the point-spread function of the
                observations. If ``aperture`` is None, this should be
                the effective smoothing kernel for the kinematic
                fields. Otherwise, this is the on-sky seeing kernel
                and the effective smoothing kernel is constructed as
                the convolution of this image with ``aperture``. If
                None, the kernel will be set to the ``aperture``
                value (if provided) or None.
            aperture (`numpy.ndarray`_):
                Monochromatic image of the spectrograph aperture. If
                ``psf`` is None, this should be the effective
                smoothing kernel for the kinematic fields. Otherwise,
                this is the on-sky representation of the spectrograph
                aperture and the effective smoothing kernel is
                constructed as the convolution of this image with
                ``psf``. If None, the kernel will be set to the
                ``psf`` value (if provided) or None.
        """
        if psf is None and aperture is None:
            self.beam = None
            self.beam_fft = None
            return
        if psf is None:
            self.beam = aperture/np.sum(aperture)
            self.beam_fft = np.fft.fftn(np.fft.ifftshift(aperture))
            return
        if aperture is None:
            self.beam = psf/np.sum(psf)
            self.beam_fft = np.fft.fftn(np.fft.ifftshift(psf))
            return
        self.beam_fft = construct_beam(psf/np.sum(psf), aperture/np.sum(aperture), return_fft=True)
        self.beam = np.fft.fftshift(np.fft.ifftn(self.beam_fft).real)

    def _ingest(self, data, ivar, mask):
        """
        Check the data for ingestion into the object.

        Args:
            data (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
                Kinematic measurements. Can be None.
            ivar (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
                Inverse variance in the kinematic measurements.
                Regardless of the input, any pixel with an inverse
                variance that is not greater than 0 is automatically
                masked.
            mask (`numpy.ndarray`_):
                A boolean bad-pixel mask (i.e., values to ignore are
                set to True). This is the baseline mask that is
                combined with any masks provide by ``data`` and
                ``ivar`` if either are provided as
                `numpy.ma.MaskedArray`_ objects. The returned mask
                also automatically masks any bad inverse-variance
                values. If None, the baseline mask is set to be False
                for all pixels.

        Returns:
            :obj:`tuple`: Return three `numpy.ndarray`_ objects with the
            ingested data, inverse variance, and boolean mask. The first two
            arrays are forced to have type ``numpy.float64``.
        """
        if data is None:
            # No data, so do nothing
            return None, None, None

        # Initialize the mask
        _mask = np.zeros(self.spatial_shape, dtype=bool) if mask is None else mask.copy()

        # Set the data and incorporate the mask for a masked array
        if isinstance(data, np.ma.MaskedArray):
            _mask |= np.ma.getmaskarray(data)
            _data = data.data.astype(np.float64)
        else:
            _data = data.astype(np.float64)

        # Set the error and incorporate the mask for a masked array
        if ivar is None:
            # Don't instantiate the array if we don't need to.
            _ivar = None
        elif isinstance(ivar, np.ma.MaskedArray):
            _mask |= np.ma.getmaskarray(ivar)
            _ivar = ivar.data.astype(np.float64)
        else:
            _ivar = ivar.astype(np.float64)
        # Make sure to mask any measurement with ivar <= 0
        if _ivar is not None:
            _mask |= np.logical_not(_ivar > 0)

        return _data, _ivar, _mask

    def _ingest_covar(self, covar, mask=None, positive_definite=True, quiet=False):
        """
        Ingest an input covariance matrix for use when fitting the data.

        The covariance matrix is forced to be positive everywhere (any negative
        values are set to 0) and to be identically symmetric.  

        Args:
            covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance matrix. It's shape must match the input map shape.
                If None, the returned value is also None.
            positive_definite (:obj:`bool`, optional):
                Use :func:`~nirvana.data.util.impose_positive_definite` to force
                the provided covariance matrix to be positive definite.
            quiet (:obj:`bool`, optional):
                Suppress output to stdout.

        Returns:
            `scipy.sparse.csr_matrix`_: The covariance matrix of the good bin
            values.
        """
        if covar is None:
            return None

        if not quiet:
            print('Ingesting covariance matrix ... ')

        nspax = np.prod(self.spatial_shape)
        if covar.shape != (nspax,nspax):
            raise ValueError('Input covariance matrix has incorrect shape: {0}'.format(covar.shape))

        _covar = covar.copy() if isinstance(covar, sparse.csr.csr_matrix) \
                    else sparse.csr_matrix(covar)

        # It should be the case that, on input, the covariance matrix should
        # demonstrate that map values in the same bin are perfectly correlated.
        # The matrix operation below constructs the covariance in the *binned*
        # data, but you should also be able to obtain this just by selecting the
        # appropriate rows/columns of the covariance matrix. You should be able
        # to recover the input covariance matrix (or at least the populated
        # regions of it) like so:

        #   # Covariance matrix of the binned data
        #   vc = self.bin_transform.dot(vel_covar.dot(self.bin_transform.T))
        #   # Revert
        #   gpm = np.logical_not(vel_mask)
        #   _bt = self.bin_transform[:,gpm.ravel()].T.copy()
        #   _bt[_bt > 0] = 1.
        #   ivc = _bt.dot(vc.dot(_bt.T))
        #   assert np.allclose(ivc.toarray(),
        #                      vel_covar[np.ix_(gpm.ravel(), gpm.ravel())].toarray())

        _covar = self.binner.bin_covar(_covar)

        # Deal with possible numerical error
        # - Force it to be positive
        _covar[_covar < 0] = 0.
        # - Force it to be identically symmetric
        _covar = (_covar + _covar.T)/2
        # - Force it to be positive definite if requested
        return impose_positive_definite(_covar) if positive_definite else _covar

    def copy(self):
        """
        Perform a *deep* copy of the instance.

        .. warning::
            
            I'm generally wary of ``deepcopy``, but using it is so much faster
            than trying to instantiate a new object using the internal data, and
            it avoids numerical error for the covariance matrix, which can lead
            to them not being positive definite.

        """
        return copy.deepcopy(self)

    def update_sigma(self, sig=None, sig_ivar=None, sig_covar=None, sqr=False):
        """
        Update both the measured and corrected velocity dispersions and errors.

        .. warning::

            This method primarily exists for creating/handling mock
            observations.  It should should never be called with real data once
            the object is instantiated.

        Args:
            sig (`numpy.ndarray`_, optional):
                New velocity dispersion values.  If None, the velocity
                dispersions remain unchanged.  Shape must match the current
                number of bins (:attr:`nbin`).
            sig_ivar (`numpy.ndarray`_, optional):
                New velocity dispersion inverse variance.  If None, the inverse
                variance remain unchanged.  Shape must match the current number
                of bins (:attr:`nbin`).
            sig_covar (`scipy.sparse.csr_matrix`_, optional):
                New velocity dispersion covariance.  If None, the covariance
                remains unchanged.  Shape must match the current number of bins
                (:attr:`nbin`) along each axis.
            sqr (:obj:`bool`, optional):
                The provided values are for sigma-square, not sigma
        """
        if sig is not None:
            if sig.size != self.nbin:
                raise ValueError('Provided velocity dispersion has incorrect shape!')
            if sqr:
                self.sig_phys2 = sig.ravel().copy()
            else:
                self.sig = sig.ravel().copy()
        if sig_ivar is not None:
            if sig_ivar.size != self.nbin:
                raise ValueError('Provided dispersion inv. variance has incorrect shape!')
            if sqr:
                self.sig_phys2_ivar = sig_ivar.ravel().copy()
            else:
                self.sig_ivar = sig_ivar.ravel().copy()
        if sig_covar is not None:
            if sig_covar.shape != (self.nbin,self.nbin):
                raise ValueError('Provided dispersion covariance has incorrect shape!')
            if not isinstance(sig_covar, sparse.csr_matrix):
                raise TypeError('Covariance matrix must have type scipy.sparse.csr_matrix')
            if sqr:
                self.sig_phys2_covar = sig_covar.copy()
            else:
                self.sig_covar = sig_covar.copy()

        if sqr:
            # The provided data is actually of sig_phys2, not observed sigma.
            if self.sig_phys2 is None:
                self.sig = None
                self.sig_ivar = None
                self.sig_covar = None
                return

            # Calculate the observed sigma
            self.sig_mask |= self.sig_phys2 < 0.
            self.sig = np.ma.sqrt(self.sig_phys2 if self.sig_corr is None \
                                    else self.sig_phys2 + self.sig_corr**2).filled(0.0)
            # Its inverse variance
            self.sig_ivar = 4 * self.sig**2 * self.sig_phys2_ivar
            # And its covariance, if available
            if self.sig_phys2_covar is None:
                self.sig_covar = None
            else:
                jac = sparse.diags(1/(2*self.sig + (self.sig == 0.0))**2, format='csr')
                self.sig_covar = jac.dot(self.sig_phys2_covar.dot(jac.T))
            return

        if self.sig is None:
            self.sig_phys2 = None
            self.sig_phys2_ivar = None
            self.sig_phys2_covar = None
            return

        # Calculate the square of the astrophysical velocity dispersion. This is
        # just the square of the velocity dispersion if no correction is
        # provided. The error calculation assumes there is no error on the
        # correction.
        # TODO: Change this to sig2 or sigsqr
        # TODO: Need to keep track of mask...
        self.sig_phys2 = self.sig**2 if self.sig_corr is None else self.sig**2 - self.sig_corr**2
        self.sig_phys2_ivar = None if self.sig_ivar is None \
                                    else self.sig_ivar/(2*self.sig + (self.sig == 0.0))**2

        # Construct the covariance in the square of the astrophysical velocity
        # dispersion.
        if self.sig_covar is None:
            self.sig_phys2_covar = None
        else:
            jac = sparse.diags(2*self.sig, format='csr')
            self.sig_phys2_covar = jac.dot(self.sig_covar.dot(jac.T))

    def remap(self, data, mask=None, masked=True, fill_value=0):
        """
        Remap the requested attribute to the full 2D array.

        Args:
            data (`numpy.ndarray`_, :obj:`str`):
                The data or attribute to remap. If the object is a
                string, the string must be a valid attribute.
            mask (`numpy.ndarray`_, optional):
                Boolean mask with the same shape as ``data`` or the selected
                ``data`` attribute. If ``data`` is provided as a
                `numpy.ndarray`_, this provides an associated mask. If
                ``data`` is provided as a string, this is a mask is used *in
                addition to* any mask associated with selected attribute.
                Ignored if set to None.
            masked (:obj:`bool`, optional):
                Return data as a masked array, where data that are not filled
                by the provided data. If ``data`` is a string selecting an
                attribute and an associated mask exists for that attribute
                (called "{data}_mask"), also include the mask in the output.
            fill_value (scalar-like, optional):
                Value used to fill the masked pixels, if a masked array is
                *not* requested. Warning: The value is automatically
                converted to be the same data type as the input array or
                attribute.

        Returns:
            `numpy.ndarray`_, `numpy.ma.MaskedArray`_: 2D array with
            the attribute remapped to the original on-sky locations.

        Raises:
            ValueError:
                Raised if ``data`` is a `numpy.ndarray`_ and the
                shape does not match the expected 1d shape.
            AttributeError:
                Raised if ``data`` is a string and the requested
                attribute is invalid.

        """
        if isinstance(data, str):
            # User attempting to select an attribute. First check it exists.
            if not hasattr(self, data):
                raise AttributeError('No attribute called {0}.'.format(data))
            # Get the data
            d = getattr(self, data)
            if d is None:
                # There is no data, so just return None
                return None
            # Try to find the mask
            m = '{0}_mask'.format(data)
            if not masked or not hasattr(self, m) or getattr(self, m) is None:
                # If there user doesn't want the mask, there is no mask, or the
                # mask is None, ignore it
                m = None if mask is None else mask
            else:
                # Otherwise, get it
                m = getattr(self, m)
                if mask is not None:
                    m |= mask
        else:
            # User provided arrays directly
            d = data
            m = mask

        return self.binner.remap(d, mask=m, masked=masked, fill_value=fill_value)

    def remap_covar(self, covar):
        """
        Remap a covariance matrix from the binned data to the individual
        spaxels.

        Spaxels in the same bin are perfectly correlated.

        This is a simple wrapper for :func:`~nirvana.util.bin2d.Bin2D.remap_covar` using
        the :class:`~nirvana.util.bin2d.Bin2D` instance :attr:`binner`.

        Args:
            covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_, :obj:`str`):
                Covariance matrix to remap.  If a string is provided, it should
                directly map to one of the class attributes (e.g.,
                ``'vel_covar'``).

        Returns:
            `scipy.sparse.csr_matrix`_: A sparse matrix for the remapped
            covariance.
        """
        _covar = getattr(self, covar) if isinstance(covar, str) else covar
        return None if _covar is None else self.binner.remap_covar(_covar)

    def bin(self, data):
        """
        Provided a set of mapped data, rebin it to match the internal vectors.

        This is a simple wrapper for :func:`~nirvana.util.bin2d.Bin2D.bin` using
        the :class:`~nirvana.util.bin2d.Bin2D` instance :attr:`binner`.
        """
        return self.binner.bin(data)

    def deriv_bin(self, data, deriv):
        """
        Provided a set of mapped data, rebin it to match the internal vectors
        and propagate the derivatives in the data.

        This is a simple wrapper for :func:`~nirvana.util.bin2d.Bin2D.deriv_bin`
        using the :class:`~nirvana.util.bin2d.Bin2D` instance :attr:`binner`.
        """
        return self.binner.deriv_bin(data, deriv)

    # TODO: Include error calculations?
    def bin_moments(self, norm, center, stddev):
        r"""
        Bin a set of Gaussian moments.

        Assuming the provided data are the normalization, mean, and standard
        deviation of a set of Gaussian profiles, this method performs a nominal
        calculation of the moments of the summed Gaussian profile.

        This is a simple wrapper for
        :func:`~nirvana.util.bin2d.Bin2D.bin_moments` using the
        :class:`~nirvana.util.bin2d.Bin2D` instance :attr:`binner`.

        .. note::

            Any of the input arguments can be None, but at least one of them
            cannot be!

        Args:
            norm (`numpy.ndarray`_):
                Gaussian normalization.  Shape must match :attr:`spatial_shape`.
            center (`numpy.ndarray`_):
                Gaussian center.  Shape must match :attr:`spatial_shape`.
            stddev (`numpy.ndarray`_, optional):
                Gaussian standard deviation.  Shape must match
                :attr:`spatial_shape`.

        Returns:
            :obj:`tuple`: A tuple of three `numpy.ndarray`_ objects with the
            binned normalization, mean, and standard deviation of the summed
            profile.  If ``norm`` is None on input, the returned 0th moment is 1
            everywhere.  If ``center`` is None on input, the returned 1st moment
            is 0 everywhere.  If ``stddev`` is None on input, the returned 2nd
            moment is 1 everywhere.
        """
        return self.binner.bin_moments(norm, center, stddev)

    def deriv_bin_moments(self, norm, center, stddev, dnorm, dcenter, dstddev):
        r"""
        Bin a set of Gaussian moments and propagate the calculation for the
        derivatives.

        This method is identical to :func:`bin_moments`, except it includes the
        propagation of the derivatives.

        .. note::

            Any of the first three input arguments can be None, but at least one of them
            cannot be!

        This is a simple wrapper for
        :func:`~nirvana.util.bin2d.Bin2D.deriv_bin_moments` using the
        :class:`~nirvana.util.bin2d.Bin2D` instance :attr:`binner`.

        Args:
            norm (`numpy.ndarray`_):
                Gaussian normalization.  Shape must match :attr:`spatial_shape`.
                Can be None.
            center (`numpy.ndarray`_):
                Gaussian center.  Shape must match :attr:`spatial_shape`.  Can
                be None.
            stddev (`numpy.ndarray`_, optional):
                Gaussian standard deviation.  Shape must match
                :attr:`spatial_shape`.  Can be None.
            dnorm (`numpy.ndarray`_):
                Derivative of the Gaussian normalization with respect to model
                parameters.  Shape of the first two axes must match
                :attr:`spatial_shape`.  Can be None.
            dcenter (`numpy.ndarray`_):
                Derivative of the Gaussian center with respect to model
                parameters.  Shape of the first two axes must match
                :attr:`spatial_shape`.  Can be None.
            dstddev (`numpy.ndarray`_, optional):
                Derivative of the Gaussian standard deviation with respect to
                model parameters.  Shape of the first two axes must match
                :attr:`spatial_shape`.  Can be None.

        Returns:
            :obj:`tuple`: A tuple of three `numpy.ndarray`_ objects with the
            binned normalization, mean, and standard deviation of the summed
            profile.  If ``norm`` is None on input, the returned 0th moment is 1
            everywhere.  If ``center`` is None on input, the returned 1st moment
            is 0 everywhere.  If ``stddev`` is None on input, the returned 2nd
            moment is 1 everywhere.
        """
        return self.binner.deriv_bin_moments(norm, center, stddev, dnorm, dcenter, dstddev)

    def unique(self, data):
        """
        Provided a 2D array of binned data, select and return the unique values
        from the map.

        This is a simple wrapper for :func:`~nirvana.util.bin2d.Bin2D.unique`
        using the :class:`~nirvana.util.bin2d.Bin2D` instance :attr:`binner`.
        """
        return self.binner.unique(data)

    def max_radius(self):
        """
        Calculate and return the maximum *on-sky* radius of the valid data.
        Note this not the in-plane disk radius; however, the two are the same
        along the major axis.
        """
        minx = np.amin(self.x)
        maxx = np.amax(self.x)
        miny = np.amin(self.y)
        maxy = np.amax(self.y)
        return np.sqrt(max(abs(minx), maxx)**2 + max(abs(miny), maxy)**2)

    def reject(self, vel_rej=None, sig_rej=None):
        r"""
        Reject/Mask data.

        This is a simple wrapper that incorporates the provided vectors into
        the kinematic masks.

        Args:
            vel_rej (`numpy.ndarray`_, optional):
                Boolean vector selecting the velocity measurements to reject.
                Shape must be :math:`N_{\rm bin}`. If None, no additional
                data are rejected.
            sig_rej (`numpy.ndarray`_, optional):
                Boolean vector selecting the velocity dispersion measurements
                to reject. Shape must be :math:`N_{\rm bin}`. If None, no
                additional data are rejected. Ignored if :attr:`sig` is None.
        """
        if vel_rej is not None:
            self.vel_mask |= vel_rej
        if self.sig is not None and sig_rej is not None:
            self.sig_mask |= sig_rej

    def remask(self, mask):
        '''
        Apply a given mask to the masks that are already in the object.

        Args:
            mask (`numpy.ndarray`):
                Mask to apply to the data. Should be the same shape as the
                data (either 1D binned or 2D). Will be interpreted as boolean.

        Raises:
            ValueError:
                Thrown if input mask is not the same shape as the data.
        '''

        if mask.ndim > 1 and mask.shape != self.spatial_shape:
            raise ValueError('Mask is not the same shape as data.')
        if mask.ndim == 1 and len(mask) != len(self.vel):
            raise ValueError('Mask is not the same length as data')

        for m in ['sb_mask', 'vel_mask', 'sig_mask']:
            if m is None: continue
            if mask.ndim > 1: mask = self.bin(mask)
            setattr(self, m, np.array(getattr(self, m) + mask, dtype=bool))

    def clip_err(self, max_vel_err=None, max_sig_err=None, save=True):
        """
        Reject data with large errors.

        By default, the rejection is propagated to :attr:`vel_mask` and
        :attr:`sig_mask`; set ``save`` to False to force these attributes to
        remain unaltered.

        Args:
            max_vel_err (:obj:`float`, optional):
                Maximum allowed velocity error. If None, no additional
                masking is performed.
            max_sig_err (:obj:`float`, optional):
                Maximum allowed *observed* velocity dispersion error. I.e.,
                this is the measurement error before any velocity dispersion
                correction.  If None, no additional masking is performed.
            save (:obj:`bool`, optional):
                Propagate the rejection flags to the internal masks.  If False,
                :attr:`vel_mask` and :attr:`sig_mask` remain unaltered.

        Returns:
            :obj:`tuple`: Two objects are returned selecting the data to be If
            :attr:`sig` is None, the returned object selecting the velocity
            dispersion data that were rejected is also None.
        """
        vel_rej = np.zeros(self.vel.size, dtype=bool) if max_vel_err is None else \
                    self.vel_ivar < 1/max_vel_err**2
        sig_rej = None if self.sig is None else \
                    (np.zeros(self.sig.size, dtype=bool) if max_sig_err is None else
                     self.sig_ivar < 1/max_sig_err**2)
        if save:
            self.reject(vel_rej=vel_rej, sig_rej=sig_rej)
        return vel_rej, sig_rej

    # TODO: Include a separate S/N measurement, like as done with A/N for the
    # gas.
    def clip_snr(self, min_vel_snr=None, min_sig_snr=None, save=True):
        """
        Reject data with low S/N.

        The S/N of a given spaxel is given by the ratio of its surface
        brightness to the error in the surface brightness. An exception is
        raised if the surface-brightness or surface-brightness error are not
        defined.

        By default, the rejection is directly incorporated into :attr:`vel_mask`
        and :attr:`sig_mask`; set ``save`` to False to force these attributes to
        remain unaltered.

        Args:
            min_vel_snr (:obj:`float`, optional):
                Minimum S/N for a spaxel to use for velocity measurements. If
                None, no additional masking is performed.
            min_sig_snr (:obj:`float`, optional):
                Minimum S/N for a spaxel to use for dispersion measurements.
                If None, no additional masking is performed.
            save (:obj:`bool`, optional):
                Propagate the rejection flags to the internal masks.  If False,
                :attr:`vel_mask` and :attr:`sig_mask` remain unaltered.

        Returns:
            :obj:`tuple`: Two objects are returned selecting the data that were
            rejected. If :attr:`sig` is None, the returned object selecting the
            velocity dispersion data that were rejected is also None.
        """
        if self.sb is None or self.sb_ivar is None:
            raise ValueError('Cannot perform S/N rejection; no surface brightness and/or '
                             'surface brightness error data.')
        snr = self.sb * np.sqrt(self.sb_ivar)
        vel_rej = np.zeros(self.vel.size, dtype=bool) if min_vel_snr is None else snr < min_vel_snr
        sig_rej = None if self.sig is None else \
                    (np.zeros(self.sig.size, dtype=bool) if min_sig_snr is None else
                     snr < min_sig_snr)
        if save:
            self.reject(vel_rej=vel_rej, sig_rej=sig_rej)
        return vel_rej, sig_rej

    def clip_spatial(self, vel_mask=None, sig_mask=None, save=True):
        """
        Reject pixels that are not consituents of the largest coherent patch,
        independently for the velocity and dispersion measurements.  See
        :func:`~nirvana.data.util.find_largest_coherent_region`.

        By default, the rejection is directly incorporated into :attr:`vel_mask`
        and :attr:`sig_mask`; set ``save`` to False to force these attributes to
        remain unaltered.

        If all values are masked on input (either ``vel_mask`` or ``sig_mask``
        is all True), the returned arrays will indicate that none of the values
        were rejected because of their spatial distribution.

        Args:
            vel_mask (`numpy.ndarray`_, optional):
                Velocity mask.  Shape must match :attr:`vel`.  Can be a
                boolean (bad-pixel) mask or a bit mask; for the latter, all
                values greater than zero are assumed to be masked.  If None,
                :attr:`vel_mask` is used.
            sig_mask (`numpy.ndarray`_, optional):
                Velocity dispersion mask.  Shape must match :attr:`sig`.
                Can be a boolean (bad-pixel) mask or a bit mask; for the latter,
                all values greater than zero are assumed to be masked.  If None,
                :attr:`sig_mask` is used.
            save (:obj:`bool`, optional):
                Propagate the rejection flags to the internal masks.  If False,
                :attr:`vel_mask` and :attr:`sig_mask` remain unaltered.

        Returns:
            :obj:`tuple`: Two objects are returned selecting the data that were
            rejected. If :attr:`sig` is None, the returned object selecting the
            velocity dispersion data that were rejected is also None.
        """
        _vel_mask = self.vel_mask.copy() if vel_mask is None else vel_mask.copy()
        if _vel_mask is not None and self.vel is not None and _vel_mask.shape != self.vel.shape:
            raise ValueError(f'Velocity mask has incorrect shape; found {_vel_mask.shape}, '
                             f'expected {self.vel.shape}.')
        _sig_mask = self.sig_mask.copy() if sig_mask is None else sig_mask.copy()
        if _sig_mask is not None and self.sig is not None and _sig_mask.shape != self.sig.shape:
            raise ValueError(f'Dispersion mask has incorrect shape; found {_sig_mask.shape}, '
                             f'expected {self.sig.shape}.')
        if _vel_mask is None and _sig_mask is None:
            warnings.warn('No masks available from which to return coherent region.  No '
                          'rejections to perform.')
            vel_rej = None if self.vel is None else np.zeros(self.vel.shape, dtype=bool)
            sig_rej = None if self.sig is None else np.zeros(self.sig.shape, dtype=bool)
            return vel_rej, sig_rej

        if _vel_mask is None:
            vel_rej = None
        else:
            _vel_mask = _vel_mask.astype(int)
            gpm = np.logical_not(self.remap(_vel_mask, masked=False, fill_value=1).astype(bool))
            if not np.any(gpm):
                vel_rej = np.zeros(self.vel.shape, dtype=bool)
            else:
                indx = find_largest_coherent_region(gpm.astype(int)).astype(int)
                vel_rej = np.logical_not(self.bin(indx).astype(bool)) & (_vel_mask == 0)

        if _sig_mask is None:
            sig_rej = None
        else:
            _sig_mask = _sig_mask.astype(int)
            gpm = np.logical_not(self.remap(_sig_mask, masked=False, fill_value=1).astype(bool))
            if not np.any(gpm):
                sig_rej = np.zeros(self.sig.shape, dtype=bool)
            else:
                indx = find_largest_coherent_region(gpm.astype(int)).astype(int)
                sig_rej = np.logical_not(self.bin(indx).astype(bool)) & (_sig_mask == 0)
        if save:
            self.reject(vel_rej=vel_rej, sig_rej=sig_rej)
        return vel_rej, sig_rej

    # TODO: provide a keyword parameter providing the names of the mask bits to be used?
    def init_fitting_masks(self, bitmask=None, max_vel_err=None, max_sig_err=None,
                           min_vel_snr=None, min_sig_snr=None, select_coherent=False,
                           save=True, verbose=False):
        """
        Initialize masks to be used when fitting the kinematic data.

        This is a convenience function that combines :func:`clip_err`,
        :func:`clip_snr`, and :func:`clip_spatial`.  It can also propagate the
        results to a bitmask; see ``bitmask``.

        Args:
            bitmask (:class:`~nirvana.models.thindisk.ThinDiskFitBitMask`, optional):
                The bitmask instance used to flag bits.  If None, returned
                arrays are boolean.
            max_vel_err (:obj:`float`, optional):
                Maximum allowed velocity error. If None, no additional
                masking is performed.
            max_sig_err (:obj:`float`, optional):
                Maximum allowed *observed* velocity dispersion error. I.e.,
                this is the measurement error before any velocity dispersion
                correction.  If None, no additional masking is performed.
            min_vel_snr (:obj:`float`, optional):
                Minimum S/N for a spaxel to use for velocity measurements. If
                None, no additional masking is performed.
            min_sig_snr (:obj:`float`, optional):
                Minimum S/N for a spaxel to use for dispersion measurements.
                If None, no additional masking is performed.
            select_coherent (:obj:`bool`, optional):
                Mask any measurements that are not included in the largest
                coherent region of adjacent measurements.  This assessment is
                done after applying the error and S/N rejections.
            save (:obj:`bool`, optional):
                Propagate the rejection flags to the internal masks.  If False,
                :attr:`vel_mask` and :attr:`sig_mask` remain unaltered.
            verbose (:obj:`bool`, optional):
                Include print statements reporting the number of rejections
                performed.

        Returns:
            :obj:`tuple`: Two objects are returned selecting the data to be
            masked.  If :attr:`sig` is None, the returned object selecting the
            velocity dispersion data that were rejected is also None.  If a
            ``bitmask`` is provided, the returned arrays have the appropriate
            integer type; otherwise, boolean arrays are returned that select
            pixels rejected for any reason.
        """
        # Everything that is currently masked is never used
        vel_didnotuse = self.vel_mask.copy() 
        sig_didnotuse = None if self.sig is None else self.sig_mask.copy()

        # Find the data with large errors
        vel_largeerr, sig_largeerr = self.clip_err(max_vel_err=max_vel_err,
                                                   max_sig_err=max_sig_err, save=False)
        # Find the data with low S/N        
        vel_lowsnr, sig_lowsnr = self.clip_snr(min_vel_snr=min_vel_snr, min_sig_snr=min_sig_snr,
                                               save=False)

        # Collate all the rejections
        vel_rej = vel_didnotuse | vel_largeerr | vel_lowsnr
        sig_rej = None if self.sig is None else (sig_didnotuse | sig_largeerr | sig_lowsnr)

        # Select coherent regions, if requested
        if select_coherent:
            vel_disjoint, sig_disjoint = self.clip_spatial(vel_mask=vel_rej, sig_mask=sig_rej,
                                                           save=False)
            vel_rej |= vel_disjoint
            if self.sig is not None:
                sig_rej |= sig_disjoint
        else:
            vel_disjoint = None
            sig_disjoint = None

        if verbose:
            any_removed = np.any(vel_largeerr) or np.any(vel_lowsnr) or np.any(vel_disjoint)
            if any_removed:
                print('Velocity measurements removed:')
            else:
                print('Velocity measurements removed: NONE')
            if np.any(vel_largeerr):
                print(f'    Large errors: {np.sum(vel_largeerr)}')
            if np.any(vel_lowsnr):
                print(f'         Low S/N: {np.sum(vel_lowsnr)}')
            if np.any(vel_disjoint):
                print(f'        Disjoint: {np.sum(vel_disjoint)}')

            any_removed = np.any(sig_largeerr) or np.any(sig_lowsnr) or np.any(sig_disjoint)
            if any_removed:
                print('Dispersion measurements removed:')
            else:
                print('Dispersion measurements removed: NONE')
            if np.any(sig_largeerr):
                print(f'    Large errors: {np.sum(sig_largeerr)}')
            if np.any(sig_lowsnr):
                print(f'         Low S/N: {np.sum(sig_lowsnr)}')
            if np.any(sig_disjoint):
                print(f'        Disjoint: {np.sum(sig_disjoint)}')

        # Save the rejections if requested
        if save:
            self.reject(vel_rej=vel_rej, sig_rej=sig_rej)

        # If the bitmask is not defined, just return the boolean arrays
        if bitmask is None:
            return vel_rej, sig_rej

        # Collect the velocity bitmask flags
        vel_mask = bitmask.init_mask_array(self.vel.shape)
        vel_mask[vel_didnotuse] = bitmask.turn_on(vel_mask[vel_didnotuse], 'DIDNOTUSE')
        vel_mask[vel_largeerr] = bitmask.turn_on(vel_mask[vel_largeerr], 'REJ_ERR')
        vel_mask[vel_lowsnr] = bitmask.turn_on(vel_mask[vel_lowsnr], 'REJ_SNR')
        if select_coherent:
            vel_mask[vel_disjoint] = bitmask.turn_on(vel_mask[vel_disjoint], 'DISJOINT')

        # If there are not sigma measurements, we're done
        if self.sig is None:
            return vel_mask, None

        # Collect the dispersion bitmask flags
        sig_mask = bitmask.init_mask_array(self.sig.shape)
        sig_mask[sig_didnotuse] = bitmask.turn_on(sig_mask[sig_didnotuse], 'DIDNOTUSE')
        sig_mask[sig_largeerr] = bitmask.turn_on(sig_mask[sig_largeerr], 'REJ_ERR')
        sig_mask[sig_lowsnr] = bitmask.turn_on(sig_mask[sig_lowsnr], 'REJ_SNR')
        if select_coherent:
            sig_mask[sig_disjoint] = bitmask.turn_on(sig_mask[sig_disjoint], 'DISJOINT')

        # Done    
        return vel_mask, sig_mask

    def deviate(self, size=None, rng=None, sigma='draw'):
        r"""
        Draw Gaussian deviates from the velocity and velocity dispersion error
        distributions.

        This is basically a wrapper for
        :func:`~nirvana.data.util.gaussian_deviates`.  One deviate is drawn for
        each of the valid velocity and velocity dispersion measurements.
        Multiple sets of deviates can be drawn using ``size``.

        This function is primarily for use with mock observations.  For example,
        if you have a :class:`Kinematics` object (``kin``) with an observed set
        of data, you can generate a mock dataset based on the
        :class:`~nirvana.models.axisym.AxisymmetricDisk`::

            import numpy as np
            from nirvana.models.oned import HyperbolicTangent, Exponential
            from nirvana.models.axisym import AxisymmetricDisk
            disk = AxisymmetricDisk(rc=HyperbolicTangent(), dc=Exponential())
            p0 = np.array([-0.2, -0.08, 166.3, 53.0, 25.6, 217.0, 2.82, 189.7, 16.2])
            noisefree_mock = disk.mock_observation(p0, kin=kin)

        And then generate deviates using this method and add them to a new
        kinematics object::

            # Generate 10 sets of deviates
            vgpm, dv, sgpm, ds \
                    = noisefree_mock.deviate(size=10, sigma='ignore' if disk.dc is None else 'draw')
            # Use them in a simulation of fitting the mock data
            _vel = noisefree_mock.vel.copy()
            _sig = noisefree_mock.sig.copy()
            noisy_mock = noisefree_mock.copy()
            for i in range(10):
                noisy_mock.vel[vgpm] = _vel[vgpm] + dv[i]
                if disk.dc is not None:
                    _sig[sgpm] += ds[i]
                    noisy_mock.update_sigma(sig=_sig)
                    _sig[sgpm] -= ds[i]
                disk.lsq_fit(noisy_mock)
        
        Args:
            size (:obj:`int`, optional):
                The number of draws to make.  Each draw generates a deviate for
                each of the valid measurements.  None is identical to
                ``size=1``.
            rng (`numpy.random.Generator`, optional):
                Generator object to use.  If None, a new Generator is
                instantiated.
            sigma (:obj:`str`, optional):
                Treatment for the velocity dispersion.  Options are:

                    - 'draw': Draw Gaussian deviates from ``sig_ivar`` or
                      ``sig_covar``, if they exist.
                    - 'drawsqr': Draw Gaussian deviates from ``sig_phys2_ivar``
                      or ``sig_phys2_covar``, if they exist.
                    - 'ignore': Do not draw any deviates for the dispersion,
                      even if the error distributions exist.
                
        Returns:
            :obj:`tuple`: Good-value masks and deviates for the velocity and
            velocity dispersion, respectively.  If neither :attr:`vel_ivar` nor
            `vel_covar` are available, the first two objects returned are None;
            similarly for the second two objects if dispersion errors are not
            available or ignored.  The shape of the good pixel masks is always
            the same as the internal velocity and dispersion arrays.  The shape
            of the returned deviate arrays is ``(size,n_good)``, where ``size``
            is the provided keyword argument and ``n_good`` is the number of
            valid kinematic measurements.
        """
        if sigma == 'ignore':
            s_ret = (None, None)
        elif sigma == 'draw':
            s_ret = gaussian_deviates(ivar=self.sig_ivar, mask=self.sig_mask, covar=self.sig_covar,
                                      size=size, rng=rng)
        elif sigma == 'drawsqr':
            s_ret = gaussian_deviates(ivar=self.sig_phys2_ivar, mask=self.sig_mask,
                                      covar=self.sig_phys2_covar, size=size, rng=rng)
        else:
            raise ValueError('Value for sigma must be ignore, draw, or drawsqr.')
        return gaussian_deviates(ivar=self.vel_ivar, mask=self.vel_mask, covar=self.vel_covar,
                                 size=size, rng=rng) + s_ret

    # TODO:
    #   - Allow the plot to be constructed from the fits file written by
    #     axisym_fit_data
    def asymmetry_plot(self, galmeta=None, xc=0., yc=0., pa=0., inc=0., vsys=0., ell_r=None,
                       fwhm=None, vel_mask=None, sig_mask=None, ofile=None):
        r"""
        Construct a plot showing reflective asymmetries in the velocity and
        velocity dispersion.

        Args:
            galmeta (:class:`~nirvana.data.meta.GlobalPar`, optional):
                Object with metadata for the galaxy to be fit.
            xc (:obj:`float`, optional):
                Kinematic center relative to :attr:`grid_x`.
            yc (:obj:`float`, optional):
                Kinematic center relative to :attr:`grid_y`.
            pa (:obj:`float`, optional):
                Position angle of the major axis, from N through E in degrees.
            inc (:obj:`float`, optional):
                An inclination used to create an ellipse that marks a specific
                in-plane radius; see ``ell_r``.  If ``ell_r`` is None, no
                ellipse is plotted.  
            vsys (:obj:`float`, optional):
                Systemic velocity.
            ell_r (:obj:`float`, optional):
                The radius of an ellipse used to mark a specific in-plane
                radius.  If None, the maximum radius within a :math:`\pm 10` deg
                wedge along the major axis is used.  The ellipticity, center,
                and orientation of the ellipse are set by the provided
                inclination (``inc``), center (``xc``, ``yc``), and position
                angle (``pa``).
            fwhm (:obj:`float`, optional):
                FWHM of the PSF in arcsec.  If provided, this is used to add the
                nominal beam size to the map plots.
            vel_mask (`numpy.ndarray`_, optional):
                A boolean bad-value mask (True is bad) for the velocity
                measurements to use *instead* of the internal :attr:`vel_mask`.
                If None, the internal :attr:`vel_mask` is used.  If provided,
                the shape must be the same as :attr:`vel`.
            sig_mask (`numpy.ndarray`_, optional):
                A boolean bad-value mask (True is bad) for the velocity
                dispersion measurements to use *instead* of the internal
                :attr:`sig_mask`.  If None, the internal :attr:`sig_mask` is
                used.  If provided, the shape must be the same as :attr:`sig`.
                Ignored if :attr:`sig` is None.
            ofile (:obj:`str`, optional):
                Output filename for the plot.  If None, the plot is shown to the
                screen.
        """
        # Check the input
        if vel_mask is not None and vel_mask.shape != self.vel.shape:
            raise ValueError(f'Provided vel_mask has incorrect shape; found {vel_mask.shape}, '
                             f'expected {self.vel.shape}.')
        if self.sig is not None and sig_mask is not None and sig_mask.shape != self.sig.shape:
            raise ValueError(f'Provided sig_mask has incorrect shape; found {sig_mask.shape}, '
                             f'expected {self.sig.shape}.')

        # Change the style
        logformatter = plot.get_logformatter()
        rc('font', size=8)

        # Get the velocity maps
        vel = self.remap('vel', mask=self.vel_mask if vel_mask is None else vel_mask)
        vel_x, vel_y, vel_xy = asymmetry.onsky_asymmetry_maps(self.grid_x-xc, self.grid_y-yc,
                                                              vel.data-vsys, pa=pa, mask=vel.mask,
                                                              odd=True, maxd=0.4)

        # Sigma square maps
        if self.sig is None:
            sig, sig_x, sig_y, sig_xy = [None]*4
        else:
            sigsqr = self.remap('sig_phys2', mask=self.sig_mask if sig_mask is None else sig_mask)
            sig = sigsqr/np.sqrt(np.absolute(sigsqr))
            sig_x, sig_y, sig_xy = asymmetry.onsky_asymmetry_maps(self.grid_x-xc, self.grid_y-yc,
                                                                  sig.data, pa=pa, mask=sig.mask,
                                                                  maxd=0.4)

        # Elliptical radius used in plotting and asymmetry RMS calculation
        r, th = projected_polar(self.grid_x-xc, self.grid_y-yc, *np.radians([pa, inc]))
        if ell_r is None:
            major_gpm = select_kinematic_axis(r, th, which='major', r_range='all', wedge=10.)
            major_gpm &= np.logical_not(np.ma.getmaskarray(vel))
            ell_r = np.amax(r[major_gpm]) if np.any(major_gpm) else -1
        de_x, de_y = disk_ellipse(ell_r, np.radians(pa), np.radians(inc), xc=xc, yc=yc)
        ellip_gpm = None if ell_r < 0 else r < ell_r

        # Asymmetry metrics
        fid_grw = np.array([50., 80., 90.])

        abs_vel_x, grw_vel_x, fid_vel_x = asymmetry.asymmetry_metrics(vel_x, fid_grw)
        abs_vel_y, grw_vel_y, fid_vel_y = asymmetry.asymmetry_metrics(vel_y, fid_grw)
        abs_vel_xy, grw_vel_xy, fid_vel_xy = asymmetry.asymmetry_metrics(vel_xy, fid_grw)

        abs_sig_x, grw_sig_x, fid_sig_x = asymmetry.asymmetry_metrics(sig_x, fid_grw)
        abs_sig_y, grw_sig_y, fid_sig_y = asymmetry.asymmetry_metrics(sig_y, fid_grw)
        abs_sig_xy, grw_sig_xy, fid_sig_xy = asymmetry.asymmetry_metrics(sig_xy, fid_grw)

        if ellip_gpm is not None:
            ell_abs_vel_x, ell_grw_vel_x, ell_fid_vel_x \
                    = asymmetry.asymmetry_metrics(vel_x, fid_grw, gpm=ellip_gpm)
            ell_abs_vel_y, ell_grw_vel_y, ell_fid_vel_y \
                    = asymmetry.asymmetry_metrics(vel_y, fid_grw, gpm=ellip_gpm)
            ell_abs_vel_xy, ell_grw_vel_xy, ell_fid_vel_xy \
                    = asymmetry.asymmetry_metrics(vel_xy, fid_grw, gpm=ellip_gpm)

            ell_abs_sig_x, ell_grw_sig_x, ell_fid_sig_x \
                    = asymmetry.asymmetry_metrics(sig_x, fid_grw, gpm=ellip_gpm)
            ell_abs_sig_y, ell_grw_sig_y, ell_fid_sig_y \
                    = asymmetry.asymmetry_metrics(sig_y, fid_grw, gpm=ellip_gpm)
            ell_abs_sig_xy, ell_grw_sig_xy, ell_fid_sig_xy \
                    = asymmetry.asymmetry_metrics(sig_xy, fid_grw, gpm=ellip_gpm)

        # Set the extent for the 2D maps
        extent = [np.amax(self.grid_x), np.amin(self.grid_x),
                  np.amin(self.grid_y), np.amax(self.grid_y)]
        Dx = max(extent[0]-extent[1], extent[3]-extent[2]) # *1.01
        skylim = np.array([ (extent[0]+extent[1] - Dx)/2., 0.0 ])
        skylim[1] = skylim[0] + Dx

        # Create the plot
        w,h = pyplot.figaspect(1)
        fig = pyplot.figure(figsize=(2*w,2*h))

        #-------------------------------------------------------------------
        # Velocity
        vel_lim = growth_lim(vel, 0.90, 1.05, midpoint=vsys)
        ax = plot.init_ax(fig, [0.015, 0.725, 0.24, 0.24])
        cax = fig.add_axes([0.055, 0.97, 0.195, 0.005])
        cax.tick_params(which='both', direction='in')
        ax.set_xlim(skylim[::-1])
        ax.set_ylim(skylim)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        if fwhm is not None:
            ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2,
                         transform=ax.transAxes, facecolor='0.7', edgecolor='k', zorder=4))
        im = ax.imshow(vel, origin='lower', interpolation='nearest', cmap='RdBu_r',
                       extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
        # Mark the fitted dynamical center and the position angle
        ax.scatter(xc, yc, marker='+', color='k', s=40, lw=1, zorder=5)
        ax.arrow(xc, yc, skylim[1]*0.7*np.cos(np.radians(90-pa)),
                 skylim[1]*0.7*np.sin(np.radians(90-pa)), zorder=5, length_includes_head=True,
                 head_length=2., head_width=1.0, facecolor='k', edgecolor='k')
        if de_x is not None:
            ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cax.text(-0.05, 1.1, 'V', ha='right', va='center', transform=cax.transAxes)


        #-------------------------------------------------------------------
        # Velocity Reflection Asymmetries
        v_res_lim = growth_lim(np.ma.concatenate((vel_x, vel_y, vel_xy)), 0.90, 1.15, midpoint=0.0)

        ax = plot.init_ax(fig, [0.260, 0.725, 0.24, 0.24])
        cax = fig.add_axes([0.300, 0.97, 0.195, 0.005])
        cax.tick_params(which='both', direction='in')
        ax.set_xlim(skylim[::-1])
        ax.set_ylim(skylim)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        if fwhm is not None:
            ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2,
                         transform=ax.transAxes, facecolor='0.7', edgecolor='k', zorder=4))
        im = ax.imshow(vel_x, origin='lower', interpolation='nearest', cmap='RdBu_r',
                       extent=extent, vmin=v_res_lim[0], vmax=v_res_lim[1], zorder=4)
        if de_x is not None:
            ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cax.text(-0.05, 1.1, r'$\Delta V_x$', ha='right', va='center', transform=cax.transAxes)
 

        ax = plot.init_ax(fig, [0.505, 0.725, 0.24, 0.24])
        cax = fig.add_axes([0.545, 0.97, 0.195, 0.005])
        cax.tick_params(which='both', direction='in')
        ax.set_xlim(skylim[::-1])
        ax.set_ylim(skylim)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        if fwhm is not None:
            ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2,
                         transform=ax.transAxes, facecolor='0.7', edgecolor='k', zorder=4))
        im = ax.imshow(vel_y, origin='lower', interpolation='nearest', cmap='RdBu_r',
                       extent=extent, vmin=v_res_lim[0], vmax=v_res_lim[1], zorder=4)
        if de_x is not None:
            ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cax.text(-0.05, 1.1, r'$\Delta V_y$', ha='right', va='center', transform=cax.transAxes)
 

        ax = plot.init_ax(fig, [0.750, 0.725, 0.24, 0.24])
        cax = fig.add_axes([0.790, 0.97, 0.195, 0.005])
        cax.tick_params(which='both', direction='in')
        ax.set_xlim(skylim[::-1])
        ax.set_ylim(skylim)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        if fwhm is not None:
            ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2,
                         transform=ax.transAxes, facecolor='0.7', edgecolor='k', zorder=4))
        im = ax.imshow(vel_xy, origin='lower', interpolation='nearest', cmap='RdBu_r',
                       extent=extent, vmin=v_res_lim[0], vmax=v_res_lim[1], zorder=4)
        if de_x is not None:
            ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cax.text(-0.05, 1.1, r'$\Delta V_{xy}$', ha='right', va='center', transform=cax.transAxes)

        if sig is not None:
            #-------------------------------------------------------------------
            # Velocity Dispersion
            sig_lim = growth_lim(sig, 0.90, 1.05)

            ax = plot.init_ax(fig, [0.015, 0.480, 0.24, 0.24])
            cax = fig.add_axes([0.055, 0.47, 0.195, 0.005])
            cax.tick_params(which='both', direction='in')
            ax.set_xlim(skylim[::-1])
            ax.set_ylim(skylim)
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            if fwhm is not None:
                ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2,
                            transform=ax.transAxes, facecolor='0.7', edgecolor='k', zorder=4))
            im = ax.imshow(sig, origin='lower', interpolation='nearest', cmap='viridis',
                        extent=extent, vmin=sig_lim[0], vmax=sig_lim[1], zorder=4)
            # Mark the fitted dynamical center
            ax.scatter(xc, yc, marker='+', color='k', s=40, lw=1, zorder=5)
            if de_x is not None:
                ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cax.text(-0.05, 0.1, r'$\sigma$', ha='right', va='center', transform=cax.transAxes)

            #-------------------------------------------------------------------
            # Velocity Dispersion Reflection Asymmetries
            s_res_lim = growth_lim(np.ma.concatenate((sig_x, sig_y, sig_xy)), 0.90, 1.15,
                                   midpoint=0.0)

            ax = plot.init_ax(fig, [0.260, 0.480, 0.24, 0.24])
            cax = fig.add_axes([0.300, 0.47, 0.195, 0.005])
            cax.tick_params(which='both', direction='in')
            ax.set_xlim(skylim[::-1])
            ax.set_ylim(skylim)
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            if fwhm is not None:
                ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2,
                            transform=ax.transAxes, facecolor='0.7', edgecolor='k', zorder=4))
            im = ax.imshow(sig_x, origin='lower', interpolation='nearest', cmap='RdBu_r',
                           extent=extent, vmin=s_res_lim[0], vmax=s_res_lim[1], zorder=4)
            if de_x is not None:
                ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cax.text(-0.05, 0.1, r'$\Delta \sigma_x$', ha='right', va='center',
                     transform=cax.transAxes)
 

            ax = plot.init_ax(fig, [0.505, 0.480, 0.24, 0.24])
            cax = fig.add_axes([0.545, 0.47, 0.195, 0.005])
            cax.tick_params(which='both', direction='in')
            ax.set_xlim(skylim[::-1])
            ax.set_ylim(skylim)
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            if fwhm is not None:
                ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2,
                            transform=ax.transAxes, facecolor='0.7', edgecolor='k', zorder=4))
            im = ax.imshow(sig_y, origin='lower', interpolation='nearest', cmap='RdBu_r',
                           extent=extent, vmin=s_res_lim[0], vmax=s_res_lim[1], zorder=4)
            if de_x is not None:
                ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cax.text(-0.05, 0.1, r'$\Delta \sigma_y$', ha='right', va='center',
                     transform=cax.transAxes)
 

            ax = plot.init_ax(fig, [0.750, 0.480, 0.24, 0.24])
            cax = fig.add_axes([0.790, 0.47, 0.195, 0.005])
            cax.tick_params(which='both', direction='in')
            ax.set_xlim(skylim[::-1])
            ax.set_ylim(skylim)
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())
            if fwhm is not None:
                ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2,
                            transform=ax.transAxes, facecolor='0.7', edgecolor='k', zorder=4))
            im = ax.imshow(sig_xy, origin='lower', interpolation='nearest', cmap='RdBu_r',
                           extent=extent, vmin=s_res_lim[0], vmax=s_res_lim[1], zorder=4)
            if de_x is not None:
                ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cax.text(-0.05, 0.1, r'$\Delta \sigma_{xy}$', ha='right', va='center',
                     transform=cax.transAxes)
 
        #-------------------------------------------------------------------
        # SDSS image
        ax = fig.add_axes([0.03, 0.14, 0.28, 0.28])
        if self.image is not None:
            ax.imshow(self.image)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', transform=ax.transAxes,
                    fontsize=20)

        ax.text(0.5, 1.05, 'SDSS gri Composite', ha='center', va='center', transform=ax.transAxes,
                fontsize=10)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # TODO: This is MaNGa specific!
        if galmeta is not None:
            ax.text(0.00, -0.05, 'MaNGA ID:', ha='left', va='center', transform=ax.transAxes,
                    fontsize=10)
            ax.text(1.01, -0.05, f'{galmeta.mangaid}', ha='right', va='center',
                    transform=ax.transAxes, fontsize=10)
            # Observation
            ax.text(0.00, -0.12, 'Observation:', ha='left', va='center', transform=ax.transAxes,
                    fontsize=10)
            ax.text(1.01, -0.12, galmeta.plateifu, ha='right', va='center',
                    transform=ax.transAxes, fontsize=10)
        # Observation
        ax.text(0.00, -0.19, r'$V_{\rm asym}$ (all: x, y, xy)', ha='left', va='center',
                transform=ax.transAxes, fontsize=10)
        ax.text(1.01, -0.19, f'{fid_vel_x[-1]:.1f}, {fid_vel_y[-1]:.1f}, {fid_vel_xy[-1]:.1f}',
                ha='right', va='center', transform=ax.transAxes, fontsize=10)
        if ellip_gpm is not None:
            ax.text(0.00, -0.26, r'$V_{\rm asym}$ (ell.: x, y, xy)', ha='left', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.text(1.01, -0.26,
                    f'{ell_fid_vel_x[-1]:.1f}, {ell_fid_vel_y[-1]:.1f}, {ell_fid_vel_xy[-1]:.1f}',
                    ha='right', va='center', transform=ax.transAxes, fontsize=10)
        ax.text(0.00, -0.33, r'$\sigma_{\rm asym}$ (all: x, y, xy)', ha='left', va='center',
                transform=ax.transAxes, fontsize=10)
        ax.text(1.01, -0.33, f'{fid_sig_x[-1]:.1f}, {fid_sig_y[-1]:.1f}, {fid_sig_xy[-1]:.1f}',
                ha='right', va='center', transform=ax.transAxes, fontsize=10)
        if ellip_gpm is not None:
            ax.text(0.00, -0.40, r'$\sigma_{\rm asym}$ (ell.: x, y, xy)', ha='left', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.text(1.01, -0.40,
                    f'{ell_fid_sig_x[-1]:.1f}, {ell_fid_sig_y[-1]:.1f}, {ell_fid_sig_xy[-1]:.1f}',
                    ha='right', va='center', transform=ax.transAxes, fontsize=10)

        rc('font', size=10)

        # Velocity asymmetry growth
        ax = plot.init_ax(fig, [0.38, 0.06, 0.29, 0.36], majlen=6, minlen=3, facecolor='0.95',
                          gridcolor='0.9')
        asym90 = np.array([fid_vel_x[-2], fid_vel_y[-2], fid_vel_xy[-2]])
        if ellip_gpm is not None:
            asym90 = np.append(asym90, [ell_fid_vel_x[-2], ell_fid_vel_y[-2], ell_fid_vel_xy[-2]])
        ax.set_xlim([0., np.amax(asym90)*1.2])
        ax.set_ylim([3e-2, 1.01])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(logformatter)

        ax.step(abs_vel_x, grw_vel_x, color='C0', where='post', zorder=3)
        ax.scatter(fid_vel_x[:3], 1-fid_grw/100, marker='.', s=200, color='C0', zorder=4)
        ax.step(abs_vel_y, grw_vel_y, color='C1', where='post', zorder=3)
        ax.scatter(fid_vel_y[:3], 1-fid_grw/100, marker='.', s=200, color='C1', zorder=4)
        ax.step(abs_vel_xy, grw_vel_xy, color='C2', where='post', zorder=3)
        ax.scatter(fid_vel_xy[:3], 1-fid_grw/100, marker='.', s=200, color='C2', zorder=4)

        if ellip_gpm is not None:
            ax.step(ell_abs_vel_x, ell_grw_vel_x,
                    color='C0', where='post', zorder=3, ls='--', lw=0.5)
            ax.scatter(ell_fid_vel_x[:3], 1-fid_grw/100,
                       marker='o', s=100, facecolor='none', edgecolor='C0', zorder=4)
            ax.step(ell_abs_vel_y, ell_grw_vel_y,
                    color='C1', where='post', zorder=3, ls='--', lw=0.5)
            ax.scatter(ell_fid_vel_y[:3], 1-fid_grw/100,
                       marker='o', s=100, facecolor='none', edgecolor='C1', zorder=4)
            ax.step(ell_abs_vel_xy, ell_grw_vel_xy,
                    color='C2', where='post', zorder=3, ls='--', lw=0.5)
            ax.scatter(ell_fid_vel_xy[:3], 1-fid_grw/100,
                       marker='o', s=100, facecolor='none', edgecolor='C2', zorder=4)

        ax.text(-0.15, 0.5, '1-Growth', ha='center', va='center', transform=ax.transAxes,
                rotation='vertical')
        ax.text(0.5, -0.08, r'$\Delta V$ [km/s]', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.85, 0.9, r'$\Delta V_x$', ha='left', va='center', transform=ax.transAxes,
                color='C0', fontsize=12)
        ax.text(0.85, 0.82, r'$\Delta V_y$', ha='left', va='center', transform=ax.transAxes,
                color='C1', fontsize=12)
        ax.text(0.85, 0.74, r'$\Delta V_{xy}$', ha='left', va='center', transform=ax.transAxes,
                color='C2', fontsize=12)

        # Velocity dispersion asymmetry growth
        ax = plot.init_ax(fig, [0.68, 0.06, 0.29, 0.36], majlen=6, minlen=3, facecolor='0.95',
                          gridcolor='0.9')
        asym90 = np.array([fid_sig_x[-2], fid_sig_y[-2], fid_sig_xy[-2]])
        if ellip_gpm is not None:
            asym90 = np.append(asym90, [ell_fid_sig_x[-2], ell_fid_sig_y[-2], ell_fid_sig_xy[-2]])
        ax.set_xlim([0., np.amax(asym90)*1.2])
        ax.set_ylim([3e-2, 1.01])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.NullFormatter())

        ax.step(abs_sig_x, grw_sig_x, color='C0', where='post', zorder=3)
        ax.scatter(fid_sig_x[:3], 1-fid_grw/100, marker='.', s=200, color='C0', zorder=4)
        ax.step(abs_sig_y, grw_sig_y, color='C1', where='post', zorder=3)
        ax.scatter(fid_sig_y[:3], 1-fid_grw/100, marker='.', s=200, color='C1', zorder=4)
        ax.step(abs_sig_xy, grw_sig_xy, color='C2', where='post', zorder=3)
        ax.scatter(fid_sig_xy[:3], 1-fid_grw/100, marker='.', s=200, color='C2', zorder=4)

        if ellip_gpm is not None:
            ax.step(ell_abs_sig_x, ell_grw_sig_x,
                    color='C0', where='post', zorder=3, ls='--', lw=0.5)
            ax.scatter(ell_fid_sig_x[:3], 1-fid_grw/100,
                       marker='o', s=100, facecolor='none', edgecolor='C0', zorder=4)
            ax.step(ell_abs_sig_y, ell_grw_sig_y,
                    color='C1', where='post', zorder=3, ls='--', lw=0.5)
            ax.scatter(ell_fid_sig_y[:3], 1-fid_grw/100,
                       marker='o', s=100, facecolor='none', edgecolor='C1', zorder=4)
            ax.step(ell_abs_sig_xy, ell_grw_sig_xy,
                    color='C2', where='post', zorder=3, ls='--', lw=0.5)
            ax.scatter(ell_fid_sig_xy[:3], 1-fid_grw/100,
                       marker='o', s=100, facecolor='none', edgecolor='C2', zorder=4)

        ax.text(0.5, -0.08, r'$\Delta \sigma$ [km/s]', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.85, 0.9, r'$\Delta \sigma_x$', ha='left', va='center', transform=ax.transAxes,
                color='C0', fontsize=12)
        ax.text(0.85, 0.82, r'$\Delta \sigma_y$', ha='left', va='center', transform=ax.transAxes,
                color='C1', fontsize=12)
        ax.text(0.85, 0.74, r'$\Delta \sigma_{xy}$', ha='left', va='center', transform=ax.transAxes,
                color='C2', fontsize=12)

        if ofile is None:
            pyplot.show()
        else:
            fig.canvas.print_figure(ofile, bbox_inches='tight')
        fig.clear()
        pyplot.close(fig)

        # Reset to default style
        pyplot.rcdefaults()

    # TODO: 
    #   - Get the radial profile of the surface brightness, as well?
    #   - Bin sigma^2 and then take the sqrt instead of binning sigma?
    #   - Include covariance
    def radial_profiles(self, rstep, xc=0., yc=0., pa=0., inc=0., vsys=0., maj_wedge=30.,
                        min_wedge=10., vel_mod=None, sig_mod=None, vel_beam_corr=None,
                        sig_beam_corr=None):
        r"""
        Construct azimuthally averaged radial profiles of the kinematics.

        Radial profiles can be construct for best-fiting model data and/or after
        applying beam-smearing corrections.

        Beam-smearing corrections are applied as follows:

        .. math::

            V = V_{\rm obs} - V_{\rm beam}

        and

        .. math::

            \sigma^2 = \sigma^2_{\rm obs} - \sigma^2_{\rm beam},

        where :math:`V_{\rm beam}` and :math:`\sigma^2_{\rm beam}` are provided
        by the ``vel_beam_corr`` and ``sig_beam_corr`` parameters, respectively.
        Importantly, if model data are provided, **both** the observed and model
        data are beam corrected.

        The binned quantities are the projected tangential speed

        .. math::

            V_\theta = (V - V_{\rm sys}) / \cos\theta

        for data within the major axis wedge (see ``maj_wedge``), the projected
        radial speed

        .. math::

            V_R = (V - V_{\rm sys}) / \sin\theta

        for data within the minor axis wedge (see ``min_wedge``), and the
        observed velocity dispersion at all in-plane azimuths, within the major
        axis wedge, and within the minor axis wedge.

        .. warning::

            Velocity dispersion averages are currently done linearly.  I.e., the
            function first computes the sqrt of :attr:`sig_phys2` (ignoring
            negative values) and then computes the binned data.  This may change
            to computing the binned data for :math:`\sigma^2`.

        Args:
            rstep (:obj:`float`):
                The bin size in arcsec for each radial bin.
            xc (:obj:`float`, optional):
                The on-sky x coordinate of the dynamical center.
            yc (:obj:`float`, optional):
                The on-sky y coordinate of the dynamical center.
            pa (:obj:`float`, optional):
                Galaxy position angle, on-sky from N (+y) through E (+x), in
                degrees.
            inc (:obj:`float`, optional):
                Galaxy inclination in degrees.
            maj_wedge (:obj:`float`, optional):
                Half wedge size, in degrees, used to select data near the major
                axis.  For example, ``maj_wedge=30.`` selects data within a +/-
                30 degree wedge centered on the major axis.
            min_wedge (:obj:`float`, optional):
                Half wedge size, in degrees, used to select data near the minor
                axis.  For example, ``min_wedge=10.`` selects data within a +/-
                10 degree wedge centered on the minor axis.
            vel_mod (`numpy.ndarray`_, optional):
                Model velocities measured at exactly the same positions as the
                observed data.  Shape must match :attr:`vel`.  Used to construct
                binned model data weighted and averaged in an identical way to
                the observed data.  If beam-smearing corrections are provided
                (``vel_beam_corr``),
                **they are also applied to the provided model data!**
            sig_mod (`numpy.ndarray`_, optional):
                Model velocity dispersions measured at exactly the same
                positions as the observed data.  Shape must match
                :attr:`sig_phys2`.  Used to construct binned model data weighted
                and averaged in an identical way to the observed data.  If
                beam-smearing corrections are provided (``sig_beam_corr``),
                **they are also applied to the provided model data!**
            vel_beam_corr (`numpy.ndarray`_, optional):
                Beam-smearing corrections to be applied to the velocity
                measurements.  See description above.  Shape must match
                :attr:`vel`. If model data are provided (``vel_mod``), these
                corrections are applied to **both the observed data and the
                model**.
            sig_beam_corr (`numpy.ndarray`_, optional):
                Beam-smearing corrections to be applied to the velocity
                dispersion measurements.  See description above.  Shape must
                match :attr:`sig_phys2`. If model data are provided
                (``sig_mod``), these corrections are applied to
                **both the observed data and the model**.

        Returns:
            :obj:`tuple`: The function returns 31 `numpy.ndarray`_ objects.  The
            first provides the radius of each bin, which is followed by groups
            of 6 arrays for each of the following quantities: (1) the mean
            tangential velocity along the major axis, (2) the mean radial
            velocity along the minor axis, (3) the mean LOS velocity dispersion
            at all azimuths, (4) the mean LOS velocity dispersion near the major
            axis, (5) the mean LOS velocity dispersion near the minor axis.  The
            6 arrays for each of these quantities provide the error-weighted
            mean, the error-weighted standard deviation, the total number of
            measurements in the radial bin, the number included after rejecting
            outliers, the error-weighted mean model value, and the
            error-weighted standard deviation of the model values.  Data
            requiring the model data will be None if the relevant model data are
            not provided, and all velocity dispersion releted quantities will be
            None if no velocity dispersion measurements are available.
        """

        # Check input
        if vel_mod is not None and vel_mod.size != self.vel.size:
            raise ValueError('Number of model velocities does not match number of measurements.')
        if sig_mod is not None and sig_mod.size != self.sig_phys2.size:
            raise ValueError('Number of model dispersions does not match number of measurements.')
        if vel_beam_corr is not None and vel_beam_corr.size != self.vel.size:
            raise ValueError('Number of velocity beam-smearing corrections does not match number '
                             'of measurements.')
        if sig_beam_corr is not None and sig_beam_corr.size != self.sig_phys2.size:
            raise ValueError('Number of velocity dispersion beam-smearing corrections does not '
                             'match number of measurements.')

        # Disk-plane coordinates
        r, th = projected_polar(self.x - xc, self.y - yc, np.radians(pa), np.radians(inc))

        # Apply beam-smearing corrections if provided
        _vel = self.vel if vel_beam_corr is None else self.vel - vel_beam_corr
        _vel_mod = None if vel_mod is None \
                    else (vel_mod if vel_beam_corr is None else vel_mod - vel_beam_corr)

        # Mask for data along the major and minor axes
        major_gpm = select_kinematic_axis(r, th, which='major', r_range='all', wedge=maj_wedge)
        minor_gpm = select_kinematic_axis(r, th, which='minor', r_range='all', wedge=min_wedge)

        # Projected rotation velocities (within the major axis wedge)
        indx = major_gpm & np.logical_not(self.vel_mask)
        vrot_r = r[indx]
        vrot = (_vel[indx] - vsys)/np.cos(th[indx])
        vrot_wgt = self.vel_ivar[indx]*np.cos(th[indx])**2
        vrot_mod = None if vel_mod is None else (_vel_mod[indx] - vsys)/np.cos(th[indx])

        # Projected radial velocities (within the minor axis wedge)
        indx = minor_gpm & np.logical_not(self.vel_mask)
        vrad_r = r[indx]
        vrad = (_vel[indx] - vsys)/np.sin(th[indx])
        vrad_wgt = self.vel_ivar[indx]*np.sin(th[indx])**2
        vrad_mod = None if vel_mod is None else (_vel_mod[indx] - vsys)/np.sin(th[indx])

        # Get the binned data
        # NOTE: ad hoc maximum radius is meant to mitigate effect of minor axis
        # points on number radial bins.  This will limit the number of off-axis
        # points included in galaxies with inclinations > 75 deg.
        maxr = min(4*np.amax(vrot_r), np.amax(r)) if len(vrot_r) > 0 else np.amax(r)
        binr = np.array([rstep/2]) if maxr < rstep/2 else np.arange(rstep/2, maxr, rstep)
#        binr = np.arange(rstep/2, min(4*np.amax(vrot_r), np.amax(r)), rstep)
        binw = np.full(binr.size, rstep, dtype=float)

        # Projected Velocity profiles
        _, _, _, _, _, _, _, vrot_ewmean, vrot_ewsdev, _, vrot_ntot, vrot_nbin, vrot_bin_gpm \
                = bin_stats(vrot_r, vrot, binr, binw, wgts=vrot_wgt, fill_value=0.0) 
        _, _, _, _, _, _, _, vrad_ewmean, vrad_ewsdev, _, vrad_ntot, vrad_nbin, vrad_bin_gpm \
                = bin_stats(vrad_r, vrad, binr, binw, wgts=vrad_wgt, fill_value=0.0) 
        if vel_mod is None:
            vrotm_ewmean = vrotm_ewsdev = None
            vradm_ewmean = vradm_ewsdev = None
        else:
            _, _, _, _, _, _, _, vrotm_ewmean, vrotm_ewsdev, _, _, _, _ \
                    = bin_stats(vrot_r[vrot_bin_gpm], vrot_mod[vrot_bin_gpm], binr, binw,
                                wgts=vrot_wgt[vrot_bin_gpm], fill_value=0.0)
            _, _, _, _, _, _, _, vradm_ewmean, vradm_ewsdev, _, _, _, _ \
                    = bin_stats(vrad_r[vrad_bin_gpm], vrad_mod[vrad_bin_gpm], binr, binw,
                                wgts=vrad_wgt[vrad_bin_gpm], fill_value=0.0) 

        if self.sig is None:
            # No dispersion data, so we're done
            return binr, vrot_ewmean, vrot_ewsdev, vrot_ntot, vrot_nbin, \
                         vrotm_ewmean, vrotm_ewsdev, \
                         vrad_ewmean, vrad_ewsdev, vrad_ntot, vrad_nbin, \
                         vradm_ewmean, vradm_ewsdev, \
                         None, None, None, None, \
                         None, None, \
                         None, None, None, None, \
                         None, None, \
                         None, None, None, None, \
                         None, None

        # Apply beam-smearing corrections if provided
        _sig_phys2 = self.sig_phys2 if sig_beam_corr is None else self.sig_phys2 - sig_beam_corr
        _sig_mod = None if sig_mod is None \
                    else (sig_mod**2 if sig_beam_corr is None else sig_mod**2 - sig_beam_corr)

        # TODO: bin sigma^2 and then take sqrt instead?
        # Convert to corrected dispersion
        sig = np.ma.sqrt(_sig_phys2).filled(0.0)
        sigwgt = 4*np.absolute(_sig_phys2)*self.sig_phys2_ivar

        # GPM for all sigma measurements
        sig_gpm = np.logical_not(self.sig_mask) & (_sig_phys2 > 0)

        # Get all measurements, independent of azimuth
        indx = sig_gpm
        sprof_r = r[indx]
        sprof = sig[indx]
        sprof_wgt = sigwgt[indx]
        sprof_mod = None if sig_mod is None else np.ma.sqrt(_sig_mod[indx]).filled(0.0)

        # Get along the major axis
        indx = sig_gpm & major_gpm
        smaj_r = r[indx]
        smaj = sig[indx]
        smaj_wgt = sigwgt[indx]
        smaj_mod = None if sig_mod is None else np.ma.sqrt(_sig_mod[indx]).filled(0.0)

        # Get along the minor axis
        indx = sig_gpm & minor_gpm
        smin_r = r[indx]
        smin = sig[indx]
        smin_wgt = sigwgt[indx]
        smin_mod = None if sig_mod is None else np.ma.sqrt(_sig_mod[indx]).filled(0.0)

        # Dispersion profiles
        _, _, _, _, _, _, _, sprof_ewmean, sprof_ewsdev, _, sprof_ntot, sprof_nbin, sprof_bin_gpm \
                = bin_stats(sprof_r, sprof, binr, binw, wgts=sprof_wgt, fill_value=0.0) 
        _, _, _, _, _, _, _, smaj_ewmean, smaj_ewsdev, _, smaj_ntot, smaj_nbin, smaj_bin_gpm \
                = bin_stats(smaj_r, smaj, binr, binw, wgts=smaj_wgt, fill_value=0.0) 
        _, _, _, _, _, _, _, smin_ewmean, smin_ewsdev, _, smin_ntot, smin_nbin, smin_bin_gpm \
                = bin_stats(smin_r, smin, binr, binw, wgts=smin_wgt, fill_value=0.0) 
        if sig_mod is None:
            sprofm_ewmean = sprofm_ewsdev = None
            smajm_ewmean = smajm_ewsdev = None
            sminm_ewmean = sminm_ewsdev = None
        else:
            _, _, _, _, _, _, _, sprofm_ewmean, sprofm_ewsdev, _, _, _, _ \
                    = bin_stats(sprof_r[sprof_bin_gpm], sprof_mod[sprof_bin_gpm], binr, binw,
                                wgts=sprof_wgt[sprof_bin_gpm], fill_value=0.0) 
            _, _, _, _, _, _, _, smajm_ewmean, smajm_ewsdev, _, _, _, _ \
                    = bin_stats(smaj_r[smaj_bin_gpm], smaj_mod[smaj_bin_gpm], binr, binw,
                                wgts=smaj_wgt[smaj_bin_gpm], fill_value=0.0) 
            _, _, _, _, _, _, _, sminm_ewmean, sminm_ewsdev, _, _, _, _ \
                    = bin_stats(smin_r[smin_bin_gpm], smin_mod[smin_bin_gpm], binr, binw,
                                wgts=smin_wgt[smin_bin_gpm], fill_value=0.0) 

        return binr, vrot_ewmean, vrot_ewsdev, vrot_ntot, vrot_nbin, \
                     vrotm_ewmean, vrotm_ewsdev, \
                     vrad_ewmean, vrad_ewsdev, vrad_ntot, vrad_nbin, \
                     vradm_ewmean, vradm_ewsdev, \
                     sprof_ewmean, sprof_ewsdev, sprof_ntot, sprof_nbin, \
                     sprofm_ewmean, sprofm_ewsdev, \
                     smaj_ewmean, smaj_ewsdev, smaj_ntot, smaj_nbin, \
                     smajm_ewmean, smajm_ewsdev, \
                     smin_ewmean, smin_ewsdev, smin_ntot, smin_nbin, \
                     sminm_ewmean, sminm_ewsdev

