"""
Two-dimensional binning routines.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""
import warnings

import numpy as np
from matplotlib import pyplot

from vorbin.voronoi_2d_binning import voronoi_2d_binning
from .util import get_map_bin_transformations, fill_matrix

class VoronoiBinning:
    """
    Class that wraps Voronoi binning code.

    This is (mostly) copied from the MaNGA Data Analysis Pipeline.
    """
    def __init__(self):
        self.covar = None

    def sn_calculation_no_covariance(self, index, signal, noise): 
        """
        S/N calculation for independent data.

        Args:
            index (`numpy.ndarray`_):
                Indices of the measurements in a single bin.
            signal (`numpy.ndarray`_):
                The signal measurements.
            noise (`numpy.ndarray`_):
                The noise measurements.

        Returns:
            :obj:`float`: The nominal signal-to-noise reached by
            summing the measurements selected by ``index``.
        """
        return np.sum(signal[index]) / np.sqrt(np.sum(noise[index]**2))

    def sn_calculation_covariance_matrix(self, index, signal, noise):
        """
        Calculate the S/N using a full covariance matrix.

        The method uses the internal :attr:`covar`.

        Args:
            index (`numpy.ndarray`_):
                Indices of the measurements in a single bin.
            signal (`numpy.ndarray`_):
                The signal measurements.
            noise (`numpy.ndarray`_):
                The noise measurements.

        Returns:
            :obj:`float`: The nominal signal-to-noise reached by
            summing the measurements selected by ``index``, including
            any covariance.
        """
        return np.sum(signal[index]) / np.sqrt(np.sum(self.covar[np.ix_(index, index)]))

    @classmethod
    def bin_index(cls, x, y, signal, noise, target_snr, show=False):
        """
        Bin the data and return the indices of the bins.

        Args:
            x (`numpy.ndarray`_):
                Fiducial Cartesian X position.  Shape must match ``signal``.
            y (`numpy.ndarray`_):
                Fiducial Cartesian Y position.  Shape must match ``signal``.
            signal (`numpy.ndarray`_):
                The signal measurements.  Shape must be 1D.
            noise (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                The noise measurements.  Shape can be 1D or 2D.  If 2D, assumed
                to be a covariance matrix.  The 1D size or the length along one
                axis of a 2D array must match ``signal``.
            target_snr (:obj:`float`):
                Target minimum S/N for the bins.
            show (:obj:`bool`, optional):
                Show the default plot with the binning results.

        Returns:
            `numpy.ndarray`_: An integer bin index for each position.

        Raises:
            ValueError:
                Raised if the sizes of ``x`` and ``y`` do not match, or if
                various checks of the signal, noise, and/or covariance elements
                are incorrectly matched.
        """
        self = cls()

        # Check the position input
        if signal.ndim != 1:
            raise ValueError('Signal values must be in a 1D array.')
        if signal.shape != x.shape:
            raise ValueError('Shape of signal does not match coordinates.')
        if x.shape != y.shape:
            raise ValueError('Shape of x and y coordinates do not match!')
        if noise.ndim == 1:
            if noise.shape != x.shape:
                raise ValueError('Shape of noise does not match coordinates.')
            _noise = np.atleast_1d(noise)
            self.covar = None
            sn_func = self.sn_calculation_no_covariance
        if noise.ndim == 2:
            if noise.shape[0] != x.size:
                raise ValueError('Single axis length of covariance does not match coordinates.')
            if noise.shape[0] != noise.shape[1]:
                raise ValueError('Covariance arrays must be square.')
            _noise = np.sqrt(np.diag(noise))
            self.covar = noise.copy()
            sn_func = self.sn_calculation_covariance_matrix

        # All spaxels have S/N greater than threshold, so return each
        # spaxel in its own "bin"
        if np.min(signal/_noise) > target_snr:
            warnings.warn('All pixels have enough S/N. Binning is not needed')
            return np.arange(signal.size)

        # Cannot reach the S/N using all spaxels, so return all spaxels
        # in a single bin
        sn_total = sn_func(np.arange(signal.size), signal, _noise)
        if sn_total < target_snr:
            warnings.warn('Cannot reach target S/N using all data; all data included in one bin.')
            return np.zeros(signal.size)

        # Bin the data
        try:
            binid, xNode, yNode, xBar, yBar, sn, area, scale = \
                    voronoi_2d_binning(x, y, signal, _noise, target_snr, sn_func=sn_func,
                                       plot=show)
            if show:
                pyplot.show()
        except:
            warnings.warn('Binning algorithm has raised an exception.  Assume this is because '
                          'all the spaxels should be in the same bin.')
            binid = numpy.zeros(signal.size)

        return binid


# TODO:
#   - Include an optional weight map?
#   - Include covariance
class Bin2D:
    r"""
    A utility class for handling two-dimensional binning.

    The core functionality of the class is to compute a set of transformations
    using :func:`~nirvana.data.util.get_map_bin_transformations` and provide
    convenience methods that apply and revert those transformations.  See that
    function for further documentation and argument/attribute descriptions.

    Args:
        spatial_shape (:obj:`tuple`, optional):
            The 2D spatial shape of the mapped data.  Ignored if ``binid`` is
            provided.
        binid (`numpy.ndarray`_, optional):
            The 2D array providing the 0-indexed bin ID number associated with
            each map element. Bin IDs of -1 are assumed to be ignored; no bin ID
            can be less than -1. Shape is ``spatial_shape`` and its size (i.e.
            the number of grid points in the map) is :math:`N_{\rm spaxel}`.

    Attributes:
        spatial_shape (:obj:`tuple`):
            2D array shape
        ubinid (`numpy.ndarray`_):
            1D vector with the sorted list of *unique* bin IDs. Shape is
            :math:`(N_{\rm bin},)`. If ``binid`` is not provided on
            instantiation, this is None.
        nbin (`numpy.ndarray`_):
            1D vector with the number of spaxels in each bin. Shape is
            :math:`(N_{\rm bin},)`. If ``binid`` is not provided on
            instantiation, this is just a vector of ones. The number of bins can
            also be determined from :attr:`bin_transform`.
        ubin_indx (`numpy.ndarray`_):
            The index vector used to select the unique bin values from a
            flattened map of binned data, *excluding* any element with ``binid
            == -1``. Shape is :math:`(N_{\rm bin},)`. If ``binid`` is not
            provided on instantiation, this is identical to :attr:`grid_indx`.
        grid_indx (`numpy.ndarray`_):
            The index vector used to select valid grid cells in the input maps;
            i.e., any grid point with a valid bin ID (``binid != -1``). Shape is
            :math:`(N_{\rm valid},)`.
        bin_inverse (`numpy.ndarray`_):
            The index vector applied to a recover the mapped data given the
            unique quantities, when used in combination with :attr:`grid_indx`.
            Shape is :math:`(N_{\rm valid},)`.
        bin_transform (`scipy.sparse.csr_matrix`_):
            A sparse matrix used to construct the binned (averaged) quantities
            from a full 2D map. Shape is :math:`(N_{\rm bin}, N_{\rm spaxel})`.
        unravel (:obj:`tuple`):
            The :obj:`tuple` of `numpy.ndarray`_ objects that provide the
            indices of :attr:`grid_indx` in the 2D array.
    """
    def __init__(self, spatial_shape=None, binid=None):
        self.spatial_shape = spatial_shape if binid is None else binid.shape
        self.ubinid, self.nbin, self.ubin_indx, self.grid_indx, self.bin_inverse, \
                self.bin_transform = get_map_bin_transformations(spatial_shape=spatial_shape,
                                                                 binid=binid)
        self.unravel = np.unravel_index(self.grid_indx, self.spatial_shape)

    def bin(self, data):
        """
        Provided a set of mapped data, bin it according to the internal bin ID
        map.

        Based on the construction of :attr:`bin_transform` by 
        :func:`~nirvana.data.util.get_map_bin_transformations`, this computes
        the average value of the data in each bin.

        Args:
            data (`numpy.ndarray`_):
                Data to bin. Shape must match :attr:`spatial_shape`.

        Returns:
            `numpy.ndarray`_: A vector with the binned data.

        Raises:
            ValueError:
                Raised if the shape of the input array is incorrect.
        """
        if data.shape != self.spatial_shape:
            raise ValueError('Data to rebin has incorrect shape; expected {0}, found {1}.'.format(
                              self.spatial_shape, data.shape))
        return self.bin_transform.dot(data.ravel())

    def deriv_bin(self, data, deriv):
        """
        Provided a set of mapped data, rebin it to match the internal vectors
        and propagate the derivatives in the data.

        This method is most often used to bin maps of model data to match the
        binning of observed data.

        This method is identical to :func:`bin`, except that it allows for
        propagation of derivatives of the provided model with respect to its
        parameters.  The propagation of derivatives for any single parameter is
        identical to calling :func:`bin` on that derivative map.
        
        Args:
            data (`numpy.ndarray`_):
                Data to rebin. Shape must match :attr:`spatial_shape`.
            deriv (`numpy.ndarray`_):
                If the input data is a model, this provides the derivatives of
                model w.r.t. its parameters.  The shape must be 3D and the first
                two axes of the array must have a shape that matches
                :attr:`spatial_shape`.

        Returns:
            :obj:`tuple`: Two `numpy.ndarray`_ arrays.  The first provides the
            vector with the data rebinned to match the number of unique
            measurements available, and the second is a 2D array with the binned
            derivatives for each model parameter.

        Raises:
            ValueError:
                Raised if the spatial shapes of the input arrays are incorrect.
        """
        if data.shape != self.spatial_shape:
            raise ValueError('Data to rebin has incorrect shape; expected {0}, found {1}.'.format(
                              self.spatial_shape, data.shape))
        if deriv.shape[:2] != self.spatial_shape:
            raise ValueError('Derivative shape is incorrect; expected {0}, found {1}.'.format(
                              self.spatial_shape, deriv.shape[:2]))
        return self.bin_transform.dot(data.ravel()), \
                    np.stack(tuple([self.bin_transform.dot(deriv[...,i].ravel())
                                    for i in range(deriv.shape[-1])]), axis=-1)

    def bin_covar(self, covar):
        """
        Calculate the covariance in the binned data provided the unbinned
        covariance.

        Args:
            covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance in the unbinned data.

        Returns:
            `scipy.sparse.csr_matrix`_: Covariance in the binned data.
        """
        return self.bin_transform.dot(covar.dot(self.bin_transform.T))

    # TODO: Include error calculations?
    def bin_moments(self, norm, center, stddev):
        r"""
        Bin a set of Gaussian moments.

        Assuming the provided data are the normalization, mean, and standard
        deviation of a set of Gaussian profiles, this method performs a nominal
        calculation of the moments of the summed Gaussian profile.

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
        if all([x is None for x in [norm, center, stddev]]):
            raise ValueError('At least one parameter must not be None.')
        shape = [x.shape for x in [norm, center, stddev] if x is not None][0]
        if not all([x is None or x.shape == shape for x in [norm, center, stddev]]):
            raise ValueError('Shape of all input arrays must match.')
        _norm = np.ones(shape, dtype=float) if norm is None else norm
        mom0 = self.bin(_norm)
        if center is None and stddev is None:
            return mom0, None, None
        inv_mom0 = 1/(mom0 + (mom0 == 0.))
        _center = np.zeros(shape, dtype=float) if center is None else center
        mom1 = self.bin(_norm*_center) * inv_mom0
        if stddev is None:
            return mom0, mom1, None
        _var = _center**2 + stddev**2
        mom2 = self.bin(_norm*_var) * inv_mom0 - mom1**2
        mom2[mom2 < 0] = 0.
        return mom0, mom1, np.sqrt(mom2)

    def deriv_bin_moments(self, norm, center, stddev, dnorm, dcenter, dstddev):
        r"""
        Bin a set of Gaussian moments and propagate the calculation for the
        derivatives.

        This method is identical to :func:`bin_moments`, except it includes the
        propagation of the derivatives.

        .. note::

            Any of the first three input arguments can be None, but at least one of them
            cannot be!

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
        # Check input
        if all([x is None for x in [norm, center, stddev]]):
            raise ValueError('At least one parameter must not be None.')
        shape = [x.shape for x in [norm, center, stddev] if x is not None][0]
        if not all([x is None or x.shape == shape for x in [norm, center, stddev]]):
            raise ValueError('Shape of all input arrays must match.')
        if norm is None and dnorm is not None:
            raise ValueError('Must provide normalization if providing its derivative.')
        if center is None and dcenter is not None:
            raise ValueError('Must provide center if providing its derivative.')
        if stddev is None and dstddev is not None:
            raise ValueError('Must provide standard deviation if providing its derivative.')
        npar = [x.shape[-1] for x in [dnorm, dcenter, dstddev] if x is not None][0]
        if any([x.shape != shape + (npar,) for x in [dnorm, dcenter, dstddev] if x is not None]):
            raise ValueError('All derivative arrays must have the same shape.')

        _norm = np.ones(shape, dtype=float) if norm is None else norm
        mom0 = self.bin(_norm)
        dmom0 = None if dnorm is None else \
                    np.stack(tuple([self.bin_transform.dot(dnorm[...,i].ravel())
                                        for i in range(npar)]), axis=-1)
        if center is None and stddev is None:
            # Everything else is None, so we're done.
            return mom0, None, None, dmom0, None, None
        
        inv_mom0 = 1/(mom0 + (mom0 == 0.))
        _center = np.zeros(shape, dtype=float) if center is None else center
        mom1 = self.bin(_norm*_center) * inv_mom0
        dmom1 = None
        if dnorm is not None:
            dmom1 = np.stack(tuple([(self.bin_transform.dot((_center*dnorm[...,i]).ravel())
                                      - mom1 * dmom0[...,i]) * inv_mom0
                                     for i in range(npar)]), axis=-1)
        if dcenter is not None:
            _dmom1 = np.stack(tuple([self.bin_transform.dot((_norm*dcenter[...,i]).ravel())
                                     * inv_mom0 for i in range(npar)]), axis=-1)
            dmom1 = _dmom1 if dmom1 is None else dmom1 + _dmom1
        if stddev is None:
            return mom0, mom1, None, dmom0, dmom1, None

        _var = _center**2 + stddev**2
        mom2 = self.bin(_norm*_var) * inv_mom0 - mom1**2
        mom2[mom2 < 0] = 0.
        _mom2 = np.sqrt(mom2)
        dmom2 = None
        if dnorm is not None:
            dmom2 = np.stack(tuple([(self.bin_transform.dot((_var*dnorm[...,i]).ravel())
                                     - mom2 * dmom0[...,i]) * inv_mom0
                                    for i in range(npar)]), axis=-1) 
        if dcenter is not None:
            _dmom2 = np.stack(tuple([self.bin_transform.dot(
                                        (2*_norm*_center*dcenter[...,i]).ravel()) * inv_mom0
                                     - 2 * mom1 * dmom1[...,i] for i in range(npar)]), axis=-1)
            dmom2 = _dmom2 if dmom2 is None else dmom2 + _dmom2
        if dstddev is not None:
            _dmom2 = np.stack(tuple([self.bin_transform.dot(
                                        (2*_norm*stddev*dstddev[...,i]).ravel()) * inv_mom0
                                    for i in range(npar)]), axis=-1) 
            dmom2 = _dmom2 if dmom2 is None else dmom2 + _dmom2
        if dmom2 is not None:
            _inv_mom2 = 1./(_mom2 + (_mom2 == 0.0))
            dmom2 = dmom2 * _inv_mom2[...,None] / 2
        return mom0, mom1, _mom2, dmom0, dmom1, dmom2

    def remap(self, data, mask=None, masked=True, fill_value=0):
        """
        Provided the vector of binned data, reconstruct the 2D map filling each
        pixel in the same bin with the binned value.

        Args:
            data (`numpy.ndarray`_):
                The data to remap.  Shape must match the number of unique bin
                IDs.
            mask (`numpy.ndarray`_, optional):
                Boolean mask with the same shape as ``data``.  If None, all data
                are unmasked.
            masked (:obj:`bool`, optional):
                Return data as a masked array, where any pixel not associated
                with a bin is masked (in addition to the provided ``mask``.)
            fill_value (scalar-like, optional):
                Value used to fill the masked pixels, if a masked array is
                *not* requested. Warning: The value is automatically
                converted to be the same data type as the input array or
                attribute.

        Returns:
            `numpy.ndarray`_, `numpy.ma.MaskedArray`_: 2D array with
            the data remapped to a 2D array.

        Raises:
            ValueError:
                Raised if shape of ``data`` does not match the expected 1d
                shape.
        """
        # Check the shapes
        if data.shape != self.nbin.shape:
            raise ValueError('To remap, data must have the same shape as the internal data '
                             'attributes: {0}'.format(self.nbin.shape))
        if mask is not None and mask.shape != self.nbin.shape:
            raise ValueError('To remap, mask must have the same shape as the internal data '
                             'attributes: {0}'.format(self.nbin.shape))

        # Construct the output map
        # NOTE: np.ma.masked_all sets the initial data array to 2.17506892e-314,
        # which just leads to trouble. I've instead used the line below to make
        # sure that the initial value is just 0.
        _data = np.ma.MaskedArray(np.zeros(self.spatial_shape, dtype=data.dtype), mask=True)
        _data[self.unravel] = data[self.bin_inverse]
        if mask is not None:
            np.ma.getmaskarray(_data)[self.unravel] = mask[self.bin_inverse]
        # Return a masked array if requested; otherwise, fill the masked values
        # with the type-cast fill value. WARNING: the default value of
        # fill_value=0 will mean fill_value=False for a boolean array.
        return _data if masked else _data.filled(_data.dtype.type(fill_value))

    def remap_covar(self, covar):
        """
        Remap a covariance matrix from the binned data to the unbinned map.

        Pixels in the same bin are perfectly correlated.

        Args:
            covar (`numpy.ndarray`_, `scipy.sparse.csr_matrix`_):
                Covariance matrix to remap.

        Returns:
            `scipy.sparse.csr_matrix`_: A sparse matrix for the remapped
            covariance.
        """
        gpm = np.ones(self.nbin, dtype=bool) if self.ubinid is None \
                    else self.remap(self.ubinid, masked=False, fill_value=-1) > -1
        _bt = self.bin_transform[:,gpm.ravel()].T
        _bt[_bt > 0] = 1.
        return fill_matrix(_bt.dot(covar.dot(_bt.T)), gpm.ravel())

    def unique(self, data):
        """
        Provided a 2D array of data binned according to :attr:`binid`, select
        and return the unique values from the map.

        The difference between this method and :func:`bin` is that, instead of
        averaging all the data within a bin, this method simply returns the
        value for a single pixel within the bin region.

        Args:
            data (`numpy.ndarray`_):
                The 2D data array from which to extract the unique data.
                Shape must be :attr:`spatial_shape`.

        Returns:
            `numpy.ndarray`_: The 1D vector with the unique data.

        Raises:
            ValueError:
                Raised if the input array shape is wrong.
        """
        if data.shape != self.spatial_shape:
            raise ValueError(f'Input has incorrect shape; found {data.shape}, '
                             f'expected {self.spatial_shape}.')
        return data.flat[self.ubin_indx]


