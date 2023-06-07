"""
Module with classes and functions used to fit an axisymmetric disk to a set of kinematics.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

import os
import warnings

from IPython import embed

import numpy as np
from matplotlib import pyplot, rc, patches, ticker, colors

from astropy.io import fits

from .geometry import projected_polar, deriv_projected_polar
from .beam import smear, deriv_smear
from . import oned 
from . import asymmetry
from ..data.kinematics import Kinematics
from ..data.scatter import IntrinsicScatter
from ..data.util import inverse, find_largest_coherent_region
from ..data.util import select_kinematic_axis, bin_stats, growth_lim, atleast_one_decade
from ..util import plot
from ..util import fileio

from .thindisk import ThinDisk

#warnings.simplefilter('error', RuntimeWarning)

class AxisymmetricDisk(ThinDisk):
    r"""
    Simple model for an axisymmetric disk.

    The model assumes the disk is infinitely thin and has a single set of
    geometric parameters:

        - :math:`x_c, y_c`: The coordinates of the galaxy dynamical center.
        - :math:`\phi`: The position angle of the galaxy (the angle from N
          through E)
        - :math:`i`: The inclination of the disk; the angle of the disk
          normal relative to the line-of-sight such that :math:`i=0` is a
          face-on disk.
        - :math:`V_{\rm sys}`: The systemic (bulk) velocity of the galaxy
          taken as the line-of-sight velocity at the dynamical center.

    In addition to these parameters, the model instantiation requires class
    instances that define the rotation curve and velocity dispersion profile.
    These classes must have:

        - an ``np`` attribute that provides the number of parameters in the
          model
        - a ``guess_par`` method that provide initial guess parameters for
          the model, and
        - ``lb`` and ``ub`` attributes that provide the lower and upper
          bounds for the model parameters.

    Importantly, note that the model fits the parameters for the *projected*
    rotation curve. I.e., that amplitude of the fitted function is actually
    :math:`V_\theta\ \sin i`.

    .. todo::
        Describe the attributes

    Args:
        rc (:class:`~nirvana.models.oned.Func1D`, optional):
            The parameterization to use for the disk rotation curve.  If None,
            defaults to :class:`~nirvana.models.oned.HyperbolicTangent`.
        dc (:class:`~nirvana.models.oned.Func1D`, optional):
            The parameterization to use for the disk dispersion profile.  If
            None, the dispersion profile is not included in the fit!

    """
    def __init__(self, rc=None, dc=None):
        # Rotation curve
        self.rc = oned.HyperbolicTangent() if rc is None else rc
        # Velocity dispersion curve (can be None)
        self.dc = dc

        # Instantiate the base class, which basically keeps all of the geometric
        # parameters.  NOTE: The parametric curves above need to be defined
        # first because instatiation of the base class calls reinit(), which
        # sets default guess parameters.  TODO: Consider instantiating without
        # guess parameters...
        super().__init__()

        # Total number parameters
        self.np = self.nbp + self.rc.np
        if self.dc is not None:
            self.np += self.dc.np
        # Flag which parameters are freely fit
        self.free = np.ones(self.np, dtype=bool)
        self.nfree = np.sum(self.free)

    def guess_par(self):
        """
        Return a list of generic guess parameters.

        .. todo::
            Could enable this to base the guess on the data to be fit, but at
            the moment these are hard-coded numbers.

        Returns:
            `numpy.ndarray`_: Vector of guess parameters
        """
        gp = np.concatenate((super().guess_par(), self.rc.guess_par()))
        return gp if self.dc is None else np.append(gp, self.dc.guess_par())

    def par_names(self, short=False):
        """
        Return a list of strings with the parameter names.

        Args:
            short (:obj:`bool`, optional):
                Return truncated nomenclature for the parameter names.

        Returns:
            :obj:`list`: List of parameter name strings.
        """
        base = super().par_names(short=short)
        if short:
            rc = [f'v_{p}' for p in self.rc.par_names(short=True)]
            dc = [] if self.dc is None else [f's_{p}' for p in self.dc.par_names(short=True)]
        else:
            rc = [f'RC: {p}' for p in self.rc.par_names()]
            dc = [] if self.dc is None else [f'Disp: {p}' for p in self.dc.par_names()]
        return base + rc + dc

    # These "private" functions yield slices of the parameter vector for the
    # desired set of parameters
    def _rc_slice(self):
        return slice(self.nbp, self.nbp + self.rc.np)
    def _dc_slice(self):
        s = self.nbp + self.rc.np
        return slice(s, s + self.dc.np)

    def rc_par(self, err=False):
        """
        Return the rotation curve parameters. Returns None if parameters are
        not defined yet.

        Args:
            err (:obj:`bool`, optional):
                Return the parameter errors instead of the parameter values.

        Returns:
            `numpy.ndarray`_: Vector with parameters or parameter errors for the
            rotation curve.
        """
        p = self.par_err if err else self.par
        return None if p is None else p[self._rc_slice()]

    def dc_par(self, err=False):
        """
        Return the dispersion profile parameters. Returns None if parameters
        are not defined yet or if no dispersion profile has been defined.

        Args:
            err (:obj:`bool`, optional):
                Return the parameter errors instead of the parameter values.

        Returns:
            `numpy.ndarray`_: Vector with parameters or parameter errors for the
            dispersion profile.
        """
        p = self.par_err if err else self.par
        return None if p is None or self.dc is None else p[self._dc_slice()]

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
        _base_lb, _base_ub = super().par_bounds(base_lb=base_lb, base_ub=base_ub)
        # Minimum and maximum allowed values for xc, yc, pa, inc, vsys, vrot, hrot
        lb = np.concatenate((_base_lb, self.rc.lb))
        ub = np.concatenate((_base_ub, self.rc.ub))
        return (lb, ub) if self.dc is None \
                    else (np.append(lb, self.dc.lb), np.append(ub, self.dc.ub))

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
        # Initialize the thin disk model
        self._init_model(par, x, y, sb, beam, is_fft, cnvfftw, ignore_beam)

        # Calculate the in-disk coordinates
        r, theta = projected_polar(self.x - self.par[0], self.y - self.par[1],
                                   *np.radians(self.par[2:4]))

        # NOTE: The velocity-field construction does not include the
        # sin(inclination) term because this is absorbed into the
        # rotation curve amplitude.
        vel = self.rc.sample(r, par=self.par[self._rc_slice()])*np.cos(theta) + self.par[4]
        if self.dc is None:
            # Only fitting the velocity field
            return vel if self.beam_fft is None or ignore_beam \
                        else smear(vel, self.beam_fft, beam_fft=True, sb=self.sb,
                                   cnvfftw=self.cnvfftw)[1]

        # Fitting both the velocity and velocity-dispersion field
        sig = self.dc.sample(r, par=self.par[self._dc_slice()])
        return (vel, sig) if self.beam_fft is None or ignore_beam \
                        else smear(vel, self.beam_fft, beam_fft=True, sb=self.sb, sig=sig,
                                   cnvfftw=self.cnvfftw)[1:]

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
        # Initialize the thin disk model
        self._init_model(par, x, y, sb, beam, is_fft, cnvfftw, ignore_beam)

        # Initialize the derivative arrays needed for the coordinate calculation
        dx = np.zeros(self.x.shape+(self.np,), dtype=float)
        dy = np.zeros(self.x.shape+(self.np,), dtype=float)
        dpa = np.zeros(self.np, dtype=float)
        dinc = np.zeros(self.np, dtype=float)

        dx[...,0] = -1.
        dy[...,1] = -1.
        dpa[2] = np.radians(1.)
        dinc[3] = np.radians(1.)

        r, theta, dr, dtheta = deriv_projected_polar(self.x - self.par[0], self.y - self.par[1],
                                                     *np.radians(self.par[2:4]), dxdp=dx, dydp=dy,
                                                     dpadp=dpa, dincdp=dinc)

        # NOTE: The velocity-field construction does not include the
        # sin(inclination) term because this is absorbed into the
        # rotation curve amplitude.

        # Calculate the rotation speed and its parameter derivatives
        slc = self._rc_slice()
        dvrot = np.zeros(self.x.shape+(self.np,), dtype=float)
        vrot, dvrot[...,slc] = self.rc.deriv_sample(r, par=self.par[slc])
        dvrot += self.rc.ddx(r, par=self.par[slc])[...,None]*dr

        # Calculate the line-of-sight velocity and its parameter derivatives
        cost = np.cos(theta)
        v = vrot*cost + self.par[4]
        dv = dvrot*cost[...,None] - (vrot*np.sin(theta))[...,None]*dtheta
        dv[...,4] = 1.

        if self.dc is None:
            # Only fitting the velocity field
            if self.beam_fft is None or ignore_beam:
                # Not smearing
                return v, dv
            # Smear and propagate through the derivatives
            _, v, _, _, dv, _ = deriv_smear(v, dv, self.beam_fft, beam_fft=True, sb=self.sb,
                                            cnvfftw=self.cnvfftw)
            return v, dv

        # Fitting both the velocity and velocity-dispersion field

        # Calculate the dispersion profile and its parameter derivatives
        slc = self._dc_slice()
        dsig = np.zeros(self.x.shape+(self.np,), dtype=float)
        sig, dsig[...,slc] = self.dc.deriv_sample(r, par=self.par[slc])
        dsig += self.dc.ddx(r, par=self.par[slc])[...,None]*dr

        if self.beam_fft is None or ignore_beam:
            # Not smearing
            return v, sig, dv, dsig

        # Smear and propagate through the derivatives
        _, v, sig, _, dv, dsig = deriv_smear(v, dv, self.beam_fft, beam_fft=True, sb=self.sb,
                                             sig=sig, dsig=dsig, cnvfftw=self.cnvfftw)
        return v, sig, dv, dsig

    def report(self, fit_message=None):
        """
        Report the current parameters of the model to the screen.

        Args:
            fit_message (:obj:`str`, optional):
                The status message returned by the fit optimization.
        """
        if self.par is None:
            print('No parameters to report.')
            return

        vfom, sfom = self._get_fom()(self.par, sep=True)
        parn = self.par_names()
        max_parn_len = max([len(n) for n in parn])+4

        print('-'*70)
        print(f'{"Fit Result":^70}')
        print('-'*70)
        if fit_message is not None:
            print(f'Fit status message: {fit_message}')
        if self.fit_status is not None:
            print(f'Fit status: {self.fit_status}')
        print(f'Fit success: {"True" if self.fit_status else "False"}')
        print('-'*10)
        ps = 0
        pe = self.nbp
        print(f'Base parameters:')
        for i in range(ps,pe):
            print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[i]:.1f}').format(parn[i])
                    + (f'' if self.par_err is None else f' +/- {self.par_err[i]:.1f}'))
        print('-'*10)
        ps = self.nbp
        pe = ps + self.rc.np
        print(f'Rotation curve parameters:')
        for i in range(ps,pe):
            print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[i]:.1f}').format(parn[i])
                    + (f'' if self.par_err is None else f' +/- {self.par_err[i]:.1f}'))
        if self.dc is None:
            print('-'*10)
            if self.scatter is not None:
                print(f'Intrinsic Velocity Scatter: {self.scatter[0]:.1f}')
            vchisqr = np.sum(vfom**2)
            print(f'Velocity measurements: {len(vfom)}')
            print(f'Velocity chi-square: {vchisqr}')
            print(f'Reduced chi-square: {vchisqr/(len(vfom)-self.nfree)}')
            print('-'*70)
            return
        print('-'*10)
        ps = self.nbp+self.rc.np
        pe = ps + self.dc.np
        print(f'Dispersion profile parameters:')
        for i in range(ps,pe):
            print(('{0:>'+f'{max_parn_len}'+'}'+ f': {self.par[i]:.1f}').format(parn[i])
                    + (f'' if self.par_err is None else f' +/- {self.par_err[i]:.1f}'))
        print('-'*10)
        if self.scatter is not None:
            print(f'Intrinsic Velocity Scatter: {self.scatter[0]:.1f}')
        vchisqr = np.sum(vfom**2)
        print(f'Velocity measurements: {len(vfom)}')
        print(f'Velocity chi-square: {vchisqr}')
        if self.scatter is not None:
            print(f'Intrinsic Dispersion**2 Scatter: {self.scatter[1]:.1f}')
        schisqr = np.sum(sfom**2)
        print(f'Dispersion measurements: {len(sfom)}')
        print(f'Dispersion chi-square: {schisqr}')
        print(f'Reduced chi-square: {(vchisqr + schisqr)/(len(vfom) + len(sfom) - self.nfree)}')
        print('-'*70)

# TODO:
#   - This is MaNGA-specific and needs to be abstracted
#   - Copy over the DataTable class from the DAP, or use an astropy.table.Table?
def _fit_meta_dtype(par_names, nr):
    """
    Set the data type for a `numpy.recarray`_ used to hold metadata of the
    best-fit model.

    Args:
        par_names (array-like):
            Array of strings with the short names for the model parameters.
        nr (:obj:`int`):
            Number radial bins for azimuthally averaged profiles

    Returns:
        :obj:`list`: The list of tuples providing the name, data type, and shape
        of each `numpy.recarray`_ column.
    """
    gp = [(f'G_{n}'.upper(), np.float) for n in par_names]
    bp = [(f'F_{n}'.upper(), np.float) for n in par_names]
    bpe = [(f'E_{n}'.upper(), np.float) for n in par_names]
    
    return [('MANGAID', '<U30'),
            ('PLATE', np.int16),
            ('IFU', np.int16),
            ('OBJRA', np.float),
            ('OBJDEC', np.float),
            # Redshift used by the DAP to (nominally) offset the velocity field
            # to 0 bulk velocity.
            ('Z', np.float),
            # The conversion factor used to convert arcseconds to kpc
            ('ASEC2KPC', np.float),
            # NSA-fit effective radius in arcseconds
            ('REFF', np.float),
            # NSA-fit Sersic index
            ('SERSICN', np.float),
            # NSA-fit position angle
            ('PA', np.float),
            # NSA-fit ellipticity
            ('ELL', np.float),
            # Assumed intrinsic oblateness
            ('Q0', np.float),
            # VNFIT is the total number of velocity measurements included in the
            # fit.
            ('VNFIT', np.int),
            # VNMSK is the number of velocity measurements masked for any
            # reason, including those measurements that were already masked by
            # the DAP.
            ('VNMSK', np.int),
            # VNFLAG is the number of velocity measurements masked by the fit for
            # any reason, meaning it does *not* include data already masked by
            # the DAP.
            ('VNFLAG', np.int),
            # VNREJ is the number of velocity measurements masked by the fit
            # only due to outlier rejection.
            ('VNREJ', np.int),
            # VMEDE is the median observed error in the data included in the
            # fit.
            ('VMEDE', np.float),
            # VMENR is the mean of the error-normalized residuals of the data
            # included in the fit.
            ('VMENR', np.float),
            # VSIGR is the standard deviation of the error-normalized residuals
            # of the data included in the fit.
            ('VSIGR', np.float),
            # VGRWR is the 1-, 2-, and 3-sigma growth and the maximum value of
            # the error-normalized residuals of the data included in the fit.
            ('VGRWR', np.float, (4,)),
            # VISCT is the intrinsic scatter term used in the fit.
            ('VISCT', np.float),
            # VSIGIR is the same as VSIGR but includes the intrinsic scatter
            # modification of the erro.
            ('VSIGIR', np.float),
            # VGRWIR is the same as VGRWR but includes the intrinsic scatter
            # modification of the erro.
            ('VGRWIR', np.float, (4,)),
            # VCHI2 is the reduced chi-square only for the velocity data and
            # excluding the instrinsic scatter modification of the error.
            ('VCHI2', np.float),
            # VASYM is the 50%, 80%, and 90% growth of the 3 asymmetry maps
            ('VASYM', np.float, (3,3)),
            # SNFIT is the total number of dispersion measurements included in
            # the fit.
            ('SNFIT', np.int),
            # SNMSK is the number of dispersion measurements masked for any
            # reason, including those measurements that were already masked by
            # the DAP.
            ('SNMSK', np.int),
            # SNFLAG is the number of dispersion measurements masked by the fit
            # for any reason, meaning it does *not* include data already masked
            # by the DAP.
            ('SNFLAG', np.int),
            # SNREJ is the number of dispersion measurements masked by the fit
            # only due to outlier rejection.
            ('SNREJ', np.int),
            # Same as VMEDE, but for the velocity dispersion instead of the
            # velocity.
            ('SMEDE', np.float),
            # Same as VMENR, but for the velocity dispersion instead of the
            # velocity.
            ('SMENR', np.float),
            # Same as VSIGR, but for the velocity dispersion instead of the
            # velocity.
            ('SSIGR', np.float),
            # Same as VGRWR, but for the velocity dispersion instead of the
            # velocity.
            ('SGRWR', np.float, (4,)),
            # Same as VISCT, but for the velocity dispersion instead of the
            # velocity.
            ('SISCT', np.float),
            # Same as VSIGIR, but for the velocity dispersion instead of the
            # velocity.
            ('SSIGIR', np.float),
            # Same as VGRWIR, but for the velocity dispersion instead of the
            # velocity.
            ('SGRWIR', np.float, (4,)),
            # Same as VCHI2, but for the velocity dispersion instead of the
            # velocity.
            ('SCHI2', np.float),
            # Same as VASYM but for the velocity dispersion, instead of the
            # velocity.  Note that sigma is calculated as sigma/sqrt(abs(sigma))
            # as a way of calculating the sqrt while maintaining the sign.
            # TODO: Revisit the sigma asymmetry calculations!
            ('SASYM', np.float, (3,3)),
            # The total chi-square of the fit, includeing the intrinsic scatter
            # modification of the error.
            ('CHI2', np.float),
            # Reduced chi-square
            ('RCHI2', np.float),
            # Status index of the fit returned by scipy.optimize.least_squares
            ('STATUS', np.int),
            # A simple boolean indication that the fit did not fault
            ('SUCCESS', np.int)] + gp + bp + bpe + \
           [('BINR', float, (nr,)),
            ('V_MAJ', float, (nr,)),
            ('V_MAJ_SDEV', float, (nr,)),
            ('V_MAJ_NTOT', float, (nr,)),
            ('V_MAJ_NUSE', float, (nr,)),
            ('V_MAJ_MOD', float, (nr,)),
            ('V_MAJ_MOD_SDEV', float, (nr,)),
            ('V_MIN', float, (nr,)),
            ('V_MIN_SDEV', float, (nr,)),
            ('V_MIN_NTOT', float, (nr,)),
            ('V_MIN_NUSE', float, (nr,)),
            ('V_MIN_MOD', float, (nr,)),
            ('V_MIN_MOD_SDEV', float, (nr,)),
            ('S_ALL', float, (nr,)),
            ('S_ALL_SDEV', float, (nr,)),
            ('S_ALL_NTOT', float, (nr,)),
            ('S_ALL_NUSE', float, (nr,)),
            ('S_ALL_MOD', float, (nr,)),
            ('S_ALL_MOD_SDEV', float, (nr,)),
            ('S_MAJ', float, (nr,)),
            ('S_MAJ_SDEV', float, (nr,)),
            ('S_MAJ_NTOT', float, (nr,)),
            ('S_MAJ_NUSE', float, (nr,)),
            ('S_MAJ_MOD', float, (nr,)),
            ('S_MAJ_MOD_SDEV', float, (nr,)),
            ('S_MIN', float, (nr,)),
            ('S_MIN_SDEV', float, (nr,)),
            ('S_MIN_NTOT', float, (nr,)),
            ('S_MIN_NUSE', float, (nr,)),
            ('S_MIN_MOD', float, (nr,)),
            ('S_MIN_MOD_SDEV', float, (nr,))]


# TODO: This is MaNGA-specific and needs to be abstracted
def axisym_fit_data(galmeta, kin, p0, disk, ofile, vmask, smask):
    """
    Construct a fits file with the best-fit results.

    Args:
        galmeta (:class:`~nirvana.data.meta.GlobalPar`):
            Object with metadata for the galaxy to be fit.
        kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the data to be fit
        p0 (`numpy.ndarray`_):
            Initial guess parameters of the model.
        disk (:class:`~nirvana.models.axisym.AxisymmetricDisk`):
            Object that performed the fit and has the best-fitting parameters.
        ofile (:obj:`str`):
            Output filename.  File names ending in '.gz' will be compressed.
        vmask (`numpy.ndarray`_):
            Vector with the mask bit values for each velocity measurement in
            ``kin``.
        smask (`numpy.ndarray`_):
            Vector with the mask bit values for each dispersion measurement in
            ``kin``.
    """

    # TODO: Add covariance if used

    # Rebuild the 2D maps
    #   - Bin ID
    binid = kin.remap('binid', masked=False, fill_value=-1)
    #   - Disk-plane coordinates
    r, th = projected_polar(kin.grid_x - disk.par[0], kin.grid_y - disk.par[1],
                            *np.radians(disk.par[2:4]))
    #   - Surface-brightness (in per spaxel units not per sq. arcsec).
    didnotuse = disk.mbm.minimum_dtype()(disk.mbm.turn_on(0, flag='DIDNOTUSE'))
    grid_sb = kin.grid_sb
    sb = kin.remap('sb', masked=False, fill_value=0.)
    sb_ivar = kin.remap('sb_ivar', masked=False, fill_value=0.)
    _mask = kin.remap('sb_mask', masked=False, fill_value=True)
    sb_mask = disk.mbm.init_mask_array(sb.shape)
    sb_mask[_mask] = disk.mbm.turn_on(sb_mask[_mask], flag='DIDNOTUSE')

    #   - Velocities
    vel = kin.remap('vel', masked=False, fill_value=0.)
    vel_ivar = kin.remap('vel_ivar', masked=False, fill_value=0.)
    vel_mask = kin.remap(vmask, masked=False, fill_value=didnotuse)
    vel_x, vel_y, vel_xy = asymmetry.onsky_asymmetry_maps(kin.grid_x-disk.par[0],
                                                          kin.grid_y-disk.par[1], vel-disk.par[4],
                                                          pa=disk.par[2], mask=vel_mask>0,
                                                          odd=True, maxd=0.4)
    vel_asym = np.array([vel_x.filled(0.0), vel_y.filled(0.0), vel_xy.filled(0.0)])
    vel_asym_mask = np.array([np.ma.getmaskarray(vel_x).copy(), np.ma.getmaskarray(vel_y).copy(),
                              np.ma.getmaskarray(vel_xy).copy()])

    #   - Corrected velocity dispersion squared
    if disk.dc is None:
        sigsqr = None
        sigsqr_ivar = None
        sigsqr_mask = None
        sig_asym = None
        sig_asym_mask = None
    else:
        sigsqr = kin.remap('sig_phys2', masked=False, fill_value=0.)
        sigsqr_ivar = kin.remap('sig_phys2_ivar', masked=False,fill_value=0.)
        sigsqr_mask = None if smask is None \
                        else kin.remap(smask, masked=False, fill_value=didnotuse)
        sig = sigsqr/np.sqrt(np.absolute(sigsqr))
        sig_x, sig_y, sig_xy \
                = asymmetry.onsky_asymmetry_maps(kin.grid_x-disk.par[0], kin.grid_y-disk.par[1],
                                                 sig, pa=disk.par[2], mask=sigsqr_mask>0,
                                                 maxd=0.4)
        sig_asym = np.array([sig_x.filled(0.0), sig_y.filled(0.0), sig_xy.filled(0.0)])
        sig_asym_mask = np.array([np.ma.getmaskarray(sig_x).copy(),
                                  np.ma.getmaskarray(sig_y).copy(),
                                  np.ma.getmaskarray(sig_xy).copy()])

    # Get the error-weighted mean radial profiles
    models = disk.model()
    if disk.dc is None:
        vmod = kin.bin(models)
        smod = None
    else:
        vmod = kin.bin(models[0])
        smod = kin.bin(models[1])
    fwhm = galmeta.psf_fwhm[1]  # Selects r band!
    oversample = 1.5
    maj_wedge = 30.
    min_wedge = 10.
    binr, vrot_ewmean, vrot_ewsdev, vrot_ntot, vrot_nbin, vrotm_ewmean, vrotm_ewsdev, \
        vrad_ewmean, vrad_ewsdev, vrad_ntot, vrad_nbin, vradm_ewmean, vradm_ewsdev, \
        sprof_ewmean, sprof_ewsdev, sprof_ntot, sprof_nbin, sprofm_ewmean, sprofm_ewsdev, \
        smaj_ewmean, smaj_ewsdev, smaj_ntot, smaj_nbin, smajm_ewmean, smajm_ewsdev, \
        smin_ewmean, smin_ewsdev, smin_ntot, smin_nbin, sminm_ewmean, sminm_ewsdev \
            = kin.radial_profiles(fwhm/oversample, xc=disk.par[0], yc=disk.par[1], pa=disk.par[2],
                                  inc=disk.par[3], vsys=disk.par[4], maj_wedge=maj_wedge,
                                  min_wedge=min_wedge, vel_mod=vmod, sig_mod=smod)

    # Instantiate the single-row table with the metadata:
    disk_par_names = disk.par_names(short=True)
    metadata = fileio.init_record_array(1, _fit_meta_dtype(disk_par_names, binr.size))

    # Fill the fit-independent data
    metadata['MANGAID'] = galmeta.mangaid
    metadata['PLATE'] = galmeta.plate
    metadata['IFU'] = galmeta.ifu
    metadata['OBJRA'] = galmeta.ra
    metadata['OBJDEC'] = galmeta.dec
    metadata['Z'] = galmeta.z
    metadata['ASEC2KPC'] = galmeta.kpc_per_arcsec()
    metadata['REFF'] = galmeta.reff
    metadata['SERSICN'] = galmeta.sersic_n
    metadata['PA'] = galmeta.pa
    metadata['ELL'] = galmeta.ell
    metadata['Q0'] = galmeta.q0

    # Fill the radial profiles
    metadata['BINR'] = binr

    metadata['V_MAJ'] = vrot_ewmean
    metadata['V_MAJ_SDEV'] = vrot_ewsdev
    metadata['V_MAJ_NTOT'] = vrot_ntot
    metadata['V_MAJ_NUSE'] = vrot_nbin
    metadata['V_MAJ_MOD'] = vrotm_ewmean
    metadata['V_MAJ_MOD_SDEV'] = vrotm_ewsdev
    
    metadata['V_MIN'] = vrad_ewmean
    metadata['V_MIN_SDEV'] = vrad_ewsdev
    metadata['V_MIN_NTOT'] = vrad_ntot
    metadata['V_MIN_NUSE'] = vrad_nbin
    metadata['V_MIN_MOD'] = vradm_ewmean
    metadata['V_MIN_MOD_SDEV'] = vradm_ewsdev
    
    metadata['S_ALL'] = sprof_ewmean
    metadata['S_ALL_SDEV'] = sprof_ewsdev
    metadata['S_ALL_NTOT'] = sprof_ntot
    metadata['S_ALL_NUSE'] = sprof_nbin
    metadata['S_ALL_MOD'] = sprofm_ewmean
    metadata['S_ALL_MOD_SDEV'] = sprofm_ewsdev
    
    metadata['S_MAJ'] = smaj_ewmean
    metadata['S_MAJ_SDEV'] = smaj_ewsdev
    metadata['S_MAJ_NTOT'] = smaj_ntot
    metadata['S_MAJ_NUSE'] = smaj_nbin
    metadata['S_MAJ_MOD'] = smajm_ewmean
    metadata['S_MAJ_MOD_SDEV'] = smajm_ewsdev
    
    metadata['S_MIN'] = smin_ewmean
    metadata['S_MIN_SDEV'] = smin_ewsdev
    metadata['S_MIN_NTOT'] = smin_ntot
    metadata['S_MIN_NUSE'] = smin_nbin
    metadata['S_MIN_MOD'] = sminm_ewmean
    metadata['S_MIN_MOD_SDEV'] = sminm_ewsdev
    
    # Best-fit model maps and fit-residual stats
    # TODO: Don't bin the intrinsic model?
    # TODO: Include the binned radial profiles shown in the output plot?
#    models = disk.model()
    intr_models = disk.model(ignore_beam=True)
    vfom, sfom = disk._get_fom()(disk.par, sep=True)
    if disk.dc is None:
        vel_mod = kin.remap(kin.bin(models), masked=False, fill_value=0.)
        vel_mod_intr = kin.remap(kin.bin(intr_models), masked=False, fill_value=0.)

        resid = kin.vel - kin.bin(models)
        err = np.sqrt(inverse(kin.vel_ivar))
        scat = IntrinsicScatter(resid, err=err, gpm=disk.vel_gpm, npar=disk.nfree)
        scat.sig = 0. if disk.scatter is None else disk.scatter[0]
        scat.rej = np.zeros(resid.size, dtype=bool) if vmask is None else vmask > 0

        metadata['VNFLAG'] = np.sum(disk.mbm.flagged(vmask, flag=['REJ_ERR', 'REJ_SNR', 'REJ_UNR',
                                                                  'DISJOINT', 'REJ_RESID']))        
        metadata['VNREJ'] = np.sum(disk.mbm.flagged(vmask, flag='REJ_RESID'))        
        metadata['VNFIT'], metadata['VNMSK'], metadata['VMEDE'], _, _, metadata['VMENR'], \
                metadata['VSIGR'], metadata['VGRWR'], _, _, _, metadata['VSIGIR'], \
                metadata['VGRWIR'] = scat.stats()

        metadata['VISCT'] = 0.0 if disk.scatter is None else disk.scatter[0]
        metadata['VCHI2'] = np.sum(vfom**2)

        metadata['VASYM'] = np.array([np.percentile(np.absolute(a[np.logical_not(m)]),
                                                    [50., 80., 90.])
                                for a, m in zip(vel_asym, vel_asym_mask)])

        nsig = 0.
        sig_mod = None
        sig_mod_intr = None
    else:
        vel_mod = kin.remap(kin.bin(models[0]), masked=False, fill_value=0.)
        vel_mod_intr = kin.remap(kin.bin(intr_models[0]), masked=False, fill_value=0.)

        resid = kin.vel - kin.bin(models[0])
        err = np.sqrt(inverse(kin.vel_ivar))
        scat = IntrinsicScatter(resid, err=err, gpm=disk.vel_gpm, npar=disk.nfree)
        scat.sig = 0. if disk.scatter is None else disk.scatter[0]
        scat.rej = np.zeros(resid.size, dtype=bool) if vmask is None else vmask > 0

        metadata['VNFLAG'] = np.sum(disk.mbm.flagged(vmask, flag=['REJ_ERR', 'REJ_SNR', 'REJ_UNR',
                                                                  'DISJOINT', 'REJ_RESID']))        
        metadata['VNREJ'] = np.sum(disk.mbm.flagged(vmask, flag='REJ_RESID'))        
        metadata['VNFIT'], metadata['VNMSK'], metadata['VMEDE'], _, _, metadata['VMENR'], \
                metadata['VSIGR'], metadata['VGRWR'], _, _, _, metadata['VSIGIR'], \
                metadata['VGRWIR'] = scat.stats()

        metadata['VISCT'] = 0.0 if disk.scatter is None else disk.scatter[0]
        metadata['VCHI2'] = np.sum(vfom**2)

        metadata['VASYM'] = np.array([np.percentile(np.absolute(a[np.logical_not(m)]),
                                                    [50., 80., 90.])
                                for a, m in zip(vel_asym, vel_asym_mask)])
        
        sig_mod = kin.remap(kin.bin(models[1]), masked=False, fill_value=0.)
        sig_mod_intr = kin.remap(kin.bin(intr_models[1]), masked=False, fill_value=0.)

        resid = kin.sig_phys2 - kin.bin(models[1])**2
        err = np.sqrt(inverse(kin.sig_phys2_ivar))
        scat = IntrinsicScatter(resid, err=err, gpm=disk.sig_gpm, npar=disk.nfree)
        scat.sig = 0. if disk.scatter is None else disk.scatter[1]
        scat.rej = np.zeros(resid.size, dtype=bool) if smask is None else smask > 0

        metadata['SNFLAG'] = np.sum(disk.mbm.flagged(smask, flag=['REJ_ERR', 'REJ_SNR', 'REJ_UNR',
                                                                  'DISJOINT', 'REJ_RESID']))        
        metadata['SNREJ'] = np.sum(disk.mbm.flagged(smask, flag='REJ_RESID'))        
        metadata['SNFIT'], metadata['SNMSK'], metadata['SMEDE'], _, _, metadata['SMENR'], \
                metadata['SSIGR'], metadata['SGRWR'], _, _, _, metadata['SSIGIR'], \
                metadata['SGRWIR'] = scat.stats()

        metadata['SISCT'] = 0.0 if disk.scatter is None else disk.scatter[1]
        metadata['SCHI2'] = np.sum(sfom**2)

        metadata['SASYM'] = np.array([np.percentile(np.absolute(a[np.logical_not(m)]),
                                                    [50., 80., 90.])
                                for a, m in zip(sig_asym, sig_asym_mask)])
        
    # Total fit chi-square. SCHI2 and SNFIT are 0 if sigma not fit because of
    # the instantiation value of init_record_array
    metadata['CHI2'] = metadata['VCHI2'] + metadata['SCHI2']
    metadata['RCHI2'] = metadata['CHI2'] / (metadata['VNFIT'] + metadata['SNFIT'] - disk.np)
    
    # Fit status flags
    metadata['STATUS'] = disk.fit_status
    metadata['SUCCESS'] = int(disk.fit_success)

    for n, gp, p, pe in zip(disk_par_names, p0, disk.par, disk.par_err):
        metadata[f'G_{n}'.upper()] = gp
        metadata[f'F_{n}'.upper()] = p
        metadata[f'E_{n}'.upper()] = pe

    # Build the output fits extension (base) headers
    #   - Primary header
    prihdr = fileio.initialize_primary_header(galmeta=galmeta)
    #   - Add the model types to the primary header
    prihdr['MODELTYP'] = ('AxisymmetricDisk', 'nirvana class used to fit the data')
    prihdr['RCMODEL'] = (disk.rc.__class__.__name__, 'Rotation curve parameterization')
    if disk.dc is not None:
        prihdr['DCMODEL'] = (disk.dc.__class__.__name__, 'Dispersion profile parameterization')
    prihdr['QUAL'] = (disk.global_mask, 'Global fit-quality bit')
    #   - Data map header
    maphdr = fileio.add_wcs(prihdr, kin)
    #   - PSF header
    if kin.beam is None:
        psfhdr = None
    else:
        psfhdr = prihdr.copy()
        psfhdr['PSFNAME'] = (kin.psf_name, 'Original PSF name, if known')
    #   - Table header
    tblhdr = prihdr.copy()
    tblhdr['PHOT_KEY'] = 'none' if galmeta.phot_key is None else galmeta.phot_key

    hdus = [fits.PrimaryHDU(header=prihdr),
            fits.ImageHDU(data=binid, header=fileio.finalize_header(maphdr, 'BINID'), name='BINID'),
            fits.ImageHDU(data=r, header=fileio.finalize_header(maphdr, 'R'), name='R'),
            fits.ImageHDU(data=th, header=fileio.finalize_header(maphdr, 'THETA'), name='THETA'),
            fits.ImageHDU(data=grid_sb,
                          header=fileio.finalize_header(maphdr, 'GRIDFLUX',
                                                        bunit='1E-17 erg/s/cm^2/ang/spaxel'),
                          name='GRIDFLUX'),
            fits.ImageHDU(data=sb,
                          header=fileio.finalize_header(maphdr, 'FLUX',
                                                        bunit='1E-17 erg/s/cm^2/ang/spaxel',
                                                        err=True, qual=True),
                          name='FLUX'),
            fits.ImageHDU(data=sb_ivar,
                          header=fileio.finalize_header(maphdr, 'FLUX',
                                                        bunit='(1E-17 erg/s/cm^2/ang/spaxel)^{-2}',
                                                        hduclas2='ERROR', qual=True),
                          name='FLUX_IVAR'),
            fits.ImageHDU(data=sb_mask,
                          header=fileio.finalize_header(maphdr, 'FLUX', hduclas2='QUALITY',
                                                        err=True, bm=disk.mbm),
                          name='FLUX_MASK'),
            fits.ImageHDU(data=vel,
                          header=fileio.finalize_header(maphdr, 'VEL', bunit='km/s', err=True,
                                                        qual=True),
                          name='VEL'),
            fits.ImageHDU(data=vel_ivar,
                          header=fileio.finalize_header(maphdr, 'VEL', bunit='(km/s)^{-2}',
                                                        hduclas2='ERROR', qual=True),
                          name='VEL_IVAR'),
            fits.ImageHDU(data=vel_mask,
                          header=fileio.finalize_header(maphdr, 'VEL', hduclas2='QUALITY',
                                                        err=True, bm=disk.mbm),
                          name='VEL_MASK'),
            fits.ImageHDU(data=vel_mod, header=fileio.finalize_header(maphdr, 'VEL_MOD',
                                                                      bunit='km/s'),
                          name='VEL_MOD'),
            fits.ImageHDU(data=vel_mod_intr,
                          header=fileio.finalize_header(maphdr, 'VEL_MODI', bunit='km/s'),
                          name='VEL_MODI'),
            # TODO: include channel names?
            fits.ImageHDU(data=vel_asym,
                          header=fileio.finalize_header(maphdr, 'VEL_ASYM', bunit='km/s',
                                                        qual=True),
                          name='VEL_ASYM'),
            fits.ImageHDU(data=vel_asym_mask.astype(np.uint8),
                          header=fileio.finalize_header(maphdr, 'VEL_ASYM', hduclas2='QUALITY',
                                                        bit_type=np.uint8),
                          name='VEL_ASYM_MASK')]

    if disk.dc is not None:
        hdus += [fits.ImageHDU(data=sigsqr,
                               header=fileio.finalize_header(maphdr, 'SIGSQR', bunit='(km/s)^2',
                                                             err=True, qual=True),
                               name='SIGSQR'),
                 fits.ImageHDU(data=sigsqr_ivar,
                          header=fileio.finalize_header(maphdr, 'SIGSQR', bunit='(km/s)^{-4}',
                                                        hduclas2='ERROR', qual=True),
                          name='SIGSQR_IVAR'),
            fits.ImageHDU(data=sigsqr_mask,
                          header=fileio.finalize_header(maphdr, 'SIGSQR', hduclas2='QUALITY',
                                                        err=True, bm=disk.mbm),
                          name='SIGSQR_MASK'),
            fits.ImageHDU(data=sig_mod,
                          header=fileio.finalize_header(maphdr, 'SIG_MOD', bunit='km/s'),
                          name='SIG_MOD'),
            fits.ImageHDU(data=sig_mod_intr,
                          header=fileio.finalize_header(maphdr, 'SIG_MODI', bunit='km/s'),
                          name='SIG_MODI'),
            fits.ImageHDU(data=sig_asym,
                          header=fileio.finalize_header(maphdr, 'SIG_ASYM', bunit='(km/s)^2',
                                                        qual=True),
                          name='SIG_ASYM'),
            fits.ImageHDU(data=sig_asym_mask.astype(np.uint8),
                          header=fileio.finalize_header(maphdr, 'SIG_ASYM', hduclas2='QUALITY',
                                                        bit_type=np.uint8),
                          name='SIG_ASYM_MASK')]

    if kin.beam is not None:
        hdus += [fits.ImageHDU(data=kin.beam,
                               header=fileio.finalize_header(psfhdr, 'PSF'), name='PSF')]

    hdus += [fits.BinTableHDU.from_columns([
                    fits.Column(name=n, format=fileio.rec_to_fits_type(metadata[n]),
                                dim=fileio.rec_to_fits_col_dim(metadata[n]), array=metadata[n])
                        for n in metadata.dtype.names], name='FITMETA', header=tblhdr)]

    if ofile.split('.')[-1] == 'gz':
        _ofile = ofile[:ofile.rfind('.')]
        compress = True
    else:
        _ofile = ofile

    fits.HDUList(hdus).writeto(_ofile, overwrite=True, checksum=True)
    if compress:
        fileio.compress_file(_ofile, overwrite=True)
        os.remove(_ofile)


# TODO:
#   - Add keyword for radial sampling for 1D model RCs and dispersion profiles
#   - This is MaNGA-specific and needs to be abstracted
#   - Allow the plot to be constructed from the fits file written by
#     axisym_fit_data
def axisym_fit_plot(galmeta, kin, disk, par=None, par_err=None, fix=None, ofile=None):
    """
    Construct the QA plot for the result of fitting an
    :class:`~nirvana.model.axisym.AxisymmetricDisk` model to a galaxy.

    Args:
        galmeta (:class:`~nirvana.data.meta.GlobalPar`):
            Object with metadata for the galaxy to be fit.
        kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the data to be fit
        disk (:class:`~nirvana.models.axisym.AxisymmetricDisk`):
            Object that performed the fit and has the best-fitting parameters.
        par (`numpy.ndarray`_, optional):
            The parameters of the model.  If None are provided, the parameters
            in ``disk`` are used.
        par_err (`numpy.ndarray`_, optional):
            The errors in the model parameters.  If None are provided, the
            parameter errors in ``disk`` are used.
        fix (`numpy.ndarray`_, optional):
            Flags indicating the parameters that were fixed during the fit.  If
            None, all parameters are assumed to have been free.
        ofile (:obj:`str`, optional):
            Output filename for the plot.  If None, the plot is shown to the
            screen.
    """
    logformatter = plot.get_logformatter()

    # Change the style
    rc('font', size=8)

    _par = disk.par if par is None else par
    _par_err = disk.par_err if par_err is None else par_err
    _fix = np.zeros(disk.np, dtype=bool) if fix is None else fix

    if _par.size != disk.np:
        raise ValueError('Number of provided parameters has the incorrect size.')
    if _par_err.size != disk.np:
        raise ValueError('Number of provided parameter errors has the incorrect size.')
    if _fix.size != disk.np:
        raise ValueError('Number of provided parameter fixing flags has the incorrect size.')

    disk.par = _par
    disk.par_err = _par_err

    # Get the fit statistics
    vfom, sfom = disk._get_fom()(disk.par, sep=True)
    nvel = np.sum(disk.vel_gpm)
    vsct = 0.0 if disk.scatter is None else disk.scatter[0]
    vchi2 = np.sum(vfom**2)
    if disk.dc is None:
        nsig = 0
        ssct = 0.
        schi2 = 0.
    else:
        nsig = np.sum(disk.sig_gpm)
        ssct = 0.0 if disk.scatter is None else disk.scatter[1]
        schi2 = np.sum(sfom**2)
    chi2 = vchi2 + schi2
    rchi2 = chi2 / (nvel + nsig - disk.np)

    # Rebuild the 2D maps
    sb_map = kin.remap('sb')
    snr_map = sb_map * np.ma.sqrt(kin.remap('sb_ivar', mask=kin.sb_mask))
    v_map = kin.remap('vel')
    v_err_map = np.ma.power(kin.remap('vel_ivar', mask=kin.vel_mask), -0.5)
    s_map = np.ma.sqrt(kin.remap('sig_phys2', mask=kin.sig_mask))
    s_err_map = np.ma.power(kin.remap('sig_phys2_ivar', mask=kin.sig_mask), -0.5)/2/s_map

    # Construct the model data, both binned data and maps
    models = disk.model()
    intr_models = disk.model(ignore_beam=True)
    if disk.dc is None:
        vmod = kin.bin(models)
        vmod_map = kin.remap(vmod, mask=kin.vel_mask)
        vmod_intr = kin.bin(intr_models)
        vmod_intr_map = kin.remap(vmod_intr, mask=kin.vel_mask)
        smod = None
        smod_map = None
        smod_intr = None
        smod_intr_map = None
    else:
        vmod = kin.bin(models[0])
        vmod_map = kin.remap(vmod, mask=kin.vel_mask)
        vmod_intr = kin.bin(intr_models[0])
        vmod_intr_map = kin.remap(vmod_intr, mask=kin.vel_mask)
        smod = kin.bin(models[1])
        smod_map = kin.remap(smod, mask=kin.sig_mask)
        smod_intr = kin.bin(intr_models[1])
        smod_intr_map = kin.remap(smod_intr, mask=kin.sig_mask)

    # Get the error-weighted mean radial profiles
    fwhm = galmeta.psf_fwhm[1]  # Selects r band!
    oversample = 1.5
    maj_wedge = 30.
    min_wedge = 10.
    binr, vrot_ewmean, vrot_ewsdev, vrot_ntot, vrot_nbin, vrotm_ewmean, vrotm_ewsdev, \
        vrad_ewmean, vrad_ewsdev, vrad_ntot, vrad_nbin, vradm_ewmean, vradm_ewsdev, \
        sprof_ewmean, sprof_ewsdev, sprof_ntot, sprof_nbin, sprofm_ewmean, sprofm_ewsdev, \
        smaj_ewmean, smaj_ewsdev, smaj_ntot, smaj_nbin, smajm_ewmean, smajm_ewsdev, \
        smin_ewmean, smin_ewsdev, smin_ntot, smin_nbin, sminm_ewmean, sminm_ewsdev \
            = kin.radial_profiles(fwhm/oversample, xc=disk.par[0], yc=disk.par[1], pa=disk.par[2],
                                  inc=disk.par[3], vsys=disk.par[4], maj_wedge=maj_wedge,
                                  min_wedge=min_wedge, vel_mod=vmod, sig_mod=smod)

    # Get the projected rotational velocity
    #   - Disk-plane coordinates
    r, th = projected_polar(kin.x - disk.par[0], kin.y - disk.par[1], *np.radians(disk.par[2:4]))
    #   - Mask for data along the major axis
    major_gpm = select_kinematic_axis(r, th, which='major', r_range='all', wedge=maj_wedge)
    minor_gpm = select_kinematic_axis(r, th, which='minor', r_range='all', wedge=min_wedge)
    #   - Projected rotation velocities
    indx = major_gpm & np.logical_not(kin.vel_mask)
    vrot_r = r[indx]
    vrot = (kin.vel[indx] - disk.par[4])/np.cos(th[indx])
    #   - Projected radial velocities
    indx = minor_gpm & np.logical_not(kin.vel_mask)
    vrad_r = r[indx]
    vrad = (kin.vel[indx] - disk.par[4])/np.sin(th[indx])
    if smod is not None:
        indx = np.logical_not(kin.sig_mask) & (kin.sig_phys2 > 0)
        sprof_r = r[indx]
        sprof = np.sqrt(kin.sig_phys2[indx])

    # Get the 1D model profiles
    maxr = np.amax(r)
    modelr = np.arange(0, maxr, 0.1)
    vrot_intr_model = disk.rc.sample(modelr, par=disk.rc_par())
    if smod is not None:
        sprof_intr_model = disk.dc.sample(modelr, par=disk.dc_par())

    # Set the extent for the 2D maps
    extent = [np.amax(kin.grid_x), np.amin(kin.grid_x), np.amin(kin.grid_y), np.amax(kin.grid_y)]
    Dx = max(extent[0]-extent[1], extent[3]-extent[2]) # *1.01
    skylim = np.array([ (extent[0]+extent[1] - Dx)/2., 0.0 ])
    skylim[1] = skylim[0] + Dx

    # Create the plot
    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(2*w,2*h))

    #-------------------------------------------------------------------
    # Surface-brightness
    sb_lim = np.power(10.0, growth_lim(np.ma.log10(sb_map), 0.90, 1.05))
    sb_lim = atleast_one_decade(sb_lim)
    
    ax = plot.init_ax(fig, [0.02, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sb_map, origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=sb_lim[0], vmax=sb_lim[1]), zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.par[0], disk.par[1], marker='+', color='k', s=40, lw=1, zorder=5)
    # TODO: For some reason, the combination of the use of a masked array and
    # setting the formatter to logformatter leads to weird behavior in the map.
    # Use something like the "pallete" object described here?
    #   https://matplotlib.org/stable/gallery/images_contours_and_fields/image_masked.html
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, r'$\mu$', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # S/N
    snr_lim = np.power(10.0, growth_lim(np.ma.log10(snr_map), 0.90, 1.05))
    snr_lim = atleast_one_decade(snr_lim)

    ax = plot.init_ax(fig, [0.02, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(snr_map, origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=snr_lim[0], vmax=snr_lim[1]), zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.par[0], disk.par[1], marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cax.text(-0.05, 0.1, 'S/N', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity
    vel_lim = growth_lim(np.ma.append(v_map, vmod_map), 0.90, 1.05, midpoint=disk.par[4])
    ax = plot.init_ax(fig, [0.215, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.245, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(v_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.par[0], disk.par[1], marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, 'V', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Dispersion
    _smaps = s_map if smod_map is None else np.ma.append(s_map, smod_map)
    sig_lim = np.power(10.0, growth_lim(np.ma.log10(_smaps), 0.80, 1.05))
    sig_lim = atleast_one_decade(sig_lim)

    ax = plot.init_ax(fig, [0.215, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.245, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(s_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=sig_lim[0], vmax=sig_lim[1]), zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.par[0], disk.par[1], marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cax.text(-0.05, 0.1, r'$\sigma$', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Model
    ax = plot.init_ax(fig, [0.410, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.440, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(vmod_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.par[0], disk.par[1], marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, 'V', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Dispersion Model
    ax = plot.init_ax(fig, [0.410, 0.580, 0.19, 0.19])
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    # Mark the fitted dynamical center
    ax.scatter(disk.par[0], disk.par[1], marker='+', color='k', s=40, lw=1, zorder=5)
    if disk.dc is None:
        ax.text(0.5, 0.3, 'No velocity dispersion model', ha='center', va='center',
                transform=ax.transAxes)
    else:
        im = ax.imshow(smod_map, origin='lower', interpolation='nearest', cmap='viridis',
                       extent=extent, norm=colors.LogNorm(vmin=sig_lim[0], vmax=sig_lim[1]),
                       zorder=4)
        cax = fig.add_axes([0.440, 0.57, 0.15, 0.005])
        cax.tick_params(which='both', direction='in')
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
        cax.text(-0.05, 0.1, r'$\sigma$', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Model Residuals
    v_resid = v_map - vmod_map
    v_res_lim = growth_lim(v_resid, 0.80, 1.15, midpoint=0.0)

    ax = plot.init_ax(fig, [0.605, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.635, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(v_resid, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=v_res_lim[0], vmax=v_res_lim[1], zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, r'$\Delta V$', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Dispersion Residuals
    ax = plot.init_ax(fig, [0.605, 0.580, 0.19, 0.19])
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    if disk.dc is None:
        ax.text(0.5, 0.3, 'No velocity dispersion model', ha='center', va='center',
                transform=ax.transAxes)
    else:
        s_resid = s_map - smod_map
        s_res_lim = growth_lim(s_resid, 0.80, 1.15, midpoint=0.0)
        im = ax.imshow(s_resid, origin='lower', interpolation='nearest', cmap='RdBu_r',
                    extent=extent, vmin=s_res_lim[0], vmax=s_res_lim[1], zorder=4)
        cax = fig.add_axes([0.635, 0.57, 0.15, 0.005])
        cax.tick_params(which='both', direction='in')
        cb = fig.colorbar(im, cax=cax, orientation='horizontal') #, format=logformatter)
        cax.text(-0.05, 0.1, r'$\Delta\sigma$', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Model Chi-square
    v_chi = np.ma.divide(np.absolute(v_resid), v_err_map)
    v_chi_lim = np.power(10.0, growth_lim(np.ma.log10(v_chi), 0.90, 1.15))
    v_chi_lim = atleast_one_decade(v_chi_lim)

    ax = plot.init_ax(fig, [0.800, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(v_chi, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=v_chi_lim[0], vmax=v_chi_lim[1]),
                   zorder=4)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.02, 1.1, r'$|\Delta V|/\epsilon$', ha='right', va='center',
             transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Dispersion Model Chi-square
    ax = plot.init_ax(fig, [0.800, 0.580, 0.19, 0.19])
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    if disk.dc is None:
        ax.text(0.5, 0.3, 'No velocity dispersion model', ha='center', va='center',
                transform=ax.transAxes)
    else:
        s_chi = np.ma.divide(np.absolute(s_resid), s_err_map)
        s_chi_lim = np.power(10.0, growth_lim(np.ma.log10(s_chi), 0.90, 1.15))
        s_chi_lim = atleast_one_decade(s_chi_lim)

        cax = fig.add_axes([0.830, 0.57, 0.15, 0.005])
        cax.tick_params(which='both', direction='in')
        im = ax.imshow(s_chi, origin='lower', interpolation='nearest', cmap='viridis',
                    extent=extent, norm=colors.LogNorm(vmin=s_chi_lim[0], vmax=s_chi_lim[1]),
                    zorder=4)
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
        cax.text(-0.02, 0.4, r'$|\Delta \sigma|/\epsilon$', ha='right', va='center',
                transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Intrinsic Velocity Model
    ax = plot.init_ax(fig, [0.800, 0.305, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.50, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    im = ax.imshow(vmod_intr_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.par[0], disk.par[1], marker='+', color='k', s=40, lw=1, zorder=5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cax.text(-0.05, 1.1, 'V', ha='right', va='center', transform=cax.transAxes)

    ax.text(0.5, 1.2, 'Intrinsic Model', ha='center', va='center', transform=ax.transAxes,
            fontsize=10)

    #-------------------------------------------------------------------
    # Intrinsic Velocity Dispersion
    ax = plot.init_ax(fig, [0.800, 0.110, 0.19, 0.19])
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    # Mark the fitted dynamical center
    ax.scatter(disk.par[0], disk.par[1], marker='+', color='k', s=40, lw=1, zorder=5)
    if disk.dc is None:
        ax.text(0.5, 0.3, 'No velocity dispersion model', ha='center', va='center',
                transform=ax.transAxes)
    else:
        im = ax.imshow(smod_intr_map, origin='lower', interpolation='nearest', cmap='viridis',
                       extent=extent, norm=colors.LogNorm(vmin=sig_lim[0], vmax=sig_lim[1]),
                       zorder=4)
        cax = fig.add_axes([0.830, 0.10, 0.15, 0.005])
        cax.tick_params(which='both', direction='in')
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
        cax.text(-0.05, 0.1, r'$\sigma$', ha='right', va='center', transform=cax.transAxes)

    #-------------------------------------------------------------------
    # Annotate with the intrinsic scatter included
    ax.text(0.00, -0.2, r'V scatter, $\epsilon_v$:', ha='left', va='center',
            transform=ax.transAxes, fontsize=10)
    ax.text(1.00, -0.2, f'{vsct:.1f}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)
    if disk.dc is not None:
        ax.text(0.00, -0.3, r'$\sigma^2$ scatter, $\epsilon_{\sigma^2}$:', ha='left', va='center',
                transform=ax.transAxes, fontsize=10)
        ax.text(1.00, -0.3, f'{ssct:.1f}', ha='right', va='center', transform=ax.transAxes,
                fontsize=10)

    #-------------------------------------------------------------------
    # SDSS image
    ax = fig.add_axes([0.01, 0.29, 0.23, 0.23])
    if kin.image is not None:
        ax.imshow(kin.image)
    else:
        ax.text(0.5, 0.5, 'No Image', ha='center', va='center', transform=ax.transAxes,
                fontsize=20)

    ax.text(0.5, 1.05, 'SDSS gri Composite', ha='center', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if galmeta.primaryplus:
        sample='Primary+'
    elif galmeta.secondary:
        sample='Secondary'
    elif galmeta.ancillary:
        sample='Ancillary'
    else:
        sample='Filler'

    # MaNGA ID
    ax.text(0.00, -0.05, 'MaNGA ID:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.05, f'{galmeta.mangaid}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)
    # Observation
    ax.text(0.00, -0.13, 'Observation:', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.13, f'{galmeta.plate}-{galmeta.ifu}', ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Sample selection
    ax.text(0.00, -0.21, 'Sample:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.21, f'{sample}', ha='right', va='center', transform=ax.transAxes, fontsize=10)
    # Redshift
    ax.text(0.00, -0.29, 'Redshift:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.29, '{0:.4f}'.format(galmeta.z), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Mag
    ax.text(0.00, -0.37, 'Mag (N,r,i):', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    if galmeta.mag is None:
        ax.text(1.01, -0.37, 'Unavailable', ha='right', va='center',
                transform=ax.transAxes, fontsize=10)
    else:
        ax.text(1.01, -0.37, '{0:.1f}/{1:.1f}/{2:.1f}'.format(*galmeta.mag), ha='right',
                va='center', transform=ax.transAxes, fontsize=10)
    # PSF FWHM
    ax.text(0.00, -0.45, 'FWHM (g,r):', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.45, '{0:.2f}, {1:.2f}'.format(*galmeta.psf_fwhm[:2]), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Sersic n
    ax.text(0.00, -0.53, r'Sersic $n$:', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.53, '{0:.2f}'.format(galmeta.sersic_n), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Stellar Mass
    ax.text(0.00, -0.61, r'$\log(\mathcal{M}_\ast/\mathcal{M}_\odot$):', ha='left', va='center',
            transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.61, '{0:.2f}'.format(np.log10(galmeta.mass)), ha='right', va='center',
            transform=ax.transAxes, fontsize=10)
    # Phot Inclination
    ax.text(0.00, -0.69, r'$i_{\rm phot}$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.69, '{0:.1f}'.format(galmeta.guess_inclination(lb=1., ub=89.)),
            ha='right', va='center', transform=ax.transAxes, fontsize=10)
    # Fitted center
    ax.text(0.00, -0.77, r'$x_0$ [arcsec]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[0] else 'k')
    xstr = r'{0:.2f}'.format(disk.par[0]) if _fix[0] \
            else r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[0], disk.par_err[0])
    ax.text(1.01, -0.77, xstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[0] else 'k')
    ax.text(0.00, -0.85, r'$y_0$ [arcsec]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[1] else 'k')
    ystr = r'{0:.2f}'.format(disk.par[1]) if _fix[1] \
            else r'{0:.2f} $\pm$ {1:.2f}'.format(disk.par[1], disk.par_err[1])
    ax.text(1.01, -0.85, ystr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[1] else 'k')
    # Position angle
    ax.text(0.00, -0.93, r'$\phi_0$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[2] else 'k')
    pastr = r'{0:.1f}'.format(disk.par[2]) if _fix[2] \
            else r'{0:.1f} $\pm$ {1:.1f}'.format(disk.par[2], disk.par_err[2])
    ax.text(1.01, -0.93, pastr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[2] else 'k')
    # Kinematic Inclination
    ax.text(0.00, -1.01, r'$i_{\rm kin}$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[3] else 'k')
    incstr = r'{0:.1f}'.format(disk.par[3]) if _fix[3] \
            else r'{0:.1f} $\pm$ {1:.1f}'.format(disk.par[3], disk.par_err[3])
    ax.text(1.01, -1.01, incstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[3] else 'k')
    # Systemic velocity
    ax.text(0.00, -1.09, r'$V_{\rm sys}$ [km/s]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[4] else 'k')
    vsysstr = r'{0:.1f}'.format(disk.par[4]) if _fix[4] \
            else r'{0:.1f} $\pm$ {1:.1f}'.format(disk.par[4], disk.par_err[4])
    ax.text(1.01, -1.09, vsysstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[4] else 'k')
    # Reduced chi-square
    ax.text(0.00, -1.17, r'$\chi^2_\nu$', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -1.17, f'{rchi2:.2f}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)

    #-------------------------------------------------------------------
    # Radial plot radius limits
    # Select bins with sufficient data
    vrot_indx = vrot_nbin > 5
    if not np.any(vrot_indx):
        vrot_indx = vrot_nbin > 0
    vrad_indx = vrad_nbin > 5
    if not np.any(vrad_indx):
        vrad_indx = vrad_nbin > 0
    if disk.dc is not None:
        sprof_indx = sprof_nbin > 5
        if not np.any(sprof_indx):
            sprof_indx = sprof_nbin > 0

    concat_r = binr[vrot_indx] if np.any(vrot_indx) else np.array([])
    if disk.dc is not None and np.any(sprof_indx):
        concat_r = np.append(concat_r, binr[sprof_indx])
    if len(concat_r) == 0:
        warnings.warn('No valid bins of velocity or sigma data.  Skipping radial bin plots!')

        # Close off the plot
        if ofile is None:
            pyplot.show()
        else:
            fig.canvas.print_figure(ofile, bbox_inches='tight')
        fig.clear()
        pyplot.close(fig)

        # Reset to default style
        pyplot.rcdefaults()
        return

    # Set the radius limits for the radial plots
    r_lim = [0.0, np.amax(concat_r)*1.1]

    #-------------------------------------------------------------------
    # Rotation curve
#    maxrc = np.amax(np.append(vrot_ewmean[vrot_indx], vrotm_ewmean[vrot_indx])) \
#                if np.any(vrot_indx) else np.amax(vrot_intr_model)
#    rc_lim = [0.0, maxrc*1.1]

    rc_lim = growth_lim(np.concatenate((vrot_ewmean[vrot_indx], vrotm_ewmean[vrot_indx],
                                        vrad_ewmean[vrad_indx], vradm_ewmean[vrad_indx])),
                        0.99, 1.3)

    reff_lines = np.arange(galmeta.reff, r_lim[1], galmeta.reff) if galmeta.reff > 1 else None

    ax = plot.init_ax(fig, [0.27, 0.27, 0.51, 0.23], facecolor='0.9', top=False, right=False)
    ax.set_xlim(r_lim)
    ax.set_ylim(rc_lim)
    plot.rotate_y_ticks(ax, 90, 'center')
    if smod is None:
        ax.text(0.5, -0.13, r'$R$ [arcsec]', ha='center', va='center', transform=ax.transAxes,
                fontsize=10)
    else:
        ax.xaxis.set_major_formatter(ticker.NullFormatter())

    indx = vrot_nbin > 0
    ax.scatter(vrot_r, vrot, marker='.', color='k', s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(indx):
        ax.scatter(binr[indx], vrot_ewmean[indx], marker='o', edgecolors='none', s=100,
                   alpha=1.0, facecolors='0.5', zorder=3)
        ax.scatter(binr[indx], vrotm_ewmean[indx], edgecolors='C3', marker='o', lw=3, s=100,
                   alpha=1.0, facecolors='none', zorder=4)
        ax.errorbar(binr[indx], vrot_ewmean[indx], yerr=vrot_ewsdev[indx], color='0.5', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    indx = vrad_nbin > 0
    ax.scatter(vrad_r, vrad, marker='.', color='0.6', s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(indx):
        ax.scatter(binr[indx], vrad_ewmean[indx], marker='o', edgecolors='none', s=100,
                   alpha=1.0, facecolors='0.7', zorder=3)
        ax.scatter(binr[indx], vradm_ewmean[indx], edgecolors='C1', marker='o', lw=3, s=100,
                   alpha=1.0, facecolors='none', zorder=4)
        ax.errorbar(binr[indx], vrad_ewmean[indx], yerr=vrad_ewsdev[indx], color='0.7', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, vrot_intr_model, color='C3', zorder=5, lw=0.5)
    if reff_lines is not None:
        for l in reff_lines:
            ax.axvline(x=l, linestyle='--', lw=0.5, zorder=2, color='k')

    asec2kpc = galmeta.kpc_per_arcsec()
    if asec2kpc > 0:
        axt = plot.get_twin(ax, 'x')
        axt.set_xlim(np.array(r_lim) * galmeta.kpc_per_arcsec())
        axt.set_ylim(rc_lim)
        ax.text(0.5, 1.14, r'$R$ [$h^{-1}$ kpc]', ha='center', va='center', transform=ax.transAxes,
                fontsize=10)
    else:
        ax.text(0.5, 1.05, 'kpc conversion unavailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)

    kin_inc = disk.par[3]
    axt = plot.get_twin(ax, 'y')
    axt.set_xlim(r_lim)
    axt.set_ylim(np.array(rc_lim)/np.sin(np.radians(kin_inc)))
    plot.rotate_y_ticks(axt, 90, 'center')
    axt.spines['right'].set_color('0.4')
    axt.tick_params(which='both', axis='y', colors='0.4')
    axt.yaxis.label.set_color('0.4')

    ax.add_patch(patches.Rectangle((0.66,0.55), 0.32, 0.19, facecolor='w', lw=0, edgecolor='none',
                                   zorder=5, alpha=0.7, transform=ax.transAxes))
    ax.text(0.97, 0.65, r'$V\ \sin i$ [km/s; left axis]', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=10, zorder=6)
    ax.text(0.97, 0.56, r'$V$ [km/s; right axis]', ha='right', va='bottom', color='0.4',
            transform=ax.transAxes, fontsize=10, zorder=6)

    #-------------------------------------------------------------------
    # Velocity Dispersion profile
    if smod is not None:
        concat_s = np.append(sprof_ewmean[sprof_indx], sprofm_ewmean[sprof_indx]) \
                        if np.any(sprof_indx) else sprof_intr_model
        sprof_lim = np.power(10.0, growth_lim(np.ma.log10(concat_s), 0.9, 1.5))
        sprof_lim = atleast_one_decade(sprof_lim)

        ax = plot.init_ax(fig, [0.27, 0.04, 0.51, 0.23], facecolor='0.9')
        ax.set_xlim(r_lim)
        ax.set_ylim(sprof_lim)#[10,275])
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(logformatter)
        plot.rotate_y_ticks(ax, 90, 'center')

        indx = sprof_nbin > 0
        ax.scatter(sprof_r, sprof, marker='.', color='k', s=30, lw=0, alpha=0.6, zorder=1)
        if np.any(indx):
            ax.scatter(binr[indx], sprof_ewmean[indx], marker='o', edgecolors='none', s=100,
                       alpha=1.0, facecolors='0.5', zorder=3)
            ax.scatter(binr[indx], sprofm_ewmean[indx], edgecolors='C3', marker='o', lw=3, s=100,
                       alpha=1.0, facecolors='none', zorder=4)
            ax.errorbar(binr[indx], sprof_ewmean[indx], yerr=sprof_ewsdev[indx], color='0.6',
                        capsize=0, linestyle='', linewidth=1, alpha=1.0, zorder=2)
        ax.plot(modelr, sprof_intr_model, color='C3', zorder=5, lw=0.5)
        if reff_lines is not None:
            for l in reff_lines:
                ax.axvline(x=l, linestyle='--', lw=0.5, zorder=2, color='k')

        ax.text(0.5, -0.13, r'$R$ [arcsec]', ha='center', va='center', transform=ax.transAxes,
                fontsize=10)

        ax.add_patch(patches.Rectangle((0.81,0.86), 0.17, 0.09, facecolor='w', lw=0,
                                       edgecolor='none', zorder=5, alpha=0.7,
                                       transform=ax.transAxes))
        ax.text(0.97, 0.87, r'$\sigma_{\rm los}$ [km/s]', ha='right', va='bottom',
                transform=ax.transAxes, fontsize=10, zorder=6)

    # TODO:
    #   - Add errors (if available)?
    #   - Surface brightness units?

    if ofile is None:
        pyplot.show()
    else:
        fig.canvas.print_figure(ofile, bbox_inches='tight')
    fig.clear()
    pyplot.close(fig)

    # Reset to default style
    pyplot.rcdefaults()


def reset_to_base_flags(disk, kin, vel_mask, sig_mask):
    """
    Reset the masks to only include the "base" flags.
    
    As the best-fit parameters change over the course of a set of rejection
    iterations, the residuals with respect to the model change. This method
    resets the flags back to the base-level rejection (i.e., independent of
    the model), allowing the rejection to be based on the most recent set of
    parameters and potentially recovering good data that was previously
    rejected because of a poor model fit.

    .. warning::
        The objects are *all* modified in place.

    Args:
        disk (:class:`~nirvana.models.axisym.AxisymmetricDisk`):
            Object that performed the fit and has the best-fitting parameters.
        kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the data being fit.
        vel_mask (`numpy.ndarray`_):
            Bitmask used to track velocity rejections.
        sig_mask (`numpy.ndarray`_):
            Bitmask used to track dispersion rejections. Can be None.
    """
    # Turn off the relevant rejection for all pixels
    vel_mask = disk.mbm.turn_off(vel_mask, flag='REJ_RESID')
    # Reset the data mask held by the Kinematics object
    kin.vel_mask = disk.mbm.flagged(vel_mask, flag=disk.mbm.base_flags())
    if sig_mask is not None:
        # Turn off the relevant rejection for all pixels
        sig_mask = disk.mbm.turn_off(sig_mask, flag='REJ_RESID')
        # Reset the data mask held by the Kinematics object
        kin.sig_mask = disk.mbm.flagged(sig_mask, flag=disk.mbm.base_flags())


def axisym_iter_fit(galmeta, kin, rctype='HyperbolicTangent', dctype='Exponential', fitdisp=True,
                    ignore_covar=True, assume_posdef_covar=True, max_vel_err=None,
                    max_sig_err=None, min_vel_snr=None, min_sig_snr=None,
                    vel_sigma_rej=[15,10,10,10], sig_sigma_rej=[15,10,10,10], fix_cen=False,
                    fix_inc=False, low_inc=None, min_unmasked=None, select_coherent=False,
                    analytic_jac=True, fit_scatter=True, verbose=0):
    r"""
    Iteratively fit kinematic data with an axisymmetric disk model.

    Constraints are as follows:

        #. The center is constrained to be in the middle third of the available
           range in x and y.

    The iterations are as follows:

        #. Fit all data but fix the inclination to the value returned by
           :func:`~nirvana.data.meta.GlobalPar.guess_inclination` and fix the
           center to be :math:`(x,y) = (0,0)`.  If available, covariance is
           ignored.

        #. Reject outliers in both velocity and velocity dispersion (if the
           latter is being fit) using :func:`disk_fit_reject`.  The rejection
           sigma used is the *first* element in the provided list.  Then refit
           the data, starting again from the initial guess parameters.  The
           intrinsic scatter estimates provided by :func:`disk_fit_reject` are
           *not* included in the fit and, if available, covariance is ignored.

        #. Reject outliers in both velocity and velocity dispersion (if the
           latter is being fit) using :func:`disk_fit_reject`.  The rejection
           sigma used is the *second* element in the provided list.  Then refit
           the data using the parameters from the previous fit as the starting
           point. This iteration also uses the intrinsic scatter estimates
           provided by :func:`disk_fit_reject`; however, covariance is still
           ignored.

        #. Recover all fit rejections (i.e., keep any masks in place that are
           tied to the data quality, but remove any masks associated with fit
           quality).  Then use :func:`disk_fit_reject` to perform a fresh
           rejection based on the most recent model; the rejection sigma is the
           *second* element in the provided list.  The resetting of the
           fit-outliers and re-rejection is done on the off chance that
           rejections from the first few iterations were driven by a bad model.
           Refit the data as in the previous iteration, using the parameters
           from the previous fit as the starting point and use the intrinsic
           scatter estimates provided by :func:`disk_fit_reject`.  Covariance is
           still ignored.

        #. Reject outliers in both velocity and velocity dispersion (if the
           latter is being fit) using :func:`disk_fit_reject`.  The rejection
           sigma used is the *third* element in the provided list.  Then refit
           the data, but fix or free the center and inclination based on the
           provided keywords (``fix_cen`` and ``fix_inc``).  Also, as in all
           previous iterations, the covariance is ignored in the outlier
           rejection and intrinsic scatter determination; however, the
           covariance *is* used by the fit, as available and if ``ignore_covar``
           is False.

        #. Redo the previous iteration in exactly the same way, except outlier
           rejection and intrinsic-scatter determination now use the covariance,
           as available and if ``ignore_covar`` is False.  The rejection sigma
           used is the *fourth* element in the provided list.

        #. If a lower inclination threshold is set (see ``low_inc``) and the
           best-fitting inclination is below this value (assuming the
           inclination is freely fit), a final iteration refits the data by
           fixing the inclination at the value set by
           :func:`~nirvana.data.meta.GlobalPar.guess_inclination`.  The code
           issues a warning, and the global fit-quality bit is set to include
           the ``LOWINC`` bit.
        
    .. todo::
        - Enable more rotation curve and dispersion profile functions.
        - Allow guess RC and DC parameters and bounds to be input, or switch to
          requiring the 1D model class instances to be provided, like in
          :class:`~nirvana.models.axisym.AxisymmetricDisk`.

    Args:
        galmeta (:class:`~nirvana.data.meta.GlobalPar`):
            Object with metadata for the galaxy to be fit.
        kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the data to be fit
        rctype (:obj:`str`, optional):
            Functional form for the rotation curve.  Must be "HyperbolicTangent"
            or "PolyEx".
        dctype (:obj:`str`, optional):
            Functional form for the dispersion profile.  Must be "Exponential",
            "ExpBase", or "Const".
        fitdisp (:obj:`bool`, optional):
            Fit the velocity dispersion data if it is available in ``kin``.
        ignore_covar (:obj:`bool`, optional):
            If ``kin`` provides the covariance between measurements, ignore it
            and fit the data assuming there is no covariance.
        assume_posdef_covar (:obj:`bool`, optional):
            If ``kin`` provides the covariance between measurements, assume the
            covariance matrices are positive definite.
        max_vel_err (:obj:`float`, optional):
            Mask measurements with velocity errors larger than this value.  If
            None, there is no upper limit on the allowed velocity error.
        max_sig_err (:obj:`float`, optional):
            Mask measurements with velocity dispersion errors larger than this
            value.  If None, there is no upper limit on the allowed velocity
            dispersion error.
        min_vel_snr (:obj:`float`, optional):
            Mask velocity measurements for spectra below this S/N.  If None,
            there is no lower S/N limit on the allowed velocities.
        min_sig_snr (:obj:`float`, optional):
            Mask velocity dispersion measurements for spectra below this S/N.
            If None, there is no lower S/N limit on the allowed velocity
            dispersions.
        vel_sigma_rej (:obj:`float`, :obj:`list`, optional):
            Sigma values used for rejection of velocity measurements.  Must be a
            single float or a *four-element* list.  If None, no rejections are
            performed.  The description above provides which value is used in
            each iteration.
        sig_sigma_rej (:obj:`float`, :obj:`list`, optional):
            Sigma values used for rejection of dispersion measurements.  Must be
            a single float or a *four-element* list.  If None, no rejections are
            performed.  The description above provides which value is used in
            each iteration.
        fix_cen (:obj:`bool`, optional):
            Fix the dynamical center of the fit to 0,0 in the final fit
            iteration.
        fix_inc (:obj:`bool`, optional):
            Fix the kinematic inclination of the fit to estimate provided by the
            :func:`~nirvana.data.meta.GlobalPar.guess_inclination` method of
            ``galmeta``.
        low_inc (scalar-like, optional):
            If the inclination is free and the best-fitting inclination from the
            final fit iteration is below this value, flag the global bitmask of
            the fit as having a low inclination and refit the data using a fixed
            inclination set by
            :func:`~nirvana.data.meta.GlobalPar.guess_inclination` (i.e., this
            is the same as when setting ``fix_inc`` to True).  If None, no
            minimum is set on the viable inclination (apart from the fit
            boundaries).
        min_unmasked (:obj:`int`, optional):
            The minimum of velocity measurements (and velocity dispersion
            measurements, if they are available and being fit) required to
            proceed with the fit, after applying all masking.
        select_coherent (:obj:`bool`, optional):
            After masking the measurement values, mask any measurements that are
            not included in the largest coherent region of adjacent
            measurements.  See
            :func:`~nirvana.data.util.find_largest_coherent_region`.
        analytic_jac (:obj:`bool`, optional):
            Use the analytic calculation of the Jacobian matrix during the fit
            optimization.  If False, the Jacobian is calculated using
            finite-differencing methods provided by
            `scipy.optimize.least_squares`_.
        fit_scatter (:obj:`bool`, optional):
            Model the intrinsic scatter in the data about the model during the
            fit optimization.
        verbose (:obj:`int`, optional):
            Verbosity level: 0=only status output written to terminal; 1=show 
            fit result QA plot; 2=full output

    Returns:
        :obj:`tuple`: Returns 5 objects: (1) the
        :class:`~nirvana.models.axisym.AxisymmetricDisk` instance used during
        the fit, (2) a `numpy.ndarray`_ with the input guess parameters, (3) a
        boolean `numpy.ndarray`_ selecting the parameters that were fixed during
        the fit, (4) a `numpy.ndarray`_ with the bad-pixel mask for the velocity
        measurements used in the fit, and (5) a `numpy.ndarray`_ with the
        bad-pixel mask for the velocity dispersion measurements used in the fit.
    """
    # Running in "debug" mode
    debug = verbose > 1

    # Check input
    _vel_sigma_rej = None if vel_sigma_rej is None else list(vel_sigma_rej)
    if _vel_sigma_rej is not None and len(_vel_sigma_rej) == 1:
        _vel_sigma_rej *= 4
    if _vel_sigma_rej is not None and len(_vel_sigma_rej) != 4:
        raise ValueError('Length of vel_sigma_rej list must be 4!')
    _sig_sigma_rej = None if sig_sigma_rej is None else list(sig_sigma_rej)
    if _sig_sigma_rej is not None and len(_sig_sigma_rej) == 1:
        _sig_sigma_rej *= 4
    if _sig_sigma_rej is not None and len(_sig_sigma_rej) != 4:
        raise ValueError('Length of sig_sigma_rej list must be 4!')

    #---------------------------------------------------------------------------
    # Get the guess parameters and the model parameterizations
    print('Setting up guess parameters and parameterization classes.')
    #   - Geometry
    pa, vproj = galmeta.guess_kinematic_pa(kin.grid_x, kin.grid_y, kin.remap('vel'),
                                           return_vproj=True)
    p0 = np.array([0., 0., pa, galmeta.guess_inclination(lb=1., ub=89.), 0.])

    #   - Rotation Curve
    rc = None
    if rctype == 'HyperbolicTangent':
        # TODO: Maybe want to make the guess hrot based on the effective radius...
        p0 = np.append(p0, np.array([min(900., vproj), 1.]))
        rc = oned.HyperbolicTangent(lb=np.array([0., 1e-3]),
                                    ub=np.array([1000., max(5., kin.max_radius())]))
    elif rctype == 'PolyEx':
        p0 = np.append(p0, np.array([min(900., vproj), 1., 0.1]))
        rc = oned.PolyEx(lb=np.array([0., 1e-3, -1.]),
                         ub=np.array([1000., max(5., kin.max_radius()), 1.]))
    else:
        raise ValueError(f'Unknown RC parameterization: {rctype}')

    #   - Dispersion profile
    dc = None
    if fitdisp:
        sig0 = galmeta.guess_central_dispersion(kin.grid_x, kin.grid_y, kin.remap('sig'))
        # For disks, 1 Re = 1.7 hr (hr = disk scale length). The dispersion
        # e-folding length is ~2 hr, meaning that I use a guess of 2/1.7 Re for
        # the dispersion e-folding length.
        if dctype == 'Exponential':
            p0 = np.append(p0, np.array([sig0, 2*galmeta.reff/1.7]))
            dc = oned.Exponential(lb=np.array([0., 1e-3]), ub=np.array([1000., 3*galmeta.reff]))
        elif dctype == 'ExpBase':
            p0 = np.append(p0, np.array([sig0, 2*galmeta.reff/1.7, 1.]))
            dc = oned.ExpBase(lb=np.array([0., 1e-3, 0.]),
                              ub=np.array([1000., 3*galmeta.reff, 100.]))
        elif dctype == 'Const':
            p0 = np.append(p0, np.array([sig0]))
            dc = oned.Const(lb=np.array([0.]), ub=np.array([1000.]))

    # Report
    print(f'Rotation curve parameterization class: {rc.__class__.__name__}')
    if fitdisp:
        print(f'Dispersion profile parameterization class: {dc.__class__.__name__}')
    print('Input guesses:')
    print(f'               Position angle: {pa:.1f}')
    print(f'                  Inclination: {p0[3]:.1f}')
    print(f'     Projected Rotation Speed: {vproj:.1f}')
    if fitdisp:
        print(f'  Central Velocity Dispersion: {sig0:.1f}')
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Define the fitting object
    disk = AxisymmetricDisk(rc=rc, dc=dc)

    # Constrain the center to be in the middle third of the map relative to the
    # photometric center. The mean in the calculation is to mitigate that some
    # galaxies can be off center, but the detail here and how well it works
    # hasn't been well tested.
    # TODO: Should this use grid_x instead, so that it's more uniform for all
    # IFUs?  Or should this be set as a fraction of Reff?
    dx = np.mean([abs(np.amin(kin.x)), abs(np.amax(kin.x))])
    dy = np.mean([abs(np.amin(kin.y)), abs(np.amax(kin.y))])
    lb, ub = disk.par_bounds(base_lb=np.array([-dx/3, -dy/3, -350., 1., -500.]),
                             base_ub=np.array([dx/3, dy/3, 350., 89., 500.]))
    print(f'If free, center constrained within +/- {dx/3:.1f} in X and +/- {dy/3:.1f} in Y.')

    # TODO: Handle these issues instead of faulting
    if np.any(np.less(p0, lb)):
        raise ValueError('Parameter lower bounds cannot accommodate initial guess value!')
    if np.any(np.greater(p0, ub)):
        raise ValueError('Parameter upper bounds cannot accommodate initial guess value!')

    #---------------------------------------------------------------------------
    # Setup the masks
    print('Initializing data masking')
    vel_mask = disk.mbm.init_mask_array(kin.vel.shape)
    vel_mask[kin.vel_mask] = disk.mbm.turn_on(vel_mask[kin.vel_mask], 'DIDNOTUSE')
    if fitdisp:
        sig_mask = disk.mbm.init_mask_array(kin.sig.shape)
        sig_mask[kin.sig_mask] = disk.mbm.turn_on(sig_mask[kin.sig_mask], 'DIDNOTUSE')
    else:
        sig_mask = None

    # Reject based on error
    vel_rej, sig_rej = kin.clip_err(max_vel_err=max_vel_err, max_sig_err=max_sig_err)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements removed because of large errors.')
        vel_mask[vel_rej] = disk.mbm.turn_on(vel_mask[vel_rej], 'REJ_ERR')
    if fitdisp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements removed because of large errors.')
        sig_mask[sig_rej] = disk.mbm.turn_on(sig_mask[sig_rej], 'REJ_ERR')

    # Reject based on S/N
    vel_rej, sig_rej = kin.clip_snr(min_vel_snr=min_vel_snr, min_sig_snr=min_sig_snr)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements removed because of low S/N.')
        vel_mask[vel_rej] = disk.mbm.turn_on(vel_mask[vel_rej], 'REJ_SNR')
    if fitdisp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements removed because of low S/N.')
        sig_mask[sig_rej] = disk.mbm.turn_on(sig_mask[sig_rej], 'REJ_SNR')

    if np.all(vel_mask > 0):
        raise ValueError('All velocity measurements masked!')
    if sig_mask is not None and np.all(sig_mask > 0):
        raise ValueError('All velocity dispersion measurements masked!')

    if min_unmasked is not None:
        if np.sum(np.logical_not(vel_mask > 0)) < min_unmasked:
            raise ValueError('Insufficient valid velocity measurements to continue!')
        if sig_mask is not None and np.sum(np.logical_not(sig_mask > 0)) < min_unmasked:
            raise ValueError('Insufficient valid velocity dispersion measurements to continue!')

    # Fit only the spatially coherent regions
    if select_coherent:
        gpm = np.logical_not(kin.remap(vel_mask, masked=False, fill_value=1).astype(bool))
        indx = find_largest_coherent_region(gpm.astype(int)).astype(int)
        vel_rej = np.logical_not(kin.bin(indx).astype(bool)) & (vel_mask == 0)
        if np.any(vel_rej):
            print(f'Flagging {np.sum(vel_rej)} velocities as disjoint from the main group.')
            vel_mask[vel_rej] = disk.mbm.turn_on(vel_mask[vel_rej], flag='DISJOINT')
        else:
            vel_rej = None

        sig_rej = None
        if sig_mask is not None:
            gpm = np.logical_not(kin.remap(sig_mask, masked=False, fill_value=1).astype(bool))
            indx = find_largest_coherent_region(gpm.astype(int)).astype(int)
            sig_rej = np.logical_not(kin.bin(indx).astype(bool)) & (sig_mask == 0)
            if np.any(sig_rej):
                print(f'Flagging {np.sum(sig_rej)} dispersions as disjoint from the main group.')
                sig_mask[sig_rej] = disk.mbm.turn_on(sig_mask[sig_rej], flag='DISJOINT')
            else:
                sig_rej = None

        # Propagate the rejection to the kinematics object
        kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)

    #---------------------------------------------------------------------------
    # Perform the fit iterations
    #---------------------------------------------------------------------------
    # Fit iteration 1: Fit all data but fix the inclination and center
    #                x0    y0    pa     inc   vsys    rc+dc parameters
    fix = np.append([True, True, False, True, False], np.zeros(p0.size-5, dtype=bool))
    print('Running fit iteration 1')
    # TODO: sb_wgt is always true throughout. Make this a command-line
    # parameter?
    disk.lsq_fit(kin, sb_wgt=True, p0=p0, fix=fix, lb=lb, ub=ub, ignore_covar=True,
                 assume_posdef_covar=assume_posdef_covar, analytic_jac=analytic_jac,
                 verbose=verbose)
    # Show
    if verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix) 

    #---------------------------------------------------------------------------
    # Fit iteration 2:
    #   - Reject very large outliers. This is aimed at finding data that is
    #     so descrepant from the model that it's reasonable to expect the
    #     measurements are bogus.
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk.reject(disp=fitdisp, ignore_covar=True, vel_sigma_rej=_vel_sigma_rej[0],
                          show_vel=debug, sig_sigma_rej=_sig_sigma_rej[0], show_sig=debug,
                          verbose=verbose > 1)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements rejected as unreliable.')
        vel_mask[vel_rej] = disk.mbm.turn_on(vel_mask[vel_rej], 'REJ_UNR')
    if fitdisp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements rejected as unreliable.')
        sig_mask[sig_rej] = disk.mbm.turn_on(sig_mask[sig_rej], 'REJ_UNR')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Refit, again with the inclination and center fixed. However, do not
    #     use the parameters from the previous fit as the starting point, and
    #     ignore the estimated intrinsic scatter.
    print('Running fit iteration 2')
    disk.lsq_fit(kin, sb_wgt=True, p0=p0, fix=fix, lb=lb, ub=ub, ignore_covar=True,
                 assume_posdef_covar=assume_posdef_covar, analytic_jac=analytic_jac,
                 verbose=verbose)
    # Show
    if verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    #---------------------------------------------------------------------------
    # Fit iteration 3: 
    #   - Perform a more restricted rejection
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk.reject(disp=fitdisp, ignore_covar=True, vel_sigma_rej=_vel_sigma_rej[1],
                          show_vel=debug, sig_sigma_rej=_sig_sigma_rej[1], show_sig=debug,
                          verbose=verbose > 1)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements rejected due to large residuals.')
        vel_mask[vel_rej] = disk.mbm.turn_on(vel_mask[vel_rej], 'REJ_RESID')
    if fitdisp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements rejected due to large residuals.')
        sig_mask[sig_rej] = disk.mbm.turn_on(sig_mask[sig_rej], 'REJ_RESID')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Refit again with the inclination and center fixed, but use the
    #     previous fit as the starting point and include the estimated
    #     intrinsic scatter.
    print('Running fit iteration 3')
    scatter = np.array([vel_sig, sig_sig]) if fit_scatter else None
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, ignore_covar=True,
                 assume_posdef_covar=assume_posdef_covar, scatter=scatter,
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    #---------------------------------------------------------------------------
    # Fit iteration 4: 
    #   - Recover data from the restricted rejection
    reset_to_base_flags(disk, kin, vel_mask, sig_mask)
    #   - Reject again based on the new fit parameters
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk.reject(disp=fitdisp, ignore_covar=True, vel_sigma_rej=_vel_sigma_rej[1],
                          show_vel=debug, sig_sigma_rej=_sig_sigma_rej[1], show_sig=debug,
                          verbose=verbose > 1)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements rejected due to large residuals.')
        vel_mask[vel_rej] = disk.mbm.turn_on(vel_mask[vel_rej], 'REJ_RESID')
    if fitdisp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements rejected due to large residuals.')
        sig_mask[sig_rej] = disk.mbm.turn_on(sig_mask[sig_rej], 'REJ_RESID')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Refit again with the inclination and center fixed, but use the
    #     previous fit as the starting point and include the estimated
    #     intrinsic scatter.
    print('Running fit iteration 4')
    scatter = np.array([vel_sig, sig_sig]) if fit_scatter else None
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, ignore_covar=True,
                 assume_posdef_covar=assume_posdef_covar, scatter=scatter,
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    #---------------------------------------------------------------------------
    # Fit iteration 5: 
    #   - Recover data from the restricted rejection
    reset_to_base_flags(disk, kin, vel_mask, sig_mask)
    #   - Reject again based on the new fit parameters
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk.reject(disp=fitdisp, ignore_covar=True, vel_sigma_rej=_vel_sigma_rej[2],
                          show_vel=debug, sig_sigma_rej=_sig_sigma_rej[2], show_sig=debug,
                          verbose=verbose > 1)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements rejected due to large residuals.')
        vel_mask[vel_rej] = disk.mbm.turn_on(vel_mask[vel_rej], 'REJ_RESID')
    if fitdisp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements rejected due to large residuals.')
        sig_mask[sig_rej] = disk.mbm.turn_on(sig_mask[sig_rej], 'REJ_RESID')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Now fit as requested by the user, freeing one or both of the
    #     inclination and center. Use the previous fit as the starting point
    #     and include the estimated intrinsic scatter and the covariance.
    #                    x0     y0     pa     inc    vsys
    base_fix = np.array([False, False, False, False, False])
    if fix_cen:
        base_fix[:2] = True
    if fix_inc:
        base_fix[3] = True
    fix = np.append(base_fix, np.zeros(p0.size-5, dtype=bool))
    print('Running fit iteration 5')
    scatter = np.array([vel_sig, sig_sig]) if fit_scatter else None
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, ignore_covar=ignore_covar,
                 assume_posdef_covar=assume_posdef_covar, scatter=scatter,
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    #---------------------------------------------------------------------------
    # Fit iteration 6:
    #   - Recover data from the restricted rejection
    reset_to_base_flags(disk, kin, vel_mask, sig_mask)
    #   - Reject again based on the new fit parameters.
    # TODO: Make the rejection threshold for this last iteration a keyword
    # argument?
    print('Running rejection iterations')
    vel_rej, vel_sig, sig_rej, sig_sig \
            = disk.reject(disp=fitdisp, ignore_covar=ignore_covar, vel_sigma_rej=_vel_sigma_rej[3],
                          show_vel=debug, sig_sigma_rej=_sig_sigma_rej[3], show_sig=debug,
                          verbose=verbose > 1)
    if np.any(vel_rej):
        print(f'{np.sum(vel_rej)} velocity measurements rejected due to large residuals.')
        vel_mask[vel_rej] = disk.mbm.turn_on(vel_mask[vel_rej], 'REJ_RESID')
    if fitdisp and sig_rej is not None and np.any(sig_rej):
        print(f'{np.sum(sig_rej)} dispersion measurements rejected due to large residuals.')
        sig_mask[sig_rej] = disk.mbm.turn_on(sig_mask[sig_rej], 'REJ_RESID')
    #   - Incorporate the rejection into the Kinematics object
    kin.reject(vel_rej=vel_rej, sig_rej=sig_rej)
    #   - Redo previous fit
    print('Running fit iteration 6')
    scatter = np.array([vel_sig, sig_sig]) if fit_scatter else None
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, ignore_covar=ignore_covar,
                 assume_posdef_covar=assume_posdef_covar, scatter=scatter, 
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    if fix_inc or low_inc is None or disk.par[3] > low_inc:
        # Inclination is valid, so return
        return disk, p0, fix, vel_mask, sig_mask

    #---------------------------------------------------------------------------
    # Fit iteration 7:
    #   - The best-fitting inclination is below the viable value.  Flag it.
    disk.global_mask = disk.gbm.turn_on(disk.global_mask, 'LOWINC')
    #   - Refit the data, but fix the inclination to the guess value.
    #                    x0     y0     pa     inc   vsys
    base_fix = np.array([False, False, False, True, False])
    if fix_cen:
        # Fix the center, if requested
        base_fix[:2] = True
    fix = np.append(base_fix, np.zeros(p0.size-5, dtype=bool))
    disk.par[3] = galmeta.guess_inclination(lb=1., ub=89.)
    warnings.warn(f'Best-fitting inclination is below {low_inc:.1f} degrees.  Running a final '
                  f'fit fixing the inclination to {disk.par[3]:.1f}')
    print('Running fit iteration 7')
    disk.lsq_fit(kin, sb_wgt=True, p0=disk.par, fix=fix, lb=lb, ub=ub, ignore_covar=ignore_covar,
                 assume_posdef_covar=assume_posdef_covar, scatter=scatter,
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        axisym_fit_plot(galmeta, kin, disk, fix=fix)

    return disk, p0, fix, vel_mask, sig_mask


