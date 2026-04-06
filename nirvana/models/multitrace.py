"""
Module with a class that fits multiple tracers to a single disk.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

import warnings

from astropy.io import fits
from IPython import embed
from matplotlib import pyplot, rc, patches, ticker, colors
import numpy as np
from scipy import optimize

from nirvana import log
from .geometry import projected_polar, disk_ellipse
from ..data.util import select_kinematic_axis, bin_stats, growth_lim, atleast_one_decade
from .util import cov_err
from ..util import plot, fileio
from . import thindisk
from . import axisym

#warnings.simplefilter('error', RuntimeWarning)

class MultiTracerDisk:
    """
    Define a class that enables multiple kinematic datasets to be simultaneously
    fit with ThinDisk models.

    .. todo::

        internals ...

    Args:
        disk (array-like):
            The :class:`~nirvana.models.thindisk.ThinDisk` models to use for
            each dataset; i.e., there must be one disk model per dataset even if
            the models are identical.  Importantly, these must be separate
            object instances regardless of whether or not they are identical
            models.  The :class:`~nirvana.models.thindisk.ThinDisk` classes are
            used to perform a number of operations that are specific to each
            dataset that expedite the fit; the results are saved to the
            internals of the class meaning that there must be one instance per
            kinematic database.
        tie_base (array-like, optional):
            A 5-element vector indicating which geometric projection parameters
            should tied between the disk models.  The parameters are the
            dynamical center coordinates (x, y), the position angle, the
            inclination, and the systemic velocity.
        tie_disk (array-like, optional):
            A vector indicating which model-specific parameters should be tied
            between disks.  Models to be tied must have the same type and the
            same number of parameters.  The length of this vector must match the
            number of disk model parameters *excluding* the base parameters.
    """

    gbm = thindisk.ThinDiskGlobalBitMask()
    """
    Global bitmask.
    """

    mbm = thindisk.ThinDiskFitBitMask()
    """
    Measurement-specific bitmask.
    """

    pbm = thindisk.ThinDiskParBitMask()
    """
    Parameter-specific bitmask.
    """

    def __init__(self, disk, tie_base=None, tie_disk=None):

        self.disk = np.atleast_1d(disk)
        self.ntracer = self.disk.size
        if self.ntracer == 1:
            raise ValueError('You are only fitting one dataset.  Use the disk class directly!')

        # Parameter checks
        if np.any([self.disk[i+1].nbp != self.disk[0].nbp for i in range(self.ntracer-1)]):
            raise ValueError('All of the disks should have the same number of base parameters.')
        self.nbp = self.disk[0].nbp
        self.np = np.sum([d.np for d in self.disk])

        # Setup the ties between base and disk parameters
        self.tie_base = None        # Vector indicating which base parameters to tie between disks
        self.tie_disk = None        # Vector indicating which disk parameters to tie
        self.update_tie_base(tie_base)  # Set the base tying strategy
        self.update_tie_disk(tie_disk)  # Set the disk tying strategy

        # Setup the parameters
        self.par = None             # Holds the *unique* (untied) parameters
        self.par_err = None         # Holds any estimated parameter errors
        self.par_mask = None        # Holds any parameter-specific masking
        # TODO: Set this to the bitmask dtype
        self.global_mask = 0        # Global bitmask value
        self.fit_status = None      # Detailed fit status
        self.fit_success = None     # Simple fit success flag

        # Workspace
        self.disk_fom = None        # Functions used to calculate the minimized figure-of-merit
        self.disk_jac = None        # Functions used to calculate the fit Jacobian
        self._wrkspc_parslc = None  # Fit workspace: parameter slices
        self._wrkspc_ndata = None   #   ... Number of data points per dataset
        self._wrkspc_sdata = None   #   ... Starting indices of each dataset in the concatenation
        self._wrkspc_jac = None     #   ... Full fit Jacobian

    def __getitem__(self, item):
        return self.disk[item]

    def __setitem__(self, item, value):
        raise ValueError('Cannot change disk elements using direct access.')

    def update_tie_base(self, tie_base):
        """
        Setup parameter tying between the base geometric projection parameters.

        Function sets :attr:`tie_base` and calls :func:`retie`.

        Args:
            tie_base (array-like):
                Boolean array indicating which base parameters should be tied
                between disks.
        """
        # Set the internal
        self.tie_base = np.zeros(self.nbp, dtype=bool) if tie_base is None \
                            else np.atleast_1d(tie_base)
        # Check the result
        if self.tie_base.size != self.nbp:
            raise ValueError('Number of parameters in base tying vector incorrect!  Expected '
                             f'{self.nbp}, found {self.tie_base.size}.')
        if not np.issubdtype(self.tie_base.dtype, bool):
            raise TypeError('Base tying vector should hold booleans!')

        # Reset the tying vectors
        self.retie()

    @property
    def can_tie_disks(self):
        """
        Check if the disks can be tied.

        Current criteria are:

            - disks must have same class type
            - disks must have the same number of parameters

        Returns:
            :obj:`bool`: Flag that disk parameters can be tied.
        """
        types = np.array([d.__class__.__name__ for d in self.disk])
        npar = np.array([d.np for d in self.disk])
        return np.all(types == types[0]) and np.all(npar == npar[0])

    def update_tie_disk(self, tie_disk):
        """
        Setup parameter tying between the disk kinematic parameters

        Function sets :attr:`tie_disk` and calls :func:`retie`.

        Args:
            tie_disk (array-like):
                Boolean array indicating which kinematic parameters should be
                tied between disks.
        """
        # Set the internal
        ndp = self.disk[0].np - self.nbp
        self.tie_disk = np.zeros(ndp, dtype=bool) if tie_disk is None else np.atleast_1d(tie_disk)
        # Check that the disks *can* be tied
        if np.any(self.tie_disk) and not self.can_tie_disks:
            raise ValueError('Disk parameters cannot be tied together!')
        # Check the tying flags
        if self.tie_disk.size != ndp:
            raise ValueError('Number of parameters in kinematics tying vector incorrect!  '
                             f'Expected {self.disk[0].np - self.nbp}, found {self.tie_disk.size}.')
        if not np.issubdtype(self.tie_disk.dtype, bool):
            raise TypeError('Kinematics tying vector should hold booleans!')
        # Reset the tying vectors
        self.retie()

    def retie(self):
        """
        Construct the tying and untying vectors.

        Given the current :attr:`tie_base` and :attr:`tie_disk` arrays, this
        sets the :attr:`tie` and :attr:`untie` internals.

        If ``full_par`` provides the complete set of parameters, including
        redundant parameters that are tied and ``tied_par`` par only contains
        the unique, untied parameters, one can change between them using the
        :attr:`tie` and :attr:`untie` internal arrays as follows:

        .. code-block:: python

            full_par = tied_par[disk.untie]
            tied_par = full_par[disk.tie]

        where ``disk`` is an instance of this class.
        """
        # Find the starting index for the parameters of each disk
        s = [0 if i == 0 else np.sum([d.np for d in self.disk[:i]]) for i in range(self.ntracer)]

        # Instatiate as all parameters being untied
        self.untie = np.arange(self.np)

        # Tie base parameters by replacing parameter indices in disks with the
        # ones from the first disk.
        indx = np.where(self.tie_base)[0]
        if indx.size > 0: 
            for i in range(self.ntracer-1):
                self.untie[indx + s[i+1]] = indx
            
        # Tie disk parameters by replacing parameter indices in disks with the
        # ones from the first disk.
        indx = None if self.tie_disk is None else np.where(self.tie_disk)[0]
        if indx is not None and indx.size > 0:
            for i in range(self.ntracer-1):
                self.untie[indx + s[i+1] + self.nbp] = indx + self.nbp

        # The tying vector simply uses the unique indices in the untying vector
        self.tie, self.untie = np.unique(self.untie, return_inverse=True)

        # Set the number of untied, unique parameters
        self.nup = self.tie.size
            
    def guess_par(self, full=False):
        """
        Return a set of guess parameters based on the guess parameters for each disk.

        Args:
            full (:obj:`bool`, optional):
                Flag to return the full set of parameters, not just the unique,
                untied ones.

        Returns:
            `numpy.ndarray`_: Guess parameters
        """
        gp = np.concatenate([d.guess_par() for d in self.disk])
        return gp if full else gp[self.tie]

    def _init_par(self, p0, fix):
        """
        Initialize the parameter vectors.

        This includes the parameter vector itself, :attr:`par`, and the boolean
        vector selecting the free parameters, :attr:`free`.  This also sets the
        number of free parameters, :attr:`nfree`.

        Args:
            p0 (`numpy.ndarray`_):
                The parameters to use.  The length can be either the total
                number of parameters (:attr:`np`) or the total number of
                unique/untied parameters (:attr:`nup`).  If None, the parameters
                are set by :func:`guess_par`.
            fix (`numpy.ndarray`_):
                Boolean flags to fix the parameters.  If None or if ``p0`` is
                None, all parameters are assumed to be free.  If not None, the
                length *must* match ``p0``.
        """
        # If the parameters are not provided, use the defaults
        if p0 is None:
            if fix is not None:
                warnings.warn('To fix parameter, must provide the full set of initial guess '
                              'parameters.  Ignoring the fixed parameters provided and fitting '
                              'all parameters.')
            self.par = self.guess_par()
            self.free = np.ones(self.nup, dtype=bool)
            self.nfree = self.nup
            return

        _p0 = np.atleast_1d(p0)
        _free = np.ones(_p0.size, dtype=bool) if fix is None else np.logical_not(fix)
        if _p0.size not in [self.np, self.nup]:
            raise ValueError('Incorrect number of model parameters.  Must be either '
                             f'{self.np} (full) or {self.nup} (unique).')
        if _free.size != _p0.size:
            raise ValueError('Vector selecting fixed parameters has different length from '
                             f'parameter vector: {_free.size} instead of {_p0.size}.')
        self.par = _p0.copy()
        self.par_err = None
        self.free = _free.copy()
        if self.par.size == self.np:
            self.par = self.par[self.tie]
            self.free = self.free[self.tie]
        self.nfree = np.sum(self.free)

    def _set_par(self, par):
        """
        Set the full parameter vector, accounting for any fixed and/or tied
        parameters.

        Args:
            par (`numpy.ndarray`_, optional):
                The list of parameters to use. Length must be either the total
                number of *untied* parameters (:attr:`nup`) or the total number
                of free parameters (:attr:`nfree`).
        """
        if par.ndim != 1:
            raise ValueError('Parameter array must be a 1D vector.')
        if par.size == self.nup:
            self.par = par.copy()
            return
        if par.size != self.nfree:
            raise ValueError('Must provide {0} or {1} parameters.'.format(self.nup, self.nfree))
        self.par[self.free] = par.copy()

    def disk_slice(self, index):
        """
        Return the slice selecting parameters for the specified disk from the
        *full* parameter vector.

        Args:
            index (:obj:`int`):
                The index of the disk to select.

        Returns:
            :obj:`slice`: Slice selected the relevant parameters.
        """
        s = 0 if index == 0 else np.sum([d.np for d in self.disk[:index]])
        return slice(s, s + self.disk[index].np)

    def lsq_fit(self, kin, sb_wgt=False, p0=None, fix=None, lb=None, ub=None, scatter=None,
                verbose=0, assume_posdef_covar=False, ignore_covar=True, analytic_jac=True,
                maxiter=5):
        """
        Use non-linear least-squares minimization to simultaneously fit all
        datasets.

        For all input parameter vectors (``p0``, ``fix``, ``lb``, and ``ub``),
        the length of all vectors must be the same, but they can be either the
        total number of parameters (:attr:`np`) or the number of unique (untied)
        parameters (:attr:`nup`).  If the former, note that only the values of
        the tied parameters in the first disk will be used *regardless of
        whether or not the parameters are fixed*.
                
        .. warning::

            Currently, this class *does not construct a model of the
            surface-brightness distribution*.  Instead, any weighting of the
            model during convolution with the beam profile uses the as-observed
            surface-brightness distribution, instead of a model of the intrinsic
            surface brightness distribution.  See ``sb_wgt``.

        Args:
            kin (array-like):
                A list of :class:`~nirvana.data.kinematics.Kinematics` objects
                to fit.  The number of datasets *must* match the number of disks
                (:attr:`ntracer`).
            sb_wgt (:obj:`bool`, optional):
                Flag to use the surface-brightness data provided by each dataset
                to weight the model when applying the beam-smearing.  **See the
                warning above**.
            p0 (`numpy.ndarray`_, optional):
                The initial parameters for the model.  See function description
                for the allowed shapes.  If None, the default guess parameters
                are used; see :func:`guess_par`.
            fix (`numpy.ndarray`_, optional):
                A boolean array selecting the parameters that should be fixed
                during the model fit.  See function description for the allowed
                shapes.  If None, all parameters are free.
            lb (`numpy.ndarray`_, optional):
                The lower bounds for the parameters.  See function description
                for the allowed shapes.  If None, the defaults are used (see
                :func:`par_bounds`).
            ub (`numpy.ndarray`_, optional):
                The upper bounds for the parameters.  See function description
                for the allowed shapes.  If None, the defaults are used (see
                :func:`par_bounds`).
            scatter (:obj:`float`, array-like, optional):
                Introduce a fixed intrinsic-scatter term into the model. The 
                scatter is added in quadrature to all measurement errors in the
                calculation of the merit function. If no errors are available,
                this has the effect of renormalizing the unweighted merit
                function by 1/scatter.  Can be None, which means no intrinsic
                scatter is added.  The number of provided values can be either:

                    - 1 (i.e., a single float) to apply the same scatter to all
                      kinematic moments and datasets, 
                    - the same as the number of disks to apply a separate
                      scatter to both kinematic moments for each dataset
                    - twice the number of disks to apply a separate scatter to
                      all the kinematic moments and datasets.  In this case, the
                      order is velocity scatter for disk 1, velocity dispersion
                      scatter for disk 1, velocity scatter for disk 2, etc.

            verbose (:obj:`int`, optional):
                Verbosity level to pass to `scipy.optimize.least_squares`_.
            assume_posdef_covar (:obj:`bool`, optional):
                If any of the :class:`~nirvana.data.kinematics.Kinematics`
                datasets include covariance matrices, this forces the code to
                proceed assuming the matrices are positive definite.
            ignore_covar (:obj:`bool`, optional):
                If any of the :class:`~nirvana.data.kinematics.Kinematics`
                datasets include covariance matrices, ignore them and just use
                the inverse variance.
            analytic_jac (:obj:`bool`, optional):
                Use the analytic calculation of the Jacobian matrix during the
                fit optimization.  If False, the Jacobian is calculated using
                finite-differencing methods provided by
                `scipy.optimize.least_squares`_.
            maxiter (:obj:`int`, optional):
                The call to `scipy.optimize.least_squares`_ is repeated when it
                returns best-fit parameters that are *identical* to the input
                parameters (to within a small tolerance).  This parameter sets
                the maximum number of times the fit will be repeated.  Set this
                to 1 to ignore these occurences; ``maxiter`` cannot be None.
        """

        # Check the input
        self.kin = np.atleast_1d(kin)
        if self.kin.size != self.ntracer:
            raise ValueError('Must provide the same number of kinematic databases as disks '
                             f'({self.ntracer}).')

        if scatter is not None:
            _scatter = np.atleast_1d(scatter)
            if _scatter.size not in [1, self.ntracer, 2*self.ntracer]:
                raise ValueError(f'Number of scatter terms must be 1, {self.ntracer}, or '
                                f'{2*self.ntracer}; found {_scatter.size}.')
            if _scatter.size == 1:
                _scatter = np.repeat(_scatter, 2*self.ntracer).reshape(self.ntracer,-1)
            elif _scatter.size == 2:
                _scatter = np.tile(_scatter, (self.ntracer,1))
            else:
                _scatter = scatter.reshape(self.ntracer,-1)

        # Initialize the parameters.  This checks that the parameters have the
        # correct length.
        self._init_par(p0, fix)

        # Prepare the disks for fitting
        self.disk_fom = [None]*self.ntracer
        self.disk_jac = [None]*self.ntracer

        self._wrkspc_parslc = [self.disk_slice(i) for i in range(self.ntracer)]
        for i in range(self.ntracer):
            self.disk[i]._init_par(self.par[self.untie][self._wrkspc_parslc[i]], None)
            self.disk[i]._init_model(None, self.kin[i].grid_x, self.kin[i].grid_y,
                                     self.kin[i].grid_sb if sb_wgt else None,
                                     self.kin[i].beam_fft, True, None, False)
            self.disk[i]._init_data(self.kin[i], None if scatter is None else _scatter[i],
                                    assume_posdef_covar, ignore_covar)
            self.disk_fom[i] = self.disk[i]._get_fom()
            self.disk_jac[i] = self.disk[i]._get_jac()

        if analytic_jac:
            # Set the least_squares keywords
            jac_kwargs = {'jac': self.jac}
            # Set up the workspace for dealing with tied parameters
            self._wrkspc_ndata = [np.sum(self.disk[i].vel_gpm) + (0 if self.disk[i].dc is None
                                    else np.sum(self.disk[i].sig_gpm))
                                        for i in range(self.ntracer)]
            self._wrkspc_sdata = np.append([0],np.cumsum(self._wrkspc_ndata)[:-1])
            self._wrkspc_jac = np.zeros((np.sum(self._wrkspc_ndata), self.nup), dtype=float)
        else:
            jac_kwargs = {'diff_step': np.full(self.nup, 0.01, dtype=float)[self.free]}

        # Parameter boundaries
        if lb is None or ub is None:
            _lb, _ub = self.par_bounds()
        if lb is not None:
            _lb = lb[self.tie] if lb.size == self.np else lb.copy()
        if ub is not None:
            _ub = ub[self.tie] if ub.size == self.np else ub.copy()
        if len(_lb) != self.nup or len(_ub) != self.nup:
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
            result = optimize.least_squares(self.fom, p, x_scale='jac', method='trf',
                                            xtol=1e-12, bounds=(_lb[self.free], _ub[self.free]), 
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
            warnings.warn('Parameters unchanged after fit.  Retrying...')

            # If it doesn't, something likely went wrong with the fit.  Perturb
            # the input guesses a bit and retry.
            p = _p0 + rng.normal(size=self.nfree)*(pe if pe is not None else 0.1*p0)
            p = np.clip(p, _lb[self.free], _ub[self.free])
            niter += 1

        if niter == maxiter and np.all(np.absolute(p-result.x) > 1e-3):
            warnings.warn('Parameters unchanged after fit.  Abandoning iterations...')
            # TODO: Save this to the status somehow

        # TODO: Add something to the fit status/success flags that tests if
        # niter == maxiter and/or if the input parameters are identical to the
        # final best-fit parameters?  Note that the input parameters, p0, may not
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
            self.par_err = np.zeros(self.nup, dtype=float)
            self.par_err[self.free] = pe

        # Initialize the mask
        self.par_mask = self.pbm.init_mask_array(self.nup)
        # Check if any parameters are "at" the boundary
        pm = self.par_mask[self.free]
        for v, flg in zip([-1, 1], ['LOWERBOUND', 'UPPERBOUND']):
            indx = result.active_mask == v
            if np.any(indx):
                pm[indx] = self.pbm.turn_on(pm[indx], flg)

        # Check if any parameters are within 1-sigma of the boundary
        indx = self.par[self.free] - self.par_err[self.free] < _lb[self.free]
        if np.any(indx):
            pm[indx] = self.pbm.turn_on(pm[indx], 'LOWERERR')
        indx = self.par[self.free] + self.par_err[self.free] > _ub[self.free]
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

    def par_bounds(self, base_lb=None, base_ub=None):
        """
        Return the lower and upper bounds for the unique, untied parameters.

        Returns:
            :obj:`tuple`: A two-tuple of `numpy.ndarray`_ objects with the lower
            and upper parameter boundaries.
        """
        lb, ub = np.array([list(d.par_bounds(base_lb=base_lb, base_ub=base_ub)) 
                                for d in self.disk]).transpose(1,0,2).reshape(2,-1)
        return lb[self.tie], ub[self.tie]

    def fom(self, par):
        """
        Calculate the figure-of-merit for a model by concatenating the result
        from the figure-of-merit calculations provided for each disk.  This is
        the function used by the least-squares optimization in the simultaneous
        fitting process.

        Args:
            par (`numpy.ndarray`_):
                The model parameters; see :func:`_set_par`.
    
        Returns:
            `numpy.ndarray`_: The vector with the model residuals for each
            fitted data point.
        """
        # Get the tied parameters
        self._set_par(par)
        # Untie them to get the full set
        full_par = self.par[self.untie]
        return np.concatenate([self.disk_fom[i](full_par[self._wrkspc_parslc[i]]) 
                                for i in range(self.ntracer)])

    def jac(self, par):
        """
        Compute the model Jacobian using the analytic derivatives provided by
        each disk independently.

        Args:
            par (`numpy.ndarray`_):
                The model parameters; see :func:`_set_par`.
    
        Returns:
            `numpy.ndarray`_: The array with the model Jacobian.
        """
        # Get the tied parameters
        self._set_par(par)
        # Untie them to get the full set
        full_par = self.par[self.untie]
        _jac = [self.disk_jac[i](full_par[self._wrkspc_parslc[i]]) for i in range(self.ntracer)] 
        self._wrkspc_jac[...] = 0.
        assert np.array_equal(self._wrkspc_ndata, [j.shape[0] for j in _jac]), \
                'Jacobians have incorrect shape!'
        for i in range(self.ntracer):
            sec = np.ix_(np.arange(self._wrkspc_ndata[i])+self._wrkspc_sdata[i],
                         self.untie[self._wrkspc_parslc[i]])
            self._wrkspc_jac[sec] += _jac[i]
        return self._wrkspc_jac[:,self.free]

    def distribute_par(self):
        """
        Distribute the current parameter set to the disk components.
        """
        full_par = self.par[self.untie]
        full_par_err = None if self.par_err is None else self.par_err[self.untie]
        full_free = self.free[self.untie]
        tied = np.setdiff1d(np.arange(self.np), self.tie)
        full_free[tied] = False
        if self.par_mask is None:
            full_par_mask = None
        else:
            full_par_mask = self.par_mask[self.untie]
            full_par_mask[tied] = self.pbm.turn_on(full_par_mask[tied], 'TIED')
        for i in range(self.ntracer):
            slc = self.disk_slice(i)
            self.disk[i].par = full_par[slc]
            self.disk[i].free = full_free[slc]
            self.disk[i].par_err = None if full_par_err is None else full_par_err[slc]
            self.disk[i].par_mask = None if full_par_mask is None else full_par_mask[slc]

    def report(self, fit_message=None):
        """
        Report the current parameters of the model to the screen.

        Args:
            fit_message (:obj:`str`, optional):
                The status message returned by the fit optimization.
        """
        if self.par is None:
            log.info('No parameters to report.')
            return

        log.info('-'*70)
        log.info(f'{"Fit Result":^70}')
        log.info('-'*70)
        if fit_message is not None:
            log.info(f'Fit status message: {fit_message}')
        if self.fit_status is not None:
            log.info(f'Fit status: {self.fit_status}')
        log.info(f'Fit success: {"True" if self.fit_status else "False"}')
        log.info('-'*50)
        # Indicate which parameters are tied
        if np.any(self.tie_base) or np.any(self.tie_disk):
            disk0_parn = self.disk[0].par_names()
            tied = self.tie_base if not np.any(self.tie_disk) \
                        else np.append(self.tie_base, self.tie_disk)
            log.info('Tied Parameters:')
            for i in range(tied.size):
                if not tied[i]:
                    continue
                log.info(f'{disk0_parn[i]:>30}')
        else:
            log.info('No tied parameters')

        # Print the results for each disk
        self.distribute_par()
        for i in range(self.ntracer):
            log.info('-'*50)
            log.info(f'{f"Disk {i+1}":^50}')
            log.info('-'*50)
            self.disk[i].report(component=True)
        log.info('-'*50)

        resid = self.fom(self.par)
        chisqr = np.sum(resid**2) / (resid.size - self.nfree)
        log.info(f'Total reduced chi-square: {chisqr}')
        log.info('-'*70)        


# TODO:
#   - Add keyword for radial sampling for 1D model RCs and dispersion profiles
#   - This is MaNGA-specific and needs to be abstracted
#   - Allow the plot to be constructed from the fits file written by
#     axisym_fit_data
def asymdrift_fit_plot(galmeta, kin, disk, par=None, par_err=None, fix=None, ofile=None):
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

    if disk.par is None and par is None:
        raise ValueError('No model parameters available.  Provide directly or via disk argument.')

    _par = disk.par[disk.untie] if par is None else par
    if disk.par_err is None and par_err is None:
        _par_err = np.full(_par.size, -1., dtype=float)
    else:
        _par_err = disk.par_err[disk.untie] if par_err is None else par_err

    if fix is None:
        _fix = np.logical_not(disk.free)[disk.untie]
        _fix[np.setdiff1d(np.arange(disk.np), disk.tie)] = True
    else:
        _fix = fix

    if _par.size != disk.np:
        raise ValueError('Number of provided parameters has the incorrect size.')
    if _par_err.size != disk.np:
        raise ValueError('Number of provided parameter errors has the incorrect size.')
    if _fix.size != disk.np:
        raise ValueError('Number of provided parameter fixing flags has the incorrect size.')

    mean_vsys = (_par[4] + _par[disk.disk[0].np+4])/2

    # TODO: Move these to arguments for the function?
    fwhm = galmeta.psf_fwhm[1]
    oversample = 1.5
    maj_wedge = 30.
    rstep = fwhm/oversample

    # Build the AD data
    gv_map, gv_ivar_map, gv_mod_map, gv_mod_intr_map, \
        sv_map, sv_ivar_map, sv_mod_map, sv_mod_intr_map, \
        sd_map, sd_ivar_map, sd_mod_map, sd_mod_intr_map, \
        ad_map, ad_ivar_map, ad_bc_map, ad_bc_ivar_map, ad_mod_map, ad_mod_bc_map, ad_mask_map, \
        ados_map, ados_ivar_map, ados_bc_map, ados_bc_ivar_map, ados_mod_map, ados_mod_bc_map, \
            ados_mask_map, \
        spax_ad_r, spax_ad, _, spax_ad_mask, \
        ad_binr, \
        ad_ewmean, ad_ewsdev, ad_mod_ewmean, ad_mod_ewsdev, ad_nbin, \
        ad_bc_ewmean, ad_bc_ewsdev, ad_mod_bc_ewmean, ad_mod_bc_ewsdev, ad_bc_nbin, \
        ados_ewmean, ados_ewsdev, ados_mod_ewmean, ados_mod_ewsdev, ados_nbin, \
        ados_bc_ewmean, ados_bc_ewsdev, ados_mod_bc_ewmean, ados_mod_bc_ewsdev, ados_bc_nbin \
            = asymdrift_fit_maps(kin, disk, rstep, maj_wedge=maj_wedge)

    # Surface brightness maps
    gs_map = kin[0].remap('sb')
    ss_map = kin[1].remap('sb')

    # Get the projected spaxel data and the binned radial profiles for the
    # kinematic data.
    spax_vrot_r = [None]*2
    spax_vrot = [None]*2
    spax_smaj_r = [None]*2
    spax_smaj = [None]*2

    bin_r = [None]*2
    bin_vrot = [None]*2
    bin_vrote = [None]*2
    bin_vrotn = [None]*2
    bin_smaj = [None]*2
    bin_smaje = [None]*2
    bin_smajn = [None]*2

    for i in range(2):
        # Get the projected rotational velocity
        #   - Disk-plane coordinates
        r, th = projected_polar(kin[i].x - disk.disk[i].par[0], kin[i].y - disk.disk[i].par[1],
                                *np.radians(disk.disk[i].par[2:4]))
        #   - Mask for data along the major axis
        major_gpm = select_kinematic_axis(r, th, which='major', r_range='all', wedge=maj_wedge)
        #   - Projected rotation velocities
        indx = major_gpm & np.logical_not(kin[i].vel_mask)
        spax_vrot_r[i] = r[indx]
        spax_vrot[i] = (kin[i].vel[indx] - disk.disk[i].par[4])/np.cos(th[indx])
        #   - Major axis velocity dispersions
        indx = major_gpm & np.logical_not(kin[i].sig_mask) & (kin[i].sig_phys2 > 0)
        spax_smaj_r[i] = r[indx]
        spax_smaj[i] = np.sqrt(kin[i].sig_phys2[indx])

        bin_r[i], bin_vrot[i], bin_vrote[i], _, bin_vrotn[i], _, _, _, _, _, _, _, _, \
            _, _, _, _, _, _, bin_smaj[i], bin_smaje[i], _, bin_smajn[i], _, _, _, _, _, _, _, _, \
                = kin[i].radial_profiles(rstep, xc=disk.disk[i].par[0], yc=disk.disk[i].par[1],
                                         pa=disk.disk[i].par[2], inc=disk.disk[i].par[3],
                                         vsys=disk.disk[i].par[4], maj_wedge=maj_wedge)

    # Get the 1D model profiles
    # NOTE: This catches cases when there is no data, which makes the plot
    # useless, but it avoids a fault.
    maxr = rstep if len(spax_vrot_r) == 0 and len(spax_smaj_r) == 0 \
                else np.amax(np.concatenate(spax_vrot_r+spax_smaj_r))
    modelr = np.arange(0, maxr, 0.1)
    vrotm = [None]*2
    smajm = [None]*2
    for i in range(2):
        vrotm[i] = disk.disk[i].rc.sample(modelr, par=disk.disk[i].rc_par())
        smajm[i] = disk.disk[i].dc.sample(modelr, par=disk.disk[i].dc_par())

    # Construct an ellipse that has a constant disk radius and is at the
    # best-fit center, position angle, and inclination.  Set the radius to the
    # maximum of the valid binned rotation curve measurements, selecting the
    # larger value between the gas and stars.
    vrot_indx = [vrn > 5 for vrn in bin_vrotn]
    for i in range(2):
        if not np.any(vrot_indx[i]):
            vrot_indx[i] = bin_vrotn[i] > 0
    if not np.any(np.append(*vrot_indx)):
        de_x, de_y = None, None
    else:
        # NOTE: Assumes geometric parameters are tied!
        de_r = np.amax(np.append(bin_r[0][vrot_indx[0]], bin_r[1][vrot_indx[1]]))
        de_x, de_y = disk_ellipse(de_r, *np.radians(disk[0].par[2:4]), xc=disk[0].par[0],
                                  yc=disk[0].par[1])

    resid = disk.fom(_par[disk.tie])
    rchisqr = np.sum(resid**2) / (resid.size - disk.nfree)

    gpm = np.logical_not(spax_ad_mask) & (spax_ad > 0)
    spax_ad_r = spax_ad_r[gpm]
    spax_ad = np.sqrt(spax_ad[gpm])
    bin_ad = np.ma.sqrt(ad_ewmean).filled(0.0)
    bin_ade = np.ma.divide(ad_ewsdev, 2*bin_ad).filled(0.0)

    # Set the radius limits for the radial plots
    r_lim = [0.0, maxr * 1.05]
    rc_lim = growth_lim(np.concatenate(bin_vrot+vrotm), 0.99, 1.3)
    smaj_lim = growth_lim(np.ma.log10(np.concatenate(smajm + [bin_ad])).compressed(), 0.9, 1.5)
    smaj_lim = atleast_one_decade(np.power(10.0, smaj_lim))

    # TODO: Extent may need to be adjusted by 0.25 arcsec!  extent is from the
    # edge of the pixel, not from its center.
    # Set the extent for the 2D maps
    extent = [np.amax(kin[0].grid_x), np.amin(kin[0].grid_x),
              np.amin(kin[0].grid_y), np.amax(kin[0].grid_y)]
    Dx = max(extent[0]-extent[1], extent[3]-extent[2]) # *1.01
    skylim = np.array([ (extent[0]+extent[1] - Dx)/2., 0.0 ])
    skylim[1] = skylim[0] + Dx

    sb_lim = [growth_lim(np.ma.log10(gs_map).compressed(), 0.90, 1.05),
              growth_lim(np.ma.log10(ss_map).compressed(), 0.90, 1.05)]
    sb_lim = [atleast_one_decade(np.power(10.0, sb_lim[0])),
              atleast_one_decade(np.power(10.0, sb_lim[1]))]

    vel_lim = growth_lim(np.ma.concatenate([gv_map, sv_map, gv_mod_map, sv_mod_map]).compressed(),
                         0.90, 1.05, midpoint=mean_vsys)

    ad_lim = growth_lim(np.ma.log10(np.ma.append(ad_map, ad_mod_map)).compressed(), 0.70, 1.05)
    ad_lim = atleast_one_decade(np.power(10.0, ad_lim))

    sig_map_lim = growth_lim(np.ma.log10(np.ma.append(sd_map, sd_mod_map)).compressed(), 0.70, 1.05)
    sig_map_lim = atleast_one_decade(np.power(10., sig_map_lim))

    ados_lim = growth_lim(np.ma.append(ados_map, ados_mod_map).compressed(), 0.80, 1.05)

    # Create the plot
    w,h = pyplot.figaspect(1)
    fig = pyplot.figure(figsize=(2*w,2*h))

    #-------------------------------------------------------------------
    # Gas velocity field
    ax = plot.init_ax(fig, [0.02, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(gv_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[0].par[0], disk.disk[0].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$V_g$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Gas velocity field model
    ax = plot.init_ax(fig, [0.02, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.05, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(gv_mod_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[0].par[0], disk.disk[0].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    ax.text(0.05, 0.90, r'$V_{g,m}$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Stellar velocity field
    ax = plot.init_ax(fig, [0.215, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.245, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sv_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$V_\ast$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Stellar velocity field model
    ax = plot.init_ax(fig, [0.215, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.245, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sv_mod_map, origin='lower', interpolation='nearest', cmap='RdBu_r',
                   extent=extent, vmin=vel_lim[0], vmax=vel_lim[1], zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    ax.text(0.05, 0.90, r'$V_{\ast,m}$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Measured AD
    ax = plot.init_ax(fig, [0.410, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.440, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(ad_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=ad_lim[0], vmax=ad_lim[1]), zorder=4) 
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$\sigma^2_a$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # AD Model
    ax = plot.init_ax(fig, [0.410, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.440, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(ad_mod_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=ad_lim[0], vmax=ad_lim[1]), zorder=4)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    ax.text(0.05, 0.90, r'$\sigma^2_{a,m}$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Velocity Dispersion
    ax = plot.init_ax(fig, [0.605, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.635, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sd_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=sig_map_lim[0], vmax=sig_map_lim[1]),
                   zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$\sigma^2_\ast$', ha='left', va='center', transform=ax.transAxes)
 
    #-------------------------------------------------------------------
    # Velocity Dispersion Model
    ax = plot.init_ax(fig, [0.605, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.635, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(sd_mod_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, norm=colors.LogNorm(vmin=sig_map_lim[0], vmax=sig_map_lim[1]),
                   zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    ax.text(0.05, 0.90, r'$\sigma^2_{\ast,m}$', ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # AD ratio
    ax = plot.init_ax(fig, [0.800, 0.775, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.97, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(ados_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, vmin=ados_lim[0], vmax=ados_lim[1], zorder=4)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$\sigma^2_a/\sigma^2_\ast$',
            ha='left', va='center', transform=ax.transAxes)

    #-------------------------------------------------------------------
    # Model AD ratio
    ax = plot.init_ax(fig, [0.800, 0.580, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.57, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(ados_mod_map, origin='lower', interpolation='nearest', cmap='viridis',
                   extent=extent, vmin=ados_lim[0], vmax=ados_lim[1], zorder=4)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    ax.text(0.05, 0.90, r'$\sigma^2_{a,m}/\sigma^2_{\ast,m}$', ha='left', va='center',
             transform=ax.transAxes)

    #-------------------------------------------------------------------
    # H-alpha surface-brightness
    ax = plot.init_ax(fig, [0.800, 0.305, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.50, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(gs_map, origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=sb_lim[0][0], vmax=sb_lim[0][1]),
                   zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[0].par[0], disk.disk[0].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    # TODO: For some reason, the combination of the use of a masked array and
    # setting the formatter to logformatter leads to weird behavior in the map.
    # Use something like the "pallete" object described here?
    #   https://matplotlib.org/stable/gallery/images_contours_and_fields/image_masked.html
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax.text(0.05, 0.90, r'$\mu_g$', ha='left', va='center', transform=ax.transAxes)

#    ax.text(0.5, 1.2, 'Intrinsic Model', ha='center', va='center', transform=ax.transAxes,
#            fontsize=10)

    #-------------------------------------------------------------------
    # Continuum surface brightness
    ax = plot.init_ax(fig, [0.800, 0.110, 0.19, 0.19])
    cax = fig.add_axes([0.830, 0.10, 0.15, 0.005])
    cax.tick_params(which='both', direction='in')
    ax.set_xlim(skylim[::-1])
    ax.set_ylim(skylim)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.add_patch(patches.Circle((0.1, 0.1), fwhm/np.diff(skylim)[0]/2, transform=ax.transAxes,
                                facecolor='0.7', edgecolor='k', zorder=4))
    im = ax.imshow(ss_map, origin='lower', interpolation='nearest', cmap='inferno',
                   extent=extent, norm=colors.LogNorm(vmin=sb_lim[1][0], vmax=sb_lim[1][1]),
                   zorder=4)
    # Mark the fitted dynamical center
    ax.scatter(disk.disk[1].par[0], disk.disk[1].par[1],
               marker='+', color='k', s=40, lw=1, zorder=5)
    # Plot the ellipse with constant disk radius
    if de_x is not None:
        ax.plot(de_x, de_y, color='w', lw=2, zorder=6, alpha=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', format=logformatter)
    ax.text(0.05, 0.90, r'$\mu_\ast$', ha='left', va='center', transform=ax.transAxes)

#    #-------------------------------------------------------------------
#    # Annotate with the intrinsic scatter included
#    ax.text(0.00, -0.2, r'V scatter, $\epsilon_v$:', ha='left', va='center',
#            transform=ax.transAxes, fontsize=10)
#    ax.text(1.00, -0.2, f'{vsct:.1f}', ha='right', va='center', transform=ax.transAxes,
#            fontsize=10)
#    if disk.dc is not None:
#        ax.text(0.00, -0.3, r'$\sigma^2$ scatter, $\epsilon_{\sigma^2}$:', ha='left', va='center',
#                transform=ax.transAxes, fontsize=10)
#        ax.text(1.00, -0.3, f'{ssct:.1f}', ha='right', va='center', transform=ax.transAxes,
#                fontsize=10)

    #-------------------------------------------------------------------
    # SDSS image
    ax = fig.add_axes([0.01, 0.29, 0.23, 0.23])
    if kin[0].image is not None:
        ax.imshow(kin[0].image)
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

    # Assume center, PA, and inclination are tied, and that the systemic velocities are not.

    # MaNGA ID
    ax.text(0.00, -0.05, 'MaNGA ID:', ha='left', va='center', transform=ax.transAxes, fontsize=10)
    ax.text(1.01, -0.05, f'{galmeta.mangaid}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)
    # Observation
    ax.text(0.00, -0.13, 'Observation:', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -0.13, galmeta.plateifu, ha='right', va='center',
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
    xstr = r'{0:.2f}'.format(_par[0]) if _fix[0] \
            else r'{0:.2f} $\pm$ {1:.2f}'.format(_par[0], _par_err[0])
    ax.text(1.01, -0.77, xstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[0] else 'k')
    ax.text(0.00, -0.85, r'$y_0$ [arcsec]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[1] else 'k')
    ystr = r'{0:.2f}'.format(_par[1]) if _fix[1] \
            else r'{0:.2f} $\pm$ {1:.2f}'.format(_par[1], _par_err[1])
    ax.text(1.01, -0.85, ystr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[1] else 'k')
    # Position angle
    ax.text(0.00, -0.93, r'$\phi_0$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[2] else 'k')
    pastr = r'{0:.1f}'.format(_par[2]) if _fix[2] \
            else r'{0:.1f} $\pm$ {1:.1f}'.format(_par[2], _par_err[2])
    ax.text(1.01, -0.93, pastr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[2] else 'k')
    # Kinematic Inclination
    ax.text(0.00, -1.01, r'$i_{\rm kin}$ [deg]', ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='C3' if _fix[3] else 'k')
    incstr = r'{0:.1f}'.format(_par[3]) if _fix[3] \
            else r'{0:.1f} $\pm$ {1:.1f}'.format(_par[3], _par_err[3])
    ax.text(1.01, -1.01, incstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10,
            color='C3' if _fix[3] else 'k')
    # Systemic velocity
    ax.text(0.00, -1.09, r'$\langle V_{\rm sys}\rangle$ [km/s]',
            ha='left', va='center', transform=ax.transAxes,
            fontsize=10, color='k')
    vsysstr = r'{0:.1f}'.format(mean_vsys)
    ax.text(1.01, -1.09, vsysstr,
            ha='right', va='center', transform=ax.transAxes, fontsize=10, color='k')
    # Reduced chi-square
    ax.text(0.00, -1.17, r'$\chi^2_\nu$', ha='left', va='center', transform=ax.transAxes,
            fontsize=10)
    ax.text(1.01, -1.17, f'{rchisqr:.2f}', ha='right', va='center', transform=ax.transAxes,
            fontsize=10)

    #-------------------------------------------------------------------
    # Radial plot radius limits
    # Select bins with sufficient data
    vrot_indx = [vrn > 5 for vrn in bin_vrotn]
    smaj_indx = [smn > 5 for smn in bin_smajn]
    for i in range(2):
        if not np.any(vrot_indx[i]):
            vrot_indx[i] = bin_vrotn[i] > 0
        if not np.any(smaj_indx[i]):
            smaj_indx[i] = bin_smajn[i] > 0
    ad_indx = ad_nbin > 5
    if not np.any(ad_indx):
        ad_indx = ad_nbin > 0

    #-------------------------------------------------------------------
    # Rotation curves
    reff_lines = np.arange(galmeta.reff, r_lim[1], galmeta.reff) if galmeta.reff > 1 else None

    ax = plot.init_ax(fig, [0.27, 0.27, 0.51, 0.23], facecolor='0.9', top=False, right=False)
    ax.set_xlim(r_lim)
    ax.set_ylim(rc_lim)
    plot.rotate_y_ticks(ax, 90, 'center')
    ax.xaxis.set_major_formatter(ticker.NullFormatter())

    # Gas
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('C3')])
    ax.scatter(spax_vrot_r[0], spax_vrot[0],
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(vrot_indx[0]):
        ax.scatter(bin_r[0][vrot_indx[0]], bin_vrot[0][vrot_indx[0]],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_r[0][vrot_indx[0]], bin_vrot[0][vrot_indx[0]],
                   marker='o', s=90, alpha=1.0, color='C3', zorder=4)
        ax.errorbar(bin_r[0][vrot_indx[0]], bin_vrot[0][vrot_indx[0]],
                    yerr=bin_vrote[0][vrot_indx[0]], color='C3', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, vrotm[0], color='C3', zorder=5, lw=1)
    # Stars
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('C0')])
    ax.scatter(spax_vrot_r[1], spax_vrot[1],
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(vrot_indx[1]):
        ax.scatter(bin_r[1][vrot_indx[1]], bin_vrot[1][vrot_indx[1]],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_r[1][vrot_indx[1]], bin_vrot[1][vrot_indx[1]],
                   marker='o', s=90, alpha=1.0, color='C0', zorder=4)
        ax.errorbar(bin_r[1][vrot_indx[1]], bin_vrot[1][vrot_indx[1]],
                    yerr=bin_vrote[1][vrot_indx[1]], color='C0', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, vrotm[1], color='C0', zorder=5, lw=1)

    if reff_lines is not None:
        for l in reff_lines:
            ax.axvline(x=l, linestyle='--', lw=0.5, zorder=1, color='k')

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

#    kin_inc = disk.par[3]
#    axt = plot.get_twin(ax, 'y')
#    axt.set_xlim(r_lim)
#    axt.set_ylim(np.array(rc_lim)/np.sin(np.radians(kin_inc)))
#    plot.rotate_y_ticks(axt, 90, 'center')
#    axt.spines['right'].set_color('0.4')
#    axt.tick_params(which='both', axis='y', colors='0.4')
#    axt.yaxis.label.set_color('0.4')

    ax.add_patch(patches.Rectangle((0.79,0.45), 0.19, 0.09, facecolor='w', lw=0, edgecolor='none',
                                   zorder=5, alpha=0.7, transform=ax.transAxes))
    ax.text(0.97, 0.451, r'$V\ \sin i$ [km/s]', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=10, zorder=6)
#    ax.text(0.97, 0.56, r'$V$ [km/s; right axis]', ha='right', va='bottom', color='0.4',
#            transform=ax.transAxes, fontsize=10, zorder=6)


    #-------------------------------------------------------------------
    # Velocity Dispersion profile
    ax = plot.init_ax(fig, [0.27, 0.04, 0.51, 0.23], facecolor='0.9')
    ax.set_xlim(r_lim)
    ax.set_ylim(smaj_lim)#[10,275])
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(logformatter)
    plot.rotate_y_ticks(ax, 90, 'center')

    # Gas
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('C3')])
    ax.scatter(spax_smaj_r[0], spax_smaj[0],
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(smaj_indx[0]):
        ax.scatter(bin_r[0][smaj_indx[0]], bin_smaj[0][smaj_indx[0]],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_r[0][smaj_indx[0]], bin_smaj[0][smaj_indx[0]],
                   marker='o', s=90, alpha=1.0, color='C3', zorder=4)
        ax.errorbar(bin_r[0][smaj_indx[0]], bin_smaj[0][smaj_indx[0]],
                    yerr=bin_smaje[0][smaj_indx[0]], color='C3', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, smajm[0], color='C3', zorder=5, lw=1)
    # Stars
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('C0')])
    ax.scatter(spax_smaj_r[1], spax_smaj[1],
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(smaj_indx[1]):
        ax.scatter(bin_r[1][smaj_indx[1]], bin_smaj[1][smaj_indx[1]],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(bin_r[1][smaj_indx[1]], bin_smaj[1][smaj_indx[1]],
                   marker='o', s=90, alpha=1.0, color='C0', zorder=4)
        ax.errorbar(bin_r[1][smaj_indx[1]], bin_smaj[1][smaj_indx[1]],
                    yerr=bin_smaje[1][smaj_indx[1]], color='C0', capsize=0,
                    linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, smajm[1], color='C0', zorder=5, lw=1)
    # Sigma AD
    _c = tuple([(1-x)*0.2+x for x in colors.to_rgb('k')])
    ax.scatter(spax_ad_r, spax_ad,
               marker='.', color=_c, s=30, lw=0, alpha=0.6, zorder=1)
    if np.any(ad_indx):
        ax.scatter(ad_binr[ad_indx], bin_ad[ad_indx],
                   marker='o', s=110, alpha=1.0, color='white', zorder=3)
        ax.scatter(ad_binr[ad_indx], bin_ad[ad_indx],
                   marker='o', s=90, alpha=1.0, color='k', zorder=4)
        ax.errorbar(ad_binr[ad_indx], bin_ad[ad_indx], yerr=bin_ade[ad_indx],
                    color='k', capsize=0, linestyle='', linewidth=1, alpha=1.0, zorder=2)
    ax.plot(modelr, np.ma.sqrt(vrotm[0]**2 - vrotm[1]**2).filled(0.0), color='k', zorder=5, lw=1)
    if reff_lines is not None:
        for l in reff_lines:
            ax.axvline(x=l, linestyle='--', lw=0.5, zorder=1, color='k')

    ax.text(0.5, -0.13, r'$R$ [arcsec]', ha='center', va='center', transform=ax.transAxes,
            fontsize=10)

    ax.add_patch(patches.Rectangle((0.81,0.86), 0.17, 0.09, facecolor='w', lw=0,
                                    edgecolor='none', zorder=5, alpha=0.7,
                                    transform=ax.transAxes))
    ax.text(0.97, 0.861, r'$\sigma_{\rm maj}$ [km/s]', ha='right', va='bottom',
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


# TODO: Figure out what's causing:
#   - UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
def asymdrift_fit_maps(kin, disk, rstep, par=None, maj_wedge=30.):
    """
    Construct azimuthally averaged radial profiles of the kinematics.
    """
    # Check input
    if disk.par is None and par is None:
        raise ValueError('No model parameters available.  Provide directly or via disk argument.')

    if disk.ntracer != 2:
        raise NotImplementedError('Must provide two disks, first gas, second stars.')
    if len(kin) != 2:
        raise NotImplementedError('Must provide two kinematic datasets, first gas, second stars.')

    if kin[0].spatial_shape != kin[1].spatial_shape:
        raise NotImplementedError('Kinematic datasets must have the same spatial shape.')

    if not np.allclose(kin[0].grid_x, kin[1].grid_x) \
            or not np.allclose(kin[0].grid_y, kin[1].grid_y):
        raise NotImplementedError('Kinematics datasets must have the same grid sampling!')

    if not np.all(disk.tie_base[:4]):
        raise NotImplementedError('Disk must have tied the xc, yc, pa, and inc for both tracers.')

    # Set the parameters and confirm it has the correct size
    _par = disk.par[disk.untie] if par is None else par
    if _par.size != disk.np:
        raise ValueError('Number of provided parameters has the incorrect size.')

    # Get the maps of gas and stellar velocities
    gv_mask = np.logical_not(disk[0].vel_gpm) | kin[0].vel_mask
    gv_map = kin[0].remap('vel', mask=gv_mask)
    sv_mask = np.logical_not(disk[1].vel_gpm) | kin[1].vel_mask
    sv_map = kin[1].remap('vel', mask=sv_mask)
    sv_ivar_map = kin[1].remap('vel_ivar', mask=sv_mask)

    # Use masks to determine the spaxels/bins that both have valid gas and
    # stellar velocity data
    ad_mask_map = np.ma.getmaskarray(gv_map) | np.ma.getmaskarray(sv_map)
    sv_map[ad_mask_map] = np.ma.masked
    sv_ivar_map[ad_mask_map] = np.ma.masked
    # Get the mask for the binned data
    # NOTE: This approach should mean that the entire bin is masked if any
    # spaxel within it is masked.  This is conservative, but it saves me having
    # to account for individual masked gas spaxels within the stellar bins!
    ad_mask = kin[1].bin(ad_mask_map.astype(int)) > 0.

    # Bin the gas velocities identically to the stars and reconstruct the map
    # using the stellar binning
    gv = kin[1].bin_moments(kin[0].grid_sb, gv_map.filled(0.0), None)[1]
    gv[ad_mask] = 0.
    gv_map = kin[1].remap(gv, mask=ad_mask)

    # Use simple error propagation to get the errors
    _msk = np.logical_not(disk[0].vel_gpm) | kin[0].vel_mask
    gv_var_map = np.ma.power(kin[0].remap('vel_ivar', mask=_msk), -1)
    gv_ivar = np.ma.power(kin[1].bin_moments(kin[0].grid_sb, gv_var_map.filled(0.0), None)[1], -1)
    gv_ivar_map = kin[1].remap(gv_ivar.filled(0.0), mask=ad_mask)

    # Get the binned stellar velocities and update the mapped properties to use
    # the new mask
    sv = np.ma.MaskedArray(kin[1].vel, mask=ad_mask)
    sv_ivar = np.ma.MaskedArray(kin[1].vel_ivar, mask=ad_mask)
    sv_map = kin[1].remap(sv)
    sv_ivar_map = kin[1].remap(sv_ivar)

    # Stellar velocity dispersion (squared) data
    sd = np.ma.MaskedArray(kin[1].sig_phys2.copy())
    sd_mask = np.logical_not(disk[1].sig_gpm) | kin[1].sig_mask \
                | np.logical_not(kin[1].sig_phys2_ivar > 0)
    sd[sd_mask] = np.ma.masked
    sd_map = kin[1].remap(sd, mask=sd_mask)
    sd_ivar = np.ma.MaskedArray(kin[1].sig_phys2_ivar, mask=sd_mask)
    sd_ivar_map = kin[1].remap(sd_ivar, mask=sd_mask)

    # Create the model data
    #   - Gas kinematics
    slc = disk.disk_slice(0)
    disk[0].par = _par[slc]
    models = disk[0].model()
    intr_models = disk[0].model(ignore_beam=True)
    gv_mod = kin[1].bin_moments(kin[0].grid_sb,
                                models if disk[0].dc is None else models[0], None)[1]
    gv_mod_map = kin[1].remap(gv_mod, mask=ad_mask)
    gv_mod_intr = kin[1].bin_moments(kin[0].grid_sb,
                                     intr_models if disk[0].dc is None else intr_models[0], None)[1]
    gv_mod_intr_map = kin[1].remap(gv_mod_intr, mask=ad_mask)
    # - Stellar kinematics
    slc = disk.disk_slice(1)
    sv_mod, sd_mod = disk[1].binned_model(_par[slc])
    sv_mod_intr, sd_mod_intr = disk[1].binned_model(_par[slc], ignore_beam=True)
    sv_mod_map = kin[1].remap(sv_mod, mask=ad_mask)
    sv_mod_intr_map = kin[1].remap(sv_mod_intr, mask=ad_mask)
    if sd_mod is None:
        sd_mod = np.ones(sv_mod.shape, dtype=float)
        sd_mod_intr = np.ones(sv_mod.shape, dtype=float)
    sd_mod = sd_mod**2
    sd_mod_intr = sd_mod_intr**2
    sd_mod_map = kin[1].remap(sd_mod, mask=sd_mask)
    sd_mod_intr_map = kin[1].remap(sd_mod, mask=sd_mask)

    # Get the beam-smearing corrections
    gv_bc = gv_mod - gv_mod_intr
    sv_bc = sv_mod - sv_mod_intr
    sd_bc = sd_mod - sd_mod_intr

    # Get the coordinates of the bins in the disk plane
    r, th = projected_polar(kin[1].x - _par[0], kin[1].y - _par[1], *np.radians(_par[2:4]))

    # Get the AD data for each bin, both beam-corrected and not
    grot = np.ma.divide(gv - _par[4], np.cos(th))
    grot_bc = np.ma.divide(gv - gv_bc - _par[4], np.cos(th))
    grot_var = np.ma.power(np.cos(th)**2 * gv_ivar, -1)
    srot = np.ma.divide(sv - _par[disk[0].np+4], np.cos(th))
    srot_bc = np.ma.divide(sv - sv_bc - _par[disk[0].np+4], np.cos(th))
    srot_var = np.ma.power(np.cos(th)**2 * sv_ivar, -1)

    ad = grot**2 - srot**2
    ad_var = (2*grot)**2 * grot_var + (2*srot)**2 * srot_var
    ad_ivar = np.ma.power(ad_var, -1)

    ad_bc = grot_bc**2 - srot_bc**2
    ad_bc_var = (2*grot_bc)**2 * grot_var + (2*srot_bc)**2 * srot_var
    ad_bc_ivar = np.ma.power(ad_bc_var, -1)

    ados = np.ma.divide(ad, sd)
    ados_ivar = np.ma.divide(np.ma.power(np.ma.power(ad_ivar * ad**2,-1)
                              + np.ma.power(sd_ivar * sd**2, -1), -1), ados**2)
    ados_mask = sd_mask | ad_mask | np.ma.getmaskarray(ados_ivar)
    ados[ados_mask] = np.ma.masked
    ados_ivar[ados_mask] = np.ma.masked

    ados_bc = np.ma.divide(ad_bc, sd - sd_bc)
    ados_bc_ivar = np.ma.divide(np.ma.power(np.ma.power(ad_bc_ivar * ad_bc**2,-1)
                                + np.ma.power(sd_ivar * (sd-sd_bc)**2, -1), -1), ados_bc**2)
    ados_bc_mask = sd_mask | ad_mask | np.ma.getmaskarray(ados_bc_ivar)
    ados_bc[ados_bc_mask] = np.ma.masked
    ados_bc_ivar[ados_bc_mask] = np.ma.masked

    # Update the masking
    if np.any((np.ma.getmaskarray(ad) | np.ma.getmaskarray(ad_ivar)) & np.logical_not(ad_mask)):
        raise ValueError('check mask')
    if np.any((np.ma.getmaskarray(ad_bc) | np.ma.getmaskarray(ad_bc_ivar)) & np.logical_not(ad_mask)):
        raise ValueError('check bc mask')
#    ad_mask |= (np.ma.getmaskarray(ad) | np.ma.getmaskarray(ad_ivar))
#    ad[ad_mask] = np.ma.masked
#    ad_ivar[ad_mask] = np.ma.masked
#    gv[ad_mask] = np.ma.masked
#    gv_ivar[ad_mask] = np.ma.masked
#    sv[ad_mask] = np.ma.masked
#    sv_ivar[ad_mask] = np.ma.masked

    # Create the maps
    ad_map = kin[1].remap(ad.filled(0.0), mask=ad_mask)
    ad_ivar_map = kin[1].remap(ad_ivar.filled(0.0), mask=ad_mask)
    ad_mask_map = kin[1].remap(ad_mask.astype(int), mask=ad_mask).filled(1).astype(bool)
    ad_bc_map = kin[1].remap(ad_bc.filled(0.0), mask=ad_mask)
    ad_bc_ivar_map = kin[1].remap(ad_bc_ivar.filled(0.0), mask=ad_mask)
    ados_map = kin[1].remap(ados.filled(0.0), mask=ados_mask)
    ados_ivar_map = kin[1].remap(ados_ivar.filled(0.0), mask=ados_mask)
    ados_mask_map = kin[1].remap(ados_mask.astype(int), mask=ados_mask).filled(1).astype(bool)
    ados_bc_map = kin[1].remap(ados_bc.filled(0.0), mask=ados_mask)
    ados_bc_ivar_map = kin[1].remap(ados_bc_ivar.filled(0.0), mask=ados_mask)

    # Get the model AD data for each bin, both beam-corrected and not
    grot_mod = np.ma.divide(gv_mod - _par[4], np.cos(th))
    grot_mod_bc = np.ma.divide(gv_mod - gv_bc - _par[4], np.cos(th))
    srot_mod = np.ma.divide(sv_mod - _par[disk[0].np+4], np.cos(th))
    srot_mod_bc = np.ma.divide(sv_mod - sv_bc - _par[disk[0].np+4], np.cos(th))
    ad_mod = grot_mod**2 - srot_mod**2
    ad_mod_bc = grot_mod_bc**2 - srot_mod_bc**2
    ados_mod = np.ma.divide(ad_mod, sd_mod)
    ados_mod_bc = np.ma.divide(ad_mod_bc, sd_mod - sd_bc)

    # Create the model maps    
    ad_mod_map = kin[1].remap(ad_mod.filled(0.0), mask=ad_mask)
    ad_mod_bc_map = kin[1].remap(ad_mod_bc.filled(0.0), mask=ad_mask)
    ados_mod_map = kin[1].remap(ados_mod.filled(0.0), mask=ados_mask)
    ados_mod_bc_map = kin[1].remap(ados_mod_bc.filled(0.0), mask=ados_mask)

    # Mask data away from the major axes
    major_gpm = select_kinematic_axis(r, th, which='major', r_range='all', wedge=maj_wedge)
    ad_indx = major_gpm & np.logical_not(ad_mask)
    ados_indx = major_gpm & np.logical_not(ados_mask)

    # Set the radial bins
    # NOTE: ad hoc maximum radius is meant to mitigate effect of minor axis
    # points on number radial bins.  This will limit the number of off-axis
    # points included in galaxies with inclinations > 75 deg.
    maxr = min(4*np.amax(r[ad_indx]), np.amax(r)) if np.any(ad_indx) else np.amax(r)
    binr = np.array([rstep/2]) if maxr < rstep/2 else np.arange(rstep/2, maxr, rstep)
    #binr = np.arange(rstep/2, min(4*np.amax(r[ad_indx]), np.amax(r)), rstep)
    binw = np.full(binr.size, rstep, dtype=float)

    # Bin the data
    _, _, _, _, _, _, _, ad_ewmean, ad_ewsdev, _, _, ad_nbin, _ \
            = bin_stats(r[ad_indx], ad.data[ad_indx], binr, binw, wgts=ad_ivar.data[ad_indx],
                        fill_value=0.0) 
    _, _, _, _, _, _, _, ad_bc_ewmean, ad_bc_ewsdev, _, _, ad_bc_nbin, _ \
            = bin_stats(r[ad_indx], ad_bc.data[ad_indx], binr, binw, wgts=ad_bc_ivar.data[ad_indx],
                        fill_value=0.0) 
    _, _, _, _, _, _, _, ados_ewmean, ados_ewsdev, _, _, ados_nbin, _ \
            = bin_stats(r[ados_indx], ados.data[ados_indx], binr, binw,
                        wgts=ados_ivar.data[ados_indx], fill_value=0.0) 
    _, _, _, _, _, _, _, ados_bc_ewmean, ados_bc_ewsdev, _, _, ados_bc_nbin, _ \
            = bin_stats(r[ados_indx], ados_bc.data[ados_indx], binr, binw,
                        wgts=ados_bc_ivar.data[ados_indx], fill_value=0.0) 

    # Bin the model identically to the data
    _, _, _, _, _, _, _, ad_mod_ewmean, ad_mod_ewsdev, _, _, _, _ \
            = bin_stats(r[ad_indx], ad_mod.data[ad_indx], binr, binw, wgts=ad_ivar.data[ad_indx],
                        fill_value=0.0) 
    _, _, _, _, _, _, _, ad_mod_bc_ewmean, ad_mod_bc_ewsdev, _, _, _, _ \
            = bin_stats(r[ad_indx], ad_mod_bc.data[ad_indx], binr, binw, wgts=ad_bc_ivar.data[ad_indx],
                        fill_value=0.0) 
    _, _, _, _, _, _, _, ados_mod_ewmean, ados_mod_ewsdev, _, _, _, _ \
            = bin_stats(r[ados_indx], ados_mod.data[ados_indx], binr, binw,
                        wgts=ados_ivar.data[ados_indx], fill_value=0.0) 
    _, _, _, _, _, _, _, ados_mod_bc_ewmean, ados_mod_bc_ewsdev, _, _, _, _ \
            = bin_stats(r[ados_indx], ados_mod_bc.data[ados_indx], binr, binw,
                        wgts=ados_bc_ivar.data[ados_indx], fill_value=0.0) 

    # Return the data
    return gv_map, gv_ivar_map, gv_mod_map, gv_mod_intr_map, \
           sv_map, sv_ivar_map, sv_mod_map, sv_mod_intr_map, \
           sd_map, sd_ivar_map, sd_mod_map, sd_mod_intr_map, \
           ad_map, ad_ivar_map, ad_bc_map, ad_bc_ivar_map, ad_mod_map, ad_mod_bc_map, \
                ad_mask_map, \
           ados_map, ados_ivar_map, ados_bc_map, ados_bc_ivar_map, ados_mod_map, ados_mod_bc_map, \
                ados_mask_map, \
           r[ad_indx], ad[ad_indx].filled(0.0), ad_ivar[ad_indx].filled(0.0), \
                np.ma.getmaskarray(ad[ad_indx]), \
           binr, \
           ad_ewmean, ad_ewsdev, ad_mod_ewmean, ad_mod_ewsdev, ad_nbin, \
           ad_bc_ewmean, ad_bc_ewsdev, ad_mod_bc_ewmean, ad_mod_bc_ewsdev, ad_bc_nbin, \
           ados_ewmean, ados_ewsdev, ados_mod_ewmean, ados_mod_ewsdev, ados_nbin, \
           ados_bc_ewmean, ados_bc_ewsdev, ados_mod_bc_ewmean, ados_mod_bc_ewsdev, ados_bc_nbin


def asymdrift_radial_profile(disk, kin, rstep, maj_wedge=30.):
    """
    Construct azimuthally averaged radial profiles of the kinematics.
    """

    if disk.ntracer != 2:
        raise NotImplementedError('Must provide two disks, first gas, second stars.')
    if len(kin) != 2:
        raise NotImplementedError('Must provide two kinematic datasets, first gas, second stars.')

    if kin[0].spatial_shape != kin[1].spatial_shape:
        raise NotImplementedError('Kinematic datasets must have the same spatial shape.')

    if not np.allclose(kin[0].grid_x, kin[1].grid_x) \
            or not np.allclose(kin[0].grid_y, kin[1].grid_y):
        raise NotImplementedError('Kinematics datasets must have the same grid sampling!')

    if not np.all(disk.tie_base[:4]):
        raise NotImplementedError('Disk must have tied the xc, yc, pa, and inc for both tracers.')

    # Get the full list of parameters
    par = disk.par[disk.untie]

    # Determine the spaxels/bins that both have valid gas and stellar velocity
    # data
    gas_vel = kin[0].remap('vel', mask=np.logical_not(disk[0].vel_gpm) | kin[0].vel_mask)
    str_vel = kin[1].remap('vel', mask=np.logical_not(disk[1].vel_gpm) | kin[1].vel_mask)
    ad_mask_map = np.ma.getmaskarray(gas_vel) | np.ma.getmaskarray(str_vel)
    ad_mask = kin[1].bin(ad_mask_map.astype(int)) > 0.

    # Bin the gas data identical to the stars
    gas_sig = np.ma.sqrt(kin[0].remap('sig_phys2',
                                      mask=np.logical_not(disk[0].sig_gpm) | kin[0].sig_mask))
    _, _gas_vel, _gas_sig = kin[1].bin_moments(kin[0].grid_sb, gas_vel.filled(0.0),
                                               gas_sig.filled(0.0))
    # Use bin_moments to get the error in the velocities
    gas_vel_var = np.ma.power(kin[0].remap('vel_ivar',
                                           mask=np.logical_not(disk[0].vel_gpm) | kin[0].vel_mask),
                              -1)
    _gas_vel_ivar = np.ma.power(kin[1].bin_moments(kin[0].grid_sb, gas_vel_var.filled(0.0), None)[1],
                                -1)

    # Get the coordinates of the bins in the disk plane
    r, th = projected_polar(kin[1].x - par[0], kin[1].y - par[1], *np.radians(par[2:4]))

    # Mask for data along the major and minor axes
    major_gpm = select_kinematic_axis(r, th, which='major', r_range='all', wedge=maj_wedge)

    # Set the radial bins
    binr = np.arange(rstep/2, np.amax(r), rstep)
    binw = np.full(binr.size, rstep, dtype=float)

    # Construct the radial profile using the binned data
    indx = major_gpm & np.logical_not(ad_mask)

    ad_r = r[indx]
    gas_vrot = (_gas_vel[indx] - par[4])/np.cos(th[indx])
    gas_vrot_ivar = _gas_vel_ivar.data[indx]*np.cos(th[indx])**2

    str_vrot = (kin[1].vel[indx] - par[disk[0].np+4])/np.cos(th[indx])
    str_vrot_ivar = kin[1].vel_ivar[indx]*np.cos(th[indx])**2

    ad = gas_vrot**2 - str_vrot**2
    ad_wgt = 1 / 2 / (gas_vrot**2/gas_vrot_ivar + str_vrot**2/str_vrot_ivar)
#    sini = np.sin(np.radians(par[3]))
#    ad /= sini
#    ad_wgt *= sini**2
    _, _, _, _, _, _, _, ad_ewmean, ad_ewsdev, _, _, ad_nbin, ad_bin_gpm \
            = bin_stats(ad_r, ad, binr, binw, wgts=ad_wgt, fill_value=0.0) 

    return ad_r, ad, binr, ad_ewmean, ad_ewsdev, ad_nbin


def _rej_iters(rej):
    _rej = None if rej is None else list(rej)
    if _rej is not None and len(_rej) == 1:
        _rej *= 4
    if _rej is not None and len(_rej) != 4:
        raise ValueError('Must provide 1 or 4 sigma rejection levels.')
    return _rej


def asymdrift_iter_fit(galmeta, gas_kin, str_kin, gas_disk, str_disk, gas_vel_mask=None,
                       gas_sig_mask=None, str_vel_mask=None, str_sig_mask=None, ignore_covar=True,
                       assume_posdef_covar=True, gas_vel_sigma_rej=[15,10,10,10],
                       gas_sig_sigma_rej=[15,10,10,10], str_vel_sigma_rej=[15,10,10,10],
                       str_sig_sigma_rej=[15,10,10,10], fix_cen=False, fix_inc=False, low_inc=None,
                       min_unmasked=None, analytic_jac=True, fit_scatter=True, verbose=0):
    r"""
    Iteratively fit a two-component disk to measure asymmetric drift.

    The constraints and iterations closely mirror the approach used by
    :func:`~nirvana.models.axisym.axisym_iter_fit`.

    The initial guess parameters are set using the provided ``gas_disk`` and
    ``str_disk`` objects.  If the ``par`` attributes of either of these objects
    are None, the guess parameters are set by
    :func:`~nirvana.models.thindisk.ThinDisk.guess_par` function of the derived
    class (e.g., :func:`~nirvana.models.axisym.AxisymmetricDisk.guess_par`).
    The initial guess for the geometric parameters are simply the mean of the
    two disk objects, except that the guess for the inclination is always set to
    the photometric inclination.

    Constraints are as follows:

        #. The center is constrained to be in the middle third of the available
           range in x and y.

        #. The center, position angle, and inclination are forced to be the same
           for both disks.  The systemic velocities are, however, allowed to be
           different.

    The iterations are as follows:

        #. Fit all data but fix the inclination to the value returned by
           :func:`~nirvana.data.meta.GlobalPar.guess_inclination` and fix the
           center to the initial guess value.  The initial guess will either be
           :math:`(x,y) = (0,0)` or the mean of the centers provided by the
           ``gas_disk`` and ``str_disk`` arguments.  If available, covariance is
           ignored.

        #. Reject outliers for all 4 kinematic measurements (gas v, gas sigma,
           stellar v, stellar sigma) using
           :func:`~nirvana.models.thindisk.ThinDisk.reject`.  The rejection
           sigma used is the *first* element in the provided lists.  Then refit
           the data, starting again from the initial guess parameters.  The
           intrinsic scatter estimates provided by
           :func:`~nirvana.models.thindisk.ThinDisk.reject` are
           *not* included in the fit and, if available, covariance is ignored.

        #. Reject outliers for all 4 kinematic measurements (gas v, gas sigma,
           stellar v, stellar sigma) using
           :func:`~nirvana.models.thindisk.ThinDisk.reject`.  The rejection
           sigma used is the *second* element in the provided lists.  Then refit
           the data using the parameters from the previous fit as the starting
           point. This iteration also uses the intrinsic scatter estimates
           provided by :func:`~nirvana.models.thindisk.ThinDisk.reject`;
           however, covariance is still ignored.

        #. Recover all fit rejections (i.e., keep any masks in place that are
           tied to the data quality, but remove any masks associated with fit
           quality).  Then use :func:`~nirvana.models.thindisk.ThinDisk.reject`
           to perform a fresh rejection based on the most recent model; the
           rejection sigma is the
           *second* element in the provided lists.  The resetting of the
           fit-outliers and re-rejection is done on the off chance that
           rejections from the first few iterations were driven by a bad model.
           Refit the data as in the previous iteration, using the parameters
           from the previous fit as the starting point and use the intrinsic
           scatter estimates provided by
           :func:`~nirvana.models.thindisk.ThinDisk.reject`.  Covariance is
           still ignored.

        #. Reject outliers for all 4 kinematic measurements (gas v, gas sigma,
           stellar v, stellar sigma) using
           :func:`~nirvana.models.thindisk.ThinDisk.reject`.  The rejection
           sigma used is the *third* element in the provided lists.  Then refit
           the data, but fix or free the center and inclination based on the
           provided keywords (``fix_cen`` and ``fix_inc``).  Also, as in all
           previous iterations, the covariance is ignored in the outlier
           rejection and intrinsic scatter determination; however, the
           covariance *is* used by the fit, as available and if ``ignore_covar``
           is False.

        #. Redo the previous iteration in exactly the same way, except outlier
           rejection and intrinsic-scatter determination now use the covariance,
           as available and if ``ignore_covar`` is False.  The rejection sigma
           used is the *fourth* element in the provided lists.

        #. If a lower inclination threshold is set (see ``low_inc``) and the
           best-fitting inclination is below this value (assuming the
           inclination is freely fit), a final iteration refits the data by
           fixing the inclination at the value set by
           :func:`~nirvana.data.meta.GlobalPar.guess_inclination`.  The code
           issues a warning, and the global fit-quality bit is set to include
           the ``LOWINC`` flag.
        
    Args:
        galmeta (:class:`~nirvana.data.meta.GlobalPar`):
            Object with metadata for the galaxy to be fit.
        gas_kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the gas data to be fit
        str_kin (:class:`~nirvana.data.kinematics.Kinematics`):
            Object with the stellar data to be fit
        gas_disk (:class:`~nirvana.models.thindisk.ThinDisk`):
            Thin disk object used to model and set the initial guess parameters
            for the gas disk (see above).
        str_disk (:class:`~nirvana.models.thindisk.ThinDisk`):
            Thin disk object used to model and set the initial guess parameters
            for the stellar disk (see above).
        gas_vel_mask (`numpy.ndarray`_, optional):
            Initial array with the mask bits for gas velocities.  If None,
            initialization uses
            :func:`~nirvana.data.kinematics.Kinematics.init_fitting_masks` for
            the gas data.
        gas_sig_mask (`numpy.ndarray`_, optional):
            Initial array with the mask bits for gas velocity dispersions.  If
            None, initialization uses
            :func:`~nirvana.data.kinematics.Kinematics.init_fitting_masks` for
            the gas data.
        str_vel_mask (`numpy.ndarray`_, optional):
            Initial array with the mask bits for stellar velocities.  If None,
            initialization uses
            :func:`~nirvana.data.kinematics.Kinematics.init_fitting_masks` for
            the stellar data.
        gas_sig_mask (`numpy.ndarray`_, optional):
            Initial array with the mask bits for stellar velocity dispersions.
            If None, initialization uses
            :func:`~nirvana.data.kinematics.Kinematics.init_fitting_masks` for
            the stellar data.
        ignore_covar (:obj:`bool`, optional):
            If ``kin`` provides the covariance between measurements, ignore it
            and fit the data assuming there is no covariance.
        assume_posdef_covar (:obj:`bool`, optional):
            If ``kin`` provides the covariance between measurements, assume the
            covariance matrices are positive definite.
        gas_vel_sigma_rej (:obj:`float`, :obj:`list`, optional):
            Sigma values used for rejection of the gas velocity measurements.
            Must be a single float or a *four-element* list.  If None, no
            rejections are performed.  The description above provides which
            value is used in each iteration.
        gas_sig_sigma_rej (:obj:`float`, :obj:`list`, optional):
            Sigma values used for rejection of gas dispersion measurements; cf.
            ``gas_vel_sigma_rej``.
        str_vel_sigma_rej (:obj:`float`, :obj:`list`, optional):
            Sigma values used for rejection of the stellar velocity
            measurements; cf. ``gas_vel_sigma_rej``.
        str_sig_sigma_rej (:obj:`float`, :obj:`list`, optional):
            Sigma values used for rejection of stellar dispersion measurements;
            cf. ``gas_vel_sigma_rej``.
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
            proceed with the fit, after applying all masking.  This is applied
            independently to both the gas and stellar data.
        analytic_jac (:obj:`bool`, optional):
            Use the analytic calculation of the Jacobian matrix during the fit
            optimization.  If False, the Jacobian is calculated using
            finite-differencing methods provided by
            `scipy.optimize.least_squares`_.
        fit_scatter (:obj:`bool`, optional):
            Model the intrinsic scatter in the data about the model during the
            fit optimization.  The scatter is modeled independently for all 4
            kinematic measurements (gas velocity and dispersion, stellar
            velocity and dispersion).
        verbose (:obj:`int`, optional):
            Verbosity level: 0=only status output written to terminal; 1=show 
            fit result QA plot; 2=full output

    Returns:
        :obj:`tuple`: Returns 9 objects: (1) the
        :class:`~nirvana.models.multitrace.MultiTracerDisk` instance used during
        the fit, (2) a `numpy.ndarray`_ with the input guess parameters, (3,4)
        `numpy.ndarray`_ objects with the lower and upper bounds imposed on the
        best-fit parameters, (5) a boolean `numpy.ndarray`_ selecting the
        parameters that were fixed during the fit, (6,7) `numpy.ndarray`_
        objects with the bad-pixel masks for the gas velocity and dispersion
        measurements used in the fit, and (8,9) `numpy.ndarray`_ objects with
        the bad-pixel masks for the stellar velocity and dispersion measurements
        used in the fit. 
    """
    # Running in "debug" mode
    debug = verbose > 1

    # Check input
    _gas_vel_sigma_rej = _rej_iters(gas_vel_sigma_rej)
    _gas_sig_sigma_rej = _rej_iters(gas_sig_sigma_rej)
    _str_vel_sigma_rej = _rej_iters(str_vel_sigma_rej)
    _str_sig_sigma_rej = _rej_iters(str_sig_sigma_rej)

    #---------------------------------------------------------------------------
    # Initialize the fitting object and set the guess parameters
    disk = MultiTracerDisk([gas_disk, str_disk])
    if gas_disk.par is None:
        gas_disk.par = gas_disk.guess_par()
    if str_disk.par is None:
        str_disk.par = str_disk.guess_par()
    p0 = np.append(gas_disk.par, str_disk.par)
    p0[:disk.nbp] = (gas_disk.par[:disk.nbp] + str_disk.par[:disk.nbp])/2.
    p0[gas_disk.np:gas_disk.np+disk.nbp] = p0[:disk.nbp]
    # Force the inclination to be the photometric inclination
    p0[3] = p0[gas_disk.np+3] = galmeta.guess_inclination(lb=1., ub=89.)

    #---------------------------------------------------------------------------
    # Define the fitting object
    # Constrain the center to be in the middle third of the map relative to the
    # photometric center. The mean in the calculation is to mitigate that some
    # galaxies can be off center, but the detail here and how well it works
    # hasn't been well tested.
    # TODO: Should this use grid_x instead, so that it's more uniform for all
    # IFUs?  Or should this be set as a fraction of Reff?
    _x = np.append(gas_kin.x, str_kin.x)
    _y = np.append(gas_kin.y, str_kin.y)
    dx = np.mean([abs(np.amin(_x)), abs(np.amax(_x))])
    dy = np.mean([abs(np.amin(_y)), abs(np.amax(_y))])
    lb, ub = disk.par_bounds(base_lb=np.array([-dx/3, -dy/3, -350., 1., -500.]),
                             base_ub=np.array([dx/3, dy/3, 350., 89., 500.]))
    log.info(f'If free, center constrained within +/- {dx/3:.1f} in X and +/- {dy/3:.1f} in Y.')

    # TODO: Handle these issues instead of faulting
    if np.any(np.less(p0, lb)):
        raise ValueError('Parameter lower bounds cannot accommodate initial guess value!')
    if np.any(np.greater(p0, ub)):
        raise ValueError('Parameter upper bounds cannot accommodate initial guess value!')

    #---------------------------------------------------------------------------
    # Setup the masks
    log.info('Initializing data masking')
    if gas_vel_mask is None or gas_sig_mask is None:
        _gas_vel_mask, _gas_sig_mask = gas_kin.init_fitting_masks(bitmask=disk.mbm, verbose=True)
    else:
        _gas_vel_mask = gas_vel_mask.copy()
        _gas_sig_mask = gas_sig_mask.copy()
    # Make sure there are sufficient data to fit!
    if min_unmasked is None:
        if np.all(_gas_vel_mask > 0):
            raise ValueError('All gas velocity measurements masked!')
        if _gas_sig_mask is not None and np.all(_gas_sig_mask > 0):
            raise ValueError('All gas velocity dispersion measurements masked!')
    else:
        if np.sum(np.logical_not(_gas_vel_mask > 0)) < min_unmasked:
            raise ValueError('Insufficient valid gas velocity measurements to continue!')
        if _gas_sig_mask is not None and np.sum(np.logical_not(_gas_sig_mask > 0)) < min_unmasked:
            raise ValueError('Insufficient valid gas dispersion measurements to continue!')

    if str_vel_mask is None or str_sig_mask is None:
        _str_vel_mask, _str_sig_mask = str_kin.init_fitting_masks(bitmask=disk.mbm, verbose=True)
    else:
        _str_vel_mask = str_vel_mask.copy()
        _str_sig_mask = str_sig_mask.copy()
    # Make sure there are sufficient data to fit!
    if min_unmasked is None:
        if np.all(_str_vel_mask > 0):
            raise ValueError('All stellar velocity measurements masked!')
        if _str_sig_mask is not None and np.all(_str_sig_mask > 0):
            raise ValueError('All stellar dispersion measurements masked!')
    else:
        if np.sum(np.logical_not(_str_vel_mask > 0)) < min_unmasked:
            raise ValueError('Insufficient valid stellar velocity measurements to continue!')
        if _str_sig_mask is not None and np.sum(np.logical_not(_str_sig_mask > 0)) < min_unmasked:
            raise ValueError('Insufficient valid stellar dispersion measurements to continue!')

    #---------------------------------------------------------------------------
    # Perform the fit iterations
    #---------------------------------------------------------------------------
    # Tie all the geometric projection parameters, but leave the systemic
    # velocities to be independent for each dataset.
    disk.update_tie_base([True, True, True, True, False])
    # Fit iteration 1: Fit all data but fix the inclination and center
    #                x0    y0    pa     inc   vsys
    fix = np.append([True, True, False, True, False], np.zeros(p0.size-5, dtype=bool))
    log.info('Running fit iteration 1')
    # TODO: sb_wgt is always true throughout. Make this a command-line
    # parameter?
    disk.lsq_fit([gas_kin, str_kin], sb_wgt=True, p0=p0, fix=fix, lb=lb, ub=ub,
                 ignore_covar=True, assume_posdef_covar=assume_posdef_covar,
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        asymdrift_fit_plot(galmeta, [gas_kin, str_kin], disk, fix=fix) 

    #---------------------------------------------------------------------------
    # Fit iteration 2:
    #   - Reject very large outliers. This is aimed at finding data that is
    #     so descrepant from the model that it's reasonable to expect the
    #     measurements are bogus.
    log.info('Running rejection iterations')
    gas_vel_rej, gas_vel_sig, gas_sig_rej, gas_sig_sig \
            = disk.disk[0].reject(vel_sigma_rej=_gas_vel_sigma_rej[0], show_vel=debug,
                                  sig_sigma_rej=_gas_sig_sigma_rej[0], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(gas_vel_rej):
        log.info(f'{np.sum(gas_vel_rej)} gas velocity measurements rejected as unreliable.')
        gas_vel_mask[gas_vel_rej] = disk.mbm.turn_on(gas_vel_mask[gas_vel_rej], 'REJ_UNR')
    if gas_sig_rej is not None and np.any(gas_sig_rej):
        log.info(f'{np.sum(gas_sig_rej)} gas dispersion measurements rejected as unreliable.')
        gas_sig_mask[gas_sig_rej] = disk.mbm.turn_on(gas_sig_mask[gas_sig_rej], 'REJ_UNR')
    gas_kin.reject(vel_rej=gas_vel_rej, sig_rej=gas_sig_rej)

    str_vel_rej, str_vel_sig, str_sig_rej, str_sig_sig \
            = disk.disk[1].reject(vel_sigma_rej=_str_vel_sigma_rej[0], show_vel=debug,
                                  sig_sigma_rej=_str_sig_sigma_rej[0], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(str_vel_rej):
        log.info(f'{np.sum(str_vel_rej)} stellar velocity measurements rejected as unreliable.')
        str_vel_mask[str_vel_rej] = disk.mbm.turn_on(str_vel_mask[str_vel_rej], 'REJ_UNR')
    if str_sig_rej is not None and np.any(str_sig_rej):
        log.info(f'{np.sum(str_sig_rej)} stellar dispersion measurements rejected as unreliable.')
        str_sig_mask[str_sig_rej] = disk.mbm.turn_on(str_sig_mask[str_sig_rej], 'REJ_UNR')
    str_kin.reject(vel_rej=str_vel_rej, sig_rej=str_sig_rej)
    #   - Refit, again with the inclination and center fixed. However, do not
    #     use the parameters from the previous fit as the starting point, and
    #     ignore the estimated intrinsic scatter.
    log.info('Running fit iteration 2')
    disk.lsq_fit([gas_kin, str_kin], sb_wgt=True, p0=p0, fix=fix, lb=lb, ub=ub,
                 ignore_covar=True, assume_posdef_covar=assume_posdef_covar,
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        asymdrift_fit_plot(galmeta, [gas_kin, str_kin], disk, fix=fix) 

    #---------------------------------------------------------------------------
    # Fit iteration 3: 
    #   - Perform a more restricted rejection
    log.info('Running rejection iterations')
    gas_vel_rej, gas_vel_sig, gas_sig_rej, gas_sig_sig \
            = disk.disk[0].reject(vel_sigma_rej=_gas_vel_sigma_rej[1], show_vel=debug,
                                  sig_sigma_rej=_gas_sig_sigma_rej[1], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(gas_vel_rej):
        log.info(f'{np.sum(gas_vel_rej)} gas velocity measurements rejected as unreliable.')
        gas_vel_mask[gas_vel_rej] = disk.mbm.turn_on(gas_vel_mask[gas_vel_rej], 'REJ_RESID')
    if gas_sig_rej is not None and np.any(gas_sig_rej):
        log.info(f'{np.sum(gas_sig_rej)} gas dispersion measurements rejected as unreliable.')
        gas_sig_mask[gas_sig_rej] = disk.mbm.turn_on(gas_sig_mask[gas_sig_rej], 'REJ_RESID')
    gas_kin.reject(vel_rej=gas_vel_rej, sig_rej=gas_sig_rej)

    str_vel_rej, str_vel_sig, str_sig_rej, str_sig_sig \
            = disk.disk[1].reject(vel_sigma_rej=_str_vel_sigma_rej[1], show_vel=debug,
                                  sig_sigma_rej=_str_sig_sigma_rej[1], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(str_vel_rej):
        log.info(f'{np.sum(str_vel_rej)} stellar velocity measurements rejected as unreliable.')
        str_vel_mask[str_vel_rej] = disk.mbm.turn_on(str_vel_mask[str_vel_rej], 'REJ_RESID')
    if str_sig_rej is not None and np.any(str_sig_rej):
        log.info(f'{np.sum(str_sig_rej)} stellar dispersion measurements rejected as unreliable.')
        str_sig_mask[str_sig_rej] = disk.mbm.turn_on(str_sig_mask[str_sig_rej], 'REJ_RESID')
    str_kin.reject(vel_rej=str_vel_rej, sig_rej=str_sig_rej)
    #   - Refit again with the inclination and center fixed, but use the
    #     previous fit as the starting point and include the estimated
    #     intrinsic scatter.
    log.info('Running fit iteration 3')
    scatter = np.array([gas_vel_sig, gas_sig_sig, str_vel_sig, str_sig_sig]) \
                if fit_scatter else None
    disk.lsq_fit([gas_kin, str_kin], sb_wgt=True, p0=disk.par[disk.untie], fix=fix, lb=lb, ub=ub,
                 ignore_covar=True, assume_posdef_covar=assume_posdef_covar, scatter=scatter,
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        asymdrift_fit_plot(galmeta, [gas_kin, str_kin], disk, fix=fix) 

    #---------------------------------------------------------------------------
    # Fit iteration 4: 
    #   - Recover data from the restricted rejection
    disk.mbm.reset_to_base_flags(gas_kin, gas_vel_mask, gas_sig_mask)
    disk.mbm.reset_to_base_flags(str_kin, str_vel_mask, str_sig_mask)
    #   - Reject again based on the new fit parameters
    log.info('Running rejection iterations')
    gas_vel_rej, gas_vel_sig, gas_sig_rej, gas_sig_sig \
            = disk.disk[0].reject(vel_sigma_rej=_gas_vel_sigma_rej[1], show_vel=debug,
                                  sig_sigma_rej=_gas_sig_sigma_rej[1], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(gas_vel_rej):
        log.info(f'{np.sum(gas_vel_rej)} gas velocity measurements rejected as unreliable.')
        gas_vel_mask[gas_vel_rej] = disk.mbm.turn_on(gas_vel_mask[gas_vel_rej], 'REJ_RESID')
    if gas_sig_rej is not None and np.any(gas_sig_rej):
        log.info(f'{np.sum(gas_sig_rej)} gas dispersion measurements rejected as unreliable.')
        gas_sig_mask[gas_sig_rej] = disk.mbm.turn_on(gas_sig_mask[gas_sig_rej], 'REJ_RESID')
    gas_kin.reject(vel_rej=gas_vel_rej, sig_rej=gas_sig_rej)

    str_vel_rej, str_vel_sig, str_sig_rej, str_sig_sig \
            = disk.disk[1].reject(vel_sigma_rej=_str_vel_sigma_rej[1], show_vel=debug,
                                  sig_sigma_rej=_str_sig_sigma_rej[1], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(str_vel_rej):
        log.info(f'{np.sum(str_vel_rej)} stellar velocity measurements rejected as unreliable.')
        str_vel_mask[str_vel_rej] = disk.mbm.turn_on(str_vel_mask[str_vel_rej], 'REJ_RESID')
    if str_sig_rej is not None and np.any(str_sig_rej):
        log.info(f'{np.sum(str_sig_rej)} stellar dispersion measurements rejected as unreliable.')
        str_sig_mask[str_sig_rej] = disk.mbm.turn_on(str_sig_mask[str_sig_rej], 'REJ_RESID')
    str_kin.reject(vel_rej=str_vel_rej, sig_rej=str_sig_rej)
    #   - Refit again with the inclination and center fixed, but use the
    #     previous fit as the starting point and include the estimated
    #     intrinsic scatter.
    log.info('Running fit iteration 4')
    scatter = np.array([gas_vel_sig, gas_sig_sig, str_vel_sig, str_sig_sig]) \
                if fit_scatter else None
    disk.lsq_fit([gas_kin, str_kin], sb_wgt=True, p0=disk.par[disk.untie], fix=fix, lb=lb, ub=ub,
                 ignore_covar=True, assume_posdef_covar=assume_posdef_covar, scatter=scatter,
                 analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        asymdrift_fit_plot(galmeta, [gas_kin, str_kin], disk, fix=fix) 

    #---------------------------------------------------------------------------
    # Fit iteration 5: 
    #   - Recover data from the restricted rejection
    disk.mbm.reset_to_base_flags(gas_kin, gas_vel_mask, gas_sig_mask)
    disk.mbm.reset_to_base_flags(str_kin, str_vel_mask, str_sig_mask)
    #   - Reject again based on the new fit parameters
    log.info('Running rejection iterations')
    gas_vel_rej, gas_vel_sig, gas_sig_rej, gas_sig_sig \
            = disk.disk[0].reject(vel_sigma_rej=_gas_vel_sigma_rej[2], show_vel=debug,
                                  sig_sigma_rej=_gas_sig_sigma_rej[2], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(gas_vel_rej):
        log.info(f'{np.sum(gas_vel_rej)} gas velocity measurements rejected as unreliable.')
        gas_vel_mask[gas_vel_rej] = disk.mbm.turn_on(gas_vel_mask[gas_vel_rej], 'REJ_RESID')
    if gas_sig_rej is not None and np.any(gas_sig_rej):
        log.info(f'{np.sum(gas_sig_rej)} gas dispersion measurements rejected as unreliable.')
        gas_sig_mask[gas_sig_rej] = disk.mbm.turn_on(gas_sig_mask[gas_sig_rej], 'REJ_RESID')
    gas_kin.reject(vel_rej=gas_vel_rej, sig_rej=gas_sig_rej)

    str_vel_rej, str_vel_sig, str_sig_rej, str_sig_sig \
            = disk.disk[1].reject(vel_sigma_rej=_str_vel_sigma_rej[2], show_vel=debug,
                                  sig_sigma_rej=_str_sig_sigma_rej[2], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(str_vel_rej):
        log.info(f'{np.sum(str_vel_rej)} stellar velocity measurements rejected as unreliable.')
        str_vel_mask[str_vel_rej] = disk.mbm.turn_on(str_vel_mask[str_vel_rej], 'REJ_RESID')
    if str_sig_rej is not None and np.any(str_sig_rej):
        log.info(f'{np.sum(str_sig_rej)} stellar dispersion measurements rejected as unreliable.')
        str_sig_mask[str_sig_rej] = disk.mbm.turn_on(str_sig_mask[str_sig_rej], 'REJ_RESID')
    str_kin.reject(vel_rej=str_vel_rej, sig_rej=str_sig_rej)
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
    log.info('Running fit iteration 5')
    scatter = np.array([gas_vel_sig, gas_sig_sig, str_vel_sig, str_sig_sig]) \
                if fit_scatter else None
    disk.lsq_fit([gas_kin, str_kin], sb_wgt=True, p0=disk.par[disk.untie], fix=fix, lb=lb, ub=ub,
                 ignore_covar=ignore_covar, assume_posdef_covar=assume_posdef_covar,
                 scatter=scatter, analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        asymdrift_fit_plot(galmeta, [gas_kin, str_kin], disk, fix=fix) 

    #---------------------------------------------------------------------------
    # Fit iteration 6:
    #   - Recover data from the restricted rejection
    disk.mbm.reset_to_base_flags(gas_kin, gas_vel_mask, gas_sig_mask)
    disk.mbm.reset_to_base_flags(str_kin, str_vel_mask, str_sig_mask)
    #   - Reject again based on the new fit parameters
    log.info('Running rejection iterations')
    gas_vel_rej, gas_vel_sig, gas_sig_rej, gas_sig_sig \
            = disk.disk[0].reject(vel_sigma_rej=_gas_vel_sigma_rej[3], show_vel=debug,
                                  sig_sigma_rej=_gas_sig_sigma_rej[3], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(gas_vel_rej):
        log.info(f'{np.sum(gas_vel_rej)} gas velocity measurements rejected as unreliable.')
        gas_vel_mask[gas_vel_rej] = disk.mbm.turn_on(gas_vel_mask[gas_vel_rej], 'REJ_RESID')
    if gas_sig_rej is not None and np.any(gas_sig_rej):
        log.info(f'{np.sum(gas_sig_rej)} gas dispersion measurements rejected as unreliable.')
        gas_sig_mask[gas_sig_rej] = disk.mbm.turn_on(gas_sig_mask[gas_sig_rej], 'REJ_RESID')
    gas_kin.reject(vel_rej=gas_vel_rej, sig_rej=gas_sig_rej)

    str_vel_rej, str_vel_sig, str_sig_rej, str_sig_sig \
            = disk.disk[1].reject(vel_sigma_rej=_str_vel_sigma_rej[3], show_vel=debug,
                                  sig_sigma_rej=_str_sig_sigma_rej[3], show_sig=debug,
                                  verbose=verbose > 1)
    if np.any(str_vel_rej):
        log.info(f'{np.sum(str_vel_rej)} stellar velocity measurements rejected as unreliable.')
        str_vel_mask[str_vel_rej] = disk.mbm.turn_on(str_vel_mask[str_vel_rej], 'REJ_RESID')
    if str_sig_rej is not None and np.any(str_sig_rej):
        log.info(f'{np.sum(str_sig_rej)} stellar dispersion measurements rejected as unreliable.')
        str_sig_mask[str_sig_rej] = disk.mbm.turn_on(str_sig_mask[str_sig_rej], 'REJ_RESID')
    str_kin.reject(vel_rej=str_vel_rej, sig_rej=str_sig_rej)
    #   - Redo previous fit
    log.info('Running fit iteration 6')
    scatter = np.array([gas_vel_sig, gas_sig_sig, str_vel_sig, str_sig_sig]) \
                if fit_scatter else None
    disk.lsq_fit([gas_kin, str_kin], sb_wgt=True, p0=disk.par[disk.untie], fix=fix, lb=lb, ub=ub,
                 ignore_covar=ignore_covar, assume_posdef_covar=assume_posdef_covar,
                 scatter=scatter, analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        asymdrift_fit_plot(galmeta, [gas_kin, str_kin], disk, fix=fix) 

    if fix_inc or low_inc is None or disk.par[3] > low_inc:
        # Inclination is valid, so return
        return disk, p0, lb, ub, fix, gas_vel_mask, gas_sig_mask, str_vel_mask, str_sig_mask

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
    # NOTE: This assumes the inclination is tied!!
    disk.par[3] = galmeta.guess_inclination(lb=1., ub=89.)
    warnings.warn(f'Best-fitting inclination is below {low_inc:.1f} degrees.  Running a final '
                  f'fit fixing the inclination to {disk.par[3]:.1f}')
    log.info('Running fit iteration 7')
    disk.lsq_fit([gas_kin, str_kin], sb_wgt=True, p0=disk.par[disk.untie], fix=fix, lb=lb, ub=ub,
                 ignore_covar=ignore_covar, assume_posdef_covar=assume_posdef_covar,
                 scatter=scatter, analytic_jac=analytic_jac, verbose=verbose)
    # Show
    if verbose > 0:
        asymdrift_fit_plot(galmeta, [gas_kin, str_kin], disk, fix=fix) 

    return disk, p0, lb, ub, fix, gas_vel_mask, gas_sig_mask, str_vel_mask, str_sig_mask


# TODO:
#   - This is MaNGA-specific and needs to be abstracted
#   - Copy over the DataTable class from the DAP, or use an astropy.table.Table?
def _ad_meta_dtype(nr):
    """
    """
    return [('MANGAID', '<U30'),
            ('PLATEIFU', '<U12'),
            # Azimuthally binned radial profiles
            ('BINR', float, (nr,)),
            ('AD', float, (nr,)),
            ('AD_SDEV', float, (nr,)),
            ('AD_MOD', float, (nr,)),
            ('AD_MOD_SDEV', float, (nr,)),
            ('AD_NUSE', int, (nr,)),
            ('AD_BC', float, (nr,)),
            ('AD_BC_SDEV', float, (nr,)),
            ('AD_BC_MOD', float, (nr,)),
            ('AD_BC_MOD_SDEV', float, (nr,)),
            ('AD_BC_NUSE', int, (nr,)),
            ('ADOS', float, (nr,)),
            ('ADOS_SDEV', float, (nr,)),
            ('ADOS_MOD', float, (nr,)),
            ('ADOS_MOD_SDEV', float, (nr,)),
            ('ADOS_NUSE', int, (nr,)),
            ('ADOS_BC', float, (nr,)),
            ('ADOS_BC_SDEV', float, (nr,)),
            ('ADOS_BC_MOD', float, (nr,)),
            ('ADOS_BC_MOD_SDEV', float, (nr,)),
            ('ADOS_BC_NUSE', int, (nr,))
           ]


def asymdrift_fit_data(galmeta, kin, disk, p0, lb, ub, gas_vel_mask, gas_sig_mask,
                       str_vel_mask, str_sig_mask, ofile=None):

    # Create the output data file
    # - Ensure the best-fitting parameters have been distributed to the disks
    disk.distribute_par()
    # - Propagate the global fit assessments/flags to each tracer-specific disk
    # object
    disk[0].global_mask = disk.global_mask
    disk[0].fit_status = disk.fit_status
    disk[0].fit_success = disk.fit_success
    disk[1].global_mask = disk.global_mask
    disk[1].fit_status = disk.fit_status
    disk[1].fit_success = disk.fit_success
    # - Get the output data for the gas
    gas_slice = disk.disk_slice(0)
    gas_hdu = axisym.axisym_fit_data(galmeta, kin[0], p0[gas_slice], lb[gas_slice], ub[gas_slice],
                                     disk[0], gas_vel_mask, gas_sig_mask)
    # - Get the output data for the stars
    str_slice = disk.disk_slice(1)
    str_hdu = axisym.axisym_fit_data(galmeta, kin[1], p0[str_slice], lb[str_slice], ub[str_slice],
                                     disk[1], str_vel_mask, str_sig_mask)
    # Get the asymmetric drift data
    fwhm = galmeta.psf_fwhm[1]
    oversample = 1.5
    rstep = fwhm/oversample
    maj_wedge = 30.

    gv_map, gv_ivar_map, gv_mod_map, gv_mod_intr_map, \
        sv_map, sv_ivar_map, sv_mod_map, sv_mod_intr_map, \
        sd_map, sd_ivar_map, sd_mod_map, sd_mod_intr_map, \
        ad_map, ad_ivar_map, ad_bc_map, ad_bc_ivar_map, ad_mod_map, ad_mod_bc_map, ad_mask_map, \
        ados_map, ados_ivar_map, ados_bc_map, ados_bc_ivar_map, ados_mod_map, ados_mod_bc_map, \
            ados_mask_map, \
        ad_spx_r, ad_spx, ad_spx_ivar, ad_spx_mask, \
        binr, \
        ad_ewmean, ad_ewsdev, ad_mod_ewmean, ad_mod_ewsdev, ad_nbin, \
        ad_bc_ewmean, ad_bc_ewsdev, ad_mod_bc_ewmean, ad_mod_bc_ewsdev, ad_bc_nbin, \
        ados_ewmean, ados_ewsdev, ados_mod_ewmean, ados_mod_ewsdev, ados_nbin, \
        ados_bc_ewmean, ados_bc_ewsdev, ados_mod_bc_ewmean, ados_mod_bc_ewsdev, ados_bc_nbin \
            = asymdrift_fit_maps(kin, disk, rstep, maj_wedge=maj_wedge)

    adprof = fileio.init_record_array(1, _ad_meta_dtype(binr.size))
    adprof['MANGAID'] = galmeta.mangaid
    adprof['PLATEIFU'] = galmeta.plateifu
    adprof['BINR'] = binr
    adprof['AD'] = ad_ewmean
    adprof['AD_SDEV'] = ad_ewsdev
    adprof['AD_MOD'] = ad_mod_ewmean
    adprof['AD_MOD_SDEV'] = ad_mod_ewsdev
    adprof['AD_NUSE'] = ad_nbin
    adprof['AD_BC'] = ad_bc_ewmean
    adprof['AD_BC_SDEV'] = ad_bc_ewsdev
    adprof['AD_BC_MOD'] = ad_mod_bc_ewmean
    adprof['AD_BC_MOD_SDEV'] = ad_mod_bc_ewsdev
    adprof['AD_BC_NUSE'] = ad_bc_nbin
    adprof['ADOS'] = ados_ewmean
    adprof['ADOS_SDEV'] = ados_ewsdev
    adprof['ADOS_MOD'] = ados_mod_ewmean
    adprof['ADOS_MOD_SDEV'] = ados_mod_ewsdev
    adprof['ADOS_NUSE'] = ados_nbin
    adprof['ADOS_BC'] = ados_bc_ewmean
    adprof['ADOS_BC_SDEV'] = ados_bc_ewsdev
    adprof['ADOS_BC_MOD'] = ados_mod_bc_ewmean
    adprof['ADOS_BC_MOD_SDEV'] = ados_mod_bc_ewsdev
    adprof['ADOS_BC_NUSE'] = ados_bc_nbin

    # - Combine the data into a single fits file
    prihdr = gas_hdu[0].header.copy()
    prihdr.remove('MODELTYP')
    prihdr.remove('RCMODEL')
    prihdr.remove('DCMODEL')
    prihdr['GMODTYP'] = gas_hdu[0].header['MODELTYP']
    prihdr['GRCMOD'] = gas_hdu[0].header['RCMODEL']
    if 'DCMODEL' in gas_hdu[0].header:
        prihdr['GDCMOD'] = gas_hdu[0].header['DCMODEL']
    prihdr['SMODTYP'] = str_hdu[0].header['MODELTYP']
    prihdr['SRCMOD'] = str_hdu[0].header['RCMODEL']
    if 'DCMODEL' in str_hdu[0].header:
        prihdr['SDCMOD'] = str_hdu[0].header['DCMODEL']
    prihdr['QUAL'] = disk.global_mask
    resid = disk.fom(disk.par)
    prihdr['CHI2'] = (np.sum(resid**2), 'Total chi-square')
    prihdr['RCHI2'] = (prihdr['CHI2']/(resid.size - disk.nfree), 'Total reduced chi-square')
    prihdr['ADWEDGE'] = (maj_wedge, 'Major axis wedge for AD')
    maphdr = fileio.add_wcs(prihdr, kin[0])
    mapmaskhdr = maphdr.copy()
    disk.mbm.to_header(mapmaskhdr)
    for h in gas_hdu[1:]:
        h.name = 'GAS_'+h.name
    for h in str_hdu[1:]:
        h.name = 'STR_'+h.name

    hdus = [fits.PrimaryHDU(header=prihdr)] + gas_hdu[1:] + str_hdu[1:] \
            + [fits.ImageHDU(data=gv_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'BIN_GAS_VEL', bunit='km/s',
                                                           err=True),
                             name='BIN_GAS_VEL'),
               fits.ImageHDU(data=gv_ivar_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'BIN_GAS_VEL',
                                                           bunit='(km/s)^{-2}', hduclas2='ERROR'),
                             name='BIN_GAS_VEL_IVAR'),
               fits.ImageHDU(data=gv_mod_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'BIN_GAS_VEL_MOD',
                                                           bunit='km/s'),
                             name='BIN_GAS_VEL_MOD'),
               fits.ImageHDU(data=gv_mod_intr_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'BIN_GAS_VEL_MODI',
                                                           bunit='km/s'),
                             name='BIN_GAS_VEL_MODI'),
               fits.ImageHDU(data=ad_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'AD', bunit='(km/s)^2',
                                                           err=True, qual=True),
                             name='AD'),
               fits.ImageHDU(data=ad_ivar_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'AD', bunit='(km/s)^{-4}',
                                                           hduclas2='ERROR', qual=True),
                             name='AD_IVAR'),
               fits.ImageHDU(data=ad_mask_map.astype(np.int16),
                             header=fileio.finalize_header(mapmaskhdr, 'AD', hduclas2='QUALITY',
                                                           err=True, bit_type=bool),
                             name='AD_MASK'),
               fits.ImageHDU(data=ad_bc_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'AD_BC', bunit='(km/s)^2',
                                                           err=True),
                             name='AD_BC'),
               fits.ImageHDU(data=ad_bc_ivar_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'AD_BC', bunit='(km/s)^{-4}',
                                                           hduclas2='ERROR'),
                             name='AD_BC_IVAR'),
               fits.ImageHDU(data=ad_mod_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'AD_MOD', bunit='(km/s)^2'),
                             name='AD_MOD'),
               fits.ImageHDU(data=ad_mod_bc_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'AD_MODI', bunit='(km/s)^2'),
                             name='AD_MODI'),
               fits.ImageHDU(data=ados_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'ADOS', err=True, qual=True),
                             name='ADOS'),
               fits.ImageHDU(data=ados_ivar_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'ADOS', hduclas2='ERROR',
                                                           qual=True),
                             name='ADOS_IVAR'),
               fits.ImageHDU(data=ados_mask_map.astype(np.int16),
                             header=fileio.finalize_header(mapmaskhdr, 'ADOS', hduclas2='QUALITY',
                                                           err=True, bit_type=bool),
                             name='ADOS_MASK'),
               fits.ImageHDU(data=ados_bc_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'ADOS_BC', err=True),
                             name='ADOS_BC'),
               fits.ImageHDU(data=ados_bc_ivar_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'ADOS_BC', hduclas2='ERROR'),
                             name='ADOS_BC_IVAR'),
               fits.ImageHDU(data=ados_mod_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'ADOS_MOD'),
                             name='ADOS_MOD'),
               fits.ImageHDU(data=ados_mod_bc_map.filled(0.0),
                             header=fileio.finalize_header(maphdr, 'ADOS_MODI'),
                             name='ADOS_MODI'),
               fits.BinTableHDU.from_columns([fits.Column(name=n,
                                                          format=fileio.rec_to_fits_type(adprof[n]),
                                                          dim=fileio.rec_to_fits_col_dim(adprof[n]),
                                                          array=adprof[n])
                                                for n in adprof.dtype.names],
                                              name='ADPROF')]

    # Construct the HDUList, write it if requested, and return
    hdu = fits.HDUList(hdus)
    if ofile is not None:
        if ofile.split('.')[-1] == 'gz':
            _ofile = ofile[:ofile.rfind('.')]
            compress = True
        else:
            _ofile = ofile
        hdu.writeto(_ofile, overwrite=True, checksum=True)
        if compress:
            fileio.compress_file(_ofile, overwrite=True, rm_original=True)
    return hdu


