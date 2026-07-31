import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from gaussn import lensingmodels
try:
    from scipy.optimize import minimize
except ImportError:
    pass
try:
    import dynesty
except ImportError:
    pass
try:
    import emcee
except ImportError:
    pass
try:
    import zeus
except ImportError:
    pass

jax.config.update('jax_enable_x64', True)

class GP:
    """
    A Gaussian Process class for modeling and regression, particularly suited for use in astronomy. Specific functionality is included for fitting time-series data in multiple bands and fitting lensed supernova time-series data.
    
    Attributes:
        kernel: The kernel function defining the covariance between data points.
        meanfunc: The mean function defining the expected value of the process.
        lensingmodel: The lensing model applied to the time and flux axes before
            evaluating the mean/covariance. Defaults to lensingmodels.NoLensing(),
            which leaves the data unchanged (i.e. an ordinary, non-lensed GP).
 
    Methods:
        __init__(self, kernel, meanfunc, lensingmodel=None): Initialize the GP with a kernel and mean function.
        _prepare_indices(self, x, band, image): Prepare indices for multi-band/multi-image data.
        _rescale_data(self, y, yerr): Rescale the y data and their errors.
        _logprior(self, params): Default uninformative (flat) prior for MCMC sampling methods.
        _loglikelihood(self, x, y, yerr, kernel_params, meanfunc_params, lensing_params): Default log likelihood of the GP model.
        jointprobability(self, params, logprior=None, fix_kernel_params=False, fix_mean_params=False, fix_lensing_params=False, invert=1): Compute the joint probability (likelihood + prior) of kernel, mean function, and lensing parameters.
        optimize_parameters(self, x, y, yerr, band=None, image=None, method='minimize', loglikelihood=None, logprior=None, ptform=None, fix_kernel_params=False, fix_mean_params=False, fix_lensing_params=False, minimize_kwargs=None, sampler_kwargs=None, run_sampler_kwargs=None, host_dust_kwargs=None, lens_dust_kwargs=None): Optimize GP parameters using different methods (minimize, emcee, zeus, dynesty).
        predict(self, x_prime, x, y, yerr, band): Predict function values at new locations given observed data.
        
    Example:
        # Create an instance of the GP class
        kernel = kernels.KernelName(kernel_params)
        meanfunc = meanfuncs.MeanFuncName(meanfunc_params)
        gp_model = GP(kernel, meanfunc)
 
        # Optimize the parameters using the minimize method
        gp_model.optimize_parameters(x_train, y_train, yerr_train, method='minimize')
 
        # Make predictions using the optimized model
        x_new = ...
        y_pred, y_pred_variance = gp_model.predict(x_new, x_train, y_train, yerr_train)
    """
    
    def __init__(self, kernel, meanfunc, lensingmodel=None):
        """
        Initialize with a kernel and mean function.
        
        Parameters:
            kernel: Kernel function defining covariance.
            meanfunc: Mean function defining expected value.
            lensingmodel: Model for lensing effects (optional). If omitted,
                defaults to lensingmodels.NoLensing(), so the GP behaves as a
                standard (non-lensed) Gaussian Process.
        """
        self.kernel = kernel
        self.meanfunc = meanfunc
        if lensingmodel != None:
            self.lensingmodel = lensingmodel
        else:
            self.lensingmodel = lensingmodels.NoLensing()
        
    def _prepare_indices(self, x, band, image):
        """
        Precompute bookkeeping needed for multi-band/multi-image data.
 
        Builds `self.indices`, a cumulative-count array marking where each
        band/image group starts and ends within the flattened data arrays
        (x, y, yerr). Group order follows np.unique(band) and, within each
        band, np.unique(image). These boundaries are used elsewhere (e.g. by
        the lensing model) to know which rows of x/y/yerr belong to which
        band/image combination.
 
        Also sets `self.n_bands`, `self.n_images`, and `self.factor` (the
        constant term len(x) * log(2*pi) used in the multivariate normal
        log-likelihood, precomputed once here for efficiency).
 
        Parameters:
            x: array-like of observation times/locations.
            band: array-like of band labels for each observation, or None
                if there is only a single band.
            image: array-like of image labels for each observation (used for
                lensed multi-image data), or None if there is only one image.
        """
        self.n_bands = len(np.unique(band))
        self.n_images = len(np.unique(image))
        
        # Walk through each band (and, within it, each image) in sorted
        # order and accumulate a running total of how many points belong to
        # each group. `indices` ends up as the group boundaries, e.g.
        # [0, n0, n0+n1, ...], so that data for group k lives in
        # x[indices[k]:indices[k+1]].
        indices = [0]
        if band is not None:
            for j, pb_id in enumerate(np.unique(band)):
                specified_band = band[band == pb_id]
                if image is not None:
                    for i, im_id in enumerate(np.unique(image)):
                        specified_image = specified_band[image[band == pb_id] == im_id]
                        indices.append(len(specified_image) + indices[-1])
                else:
                    indices.append(len(specified_band) + indices[-1])
        else:
            if image is not None:
                for i, im_id in enumerate(np.unique(image)):
                    specified_image = image[image == im_id]
                    indices.append(len(specified_image) + indices[-1])
            else:
                indices.append(len(x) + indices[-1])
        
        self.indices = jnp.array(indices)
        self.factor = (len(x) * jnp.log(2 * jnp.pi))

    def _get_initial_pos(self, fix_kernel_params, fix_mean_params, fix_lensing_params):
        """
        Assemble the initial-parameter vector used to start optimization/sampling.
 
        Free (non-fixed) parameter groups are concatenated in a fixed order:
        kernel parameters first, then mean-function parameters, then lensing-
        model parameters. Any group whose corresponding `fix_*_params` flag
        is True is left out of the vector entirely (and therefore held fixed
        at its current value during optimization/sampling). This same
        ordering convention is relied upon in `jointprobability` when
        unpacking `params` back into the individual parameter groups.
 
        Parameters:
            fix_kernel_params: bool, if True the kernel parameters are excluded.
            fix_mean_params: bool, if True the mean-function parameters are excluded.
            fix_lensing_params: bool, if True the lensing-model parameters are excluded.
 
        Returns:
            list: concatenated initial values for all free parameters.
 
        Raises:
            Exception: if every parameter group is fixed, leaving nothing to fit.
        """
        init_pos = []
        if not fix_kernel_params:
            init_pos.extend(self.kernel.params)
        if not fix_mean_params:
            init_pos.extend(self.meanfunc.params)
        if not fix_lensing_params:
            init_pos.extend(self.lensingmodel.params)
            
        if len(init_pos) < 0.5:
                raise Exception("No parameters to fit. Check fix_mean_params and fix_kernel_params to make sure there are parameters to fit.")

        return init_pos
    
    def _rescale_data(self, y, yerr):
        """
        Rescale y and yerr by dividing through by the range of y (max - min).
 
        This does not shift/center the data, only rescales it, so the result
        is not guaranteed to lie within [0, 1] unless y already starts at 0 -
        it simply compresses the *span* of y (and yerr, proportionally) to 1
        unit. This can help numerical stability of the optimizer when the
        raw flux values are very large or very small.
 
        Parameters:
            y: array-like of observed values.
            yerr: array-like of measurement uncertainties on y.
 
        Returns:
            tuple: (y_rescaled, yerr_rescaled)
        """
        factor = jnp.max(y) - jnp.min(y)
        y_rescaled = y/factor
        yerr_rescaled = yerr/factor
        return y_rescaled, yerr_rescaled

    def _split_params(self, params, fix_kernel_params, fix_mean_params, fix_lensing_params):
        """
        Slice the flat parameter vector `params` into the (kernel, meanfunc,
        lensing) sub-vectors, in that fixed order, skipping over any groups
        that are held fixed. `params` therefore only ever contains the *free*
        parameters -- this matches `_get_initial_pos`.
 
        This replaces the old 8-way if/elif chain in `jointprobability` with
        equivalent logic that reads the same either as plain Python (used by
        the interactive `jointprobability` above) or when traced by jax.jit
        (used by `_build_jitted_jointprobability` below) -- all the branching
        here is on the *static* fix_* booleans, never on `params` itself, so
        it resolves at trace time and doesn't require lax.cond.
        """
        idx = 0
        kernel_params = meanfunc_params = lensing_params = None
        if not fix_kernel_params:
            n = len(self.kernel.params)
            kernel_params = [params[idx + i] for i in range(n)]
            idx += n
        if not fix_mean_params:
            n = len(self.meanfunc.params)
            meanfunc_params = [params[idx + i] for i in range(n)]
            idx += n
        if not fix_lensing_params:
            n = len(self.lensingmodel.params)
            lensing_params = [params[idx + i] for i in range(n)]
            idx += n
        return kernel_params, meanfunc_params, lensing_params
    
    def _logprior(self, params):
        """
        Default prior: flat/uninformative over all parameter values.
 
        Always returns 0 (i.e. log(1)), so it has no effect on the posterior
        and sampling is driven purely by the likelihood. Users who need
        bounded or informative priors should pass their own `logprior`
        function to `optimize_parameters`.
        """
        return 0
    
    def _loglikelihood(self, x, y, yerr, kernel_params, meanfunc_params, lensing_params):
        """
        Compute the log-likelihood of the data under a multivariate normal
        (Gaussian Process) model, including the lensing model's effect on
        the time axis and flux normalization.
 
        Note: this method relies on `self.bands`, `self.zp`, and `self.zpsys`
        being set beforehand (this happens in `optimize_parameters`), since
        they are needed by `self.meanfunc.mean` but are not passed in as
        explicit arguments.
 
        Parameters:
            x: array-like, observation times/locations.
            y: array-like, observed flux values.
            yerr: array-like, measurement uncertainties on y.
            kernel_params: parameters for the covariance kernel.
            meanfunc_params: parameters for the mean function.
            lensing_params: parameters for the lensing model (time delays,
                magnifications, etc.).
 
        Returns:
            float: the log-likelihood of (y, yerr) given the model.
        """
        # Apply the lensing model: shifted_x are the observation times
        # de-lensed (e.g. corrected for time delays) into a common frame,
        # and b_vector holds the per-point magnification factors.
        shifted_x, b_vector = self.lensingmodel.lens(x, params=lensing_params)

        # Mean vector, scaled by the magnification of each point.
        mean = b_vector * self.meanfunc.mean(shifted_x, params=meanfunc_params, bands=self.bands, zp=self.zp, zpsys=self.zpsys)
        
        # Covariance matrix: start from the kernel's covariance evaluated in
        # the de-lensed time frame, rescale each entry by the outer product
        # of magnifications (since flux, not the underlying signal, is what
        # is magnified), zero out entries between unrelated images/bands via
        # `self.lensingmodel.mask`, and add measurement noise on the diagonal.
        K = jnp.outer(b_vector, b_vector) * self.kernel.covariance(shifted_x, params=kernel_params)
        cov = jnp.multiply(self.lensingmodel.mask, K) + jnp.diag(yerr**2)
        
        # Cholesky decomposition of the covariance matrix. Used both to get
        # the log-determinant efficiently (2 * sum(log(diag(L)))) and to
        # solve the quadratic form below without explicitly inverting cov.
        L = jnp.linalg.cholesky(cov)
        a = self.factor + ( 2 * jnp.sum(jnp.log(jnp.diag(L))) )
        
        # z = L^-1 (mean - y), so that z^T z = (mean-y)^T cov^-1 (mean-y),
        # the quadratic (Mahalanobis-distance) term in the MVN log-likelihood.
        z = solve_triangular(L, mean - y, lower=True)
        b = z.T @ z
        
        # Standard multivariate normal log-likelihood:
        # -0.5 * (N*log(2*pi) + log|cov| + (y-mean)^T cov^-1 (y-mean))
        loglike = -0.5*(a + b)
        
        return loglike
        
    def jointprobability(self, logprior = None, fix_kernel_params = False, fix_mean_params = False, fix_lensing_params=False):

        def core(params, invert):
            kernel_params, meanfunc_params, lensing_params = self._split_params(params, fix_kernel_params, fix_mean_params, fix_lensing_params)
            log_prior = logprior(params)
 
            def compute(_):
                loglike = self.loglikelihood(self.x, self.y, self.yerr, kernel_params, meanfunc_params, lensing_params)
                total = loglike + log_prior
                bad = jnp.isnan(total) | jnp.isinf(total)
                return jnp.where(bad, -jnp.inf, total)
 
            def skip(_):
                # Mirrors the original behaviour of short-circuiting out of
                # an expensive cholesky/solve when the prior already rejects
                # this point -- jax.lax.cond only executes the taken branch,
                # so this is a real compute saving, not just cosmetic.
                return -jnp.inf
 
            prior_bad = jnp.isnan(log_prior) | jnp.isinf(log_prior)
            result = jax.lax.cond(prior_bad, skip, compute, operand=None)
            return invert * result
 
        return jax.jit(core)
    
    def optimize_parameters(self, x, y, yerr, band=None, image=None, zp=27.5, zpsys='ab', method='minimize', loglikelihood=None, logprior=None, ptform=None, fix_kernel_params=False, fix_mean_params=False, fix_lensing_params=False, init_scale=1., p0=None, minimize_kwargs=None, sampler_kwargs=None, run_sampler_kwargs=None, rescale_data=False):
        """
        Optimize the parameters of the Gaussian Process (GP) for a set of observations.
 
        :param x: array-like
            Input data points (independent variable).
 
        :param y: array-like
            Observed values corresponding to the input data points.
 
        :param yerr: array-like
            Measurement uncertainties of the observed values.
            
        :param band: array-like, optional (default=None)
            Band information for multi-band data.
            
        :param image: array-like, optional (default=None)
            Image information for multi-image data.
 
        :param zp: float, optional (default=27.5)
            Zeropoint used when evaluating the mean function (e.g. to convert
            between flux and magnitude systems).
 
        :param zpsys: str, optional (default='ab')
            Magnitude system corresponding to `zp` (e.g. 'ab').
            
        :param method: str, optional (default='minimize')
            The method for optimizing parameters. Available options: 'minimize' (scipy.optimize.minimize BFGS), 'emcee' (ensemble MCMC sampler), 'zeus' (ensemble slice sampler), and 'dynesty' (nested sampling).
            
        :param loglikelihood: function, optional (default=None)
            The log-likelihood function for the GP model. If not specified, the log of a multivariate normal distribution PDF will be used.
            
        :param logprior: function, optional (default=None)
            The log-prior function for the emcee and zeus MCMC sampling methods. If not specified, no prior (uniform over all values) will be enforced.
            
        :param ptform: function, optional (default=None)
            Function to transform the prior for the dynesty nested sampling process.
            
        :param fix_mean_params: bool, optional (default=False)
            Whether to fix the parameters of the mean function during optimization/sampling.
            
        :param fix_kernel_params: bool, optional (default=False)
            Whether to fix the parameters of the kernel function during optimization/sampling.
            
        :param fix_lensing_params: bool, optional (default=False)
            Whether to fix the parameters of the lensing model during optimization/sampling.
 
        :param init_scale: int, float, or array-like, optional (default=1)
            If using emcee or zeus, the scatter introduced around the initial parameter positions to use when initializing the chains. This parameter should either be a single number (e.g., init_scale = 1.) or a list with the scale values for each parameter being fit (e.g. init_scale = [1., 1., 1.] when fitting with three free parameters). Defaults to 1 for all parameters.
            
        :param minimize_kwargs: dict, optional (default=None)
            Additional keyword arguments for the scipy.optimize.minimize function.
            
        :param sampler_kwargs: dict, optional (default=None)
            Additional keyword arguments for the emcee, zeus, or dynesty sampler.
            Also used to pop off sampler-setup options handled specially here:
            'nlive' and 'sample' for dynesty, 'nwalkers' for emcee/zeus.
            
        :param run_sampler_kwargs: dict, optional (default=None)
            Additional keyword arguments for running the emcee, zeus, or dynesty sampler.
            Also used to pop off 'nsteps' for emcee/zeus.
 
        :param rescale_data: bool, optional (default=False)
            If True, rescales y and yerr via `_rescale_data` before fitting
            (dividing through by the range of y).
            
        :return results:
            If using the 'minimize' method, returns the scipy.optimize style results.
            
        :return sampler:
            If using 'emcee', 'zeus', or 'dynesty' methods, returns the sampler in the style of the chosen optimization method.
        """

        # Handle optional arguments
        if sampler_kwargs is None:
            sampler_kwargs = {}
        if run_sampler_kwargs is None:
            run_sampler_kwargs = {}
        if minimize_kwargs is None:
            minimize_kwargs = {}
            
        # Convert data to jax arrays for faster computation, and store them
        # on self since jointprobability/_loglikelihood read self.x/self.y/self.yerr
        # rather than taking the data as arguments.
        self.x = jnp.array(x)
        if rescale_data:
            self.y, self.yerr = self._rescale_data(jnp.array(y), jnp.array(yerr))
        else:
            self.y, self.yerr = jnp.array(y), jnp.array(yerr)
        self.bands = band
        self.zp = zp
        self.zpsys = zpsys
        
        # Store n_bands, n_images, and indices information, and hand the
        # band/image bookkeeping to the lensing model so it can build its
        # masking/time-delay logic. 
        self._prepare_indices(self.x, band, image)
        self.lensingmodel.import_from_gp(self.n_bands, self.n_images, self.indices)

        # Determine the number of free dimensions being optimized/sampled,
        # based on which parameter groups are not fixed.
        self.ndim = 0
        if not fix_kernel_params:
            self.ndim += len(self.kernel.params)
        if not fix_mean_params:
            self.ndim += len(self.meanfunc.params)
        if not fix_lensing_params:
            self.ndim += len(self.lensingmodel.params)
        
        # Use the default multivariate-normal log-likelihood (JIT-compiled
        # for speed) unless the user supplied their own. Same for the prior.
        if loglikelihood == None:
            self.loglikelihood = jax.jit(self._loglikelihood)
        else:
            self.loglikelihood = loglikelihood
        if logprior == None:
            self.logprior = self._logprior
        else:
            self.logprior = logprior

        jit_logprob = self.jointprobability(self.logprior, fix_kernel_params, fix_mean_params, fix_lensing_params)
        
        if method == 'dynesty':
                
            nlive = sampler_kwargs.pop('nlive', 500)
            sample = sampler_kwargs.pop('sample', 'rslice')

            def dynesty_loglike(params):
                return float(jit_logprob(jnp.asarray(params), 1.0))
           
            sampler = dynesty.NestedSampler(dynesty_loglike, ptform, self.ndim, nlive = nlive, sample = sample, **sampler_kwargs)
            
            sampler.run_nested(**run_sampler_kwargs)
            return sampler
        
        if method == 'emcee' or method == 'zeus' or method == 'minimize':
        
            # Get vector of initial parameters, which is required for optmizing/sampling with the minimize, emcee, and zeus methods
            init_pos = self._get_initial_pos(fix_kernel_params, fix_mean_params, fix_lensing_params)

            if type(init_scale) == list and len(init_pos) != len(init_scale):
                raise ValueError("The length of the initial parameter positions does not match the length of the list with the initial parameter scatter. The init_scale parameter should either be a single number (e.g., init_scale = 1.) or a list with the scale values for each parameter being fit (e.g. init_scale = [1., 1., 1.] when fitting with three free parameters)")

            if method == 'minimize':
                # Since the whole likelihood is JAX code, we get exact
                # gradients for free via autodiff instead of scipy falling
                # back to (slow, noisy) finite differences. jac=True tells
                # scipy.optimize.minimize that the objective returns
                # (value, gradient) together.
                grad_fn = jax.jit(jax.grad(lambda p: jit_logprob(p, -1.0)))
 
                def objective(params):
                    params = jnp.asarray(params)
                    value = jit_logprob(params, -1.0)
                    grad = grad_fn(params)
                    return float(value), np.asarray(grad)
 
                minimize_kwargs.setdefault('jac', True)
                results = minimize(objective, init_pos, **minimize_kwargs)
                return results

            if np.isinf(np.any(self.logprior(init_pos))):
                raise Exception("When passed to the specified ``log_prior'' function, some or all of the parameters that the kernel and mean function were initialized with yield an indefinite value. Please check that the initial parameters used are within the bounds of the prior, as the MCMC chains are initialized, with some scatter, around these values.")

            nwalkers = sampler_kwargs.pop('nwalkers', 24)
            nsteps = run_sampler_kwargs.pop('nsteps', 1000)

            if p0 is None:
                # Initialize walkers with random initial positions around the initial guess
                p0 = np.random.normal(init_pos, init_scale, size=(nwalkers, self.ndim))
                for r, row in enumerate(p0):
                    while np.isinf(self.logprior(row)):
                        p0[r] = np.random.normal(init_pos, 0.001)

            def sampler_loglike(params):
                return float(jit_logprob(jnp.asarray(params), 1.0))
            
            if method == 'emcee':
                sampler = emcee.EnsembleSampler(nwalkers, self.ndim, sampler_loglike, **sampler_kwargs)

            if method == 'zeus':
                sampler = zeus.EnsembleSampler(nwalkers, self.ndim, sampler_loglike, **sampler_kwargs)

            # Run the sampler
            sampler.run_mcmc(p0, nsteps=nsteps, **run_sampler_kwargs)
            return sampler
    
    def predict(self, x_prime, x, y, yerr, band=None):
        """
        Predict function values at new locations given observed data.
        
        This method calculates the expectation and variance of the predicted function values at specified new locations (x_prime) based on observed data (x, y) with associated measurement uncertainties (yerr). It assumes observations are made in a single band.
        
        :param x_prime: array-like
            Desired x locations for prediction.
        
        :param x: array-like
            Observed data points (independent variable).
        
        :param y: array-like
            Observed values corresponding to the input data points.
            
        :param yerr: array-like
            Measurement uncertainties of the observed values.
            
        :param band: array-like
            Band information for the observed data.
            
        :return expectation: array-like
            Predicted function values at x_prime.
            
        :return variance: array-like
            Variance of the predicted function values at x_prime.
        """
        
        cov_UV = self.kernel.covariance(x_prime, x_prime=x)

        K_VV = self.kernel.covariance(x)
        cov_VV = K_VV + np.diagflat(yerr**2)

        cov_UU = self.kernel.covariance(x_prime)

        mu_U = self.meanfunc.mean(x_prime, bands=band)
        mu_V = self.meanfunc.mean(x, bands=band)
        
        expectation = mu_U + (cov_UV @ np.linalg.solve(cov_VV, y-mu_V))
        variance = cov_UU - (cov_UV @ np.linalg.solve(cov_VV, np.transpose(cov_UV)))

        return expectation, variance
    
    
    
