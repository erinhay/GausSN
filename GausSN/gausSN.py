import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import jax
try:
    from scipy.optimize import minimize
except:
    pass
try:
    import dynesty
except:
    pass
try:
    import emcee
except:
    pass
try:
    import zeus
except:
    pass

jax.config.update('jax_enable_x64', True)

class GP:
    """
    A Gaussian Process class for modeling and regression, particularly suited for use in astronomy. Specific functionality is included for fitting time-series data in multiple bands and fitting lensed supernova time-series data.
    
    Attributes:
        kernel: The kernel function defining the covariance between data points.
        meanfunc: The mean function defining the expected value of the process.

    Methods:
        __init__(self, kernel, meanfunc): Initialize the GP with a kernel and mean function.
        
        _get_initial_guess(self, fix_mean_params, fix_kernel_params): 
        
        logprior(self, params): Default uninformative prior for MCMC sampling methods.
        
        loglikelihood(self, x, y, yerr, log_prior, magnification_matrix=1): Compute the log likelihood of the GP model.
        
        jointprobability(self, params, logprior=None, fix_mean_params=False, fix_kernel_params=False, invert=False): Compute the joint probability of kernel and mean function parameters.
        
        optimize_parameters(self, x, y, yerr, n_bands=1, method='minimize', loglikelihood=None, logprior=None, ptform=None, fix_mean_params=False, fix_kernel_params=False, minimize_kwargs=None, sampler_kwargs=None, run_sampler_kwargs=None): Optimize GP parameters using different methods (minimize, emcee, zeus, dynesty).
        
        predict(self, x_prime, x, y, yerr): Predict function values at new locations given observed data.
        
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
    
    def __init__(self, kernel, meanfunc):
        """
        Initialize with a kernel and mean function.
        """
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.jit_loglikelihood = jax.jit(self.loglikelihood)
        
    def _prepare_indices(self, x, band, image):
        # Store n_bands/images information
        self.n_bands = len(np.unique(band))
        self.n_images = len(np.unique(image))
        
        # Store indices information
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
        
    def _get_initial_guess(self, fix_mean_params, fix_kernel_params):
        """
        Put together the vector (init_guess) of parameters which the mean function and kernel are initialized with at the starting location for the optimization/sampling process. The parameters of the kernel are stacked first, followed by the mean function parameters. For MCMC sampling, there will be some scatter enforced around the initial vector values. To set the scale of this scatter, an init_guess_scale vector is also compiled.
        """
        init_guess = []
        init_guess_scale = []
        if not fix_kernel_params:
            init_guess.extend(self.kernel.params)
            init_guess_scale.extend(self.kernel.scale)
        if not fix_mean_params:
            init_guess.extend(self.meanfunc.params)
            init_guess_scale.extend(self.meanfunc.scale)
            
        if len(init_guess) < 0.5:
                raise Exception("No parameters to fit. Check fix_mean_params and fix_kernel_params to make sure there are parameters to fit.")

        return init_guess, init_guess_scale

    def _rescale_data(self, y, yerr):
        """
        Rescale the y data and their errors so it spans only 1 unit.
        """
        factor = jnp.max(y) - jnp.min(y)
        y_rescaled = y/factor
        yerr_rescaled = yerr/factor
        return y_rescaled, yerr_rescaled
    
    def logprior(self, params):
        """
        Default uniformative prior.
        """
        return 0
    
    def loglikelihood(self, x, y, yerr, kernel_params, meanfunc_params):
        """
        Compute the log likelihood of a multivariate normal PDF.
        """
        # Compute the mean vector for the given input data points x
        self.mean = self.meanfunc.mean(x, params=meanfunc_params)
        
        # Compute the covariance matrix K for the given input data points x
        # and modify the covariance matrix to include magnification effects (if applicable) and measurement uncertainties
        self.cov = self.kernel.covariance(x, params=kernel_params) + jnp.diag(yerr**2)
        
        # Compute the logarithm of the determinant of the covariance matrix
        L = jnp.linalg.cholesky(self.cov)
        a = (len(x) * jnp.log(2 * jnp.pi)) + ( 2 * jnp.sum(jnp.log(jnp.diag(L))) )
        
        # Compute the term in the exponential of the PDF of a MVN PDF
        z = solve_triangular(L, self.mean - y, lower=True)
        b = z.T @ z
        
        # Compute the log likelihood of a MVN PDF
        loglike = -0.5*(a + b)
        
        return loglike
        
    def jointprobability(self, params, logprior = None, fix_mean_params = False, fix_kernel_params = False, invert=1):
        """
        Compute the joint probability of the kernel and mean function parameters (if applicable).
        """

        # Compute the log prior for the given parameters
        log_prior = logprior(params)
        if jnp.isinf(log_prior) or jnp.isnan(log_prior):
            return invert * -jnp.inf
        
        # Reset the kernel and/or mean function parameters
        kernel_params = None
        if not fix_kernel_params:
            kernel_params = [params[i] for i in range(len(self.kernel.params))]
        meanfunc_params = None
        if not fix_mean_params:
            meanfunc_params = [params[i+len(self.kernel.params)] for i in range(len(self.meanfunc.params))]
 
        # Compute the log likelihood for the given parameters
        # For multi-wavelength observations, we make the simplifying assumption that there is no covariance between bands
        # Therefore, we take the log likelihood of each band separately and sum them
        loglike = self.jit_loglikelihood(self.x, self.y, self.yerr, kernel_params, meanfunc_params)
        loglike += log_prior
        
        # Return the log likelihood or inverse log likelihood as either a float or jnp.inf (avoids Nans)
        if jnp.isinf(loglike) or jnp.isnan(loglike):
            return invert * -jnp.inf
            
        return invert * loglike
    
    def optimize_parameters(self, x, y, yerr, band = None, image = None, method='minimize', loglikelihood=None, logprior=None, ptform=None, fix_mean_params = False, fix_kernel_params = False, minimize_kwargs=None, sampler_kwargs=None, run_sampler_kwargs=None):
        """
        Optimize the parameters of the Gaussian Process (GP) for a set of observations.

        :param x: array-like
            Input data points (independent variable).

        :param y: array-like
            Observed values corresponding to the input data points.

        :param yerr: array-like
            Measurement uncertainties of the observed values.
            
        :param method: str, (default='minimize')
            The method for optimizing parameters. Available options: 'minimize' (scipy.optimize.minimize BFGS), 'emcee' (ensemble MCMC sampler), 'zeus' (ensemble slice sampler), and 'dynesty' (nested sampling). Defaults to 'minimize.'

        :param n_bands: int, optional (default=1)
            If fitting multi-wavelength astronomical data, specifies the number of wavelength filters through which an object has been observed. Defaults to 1.

        :param loglikelihood: function, optional (default=None)
            The log-likelihood function for the GP model. If not specified, the log of a multivariate normal distribution PDF will be used as the loglikelihood function.

        :param logprior: function, optional (default=None)
            The log-prior function for the emcee and zeus MCMC sampling methods. If not specified, no prior (uniform over all values) will be enforced.

        :param ptform: function, optional (default=None)
            Function to transform the prior for the dynesty nested sampling process.

        :param fix_mean_params: bool, optional (default=False)
            Whether to fix the parameters of the mean function during optimization/sampling. By default, the mean function parameters are fit for.

        :param fix_kernel_params: bool, optional (default=False)
            Whether to fix the parameters of the kernel function during optimization/sampling. By default, the kernel parameters are fit for.

        :param minimize_kwargs: dict, optional (default=None)
            Additional keyword arguments for the scipy.optimize.minimize function.

        :param sampler_kwargs: dict, optional (default=None)
            Additional keyword arguments for the emcee, zeus, or dynesty sampler.

        :param run_sampler_kwargs: dict, optional (default=None)
            Additional keyword arguments for running the emcee, zeus, or dynesty sampler.

        :return results:
            If using the 'minimize' method, the function will return the scipy.optimize style results.

        :return sampler:
            If using one of 'emcee,' 'zeus,' or 'dynesty' methods, the function will return the sampler in the style of the user's choice of optimization method.
        """

        # Handle optional arguments
        if sampler_kwargs is None:
            sampler_kwargs = {}
        if run_sampler_kwargs is None:
            run_sampler_kwargs = {}
        if minimize_kwargs is None:
            minimize_kwargs = {}
            
        # Convert data numpy arrays to jax arrays for faster computing later on
        self.x = jnp.array(x)
        self.y, self.yerr = self._rescale_data(jnp.array(y), jnp.array(yerr))
        
        # Store n_bands, n_images, and indices information
        self._prepare_indices(self.x, band, image)
        try:
            self.kernel.import_from_gp(self.n_bands, self.n_images, self.indices)
        except:
            pass
        
        # Determine the number of dimensions for optimization/sampling
        self.ndim = 0
        if not fix_mean_params:
            self.ndim += len(self.meanfunc.params)
        if not fix_kernel_params:
            self.ndim += len(self.kernel.params)
        
        # Set the loglikelihood/logprior to the default multi-variate normal likelihood specified within the GP class function, if not otherwise specified
        if loglikelihood == None:
            loglikelihood = self.loglikelihood
        if logprior == None:
            logprior = self.logprior
            
        # Compute mean and covariance given the specified mean function and kernel with their initial parameters
        self.mean = self.meanfunc.mean(self.x)
        self.cov = self.kernel.covariance(self.x)
        
        if method == 'dynesty':
                
            nlive = sampler_kwargs.pop('nlive', 500)
            sample = sampler_kwargs.pop('sample', 'rslice')
           
            sampler = dynesty.NestedSampler(self.jointprobability, ptform, self.ndim, logl_args = (logprior, fix_mean_params, fix_kernel_params), nlive = nlive, sample = sample, **sampler_kwargs)
            
            sampler.run_nested(**run_sampler_kwargs)
            return sampler
        
        if method == 'emcee' or method == 'zeus' or method == 'minimize':
        
            # Get vector of initial parameters, which is required for optmizing/sampling with the minimize, emcee, and zeus methods
            init_guess, init_guess_scale = self._get_initial_guess(fix_mean_params, fix_kernel_params)

            if method == 'minimize': 
                results = minimize(self.jointprobability, init_guess, args = (logprior, fix_mean_params, fix_kernel_params, -1), **minimize_kwargs)
                return results

            if np.isinf(np.any(logprior(init_guess))):
                raise Exception("When passed to the specified ``log_prior'' function, some or all of the parameters that the kernel and mean function were initialized with yield an indefinite value. Please check that the initial parameters used are within the bounds of the prior, as the MCMC chains are initialized, with some scatter, around these values.")
                
            # Initialize walkers with random initial positions around the initial guess
            p0 = np.random.normal(init_guess, init_guess_scale, size=(nwalkers, self.ndim))
            for r, row in enumerate(p0):
                while np.isinf(logprior(row)):
                    p0[r] = np.random.normal(init_guess, 0.001)

            nwalkers = sampler_kwargs.pop('nwalkers', 24)
            nsteps = run_sampler_kwargs.pop('nsteps', 1000)

            
            if method == 'emcee':
                sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.jointprobability, args = (logprior, fix_mean_params, fix_kernel_params, False), **sampler_kwargs)

            if method == 'zeus':
                sampler = zeus.EnsembleSampler(nwalkers, self.ndim, self.jointprobability, args=[logprior, fix_mean_params, fix_kernel_params, False], **sampler_kwargs)

                
            # Run the sampler
            sampler.run_mcmc(p0, nsteps=nsteps, **run_sampler_kwargs)
            return sampler
    
    def predict(self, x_prime, x, y, yerr):
        """
        For a set of observations, y, with measurement uncertainties, yerr, observed at x, give the function values at x_new.
        
        expectation = mu_U + (cov_UV * cov_VV^-1) * (y - mu_V)
        variance = cov_UU - (cov_UV * cov_VV^-1 * cov_VU)

        x_prime = desired x locations of data
        x = observed data, x
        y = observed data, y
        """
        
        cov_UV = self.kernel.covariance(x_prime, x)
        cov_VV = self.kernel.covariance(x, x) + np.diag(yerr**2)
        cov_UU = self.kernel.covariance(x_prime, x_prime)

        mu_U = self.meanfunc.mean(x_prime)
        mu_V = self.meanfunc.mean(x)
        
        expectation = mu_U + (cov_UV @ np.linalg.solve(cov_VV, y-mu_V))
        variance = cov_UU - (cov_UV @ np.linalg.solve(cov_VV, np.transpose(cov_UV)))

        return expectation, variance
    
    
    
