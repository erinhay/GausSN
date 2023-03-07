import numpy as np
from scipy.optimize import minimize
import dynesty
import emcee
import matplotlib.pyplot as plt

class GP:
    """
    Initiate with a kernel and mean function.
    """
    
    def __init__(self, kernel, meanfunc):
        self.kernel = kernel
        self.meanfunc = meanfunc
    
    def logprior(self, params):
        return 0
    
    def loglikelihood(self, x, y, yerr, log_prior, magnification_matrix):
        
        self.cov = (magnification_matrix*self.kernel.covariance(x, x)) + np.diag(yerr**2)
        self.mean = self.meanfunc.mean(x)
        
        try:
            a = np.log(np.linalg.det(2 * np.pi * self.cov))
            b = np.dot(np.transpose(self.mean - y), np.linalg.solve(self.cov, (self.mean - y)))
        except np.linalg.LinAlgError:
            return -np.inf
        
        if np.isinf(log_prior):
            return -np.inf
        loglike = -0.5*(a + b) + log_prior
        
        return loglike
        
    def jointprobability(self, params, logprior=None, lensing_model=None, fix_mean_params = False, fix_kernel_params = False, invert=False):
        
        if lensing_model != None:
            lensing_params = [params[i+self.ndim-len(lensing_model.lensing_params)] for i in range(len(lensing_model.lensing_params))]
            lensing_model.reset(lensing_params)
            x = lensing_model.time_shift(self.x)
            magnification_matrix = lensing_model.magnification_matrix()
        else:
            x = self.x
            magnification_matrix = 1
            
        if logprior == None:
            logprior = self.logprior
        
        loglike = 0
        for n in range(self.n_bands):
            
            if fix_mean_params and fix_kernel_params:
                kernel_params = self.kernel.params
                meanfunc_params = self.meanfunc.params
            elif fix_mean_params:
                offset = n * len(self.kernel.params)
                
                kernel_params = [params[i+offset] for i in range(len(self.kernel.params))]
                meanfunc_params = self.meanfunc.params
                self.kernel.reset(kernel_params)
            elif fix_kernel_params:
                offset = n * len(self.meanfunc.params)
                
                kernel_params = self.kernel.params
                meanfunc_params = [params[i+len(self.kernel.params)+offset] for i in range(len(self.meanfunc.params))]
                self.meanfunc.reset(meanfunc_params)
            else:
                offset = n * (len(self.kernel.params) + len(self.meanfunc.params))
                
                kernel_params = [params[i+offset] for i in range(len(self.kernel.params))]
                meanfunc_params = [params[i+len(self.kernel.params)+offset] for i in range(len(self.meanfunc.params))]

                self.kernel.reset(kernel_params)
                self.meanfunc.reset(meanfunc_params)
            
            log_prior = logprior(kernel_params, meanfunc_params)
            loglike += self.loglikelihood(x, self.y, self.yerr, log_prior, magnification_matrix)
            if np.isinf(loglike):
                if invert:
                    return np.inf
                return -np.inf
            
        if invert:
            return -loglike
        
        return loglike
    
    def optimize_parameters(self, x, y, yerr, n_bands = 1, method='minimize', loglikelihood=None, logprior=None, bounds=None, ptform=None, lensing_model=None, fix_mean_params = False, fix_kernel_params = False):
        
        if fix_mean_params and fix_kernel_params:
            self.ndim = 0
        elif fix_mean_params:
            self.ndim = len(self.kernel.params) * n_bands
        elif fix_kernel_params:
            self.ndim = len(self.meanfunc.params) * n_bands
        else:
            self.ndim = (len(self.kernel.params) + len(self.meanfunc.params)) * n_bands
        
        if lensing_model != None:
            self.ndim += len(lensing_model.lensing_params)
            #y, yerr = lensing_model.rescale_data(y, yerr)
            lensing_model.prepare_magnification_matrix_bases(x)
        if loglikelihood == None:
            loglikelihood = self.loglikelihood
            
        self.x = x
        self.y = y
        self.yerr = yerr
        self.n_bands = n_bands
        
        if method == 'minimize':
            if fix_mean_params and fix_kernel_params:
                init_guess = []
            elif fix_mean_params:
                init_guess = [self.kernel.params[i] for i in range(len(self.kernel.params))] * self.n_bands
            elif fix_kernel_params:
                init_guess = [self.meanfunc.params[i] for i in range(len(self.meanfunc.params))] * self.n_bands
            else:
                init_guess = ([self.kernel.params[i] for i in range(len(self.kernel.params))] + [self.meanfunc.params[i] for i in range(len(self.meanfunc.params))]) * self.n_bands
            
            if lensing_model != None:
                init_guess = init_guess + lensing_model.lensing_params
                
            if len(init_guess) < 0.5:
                raise Exception("No parameters to fit. Check fix_mean_params, fix_kernel_params, and lensing model to make sure there are parameters to fit.")
                
            results = minimize(self.jointprobability, init_guess, bounds = bounds,
                               args = (logprior, lensing_model, fix_mean_params, fix_kernel_params, True))
        
        if method == 'nested_sampling':
            sampler = dynesty.NestedSampler(self.jointprobability, ptform, self.ndim, logl_args = (logprior, lensing_model, fix_mean_params, fix_kernel_params),
                                            sample='rslice')
            sampler.run_nested(maxiter=50000)
            return sampler
            
        if method == 'MCMC':
            if fix_mean_params and fix_kernel_params:
                init_guess = []
            elif fix_mean_params:
                init_guess = [self.kernel.params[i] for i in range(len(self.kernel.params))] * self.n_bands
            elif fix_kernel_params:
                init_guess = [self.meanfunc.params[i] for i in range(len(self.meanfunc.params))] * self.n_bands
            else:
                init_guess = ([self.kernel.params[i] for i in range(len(self.kernel.params))] + [self.meanfunc.params[i] for i in range(len(self.meanfunc.params))]) * self.n_bands
            
            if lensing_model != None:
                init_guess = init_guess + lensing_model.lensing_params
                
            if len(init_guess) < 0.5:
                raise Exception("No parameters to fit. Check fix_mean_params, fix_kernel_params, and lensing model to make sure there are parameters to fit.")
            
            results = minimize(self.jointprobability, init_guess, bounds = bounds, args = (logprior, lensing_model, fix_mean_params, fix_kernel_params, True))
            
            nwalkers = 32
            p0 = np.random.randn(nwalkers, self.ndim)
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.jointprobability)
            sampler.run_mcmc(p0, 10_000, progress=True)
            return sampler
        
        return results
    
    def predict(self, x_prime, x, y, yerr):
        """
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