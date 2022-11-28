import numpy as np
from scipy.optimize import minimize

class GP:
    """
    Initiate with a kernel and mean function.
    """
    
    def __init__(self, kernel, meanfunc):
        self.kernel = kernel
        self.meanfunc = meanfunc
        
    def negloglikelihood(self, params):
        
        kernel_params = [params[n] for n in range(len(self.kernel.params))]
        meanfunc_params = [params[n+len(self.kernel.params)] for n in range(len(self.meanfunc.params))]
        
        self.kernel.reset(kernel_params)
        self.meanfunc.reset(meanfunc_params)
        
        self.cov = self.kernel.covariance(self.x, self.x) + np.diag(self.yerr**2)
        self.mean = self.meanfunc.mean(self.x)
        
        a = np.log(np.linalg.det(2 * np.pi * self.cov))
        b = np.dot(np.transpose(self.mean - self.y), np.linalg.solve(self.cov, (self.mean - self.y)))
        return 0.5*(a + b)
    
    def optimize_parameters(self, x, y, yerr, bounds=None):
        
        self.ndim = len(self.kernel.params) + len(self.meanfunc.params)
        self.x = x
        self.y = y
        self.yerr = yerr
        self.mean = self.meanfunc.mean(self.x)
        
        init_guess = [self.kernel.params[i] for i in range(len(self.kernel.params))] + [self.meanfunc.params[i] for i in range(len(self.meanfunc.params))]
        
        results = minimize(self.negloglikelihood, init_guess, bounds = bounds)
        return results
    
    def predict(self, U_x, V_x, V_y, V_yerr):
        """
        expectation = mu_U + (cov_UV * cov_VV^-1) * (V - mu_V)
        variance = cov_UU - (cov_UV * cov_VV^-1 * cov_VU)

        U_x = new data
        V_x = observed data, x
        V_y = observed data, y
        """
        cov_UV = self.kernel.covariance(U_x, V_x)
        cov_VV = self.kernel.covariance(V_x, V_x) + np.diag(V_yerr**2)
        cov_UU = self.kernel.covariance(U_x, U_x)

        mu_U = self.meanfunc.mean(U_x)
        mu_V = self.meanfunc.mean(V_x)
        
        expectation = mu_U + (cov_UV @ np.linalg.solve(cov_VV, V_y-mu_V))
        variance = cov_UU - (cov_UV @ np.linalg.solve(cov_VV, np.transpose(cov_UV)))

        return expectation, variance
