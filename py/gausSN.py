import numpy as np

class GP:
    """
    Initiate with a kernel and mean function.
    """
    
    def __init__(self, kernel, meanfunc):
        self.kernel = kernel
        self.meanfunc = meanfunc
    
    def predict(self, U, V_x, V_y):
        """
        expectation = mu_U + (cov_UV * cov_VV^-1) * (V - mu_V)
        variance = cov_UU - (cov_UV * cov_VV^-1 * cov_VU)

        U = new data
        V_x = observed data, x
        V_y = observed data, y
        """
        cov_UV = self.kernel.covariance(U, V_x)
        cov_VV = self.kernel.covariance(V_x, V_x)
        cov_UU = self.kernel.covariance(U, U)

        mu_U = self.meanfunc.mean(U)
        mu_V = self.meanfunc.mean(V_x)

        expectation = mu_U + (cov_UV @ np.linalg.inv(cov_VV) @ (V_y-mu_V))
        variance = cov_UU - (cov_UV @ np.linalg.inv(cov_VV) @ np.transpose(cov_UV))

        return expectation, variance
