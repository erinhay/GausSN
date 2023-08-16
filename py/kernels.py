import numpy as np

class ExpSquaredKernel:
    """
    Moving kernel defined as A^2 * exp(-(y-y')^2 / (2*tau^2)).
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        
    def reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        
    def covariance(self, y, y_prime):
        K = self.A**2 * np.exp(-(y[:, None] - y_prime[None, :])**2/(2*self.tau**2))
        return K

class ConstantKernel:
    def __init__(self, params):
        self.c = params[0]
        self.params = params
        
    def reset(self, params):
        self.c = params[0]
        self.params = params
        
    def covariance(self, y, y_prime):
        K = np.ones([len(y), len(y_prime)]) * self.c
        return K

class DotProductKernel:
    def __init__(self):
        self.c = 0
        
    def reset(self):
        self.c = 0
        
    def covariance(self, y, y_prime):
        K = np.zeros([len(y), len(y_prime)])
        for i in range(len(y)):
            for j in range(len(y_prime)):
                K[i,j] = np.dot(y[i], y_prime[j])
        return K

class Matern32Kernel:
    def __init__(self):
        self.c = 0
        
    def reset(self):
        self.c = 0
        
    def covariance(self, y, y_prime):
        r2 = (y[:, None] - y_prime[None, :])**2
        K = (1 + np.sqrt(3*r2)) * np.exp(-np.sqrt(3*r2))
        return K

class Matern52Kernel:
    def __init__(self):
        self.c = 0
        
    def reset(self):
        self.c = 0
        
    def covariance(self, y, y_prime):
        r2 = (y[:, None] - y_prime[None, :])**2
        K = (1 + np.sqrt(5*r2) + (5*r2/3)) * np.exp(-np.sqrt(5*r2))
        return K
