import numpy as np

class ExpSquaredKernel:
    """
    Moving kernel defined as A^2 * exp(-(y-y')^2 / (2*tau^2)).
    """
    def __init__(self, A, tau):
        self.A = A
        self.tau = tau
        
    def covariance(self, y, y_prime):
        K = np.zeros([len(y), len(y_prime)])
        for i in range(len(y)):
            for j in range(len(y_prime)):
                K[i,j] = self.A**2 * np.exp(-(y[i] - y_prime[j])**2/(2*self.tau**2))
        return K

class ConstantKernel:
    def __init__(self, c):
        self.c = c
        
    def covariance(self, y, y_prime):
        K = np.zeros([len(y), len(y_prime)])
        for i in range(len(y)):
            for j in range(len(y_prime)):
                K[i,j] = self.c
        return K

class DotProductKernel:
    def __init__(self):
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
        
    def covariance(self, y, y_prime):
        K = np.zeros([len(y), len(y_prime)])
        for i in range(len(y)):
            for j in range(len(y_prime)):
                r = np.linalg.norm(y[i]-y_prime[j])
                r2 = r**2
                K[i,j] = (1 + np.sqrt(3*r2)) * np.exp(-np.sqrt(3*r2))
        return K

class Matern52Kernel:
    def __init__(self):
        self.c = 0
        
    def covariance(self, y, y_prime):
        K = np.zeros([len(y), len(y_prime)])
        for i in range(len(y)):
            for j in range(len(y_prime)):
                r = np.linalg.norm(y[i]-y_prime[j])
                r2 = r**2
                K[i,j] = (1 + np.sqrt(5*r2) + (5*r2/3)) * np.exp(-np.sqrt(5*r2))
        return K
