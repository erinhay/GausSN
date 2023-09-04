import numpy as np
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

class ExpSquaredKernel:
    """
    Moving kernel defined as A^2 * exp(-(y-y')^2 / (2*tau^2)).
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.ndim = len(params)
        self.scale = [0.5, 5]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is None:
            A = self.A
            tau = self.tau
        else:
            A, tau = kernel_params[0], kernel_params[1]
        K = A**2 * jnp.exp(-(y[:, None] - y_prime[None, :])**2/(2*tau**2))
        return K
    
class ExponentialKernel:
    """
    Moving kernel defined as A^2 * exp(-|y-y'| / tau).
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.ndim = len(params)
        self.scale = [0.5, 5]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is None:
            A = self.A
            tau = self.tau
        else:
            A, tau = kernel_params[0], kernel_params[1]
        vector_mag = jnp.sqrt((y[:, None] - y_prime[None, :])**2)
        K = A**2 * jnp.exp(-vector_mag/tau)
        return K

class ConstantKernel:
    def __init__(self, params):
        self.c = params[0]
        self.ndim = len(params)
        self.scale = [1]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is None:
            c = self.c
        else:
            c = kernel_params[0]
        K = jnp.ones([len(y), len(y_prime)]) * c
        return K

class DotProductKernel:
    def __init__(self):
        self.scale = [1]
        
    def covariance(self, y, y_prime, kernel_params=None):
        K = np.zeros([len(y), len(y_prime)])
        for i in range(len(y)):
            for j in range(len(y_prime)):
                K[i,j] = jnp.dot(y[i], y_prime[j])
        return K

class Matern32Kernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.ndim = len(params)
        self.scale = [0.5, 5]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is None:
            A = self.A
            l = self.l
        else:
            A, l = kernel_params[0], kernel_params[1]
        r2 = (y[:, None] - y_prime[None, :])**2
        K = A * ((1 + jnp.sqrt(3*r2)/l) * jnp.exp(-jnp.sqrt(3*r2)/l))
        return K

class Matern52Kernel:
    """
    Stationary kernel defined as K = (1 + sqrt(5)d/length_scale + (5/3)d^2/length_scale^2) * exp(sqrt(5)d^2/length_scale).
    """
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.ndim = len(params)
        self.scale = [0.5, 5]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is None:
            A = self.A
            l = self.l
        else:
            A, l = kernel_params[0], kernel_params[1]
        r2 = (y[:, None] - y_prime[None, :])**2
        a = 1 + (jnp.sqrt(5*r2)/l) + (5*r2/(3*(l**2)))
        b = -jnp.sqrt(5*r2)/l
        K = A * a * jnp.exp(b)
        return K
    
class RationalQuadraticKernel:
    """
    Moving kernel defined as A^2 * (1 + (y-y')^2/(2*scale_mixture*tau^2)^(-scale_mixture).
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.scale_mixture = params[2]
        self.ndim = len(params)
        self.scale = [0.5, 5, 5]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is None:
            A = self.A
            tau = self.tau
            scale_mixture = self.scale_mixture
        else:
            A, tau, scale_mixture = kernel_params[0], kernel_params[1], kernel_params[2]
        K = A**2 * (1 + (y[:, None] - y_prime[None, :])**2/(2*scale_mixture*tau**2))
        return K
    
class GibbsKernel:
    """
    Moving kernel defined as:
        K = A^2 * ( 2*tau(y|mu, sigma)*tau(y'|mu, sigma) / (tau(y|mu, sigma)**2 + tau(y'|mu, sigma)**2) )**(1/2) * exp( (y - y')**2 / (tau(y|mu, sigma)**2 + tau(y'|mu, sigma)**2) )
    where:
        tau(y|mu, sigma) = lambda * (1 - p * N(y|mu, sigma))
    and N(y|mu, sigma) is a normal distribution with mean mu and variance sigma**2.
    """
    def __init__(self, params):
        self.A = params[0]
        self.lamdba = params[1]
        self.p = params[2]
        self.mu = params[3]
        self.sigma = params[4]
        self.ndim = len(params)
        self.scale = [0.5, 5, 2, 5, 2]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is None:
            A = self.A
            lamdba = self.lamdba
            p = self.p
            mu = self.mu
            sigma = self.sigma
        else:
            A, lamdba, p, mu, sigma = kernel_params[0], kernel_params[1], kernel_params[2], kernel_params[3], kernel_params[4]

        normal_y = jnp.exp(-(y[:, None] - mu)**2 / (2*(sigma**2))) / (sigma * jnp.sqrt(2*jnp.pi))
        normal_yprime = jnp.exp(-(y_prime[None, :] - mu)**2 / (2*(sigma**2))) / (sigma * jnp.sqrt(2*jnp.pi))
        
        tau_y = lamdba * (1 - (p * normal_y))
        tau_yprime = lamdba * (1 - (p * normal_yprime))
        
        root_num = 2 * tau_y * tau_yprime
        exp_num = (y[:, None] - y_prime[None, :])**2
        denom = (tau_y**2) + (tau_yprime**2)
        K = A**2 * jnp.sqrt(root_num/denom) * jnp.exp(-exp_num/denom)
        return K
    
class OUKernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.ndim = len(params)
        self.scale = [0.5, 5]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is None:
            A = self.A
            l = self.l
        else:
            A, l = kernel_params[0], kernel_params[1]
        K = A * jnp.exp(jnp.sqrt(jnp.sum(y[:, None] - y_prime[None, :])) / l)
        return K
        
