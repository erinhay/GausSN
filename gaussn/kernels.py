from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax

class BaseKernel(ABC):
    """Parent class for GausSN kernels. Subclasses set `param_names` and implement `_kernel_fn`."""
    param_names: list = []

    def __init__(self, params):
        self.params = params
    
    def _unpack(self, params):
        """Pure: flat params list -> dict of named kwargs for _kernel_fn. No mutation."""
        return dict(zip(self.param_names, params))

    @abstractmethod
    def _kernel_fn(self, x, x_prime, **params):
        """Pure covariance computation. The only thing each kernel needs to define."""
        raise NotImplementedError

    def covariance(self, x, x_prime=None, params=None):
        if params is None:
            params = self.params
        if x_prime is None:
            x_prime = x
        return self._kernel_fn(x, x_prime, **self._unpack(params))
    

class ExpSquaredKernel(BaseKernel):
    """Exponentiated Squared Kernel for Gaussian processes. params: [A, tau]."""
    param_names = ['A', 'tau']
 
    def _kernel_fn(self, x, x_prime, A, tau):
        return A**2 * jnp.exp(-(x[:, None] - x_prime[None, :])**2 / (2*tau**2))
    

class ExponentialKernel(BaseKernel):
    """
    Exponential Kernel for Gaussian processes.
    """  
    param_names = ['A', 'tau']
        
    def _kernel_fn(self, x, x_prime, A, tau):
        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        K = A**2 * jnp.exp(-vector_mag/tau)
        return K


class ConstantKernel(BaseKernel):
    param_names = ['c']
        
    def _kernel_fn(self, x, x_prime, c):
        return jnp.ones([len(x), len(x_prime)]) * c


class Matern32Kernel(BaseKernel):
    """
    Matern 3/2 Kernel for Gaussian processes.
    """
    param_names = ['A', 'l']
        
    def _kernel_fn(self, x, x_prime, A, l):      
        r2 = (x[:, None] - x_prime[None, :])**2
        K = A * ((1 + jnp.sqrt(3*r2)/l) * jnp.exp(-jnp.sqrt(3*r2)/l))
        return K


class Matern52Kernel(BaseKernel):
    """
    Matern 5/2 Kernel for Gaussian processes.
    """
    param_names = ['A', 'l']
        
    def _kernel_fn(self, x, x_prime, A, l):
        r2 = (x[:, None] - x_prime[None, :])**2
        a = 1 + (jnp.sqrt(5*r2)/l) + (5*r2/(3*(l**2)))
        b = -jnp.sqrt(5*r2)/l
        K = A * a * jnp.exp(b)
        return K
    

class RationalQuadraticKernel(BaseKernel):
    """
    Rational Quadratic Kernel for Gaussian processes.
    """
    param_names = ['A', 'l', 'scale_mixture']
        
    def _kernel_fn(self, x, x_prime, A, l, scale_mixture):
        return A**2 * (1 + (x[:, None] - x_prime[None, :])**2/(2*scale_mixture*tau**2))
    

class GibbsKernel(BaseKernel):
    """
    Gibbs Kernel for Gaussian processes.
    
    Defined as:
        K = A^2 * ( 2*tau(y|mu, sigma)*tau(y'|mu, sigma) / (tau(y|mu, sigma)**2 + tau(y'|mu, sigma)**2) )**(1/2) * exp( (y - y')**2 / (tau(y|mu, sigma)**2 + tau(y'|mu, sigma)**2) )
    where:
        tau(y|mu, sigma) = lambda * (1 - p * N(y|mu, sigma))
    and N(y|mu, sigma) is a normal distribution with mean mu and variance sigma**2.
    """   
    param_names = ['A', 'lamdba', 'p', 'mu', 'sigma']
        
    def _kernel_fn(self, x, x_prime, A, lamdba, p, mu, sigma):
        normal_x = jnp.exp(-(x[:, None] - mu)**2 / (2*(sigma**2))) / (sigma * jnp.sqrt(2*jnp.pi))
        normal_xprime = jnp.exp(-(x_prime[None, :] - mu)**2 / (2*(sigma**2))) / (sigma * jnp.sqrt(2*jnp.pi))
        
        tau_x = lamdba * (1 - (p * normal_x))
        tau_xprime = lamdba * (1 - (p * normal_xprime))
        
        root_num = 2 * tau_x * tau_xprime
        exp_num = (x[:, None] - x_prime[None, :])**2
        denom = (tau_x**2) + (tau_xprime**2)
        K = A**2 * jnp.sqrt(root_num/denom) * jnp.exp(-exp_num/denom)
        return K
    

class OUKernel(BaseKernel): 
    param_names = ['A', 'l']
        
    def _kernel_fn(self, x, x_prime, A, l):
        r2 = (x[:, None] - x_prime[None, :])**2
        K = A * jnp.exp(-jnp.sqrt(r2) / l)
        return K


class PeriodicKernel(BaseKernel):   
    param_names = ['A', 'l', 'p']  
        
    def _kernel_fn(self, x, x_prime, A, l, p):
        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        sin = vector_mag / p
        exponent = -2 * ( jnp.sin(jnp.pi * sin) / l)**2
        K = A * jnp.exp(exponent)
        return K


class JJKernel(BaseKernel):
    param_names = ['A', 'l', 'm', 'b']

    def _p(self, x, m, b):
        return m * x + b
        
    def _kernel_fn(self, x, x_prime, A, l, m, b):
        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        sin = vector_mag / self._p(x, m, b)
        exponent = -2 * ( jnp.sin(jnp.pi * sin) / l)**2
        K = A * jnp.exp(exponent)
        return K
        
