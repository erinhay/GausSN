import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

class ConstantLensingKernel:
    """
    Moving kernel defined as A^2 * exp(-(y-y')^2 / (2*tau^2)).
    """
    def __init__(self, params):
        self.ndim = len(params)
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::2])
        self.betas = jnp.array([1] + params[3::2])
        self.jit_time_shift = jax.jit(self._time_shift)
        self.jit_lens = jax.jit(self._lens)
        self.vmap_time_shift = jax.vmap(self.jit_time_shift)
        self.vmap_lens = jax.vmap(self.jit_lens)

    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::2])
        self.betas = jnp.array([1] + params[3::2])

    def _make_mask(self):
        self.mask = np.zeros((self.n_images, self.n_images*self.repeats))
        for n in range(self.n_images):
            start = self.repeats*n
            stop = self.repeats*(n + 1)
            self.mask[n, start : stop] = 1
        self.mask = jnp.array(self.mask)

    def _time_shift(self, x, delta):
        return x - delta

    def _lens(self, x, beta):
        return jnp.repeat(beta, len(x))
    
    def import_from_gp(self, n_images, n_bands, repeats):
        self.n_images = n_images
        self.n_bands = n_bands
        self.repeats = repeats
        self._make_mask()
        self.scale = [0.5, 5] + [5, 0.5]*(self.n_images-1)
        
    def covariance(self, x, x_prime=None, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)

        x = self.vmap_time_shift(x, self.deltas)
        beta = self.vmap_lens(x, self.betas).flatten()

        new_x = x.flatten()
        new_x_prime = new_x
        beta_prime = beta

        if x_prime != None:
            x_prime = self.vmap_time_shift(x_prime, self.deltas)
            beta_prime = self.vmap_lens(x_prime, self.betas).flatten()

            new_x_prime = x_prime.flatten()

        K = jnp.outer(beta, beta_prime) * self.A**2 * jnp.exp(-(new_x[:, None] - new_x_prime[None, :])**2/(2*self.tau**2))
        return K

class SigmoidLensingKernel:
    """
    Moving kernel defined as A^2 * exp(-(y-y')^2 / (2*tau^2)).
    """
    def __init__(self, params):
        self.ndim = len(params)
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::5])
        self.beta0s = jnp.array([1] + params[3::5])
        self.beta1s = jnp.array([1] + params[4::5])
        self.rs = jnp.array([0] + params[5::5])
        self.t0s = jnp.array([0] + params[6::5])

        self.jit_time_shift = jax.jit(self._time_shift)
        self.jit_lens = jax.jit(self._lens)
        self.vmap_time_shift = jax.vmap(self.jit_time_shift)
        self.vmap_lens = jax.vmap(self.jit_lens)

    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::5])
        self.beta0s = jnp.array([1] + params[3::5])
        self.beta1s = jnp.array([1] + params[4::5])
        self.rs = jnp.array([0] + params[5::5])
        self.t0s = jnp.array([0] + params[6::5])

    def _time_shift(self, x, delta):
        return x - delta

    def _lens(self, x, beta0, beta1, r, t0):
        denom = 1 + jnp.exp(-r * (x - t0))
        return beta0 + (beta1/denom)
    
    def import_from_gp(self, n_images, n_bands, repeats):
        self.n_images = n_images
        self.n_bands = n_bands
        self.repeats = repeats
        self.scale = [0.5, 5] + [5, 0.5, 0.5, 0.5, 10]*(self.n_images-1)
        
    def covariance(self, x, x_prime=None, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)

        x = self.vmap_time_shift(x, self.deltas)
        beta = self.vmap_lens(x, self.beta0s, self.beta1s, self.rs, self.t0s).flatten()

        new_x = x.flatten()
        new_x_prime = new_x
        beta_prime = beta

        if x_prime != None:
            x_prime = self.vmap_time_shift(x_prime, self.deltas)
            beta_prime = self.vmap_lens(x_prime, self.beta0s, self.beta1s, self.rs, self.t0s).flatten()

            new_x_prime = x_prime.flatten()
        
        K = jnp.outer(beta, beta_prime) * self.A**2 * jnp.exp(-(new_x[:, None] - new_x_prime[None, :])**2/(2*self.tau**2))
        return K

class ExpSquaredKernel:
    """
    Moving kernel defined as A^2 * exp(-(y-y')^2 / (2*tau^2)).
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.ndim = len(params)
        self.scale = [0.5, 5]

    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)
        K = self.A**2 * jnp.exp(-(y[:, None] - y_prime[None, :])**2/(2*self.tau**2)) #(beta[:, None] * beta_prime[None, :]) * 
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

    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)
        vector_mag = jnp.sqrt((y[:, None] - y_prime[None, :])**2)
        K = self.A**2 * jnp.exp(-vector_mag/self.tau)
        return K

class ConstantKernel:
    def __init__(self, params):
        self.c = params[0]
        self.ndim = len(params)
        self.scale = [1]

    def _reset(self, params):
        self.c = params[0]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)
        K = jnp.ones([len(y), len(y_prime)]) * self.c
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

    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)
        r2 = (y[:, None] - y_prime[None, :])**2
        K = self.A * ((1 + jnp.sqrt(3*r2)/self.l) * jnp.exp(-jnp.sqrt(3*r2)/self.l))
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

    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)
        r2 = (y[:, None] - y_prime[None, :])**2
        a = 1 + (jnp.sqrt(5*r2)/self.l) + (5*r2/(3*(self.l**2)))
        b = -jnp.sqrt(5*r2)/self.l
        K = self.A * a * jnp.exp(b)
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

    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.scale_mixture = params[2]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)
        K = self.A**2 * (1 + (y[:, None] - y_prime[None, :])**2/(2*self.scale_mixture*self.tau**2))
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

    def _reset(self, params):
        self.A = params[0]
        self.lamdba = params[1]
        self.p = params[2]
        self.mu = params[3]
        self.sigma = params[4]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)
        normal_y = jnp.exp(-(y[:, None] - self.mu)**2 / (2*(self.sigma**2))) / (self.sigma * jnp.sqrt(2*jnp.pi))
        normal_yprime = jnp.exp(-(y_prime[None, :] - self.mu)**2 / (2*(self.sigma**2))) / (self.sigma * jnp.sqrt(2*jnp.pi))
        
        tau_y = self.lamdba * (1 - (self.p * normal_y))
        tau_yprime = self.lamdba * (1 - (self.p * normal_yprime))
        
        root_num = 2 * tau_y * tau_yprime
        exp_num = (y[:, None] - y_prime[None, :])**2
        denom = (tau_y**2) + (tau_yprime**2)
        K = self.A**2 * jnp.sqrt(root_num/denom) * jnp.exp(-exp_num/denom)
        return K
    
class OUKernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.ndim = len(params)
        self.scale = [0.5, 5]

    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        
    def covariance(self, y, y_prime, kernel_params=None):
        if kernel_params is not None:
            self._reset(kernel_params)
        K = self.A * jnp.exp(jnp.sqrt(jnp.sum(y[:, None] - y_prime[None, :])) / self.l)
        return K
        
