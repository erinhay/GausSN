import numpy as np
import jax.numpy as jnp
import jax

class ConstantLensingKernel:
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::2])
        self.betas = jnp.array([1] + params[3::2])
        self.params = params
        self.scale = [0.5, 5]
        self.covariance = jax.jit(self._covariance)
    
    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::2])
        self.betas = jnp.array([1] + params[3::2])
        self.params = params

    def _make_mask(self):
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[self.n_images*pb]
            stop = self.indices[self.n_images*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask

    def _time_shift(self, x, delta):
        return x - delta

    def _lens(self, x, beta):
        return beta

    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta_vector = jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._lens(x, beta_vector)

        if x_prime != None:
            x_prime = self._time_shift(x_prime, delta_vector)
            b_prime = self._lens(x_prime, beta_vector)
        else:
            x_prime = x
            b_prime = b

        K = jnp.outer(b, b_prime) * self.A**2 * jnp.exp(-(x[:, None] - x_prime[None, :])**2/(2*self.tau**2))
        return jnp.multiply(self.mask, K)
    
    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class SigmoidLensingKernel:
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::5])
        self.beta0s = jnp.array([1] + params[3::5])
        self.beta1s = jnp.array([0] + params[4::5])
        self.rs = jnp.array([0] + params[5::5])
        self.t0s = jnp.array([0] + params[6::5])
        self.params = params
        self.scale = [0.5, 5]
        self.covariance = jax.jit(self._covariance)

    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::5])
        self.beta0s = jnp.array([1] + params[3::5])
        self.beta1s = jnp.array([0] + params[4::5])
        self.rs = jnp.array([0] + params[5::5])
        self.t0s = jnp.array([0] + params[6::5])
        self.params = params

    def _make_mask(self):
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[self.n_images*pb]
            stop = self.indices[self.n_images*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask
        
    def _time_shift(self, x, delta):
        return x - delta

    def _lens(self, x, beta0, beta1, r, t0):
        denom = 1 + jnp.exp(-r * (x-t0) )
        return beta0 + (beta1/denom)

    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        r_vector = jnp.repeat(jnp.tile(self.rs, self.n_bands), self.repeats)
        t0_vector = jnp.repeat(jnp.tile(self.t0s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._lens(x, beta0_vector, beta1_vector, r_vector, t0_vector)

        if x_prime != None:
            x_prime = self._time_shift(x_prime, delta_vector)
            b_prime = self._lens(x_prime, beta0_vector, beta1_vector, r_vector, t0_vector)
        else:
            x_prime = x
            b_prime = b

        K = jnp.outer(b, b_prime) * self.A**2 * jnp.exp(-(x[:, None] - x_prime[None, :])**2/(2*self.tau**2))
        return jnp.multiply(self.mask, K)
    
    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class PolynomialLensingKernel:
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::7])
        self.beta0s = jnp.array([1] + params[3::7])
        self.beta1s = jnp.array([0] + params[4::7])
        self.beta2s = jnp.array([0] + params[5::7])
        self.beta3s = jnp.array([0] + params[6::7])
        self.ls = jnp.array([0] + params[7::7])
        self.t0s = jnp.array([0] + params[8::7])
        self.params = params
        self.scale = [0.5, 5]
        self.covariance = jax.jit(self._covariance)

    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::7])
        self.beta0s = jnp.array([1] + params[3::7])
        self.beta1s = jnp.array([0] + params[4::7])
        self.beta2s = jnp.array([0] + params[5::7])
        self.beta3s = jnp.array([0] + params[6::7])
        self.ls = jnp.array([0] + params[7::7])
        self.t0s = jnp.array([0] + params[8::7])
        self.params = params

    def _make_mask(self):
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[self.n_images*pb]
            stop = self.indices[self.n_images*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask
        
    def _time_shift(self, x, delta):
        return x - delta

    def _lens(self, x, beta0, beta1, beta2, beta3, l, t0):
        b = beta0 + (beta1 * ((x-t0)/l)) + (beta2 * ((x-t0)/l)**2) + (beta3 * ((x-t0)/l)**3)
        return 10**-2 * b

    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        beta2_vector = jnp.repeat(jnp.tile(self.beta2s, self.n_bands), self.repeats)
        beta3_vector = jnp.repeat(jnp.tile(self.beta3s, self.n_bands), self.repeats)
        l_vector = jnp.repeat(jnp.tile(self.ls, self.n_bands), self.repeats)
        t0_vector = jnp.repeat(jnp.tile(self.t0s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._lens(x, beta0_vector, beta1_vector, beta2_vector, beta3_vector, l_vector, t0_vector)

        if x_prime != None:
            x_prime = self._time_shift(x_prime, delta_vector)
            b_prime = self._lens(x_prime, beta0_vector, beta1_vector, beta2_vector, beta3_vector, l_vector, t0_vector)
        else:
            x_prime = x
            b_prime = b

        K = jnp.outer(b, b_prime) * self.A**2 * jnp.exp(-(x[:, None] - x_prime[None, :])**2/(2*self.tau**2))
        return jnp.multiply(self.mask, K)
    
    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class SinLensingKernel:
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::5])
        self.beta0s = jnp.array([1] + params[3::5])
        self.beta1s = jnp.array([0] + params[4::5])
        self.ws = jnp.array([1] + params[5::5])
        self.t0s = jnp.array([0] + params[6::5])
        self.params = params
        self.covariance = jax.jit(self._covariance)

    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.deltas = jnp.array([0] + params[2::5])
        self.beta0s = jnp.array([1] + params[3::5])
        self.beta1s = jnp.array([0] + params[4::5])
        self.ws = jnp.array([1] + params[5::5])
        self.t0s = jnp.array([0] + params[6::5])
        self.params = params

    def _make_mask(self):
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[self.n_images*pb]
            stop = self.indices[self.n_images*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask
        
    def _time_shift(self, x, delta):
        return x - delta

    def _lens(self, x, beta0, beta1, w, t0):
        return beta0 + beta1*jnp.sin((x-t0)/w)

    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        w_vector = jnp.repeat(jnp.tile(self.ws, self.n_bands), self.repeats)
        t0_vector = jnp.repeat(jnp.tile(self.t0s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._lens(x, beta0_vector, beta1_vector, w_vector, t0_vector)

        if x_prime != None:
            x_prime = self._time_shift(x_prime, delta_vector)
            b_prime = self._lens(x_prime, beta0_vector, beta1_vector, w_vector, t0_vector)
        else:
            x_prime = x
            b_prime = b

        K = jnp.outer(b, b_prime) * self.A**2 * jnp.exp(-(x[:, None] - x_prime[None, :])**2/(2*self.tau**2))
        return jnp.multiply(self.mask, K)
    
    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class ExpSquaredKernel:
    """
    Moving kernel defined as A^2 * exp(-(y-y')^2 / (2*tau^2)).
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        self.scale = [0.5, 5]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime is None:
            x_prime = x

        K = self.A**2 * jnp.exp(-(x[:, None] - x_prime[None, :])**2/(2*self.tau**2))
        return K

    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
    
class ExponentialKernel:
    """
    Moving kernel defined as A^2 * exp(-|y-y'| / tau).
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        self.scale = [0.5, 5]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        K = self.A**2 * jnp.exp(-vector_mag/self.tau)
        return K

class ConstantKernel:
    def __init__(self, params):
        self.c = params[0]
        self.params = params
        self.scale = [1]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.c = params[0]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        K = jnp.ones([len(x), len(x_prime)]) * self.c
        return K

class DotProductKernel:
    def __init__(self):
        self.c = 0
        self.scale = [1]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self):
        self.c = 0
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        K = jnp.dot(x[:, None], x_prime[None, :])
        return K

class Matern32Kernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.params = params
        self.scale = [0.5, 5]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x
        
        r2 = (x[:, None] - x_prime[None, :])**2
        K = self.A * ((1 + jnp.sqrt(3*r2)/self.l) * jnp.exp(-jnp.sqrt(3*r2)/self.l))
        return K

class Matern52Kernel:
    """
    Stationary kernel defined as K = (1 + sqrt(5)d/length_scale + (5/3)d^2/length_scale^2) * exp(sqrt(5)d^2/length_scale).
    """
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.params = params
        self.scale = [0.5, 5]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x
        
        r2 = (x[:, None] - x_prime[None, :])**2
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
        self.params = params
        self.scale = [0.5, 5, 5]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.scale_mixture = params[2]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        K = self.A**2 * (1 + (x[:, None] - x_prime[None, :])**2/(2*self.scale_mixture*self.tau**2))
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
        self.params = params
        self.scale = [0.5, 5, 2, 5, 2]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.lamdba = params[1]
        self.p = params[2]
        self.mu = params[3]
        self.sigma = params[4]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        normal_x = jnp.exp(-(x[:, None] - self.mu)**2 / (2*(self.sigma**2))) / (self.sigma * jnp.sqrt(2*jnp.pi))
        normal_xprime = jnp.exp(-(x_prime[None, :] - self.mu)**2 / (2*(self.sigma**2))) / (self.sigma * jnp.sqrt(2*jnp.pi))
        
        tau_x = self.lamdba * (1 - (self.p * normal_x))
        tau_xprime = self.lamdba * (1 - (self.p * normal_xprime))
        
        root_num = 2 * tau_x * tau_xprime
        exp_num = (x[:, None] - x_prime[None, :])**2
        denom = (tau_x**2) + (tau_xprime**2)
        K = self.A**2 * jnp.sqrt(root_num/denom) * jnp.exp(-exp_num/denom)
        return K
    
class OUKernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.params = params
        self.scale = [0.5, 5]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        K = self.A * jnp.exp(jnp.sqrt(jnp.sum(x[:, None] - x_prime[None, :])) / self.l)
        return K

class PeriodicKernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.p = params[2]
        self.params = params
        self.scale = [0]
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        self.p = params[2]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        sin = vector_mag / self.p
        exponent = -2 * ( jnp.sin(jnp.pi * sin) / self.l)**2
        K = self.A * jnp.exp(exponent)
        return K

class JJKernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.m = params[2]
        self.b = params[3]
        self.params = params
        self.scale = []
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        self.m = params[2]
        self.b = params[3]
        self.params = params

    def _p(self, x):
        return self.m * x + self.b
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        sin = vector_mag / self._p(x)
        exponent = -2 * ( jnp.sin(jnp.pi * sin) / self.l)**2
        K = self.A * jnp.exp(exponent)
        return K
        
