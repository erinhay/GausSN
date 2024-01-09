import numpy as np
import jax.numpy as jnp
import jax

class NoLensing:
    def __init__(self):
        self.mask = 1
        self.lens = jax.jit(self._lens)

    def _lens(self, x, params=None):
        return x, 1

class ConstantLensing:
    def __init__(self, params):
        self.deltas = jnp.array([0] + params[0::2])
        self.betas = jnp.array([1] + params[1::2])
        self.params = params
        self.scale = [1]
        self.lens = jax.jit(self._lens)
        
    def _reset(self, params):
        self.deltas = jnp.array([0] + params[0::2])
        self.betas = jnp.array([1] + params[1::2])
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

    def _magnify(self, y, beta):
        return beta

    def _lens(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta_vector = jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta_vector)

        if x_prime != None:
            b_prime = self._magnify(x_prime, 1)
        else:
            b_prime = b

        #b_matrix = jnp.outer(b, b_prime)
        return x, b

    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()
    
class SigmoidMicrolensing:
    def __init__(self, params):
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.rs = jnp.array([0] + params[3::5])
        self.t0s = jnp.array([0] + params[4::5])
        self.params = params
        self.lens = jax.jit(self._lens)
        
    def _reset(self, params):
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.rs = jnp.array([0] + params[3::5])
        self.t0s = jnp.array([0] + params[4::5])
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

    def _magnify(self, x, beta0, beta1, r, t0):
        denom = 1 + jnp.exp(-r * (x-t0) )
        return beta0 + (beta1/denom)

    def _lens(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        r_vector = jnp.repeat(jnp.tile(self.rs, self.n_bands), self.repeats)
        t0_vector = jnp.repeat(jnp.tile(self.t0s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta0_vector, beta1_vector, r_vector, t0_vector)

        if x_prime != None:
            b_prime = self._magnify(x_prime, 1, 0, 0, 0)
        else:
            b_prime = b

        #b_matrix = jnp.outer(b, b_prime)
        return x, b

    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()


