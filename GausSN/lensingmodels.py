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

    def _magnify(self, x, beta):
        return beta

    def _lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta_vector = jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats)

        new_x = self._time_shift(x, delta_vector)
        b = self._magnify(new_x, beta_vector)

        return new_x, b

    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()
    
class SigmoidMicrolensing:
    def __init__(self, params):
        self.deltas = jnp.array([0] + params[5::5])
        self.beta0s = jnp.array(params[1::5])
        self.beta1s = jnp.array(params[2::5])
        self.rs = jnp.array(params[3::5])
        self.t0s = jnp.array(params[4::5])
        self.params = params
        self.lens = jax.jit(self._lens)
        
    def _reset(self, params):
        self.deltas = jnp.array([0] + params[5::5])
        self.beta0s = jnp.array(params[1::5])
        self.beta1s = jnp.array(params[2::5])
        self.rs = jnp.array(params[3::5])
        self.t0s = jnp.array(params[4::5])
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

    def _lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        r_vector = jnp.repeat(jnp.tile(self.rs, self.n_bands), self.repeats)
        t0_vector = jnp.repeat(jnp.tile(self.t0s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta0_vector, beta1_vector, r_vector, t0_vector)

        return x, b

    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class PolynomialLensingKernel:
    def __init__(self, params):
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.beta2s = jnp.array([0] + params[3::5])
        self.beta3s = jnp.array([0] + params[4::5])
        self.params = params
        self.scale = [0.5, 5]
        self.lens = jax.jit(self._lens)

    def _reset(self, params):
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.beta2s = jnp.array([0] + params[3::5])
        self.beta3s = jnp.array([0] + params[4::5])
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

    def _magnify(self, x, beta0, beta1, beta2, beta3):
        return beta0 + (beta1*x) + (beta2 * (x**2)) + (beta3 * (x**3))

    def _lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        beta2_vector = jnp.repeat(jnp.tile(self.beta2s, self.n_bands), self.repeats)
        beta3_vector = jnp.repeat(jnp.tile(self.beta3s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta0_vector, beta1_vector, beta2_vector, beta3_vector)
        return x, b

    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class FlexibleDust_ConstantLensingKernel:
    def __init__(self, params, n_bands):
        self.deltas = jnp.array([0] + params[0::2])
        self.betas_mask = jnp.isin(jnp.arange(len(params)), jnp.concatenate([jnp.array([0,1]), jnp.array(list(range(*slice(2,None,n_bands+1).indices(len(params)))))]), invert=True)
        self.betas = jnp.concatenate([jnp.repeat(1, n_bands), jnp.array(params)[self.betas_mask]])
        self.params = params
        self.scale = [0.5, 5]
    
    def _reset(self, params):
        self.deltas = jnp.array([0] + params[0::self.n_bands+1])
        self.betas = jnp.concatenate([jnp.repeat(1, self.n_bands), jnp.array(params)[self.betas_mask]])
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

    def _magnify(self, x, beta):
        return beta

    def _lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta_vector = jnp.repeat(self.betas, self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._lens(x, beta_vector)

        return x, b
    
    def import_from_gp(self, n_bands, n_images, indices):
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

# Spline ML model (think about locations of knots, number of knots)
# Mixture of Sigmoids
# Look at special classes of polynomials
# Numerical Recipes (spline and interpolation chapters)
