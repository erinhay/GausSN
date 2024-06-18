import numpy as np
import jax.numpy as jnp
import jax
from scipy.linalg import block_diag

class NoLensing:
    """
    NoLensing treatment for when using the Gaussian Process for cases outside of strong lensing/time-delay cosmography.
    """
    def __init__(self):
        self.mask = 1

    def lens(self, x, params=None):
        return x, 1

    def import_from_gp(self, kernel, meanfunc, bands=None, images=None, indices=None, repeats=None):
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands

class ConstantMagnification:
    """
    The constant magnification treatment for time delay estimation.
    """
    def __init__(self, params):
        """
        Initializes the ConstantMagnification class. There should be (N-1) time delays (delta) and magnifications (beta) for N images, inputted as [delta_1, beta_1, delta_2, beta_2, ...].

        Args:
            params (list): List of parameters for the ConstantMagnification class.
                delta (float): time delay.
                beta (float): magnification.
        """
        self.deltas = jnp.array([0] + params[0::2])
        self.betas = jnp.array([1] + params[1::2])
        self.params = params
        self.scale = [1]
        
    def _reset(self, params):
        self.deltas = jnp.array([0] + params[0::2])
        self.betas = jnp.array([1] + params[1::2])
        self.params = params
        
    def _make_mask(self):
        """
        Creates a mask to ensure each band in treated independently based on the indices.

        Returns:
            numpy.ndarray: Mask matrix.
        """
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[(self.n_images+1)*pb]
            stop = self.indices[(self.n_images+1)*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask

    def _time_shift(self, x, delta):
        """
        Shifts the input data in time.

        Args:
            x (numpy.ndarray): Input data.
            delta (numpy.ndarray): Time shift values.

        Returns:
            numpy.ndarray: Shifted input data.
        """
        return x - delta

    def _magnify(self, x, beta):
        """
        Applies magnification to the input data.

        Args:
            x (numpy.ndarray): Input data.
            beta (numpy.ndarray): Magnification values.

        Returns:
            numpy.ndarray: Magnified input data.
        """
        return beta

    def lens(self, x, params=None):
        """
        Applies the constant magnification effect.

        Args:
            x (numpy.ndarray): Input data.
            params: Not used in this function.

        Returns:
            new_x: time-shifted times of observations
            b: magnification of the data at times new_x
        """
        if params != None:
            self._reset(params)

        resolved_delta_vector = jnp.repeat(jnp.repeat(self.deltas, self.n_bands), self.repeats)
        resolved_beta_vector = jnp.repeat(jnp.repeat(self.betas, self.n_bands), self.repeats)

        unresolved_x = x[self.images == 'unresolved']
        unresolved_delta_vector = jnp.repeat(self.deltas, len(unresolved_x))
        unresolved_beta_vector = jnp.repeat(self.betas, len(unresolved_x))

        new_resolved_x = self._time_shift(x[self.images != 'unresolved'], resolved_delta_vector)
        new_unresolved_x = self._time_shift(jnp.repeat(unresolved_x, self.n_images), unresolved_delta_vector)
        new_x = jnp.concatenate([new_resolved_x, new_unresolved_x])

        resolved_b = self._magnify(new_resolved_x, resolved_beta_vector)
        unresolved_b = self._magnify(new_unresolved_x, unresolved_beta_vector)
        for m in range(self.n_images):
            if m == 0:
                unresolved_T = jnp.diag(unresolved_b[m*len(unresolved_x) : (m+1)*len(unresolved_x)])
            else:
                unresolved_T = jnp.hstack([unresolved_T, jnp.diag(unresolved_b[m*len(unresolved_x) : (m+1)*len(unresolved_x)])])
        T = block_diag(jnp.outer(resolved_b, resolved_b), unresolved_T)

        return new_x, T

    def import_from_gp(self, kernel, meanfunc, bands, images, indices, repeats):
        """
        Imports parameters from Gaussian process.

        Args:
            n_bands (int): Number of bands.
            n_images (int): Number of images.
            indices (numpy.ndarray): Indices.

        Returns:
            None
        """
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands
        self.n_bands = len(np.unique(bands))
        self.images = images
        self.n_images = len(np.unique(images[images != 'unresolved']))
        self.indices = indices
        self.repeats = repeats
        self.mask = self._make_mask()
    
class SigmoidMagnification:
    """
    The sigmoid magnification treatment for time delay estimation. There should be (N-1) parameters for N images, inputted as [delta_1, beta0_1, beta1_1, r_1, t0_1, delta_2, beta0_2, beta1_2, r_2, t0_2, ...].
    """
    def __init__(self, params):
        """
        Initializes the SigmoidMagnification class.

        Args:
            params (list): List of parameters for the SigmoidMagnification class.
                delta (float): time delay.
                beta0 (float): magnification before t --> - infinity (well before t0).
                beta1 (float): magnification as t --> infinity (well after t0).
                r (float): rate of change from beta0 to beta1.
                t0 (float): centering of the sigmoid magnification effect.
        """
        self.deltas = jnp.array([0] + params[5::5])
        self.beta0s = jnp.array(params[1::5])
        self.beta1s = jnp.array(params[2::5])
        self.rs = jnp.array(params[3::5])
        self.t0s = jnp.array(params[4::5])
        self.params = params
        
    def _reset(self, params):
        """
        Resets the parameters of the sigmoid magnification model.

        Args:
            params (list): List of parameters for the sigmoid magnification model.
        """
        self.deltas = jnp.array([0] + params[5::5])
        self.beta0s = jnp.array(params[1::5])
        self.beta1s = jnp.array(params[2::5])
        self.rs = jnp.array(params[3::5])
        self.t0s = jnp.array(params[4::5])
        self.params = params

    def _make_mask(self):
        """
        Creates a mask to ensure each band in treated independently based on the indices.

        Returns:
            numpy.ndarray: Mask matrix.
        """
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[self.n_images*pb]
            stop = self.indices[self.n_images*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask
        
    def _time_shift(self, x, delta):
        """
        Shifts the input data in time.

        Args:
            x (numpy.ndarray): Input data.
            delta (numpy.ndarray): Time shift values.

        Returns:
            numpy.ndarray: Shifted input data.
        """
        return x - delta

    def _magnify(self, x, beta0, beta1, r, t0):
        """
        Applies magnification to the input data.

        Args:
            x (numpy.ndarray): Input data.
            beta (numpy.ndarray): Magnification values.

        Returns:
            numpy.ndarray: Magnified input data.
        """
        denom = 1 + jnp.exp(-r * (x-t0) )
        return beta0 + (beta1/denom)

    def lens(self, x, params=None):
        """
        Applies the sigmoid magnification effect.

        Args:
            x (numpy.ndarray): Input data.
            params: Not used in this function.

        Returns:
            new_x: time-shifted times of observations
            b: magnification of the data at times new_x
        """
        if params != None:
            self._reset(params)

        delta_vector = jnp.tile(jnp.repeat(self.deltas, self.n_images), self.repeats)
        beta0_vector = jnp.tile(jnp.repeat(self.beta0s, self.n_images), self.repeats)
        beta1_vector = jnp.tile(jnp.repeat(self.beta1s, self.n_images), self.repeats)
        r_vector = jnp.tile(jnp.repeat(self.rs, self.n_images), self.repeats)
        t0_vector = jnp.tile(jnp.repeat(self.t0s, self.n_images), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta0_vector, beta1_vector, r_vector, t0_vector)

        return x, b

    def import_from_gp(self, kernel, meanfunc, bands, images, indices, repeats):
        """
        Imports parameters from Gaussian process.

        Args:
            n_bands (int): Number of bands.
            n_images (int): Number of images.
            indices (numpy.ndarray): Indices.

        Returns:
            None
        """
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands
        self.n_bands = len(np.unique(bands))
        self.images = images
        self.n_images = len(np.unique(images))
        self.indices = indices
        self.repeats = repeats
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

    def lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.tile(jnp.repeat(self.deltas, self.n_images), self.repeats)
        beta_vector = jnp.tile(self.betas, self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta_vector)

        return x, b
    
    def import_from_gp(self, kernel, meanfunc, bands, images, indices, repeats):
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands
        self.n_bands = len(np.unique(bands))
        self.images = images
        self.n_images = len(np.unique(images))
        self.indices = indices
        self.repeats = repeats
        self.mask = self._make_mask()

