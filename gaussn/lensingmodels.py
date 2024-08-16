import numpy as np
import jax.numpy as jnp
import jax

class NoLensing:
    """
    NoLensing treatment for when using the Gaussian Process for cases outside of strong lensing/time-delay cosmography.
    """
    def __init__(self, params=None):
        self.mask = 1
        self.lens = jax.jit(self._lens) #jax.jit(self._lens) self._lens

    def _lens(self, x, params=None):
        return x, 1

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
        self.lens = jax.jit(self._lens) #jax.jit(self._lens) self._lens
        
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

    def _lens(self, x, params=None):
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

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta_vector = jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats)

        new_x = self._time_shift(x, delta_vector)
        b = self._magnify(new_x, beta_vector)

        return new_x, b

    def import_from_gp(self, n_bands, n_images, indices):
        """
        Imports parameters from Gaussian process.

        Args:
            n_bands (int): Number of bands.
            n_images (int): Number of images.
            indices (numpy.ndarray): Indices.

        Returns:
            None
        """
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
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
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.rs = jnp.array([0] + params[3::5])
        self.t0s = jnp.array([0] + params[4::5])
        self.params = params
        self.lens = jax.jit(self._lens) #jax.jit(self._lens) self._lens
        
    def _reset(self, params):
        """
        Resets the parameters of the sigmoid magnification model.

        Args:
            params (list): List of parameters for the sigmoid magnification model.
        """
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.rs = jnp.array([0] + params[3::5])
        self.t0s = jnp.array([0] + params[4::5])
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

    def _lens(self, x, params=None):
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

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        r_vector = jnp.repeat(jnp.tile(self.rs, self.n_bands), self.repeats)
        t0_vector = jnp.repeat(jnp.tile(self.t0s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta0_vector, beta1_vector, r_vector, t0_vector)

        return x, b

    def import_from_gp(self, n_bands, n_images, indices):
        """
        Imports parameters from Gaussian process.

        Args:
            n_bands (int): Number of bands.
            n_images (int): Number of images.
            indices (numpy.ndarray): Indices.

        Returns:
            None
        """
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class SinusoidalMagnification:
    """
    The sigmoid magnification treatment for time delay estimation. There should be (N-1) parameters for N images, inputted as [delta_1, beta0_1, beta1_1, t0_1, T_1, delta_2, beta0_2, beta1_2, t0_2, T_2, ...].
    """
    def __init__(self, params):
        """
        Initializes the SigmoidMagnification class.

        Args:
            params (list): List of parameters for the SigmoidMagnification class.
                delta (float): time delay.
                beta0 (float): magnification before t --> - infinity (well before t0).
                beta1 (float): magnification as t --> infinity (well after t0).
                t0 (float): centering of the sinusoidal magnification effect.
                T (float): timescale of the sinusoidal magnification effect.
        """
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.t0s = jnp.array([0] + params[3::5])
        self.Ts = jnp.array([0] + params[4::5])
        self.params = params
        self.lens = jax.jit(self._lens) #jax.jit(self._lens) self._lens
        
    def _reset(self, params):
        """
        Resets the parameters of the sigmoid magnification model.

        Args:
            params (list): List of parameters for the sigmoid magnification model.
        """
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.t0s = jnp.array([0] + params[3::5])
        self.Ts = jnp.array([0] + params[4::5])
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

    def _magnify(self, x, beta0, beta1, t0, T):
        """
        Applies magnification to the input data.

        Args:
            x (numpy.ndarray): Input data.
            beta (numpy.ndarray): Magnification values.

        Returns:
            numpy.ndarray: Magnified input data.
        """
        sinusoid = jnp.sin( (x-t0) / T)
        return beta0 + (beta1 * sinusoid)

    def _lens(self, x, params=None):
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

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        t0_vector = jnp.repeat(jnp.tile(self.t0s, self.n_bands), self.repeats)
        T_vector = jnp.repeat(jnp.tile(self.Ts, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta0_vector, beta1_vector, t0_vector, T_vector)

        return x, b

    def import_from_gp(self, n_bands, n_images, indices):
        """
        Imports parameters from Gaussian process.

        Args:
            n_bands (int): Number of bands.
            n_images (int): Number of images.
            indices (numpy.ndarray): Indices.

        Returns:
            None
        """
        self.n_bands = n_bands
        self.n_images = n_images
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()


