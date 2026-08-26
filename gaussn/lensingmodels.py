from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.linalg import block_diag

class NoLensing:
    """
    NoLensing treatment for when using the Gaussian Process for cases outside of strong lensing/time-delay cosmography.
    """
    def __init__(self, params):
        self.mask = 1

    def _lens(self, x, params):
        return x, 1


class BaseLensingModel(ABC):
    """
    Parent class for lensing/time-delay models used in GausSN.
 
    Subclasses must implement `_unpack_params` and `_magnify`. Everything else
    (parameter bookkeeping, time-shifting, masking, and the resolved/unresolved
    lensing transform) is inherited.
    """
 
    def __init__(self, params):
        """
        Args:
            params (list): Flat list of model parameters, ordered per-image
                (e.g. [delta_1, beta_1, delta_2, beta_2, ...] for N images).
        """
        self.params = params
 
    @abstractmethod
    def _unpack_params(self, params):
        """
        Parse the flat `params` list into named jnp arrays, one per image.
 
        Must set:
            self.deltas          - time delays, with the reference image's value
                                    (0) prepended, e.g. jnp.array([0] + params[0::k])
            self.<...>           - any magnification-related arrays, each with the
                                    reference image's value prepended
            self.mag_param_names - list of the attribute names above (as strings,
                                    in the order _magnify expects them), e.g.
                                    ['beta0s', 'beta1s', 'rs', 't0s']
        """
        raise NotImplementedError
 
    @abstractmethod
    def _magnify(self, x, *mag_params):
        """
        Compute the magnification at (already time-shifted) times x.
 
        Args:
            x: shifted times.
            *mag_params: one jnp array per name in self.mag_param_names, in order.
 
        Returns:
            jnp.ndarray: magnification b(x).
        """
        raise NotImplementedError
 
    def _time_shift(self, x, delta):
        """Shifts input times by the per-image time delay. Identical for every model."""
        return x - delta
 
    def _gather_vectors(self, names, n_repeats, tile_bands=True):
        """Builds the per-observation parameter vectors for a given set of attribute names."""
        vectors = []
        for name in names:
            arr = getattr(self, name)
            if tile_bands:
                arr = jnp.repeat(arr, self.n_bands)
            vectors.append(jnp.repeat(arr, n_repeats))
        return vectors
 
    def _lens(self, x, params=None):
        """
        Applies the time shift + magnification, handling resolved and unresolved
        observations together and assembling the block-diagonal transform matrix.
        Identical for every model; only the parameter names being gathered change.
 
        Returns:
            shifted_x: time-shifted times of observations
            T: transform (diagonal for resolved-only data, or a mixed
               diagonal/block matrix when unresolved data is present)
        """
        if params is not None:
            self._reset(params)
 
        resolved_b = None
        unresolved_T = None
        shifted_resolved_x = shifted_unresolved_x = None
 
        if len(self.repeats) > 0:
            resolved_x = x[self.images != 'unresolved']
            resolved_delta_vector = self._gather_vectors(['deltas'], self.repeats, tile_bands=True)[0]
            resolved_mag_vectors = self._gather_vectors(self.mag_param_names, self.repeats, tile_bands=True)
 
            shifted_resolved_x = self._time_shift(resolved_x, resolved_delta_vector)
            resolved_b = self._magnify(shifted_resolved_x, *resolved_mag_vectors)
 
        if 'unresolved' in self.images:
            unresolved_x = x[self.images == 'unresolved']
            n_unresolved = len(unresolved_x)
            unresolved_delta_vector = self._gather_vectors(['deltas'], n_unresolved, tile_bands=False)[0]
            unresolved_mag_vectors = self._gather_vectors(self.mag_param_names, n_unresolved, tile_bands=False)
 
            shifted_unresolved_x = self._time_shift(jnp.tile(unresolved_x, self.n_images), unresolved_delta_vector)
            unresolved_b = self._magnify(shifted_unresolved_x, *unresolved_mag_vectors)
 
            for m in range(self.n_images):
                block = jnp.diag(unresolved_b[m * n_unresolved: (m + 1) * n_unresolved])
                unresolved_T = block if m == 0 else jnp.hstack([unresolved_T, block])
 
        if resolved_b is not None and unresolved_T is not None:
            shifted_x = jnp.concatenate([shifted_resolved_x, shifted_unresolved_x])
            T = block_diag(jnp.diag(resolved_b), unresolved_T)
        elif resolved_b is not None:
            shifted_x = shifted_resolved_x
            T = jnp.diag(resolved_b)
        else:
            shifted_x = shifted_unresolved_x
            T = unresolved_T
 
        return shifted_x, T

    def make_mask(self, bands, images):
        """Block mask marking which observations belong to the same image. Identical for every model."""
        mask = np.zeros((len(bands), len(bands)))
        for i in range(len(bands)):
            for j in range(len(bands)):
                if bands[i] == bands[j]:
                    mask[i, j] = 1
        return mask
 
    def import_from_gp(self, bands, images, n_images, indices, repeats):
        """Imports shared state from the parent GP object. Identical for every model."""
        self.bands = bands
        self.n_bands = len(np.unique(self.bands))
        self.images = images
        self.n_images = n_images
        self.indices = indices
        self.repeats = repeats


class ConstantMagnification(BaseLensingModel):
    """
    Constant magnification treatment for time delay estimation.
    (N-1) [delta, beta] pairs for N images: [delta_1, beta_1, delta_2, beta_2, ...].
    """
 
    def _unpack_params(self, params):
        self.deltas = jnp.array([0] + params[0::2])
        self.betas = jnp.array([1] + params[1::2])
        self.mag_param_names = ['betas']
 
    def _magnify(self, x, beta):
        return beta


class SigmoidMagnification(BaseLensingModel):
    """
    Sigmoid magnification treatment for time delay estimation.
    (N-1) [delta, beta0, beta1, r, t0] tuples for N images.
    """
 
    def _unpack_params(self, params):
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.rs = jnp.array([0] + params[3::5])
        self.t0s = jnp.array([0] + params[4::5])
        self.mag_param_names = ['beta0s', 'beta1s', 'rs', 't0s']
 
    def _magnify(self, x, beta0, beta1, r, t0):
        denom = 1 + jnp.exp(-r * (x - t0))
        return beta0 + (beta1 / denom)
    

class SinusoidalMagnification(BaseLensingModel):
    """
    Sinusoidal magnification treatment for time delay estimation.
    (N-1) [delta, beta0, beta1, t0, T] tuples for N images.
    """
 
    def _unpack_params(self, params):
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.t0s = jnp.array([0] + params[3::5])
        self.Ts = jnp.array([0] + params[4::5])
        self.mag_param_names = ['beta0s', 'beta1s', 't0s', 'Ts']
 
    def _magnify(self, x, beta0, beta1, t0, T):
        sinusoid = jnp.sin((x - t0) / T)
        return beta0 + (beta1 * sinusoid)


