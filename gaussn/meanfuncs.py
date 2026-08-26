from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp


class BaseMeanFunction(ABC):
    """
    Parent kernel
    """
    param_names: list = []

    def __init__(self, params):
        self.params = params

    def _unpack(self, params):
        """Pure: flat params list -> dict of named kwargs for _mean_fn. No mutation."""
        return dict(zip(self.param_names, params))
    
    @abstractmethod
    def _mean_fn(self, x, bands, images, zp, zpsys, **params):
        """Pure mean computation. The only thing each mean function needs to define."""
        raise NotImplementedError
    
    def mean(self, x, params=None, bands=None, images=None, zp=None, zpsys=None):
        if params is None:
            params = self.params
        return self._mean_fn(x, bands=bands, images=images, zp=zp, zpsys=zpsys, **self._unpack(params))
    

class UniformMean(BaseMeanFunction):
    """
    Uniform Mean function for Gaussian processes.
    """
    param_names = ['c']

    def _mean_fn(self, x, bands, images, zp, zpsys, c):
        return jnp.full(x.shape, c)


class Sin(BaseMeanFunction):
    """
    Sinusoidal mean function for Gaussian processes.
    """
    param_names = ['A', 'w', 'phi']
        
    def _mean_fn(self, x, bands, images, zp, zpsys, A, w, phi):
        return A * jnp.sin((x*w) + phi)
    

class Gaussian(BaseMeanFunction):
    """
    Gaussian mean function for Gaussian processes.
    """
    param_names = ['A', 'mu', 'sigma']
 
    def _mean_fn(self, x, bands, images, zp, zpsys, A, mu, sigma):
        exponent = -(x - mu)**2 / (2 * sigma**2)
        return A * jnp.exp(exponent) / (sigma * jnp.sqrt(2*jnp.pi))
    

class ExpFunction(BaseMeanFunction):
    """
    Exponential mean function for Gaussian processes.
    """
    param_names = ['A', 'tau']
        
    def _mean_fn(self, x, bands, images, zp, zpsys, A, tau):
        return A * jnp.exp(x*tau)


class Bazin2009(BaseMeanFunction):
    """
    Bazin (2009) mean function for Gaussian processes.
    """
    param_names = ['A', 'beta', 't0', 'Tfall', 'Trise']
    
    def _mean_fn(self, x, bands, images, zp, zpsys, A, beta, t0, Tfall, Trise):
        a = jnp.exp(-(x - t0)/Tfall)
        b = 1 + jnp.exp((x - t0)/Trise)
        mu = A * (a/b) + beta
        return mu


class Karpenka2012(BaseMeanFunction):
    """
    Karpenka (2012) mean function for Gaussian processes.
    """
    param_names = ['A', 'B', 't1', 't0', 'Tfall', 'Trise']
    
    def _mean_fn(self, x, bands, images, zp, zpsys, A, B, t1, t0, Tfall, Trise):
        a = 1 + (B * ((x - t1)**2))
        b = jnp.exp(-(x - t0)/Tfall)
        c = 1 + jnp.exp(-(x - t0)/Trise)
        mu = (A * a * (b/c))
        return mu


class Villar2019(BaseMeanFunction):
    """
    Villar (2019) mean function for Gaussian processes.
    """
    param_names = ['A', 'beta', 't1', 't0', 'Tfall', 'Trise']
        
    def _mean_fn(self, x, bands, images, zp, zpsys, A, beta, t1, t0, Tfall, Trise):
        denom = 1 + jnp.exp(-(x - t0)/Trise)
        constant = A + (beta * (t1 - t0))
    
        for i in range(len(x)):
            if x[i] < t1:
                a = A + (beta * (x[i] - t0))
                mu[i] = a
            else:
                b = jnp.exp(-(x[i] - t1)/Tfall)
                mu[i] = (constant*b)
                
        mu = (mu/denom)
        return mu


class sncosmoMean:
    """
    Mean function for Gaussian processes based on sncosmo templates.
    """
    def __init__(self, model, fixed=None):
        """
        Initializes the sncosmoMean function with given parameters.

        Args:
            model (sncosmo.Model instance): sncosmo model
            fixed (array-like, optional): list of True/False indicating whether to fix each of the sncosmo model parameters, in the order given by model.param_names; defaults to fitting all parameters
        """
        self.model = model
        if fixed is None:
            self.fixed = np.repeat(False, len(self.model.parameters))
        elif len(fixed) != len(self.model.parameters):
            raise IndexError("Array 'fixed' is not the correct length for given model. Please check the length of 'fixed' is equivalent to the length of model.param_names!")
        else:
            self.fixed = np.array(fixed)
        self.params = np.array(self.model.parameters)[~self.fixed]

    def _reset(self, params):
        """
        Resets the mean function parameters.

        Args:
            params (list): List containing parameters [redshift, t0, amp] or [t0, amp].
        """
        self.params = params
        params_dict = {np.array(self.model.param_names)[~self.fixed][k]: params[k] for k in range(len(self.params))}
        try:
            params_dict['x0'] = params_dict['x0']*1.e-8
        except:
            pass
        self.model.update(params_dict)

    def mean(self, x, params=None, bands=None, images=None, zp=27.5, zpsys='ab'):
        """
        Computes the mean flux using the sncosmo model.

        Args:
            x (numpy.ndarray): Input array of times.
            params (list, optional): List containing parameters [redshift, t0, amp] or [t0, amp].
                Defaults to None.
            bands (list, optional): List of bands. Defaults to None.

        Returns:
            numpy.ndarray: Mean flux computed using the sncosmo model.
        """
        if params != None:
            self._reset(params)

        args = np.argsort(x)
        revert_args = np.zeros(len(args), dtype=int)
        revert_args[args] = np.arange(len(args))

        reordered_x = x[args]
        if len(bands) >= 2:
            reordered_bands = bands[args]
        else:
            reordered_bands = bands

        if len(zp) >= 2:
            reordered_zp = zp[args]
        else:
            reordered_zp = zp

        if len(zpsys) >= 2:
            reordered_zpsys = zpsys[args]
        else:
            reordered_zpsys = zpsys

        flux = self.model.bandflux(reordered_bands, reordered_x, zp=reordered_zp, zpsys=reordered_zpsys)

        return flux[revert_args]

