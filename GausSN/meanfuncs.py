import numpy as np
import jax.numpy as jnp
import sncosmo

class UniformMean:
    """
    Uniform Mean function for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Uniform Mean function with given parameters.

        Args:
            params (list): List containing parameters [c].
                c (float): Constant value of the mean function.
        """
        self.c = params[0]
        self.params = params
        self.scale = [1]
        
    def _reset(self, params):
        """
        Resets the mean function parameters.

        Args:
            params (list): List containing parameters [c].
                c (float): Constant value of the mean function.
        """
        self.c = params[0]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        """
        Computes the mean value using the Uniform Mean function.

        Args:
            y (jax.numpy.ndarray): Input data.
            params (list, optional): List containing parameters [c] to reset mean function parameters.
                Defaults to None.
            bands: Not used in this function.

        Returns:
            float: Mean value computed using the Uniform Mean function.
        """
        if params != None:
            self._reset(params)
        return jnp.repeat(self.c, len(y))

class sncosmoMean:
    """
    Mean function for Gaussian processes based on sncosmo templates. This class works for all sncosmo templates which use the 'amp' and 't0' parameters.
    """
    def __init__(self, templatename, params, redshift=None):
        """
        Initializes the sncosmoMean function with given parameters.

        Args:
            templatename (str): Name of the sncosmo template.
            params (list): List containing parameters [redshift, t0, amp] or [t0, amp].
                redshift (float, optional): Redshift of the source. Defaults to None.
        """
        self.templatename = templatename
        self.params = params
        if len(params) > 2:
            self.redshift = params[0]
            self.t0 = params[1]
            self.amp = params[2]
        else:
            self.redshift = redshift
            self.t0 = params[0]
            self.amp = params[1]
        self.model = sncosmo.Model(source=self.templatename)
        self.model.set(z=self.redshift, t0=self.t0, amplitude=1.e-6*self.amp)

    def _reset(self, params):
        """
        Resets the mean function parameters.

        Args:
            params (list): List containing parameters [redshift, t0, amp] or [t0, amp].
        """
        self.params = params
        if len(params) > 2:
            self.redshift = params[0]
            self.t0 = params[1]
            self.amp = params[2]
        else:
            self.t0 = params[0]
            self.amp = params[1]
        self.model.set(z=self.redshift, t0=self.t0, amplitude=1.e-6*self.amp)

    def mean(self, x, params=None, bands=None):
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

        resolved_x = x[images != 'unresolved']
        resolved_bands = bands[images != 'unresolved']
        resolved_args = np.argsort(resolved_x)
        resolved_revert_args = np.zeros(len(resolved_args), dtype=int)
        resolved_revert_args[resolved_args] = np.arange(len(resolved_args))

        resolved_reordered_x = resolved_x[resolved_args]
        if len(resolved_bands) >= 2:
            resolved_reordered_bands = resolved_bands[resolved_args]
        else:
            resolved_reordered_bands = resolved_bands

        resolved_flux = self.model.bandflux(resolved_reordered_bands, resolved_reordered_x)

        unresolved_x = np.tile(x[images == 'unresolved'], np.unique(images[images != 'unresolved']))
        unresolved_bands = np.tile(bands[images == 'unresolved'], np.unique(images[images != 'unresolved']))
        unresolved_args = np.argsort(unresolved_x)
        unresolved_revert_args = np.zeros(len(unresolved_args), dtype=int)
        unresolved_revert_args[resolved_args] = np.arange(len(unresolved_args))

        unresolved_reordered_x = unresolved_x[unresolved_args]
        if len(unresolved_bands) >= 2:
            unresolved_reordered_bands = unresolved_bands[unresolved_args]
        else:
            unresolved_reordered_bands = unresolved_bands

        unresolved_flux = self.model.bandflux(unresolved_reordered_bands, unresolved_reordered_x)

        flux = jnp.concatenate([resolved_flux[resolved_revert_args], unresolved_flux[unresolved_revert_args]])
        return flux

class SALTMean:
    """
    Mean function for Gaussian processes based on SALT template, as implemented through sncosmo.
    """
    def __init__(self, templatename, params, redshift=None):
        """
        Initializes the SALTMean function with given parameters.

        Args:
            templatename (str): Name of the SALT template.
            params (list): List containing parameters [redshift, t0, x0, x1, c] or [t0, x0, x1, c].
                redshift (float, optional): Redshift of the source. Defaults to None.
        """
        self.templatename = templatename
        self.params = params
        if len(params) > 4:
            self.redshift = params[0]
            self.t0 = params[1]
            self.x0 = params[2]
            self.x1 = params[3]
            self.c = params[4]
        else:
            self.redshift = redshift
            self.t0 = params[0]
            self.x0 = params[1]
            self.x1 = params[2]
            self.c = params[3]
        self.model = sncosmo.Model(source=self.templatename)
        self.model.set(z=self.redshift, t0=self.t0, x0=5.e-3*self.x0, x1=self.x1, c=self.c)

    def _reset(self, params):
        """
        Resets the mean function parameters.

        Args:
            params (list): List containing parameters [redshift, t0, x0, x1, c] or [t0, x0, x1, c].
        """
        self.params = params
        if len(params) > 4:
            self.redshift = params[0]
            self.t0 = params[1]
            self.x0 = params[2]
            self.x1 = params[3]
            self.c = params[4]
        else:
            self.t0 = params[0]
            self.x0 = params[1]
            self.x1 = params[2]
            self.c = params[3]
        self.model.set(z=self.redshift, t0=self.t0, x0=5.e-3*self.x0, x1=self.x1, c=self.c)

    def mean(self, x, params=None, bands=None): #TODO: self.bands here?
        """
        Computes the mean flux using the SALT model.

        Args:
            x (numpy.ndarray): Input array of times.
            params (list, optional): List containing parameters [redshift, t0, x0, x1, c] or [t0, x0, x1, c].
                Defaults to None.
            bands (list, optional): List of bands. Defaults to None.

        Returns:
            numpy.ndarray: Mean flux computed using the SALT model.
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

        flux = self.model.bandflux(reordered_bands, reordered_x)

        return flux[revert_args]

class Sin:
    """
    Sinusoidal mean function for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Sinusoidal mean function with given parameters.

        Args:
            params (list): List containing parameters [A, w, phi].
                A (float): Amplitude parameter of the sinusoidal function.
                w (float): Frequency parameter of the sinusoidal function.
                phi (float): Phase parameter of the sinusoidal function.
        """
        self.A = params[0]
        self.w = params[1]
        self.phi = params[2]
        self.params = params
        self.scale = [0.25, 3, 10]
        
    def _reset(self, params):
        """
        Resets the sinusoidal mean function parameters.

        Args:
            params (list): List containing parameters [A, w, phi].
                A (float): Amplitude parameter of the sinusoidal function.
                w (float): Frequency parameter of the sinusoidal function.
                phi (float): Phase parameter of the sinusoidal function.
        """
        self.A = params[0]
        self.w = params[1]
        self.phi = params[2]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        """
        Computes the mean value using the sinusoidal mean function.

        Args:
            y (numpy.ndarray): Input data.
            params (list, optional): List containing parameters [A, w, phi] to reset mean function parameters.
                Defaults to None.
            bands: Not used in this function.

        Returns:
            numpy.ndarray: Mean value computed using the sinusoidal mean function.
        """
        if params != None:
            self._reset(params)
        return self.A * np.sin((y*self.w) + self.phi)
    
class Gaussian:
    """
    Gaussian mean function for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Gaussian mean function with given parameters.

        Args:
            params (list): List containing parameters [A, mu, sigma].
                A (float): Amplitude parameter of the Gaussian function.
                mu (float): Mean parameter of the Gaussian function.
                sigma (float): Standard deviation parameter of the Gaussian function.
        """
        self.A = params[0]
        self.mu = params[1]
        self.sigma = params[2]
        self.params = params
        self.scale = [0.25, 5, 0.5]
        
    def _reset(self, params):
        """
        Resets the Gaussian mean function parameters.

        Args:
            params (list): List containing parameters [A, mu, sigma].
                A (float): Amplitude parameter of the Gaussian function.
                mu (float): Mean parameter of the Gaussian function.
                sigma (float): Standard deviation parameter of the Gaussian function.
        """
        self.A = params[0]
        self.mu = params[1]
        self.sigma = params[2]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        """
        Computes the mean value using the Gaussian mean function.

        Args:
            y (numpy.ndarray): Input data.
            params (list, optional): List containing parameters [A, mu, sigma] to reset mean function parameters.
                Defaults to None.
            bands: Not used in this function.

        Returns:
            numpy.ndarray: Mean value computed using the Gaussian mean function.
        """
        if params != None:
            self._reset(params)
        exponent = -(y - self.mu)**2 / (2 * self.sigma**2)
        return self.A * jnp.exp(exponent) / (self.sigma * jnp.sqrt(2*jnp.pi))
    
class ExpFunction:
    """
    Exponential mean function for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Exponential mean function with given parameters.

        Args:
            params (list): List containing parameters [A, tau].
                A (float): Amplitude parameter of the exponential function.
                tau (float): Decay rate parameter of the exponential function.
        """
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        self.scale = [0.25, 5]

    def _reset(self, params):
        """
        Resets the Exponential mean function parameters.

        Args:
            params (list): List containing parameters [A, tau].
                A (float): Amplitude parameter of the exponential function.
                tau (float): Decay rate parameter of the exponential function.
        """
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        """
        Computes the mean value using the Exponential mean function.

        Args:
            y (numpy.ndarray): Input data.
            params (list, optional): List containing parameters [A, tau] to reset mean function parameters.
                Defaults to None.
            bands: Not used in this function.

        Returns:
            numpy.ndarray: Mean value computed using the Exponential mean function.
        """
        if params != None:
            self._reset(params)
        mu = np.zeros([len(y)])
        mu = self.A * np.exp(y*self.tau)
        return mu

class Bazin2009:
    """
    Bazin (2009) mean function for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Bazin (2009) mean function with given parameters.

        Args:
            params (list): List containing parameters [A, beta, t0, Tfall, Trise].
                A (float): Amplitude parameter of the Bazin function.
                beta (float): Beta parameter of the Bazin function.
                t0 (float): Time of maximum parameter of the Bazin function.
                Tfall (float): Fall timescale parameter of the Bazin function.
                Trise (float): Rise timescale parameter of the Bazin function.
        """
        self.A = params[0]
        self.beta = params[1]
        self.t0 = params[2]
        self.Tfall = params[3]
        self.Trise = params[4]
        self.params = params
        self.scale = [0.25, 5, 10, 5, 5]
        
    def _reset(self, params):
        """
        Resets the Bazin (2009) mean function parameters.

        Args:
            params (list): List containing parameters [A, beta, t0, Tfall, Trise].
                A (float): Amplitude parameter of the Bazin function.
                beta (float): Beta parameter of the Bazin function.
                t0 (float): Time of maximum parameter of the Bazin function.
                Tfall (float): Fall timescale parameter of the Bazin function.
                Trise (float): Rise timescale parameter of the Bazin function.
        """
        self.A = params[0]
        self.beta = params[1]
        self.t0 = params[2]
        self.Tfall = params[3]
        self.Trise = params[4]
        self.params = params
    
    def mean(self, y, params=None, bands=None):
        """
        Computes the mean value using the Bazin (2009) mean function.

        Args:
            y (numpy.ndarray): Input data.
            params (list, optional): List containing parameters [A, beta, t0, Tfall, Trise] to reset mean function parameters.
                Defaults to None.
            bands: Not used in this function.

        Returns:
            numpy.ndarray: Mean value computed using the Bazin (2009) mean function.
        """
        if params != None:
            self._reset(params)
        mu = np.zeros([len(y)])

        a = np.exp(-(y - self.t0)/self.Tfall)
        b = 1 + np.exp((y - self.t0)/self.Trise)
        mu = self.A * (a/b) + self.beta
        return mu

class Karpenka2012:
    """
    Karpenka (2012) mean function for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Karpenka (2012) mean function with given parameters.

        Args:
            params (list): List containing parameters [A, B, t1, t0, Tfall, Trise, offset].
                A (float): Amplitude parameter of the Karpenka function.
                B (float): B parameter of the Karpenka function.
                t1 (float): t1 parameter of the Karpenka function.
                t0 (float): t0 parameter of the Karpenka function.
                Tfall (float): Fall timescale parameter of the Karpenka function.
                Trise (float): Rise timescale parameter of the Karpenka function.
                offset (float): Offset parameter of the Karpenka function.
        """
        self.A = np.exp(params[0])
        self.B = np.exp(params[1])
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
        self.scale = [0.25, 5, 10, 10, 5, 5]
        
    def _reset(self, params):
        """
        Resets the Karpenka (2012) mean function parameters.

        Args:
            params (list): List containing parameters [A, B, t1, t0, Tfall, Trise, offset].
                A (float): Amplitude parameter of the Karpenka function.
                B (float): B parameter of the Karpenka function.
                t1 (float): t1 parameter of the Karpenka function.
                t0 (float): t0 parameter of the Karpenka function.
                Tfall (float): Fall timescale parameter of the Karpenka function.
                Trise (float): Rise timescale parameter of the Karpenka function.
                offset (float): Offset parameter of the Karpenka function.
        """
        self.A = np.exp(params[0])
        self.B = np.exp(params[1])
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
    
    def mean(self, y, params=None, bands=None):
        """
        Computes the mean value using the Karpenka (2012) mean function.

        Args:
            y (numpy.ndarray): Input data.
            params (list, optional): List containing parameters [A, B, t1, t0, Tfall, Trise, offset] to reset mean function parameters.
                Defaults to None.
            bands: Not used in this function.

        Returns:
            numpy.ndarray: Mean value computed using the Karpenka (2012) mean function.
        """
        if params != None:
            self._reset(params)
        mu = np.zeros([len(y)])

        a = 1 + (self.B * ((y - self.t1)**2))
        b = np.exp(-(y - self.t0)/self.Tfall)
        c = 1 + np.exp(-(y - self.t0)/self.Trise)
        mu = (self.A * a * (b/c))
        return mu

class Villar2019:
    """
    Villar (2019) mean function for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Villar (2019) mean function with given parameters.

        Args:
            params (list): List containing parameters [A, beta, t1, t0, Tfall, Trise].
                A (float): Amplitude parameter of the Villar function.
                beta (float): Beta parameter of the Villar function.
                t1 (float): t1 parameter of the Villar function.
                t0 (float): t0 parameter of the Villar function.
                Tfall (float): Fall timescale parameter of the Villar function.
                Trise (float): Rise timescale parameter of the Villar function.
        """
        self.A = params[0]
        self.beta = params[1]
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
        self.scale = [0.25, 5, 10, 10, 5, 5]
        
    def _reset(self, params):
        """
        Resets the Villar (2019) mean function parameters.

        Args:
            params (list): List containing parameters [A, beta, t1, t0, Tfall, Trise].
                A (float): Amplitude parameter of the Villar function.
                beta (float): Beta parameter of the Villar function.
                t1 (float): t1 parameter of the Villar function.
                t0 (float): t0 parameter of the Villar function.
                Tfall (float): Fall timescale parameter of the Villar function.
                Trise (float): Rise timescale parameter of the Villar function.
        """
        self.A = params[0]
        self.beta = params[1]
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        """
        Computes the mean value using the Villar (2019) mean function.

        Args:
            y (numpy.ndarray): Input data.
            params (list, optional): List containing parameters [A, beta, t1, t0, Tfall, Trise] to reset mean function parameters.
                Defaults to None.
            bands: Not used in this function.

        Returns:
            numpy.ndarray: Mean value computed using the Villar (2019) mean function.
        """
        if params != None:
            self._reset(params)
        mu = np.zeros([len(y)])
        denom = 1 + np.exp(-(y - self.t0)/self.Trise)
        constant = self.A + (self.beta * (self.t1 - self.t0))
    
        for i in range(len(y)):
            if y[i] < self.t1:
                a = self.A + (self.beta * (y[i] - self.t0))
                mu[i] = a
            else:
                b = np.exp(-(y[i] - self.t1)/self.Tfall)
                mu[i] = (constant*b)
                
        mu = (mu/denom)
        return mu

