import numpy as np
import jax.numpy as jnp

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
        
    def _reset(self, params):
        """
        Resets the mean function parameters.

        Args:
            params (list): List containing parameters [c].
                c (float): Constant value of the mean function.
        """
        self.c = params[0]
        self.params = params
        
    def mean(self, y, params=None, bands=None, zp=None, zpsys=None):
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

    def mean(self, x, params=None, bands=None, zp=27.5, zpsys='ab'):
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
        
    def mean(self, y, params=None, bands=None, zp=None, zpsys=None):
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
        
    def mean(self, y, params=None, bands=None, zp=None, zpsys=None):
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
        
    def mean(self, y, params=None, bands=None, zp=None, zpsys=None):
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
    
    def mean(self, y, params=None, bands=None, zp=None, zpsys=None):
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
    
    def mean(self, y, params=None, bands=None, zp=None, zpsys=None):
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
        
    def mean(self, y, params=None, bands=None, zp=None, zpsys=None):
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

