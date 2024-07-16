import numpy as np
import jax.numpy as jnp
import sncosmo

lt_g = np.loadtxt('/data/eeh55/Github/GausSN/filters/LT/Liverpool_IOO.SDSS-g.dat')
lt_r = np.loadtxt('/data/eeh55/Github/GausSN/filters/LT/Liverpool_IOO.SDSS-r.dat')
lt_i = np.loadtxt('/data/eeh55/Github/GausSN/filters/LT/Liverpool_IOO.SDSS-i.dat')
lt_z = np.loadtxt('/data/eeh55/Github/GausSN/filters/LT/Liverpool_IOO.SDSS-z.dat')

bandLTg = sncosmo.Bandpass(lt_g[:,0], lt_g[:,1], name='IOOg')
bandLTr = sncosmo.Bandpass(lt_r[:,0], lt_r[:,1], name='IOOr')
bandLTi = sncosmo.Bandpass(lt_i[:,0], lt_i[:,1], name='IOOi')
bandLTz = sncosmo.Bandpass(lt_z[:,0], lt_z[:,1], name='IOOz')

sncosmo.registry.register(bandLTg, force=True)
sncosmo.registry.register(bandLTr, force=True)
sncosmo.registry.register(bandLTi, force=True)
sncosmo.registry.register(bandLTz, force=True)

hawki_Y = np.loadtxt('/data/eeh55/Github/GausSN/filters/VLT/hawki_Y.dat')
hawki_J = np.loadtxt('/data/eeh55/Github/GausSN/filters/VLT/hawki_J.dat')
hawki_H = np.loadtxt('/data/eeh55/Github/GausSN/filters/VLT/hawki_H.dat')
hawki_K = np.loadtxt('/data/eeh55/Github/GausSN/filters/VLT/hawki_Knew.dat')

bandhawkiY = sncosmo.Bandpass(hawki_Y[:,0]*10, hawki_Y[:,1], name='HAWKI_Y')
bandhawkiJ = sncosmo.Bandpass(hawki_J[:,0]*10, hawki_J[:,1], name='HAWKI_J')
bandhawkiH = sncosmo.Bandpass(hawki_H[:,0]*10, hawki_H[:,1], name='HAWKI_H')
bandhawkiK = sncosmo.Bandpass(hawki_K[:,0]*10, hawki_K[:,1], name='HAWKI_K')

sncosmo.registry.register(bandhawkiY, force=True)
sncosmo.registry.register(bandhawkiJ, force=True)
sncosmo.registry.register(bandhawkiH, force=True)
sncosmo.registry.register(bandhawkiK, force=True)

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

        args = np.argsort(x)
        revert_args = np.zeros(len(args), dtype=int)
        revert_args[args] = np.arange(len(args))

        reordered_x = x[args]
        if len(bands) >= 2:
            reordered_bands = bands[args]
        else:
            reordered_bands = bands

        flux = self.model.bandflux(reordered_bands, reordered_x, zp=27.5, zpsys='ab')

        return flux[revert_args]

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
        self.model.set(z=self.redshift, t0=self.t0, x0=1.e-8*self.x0, x1=self.x1, c=self.c)

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
        self.model.set(z=self.redshift, t0=self.t0, x0=1.e-8*self.x0, x1=self.x1, c=self.c)

    def mean(self, x, params=None, bands=None):
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

        flux = self.model.bandflux(reordered_bands, reordered_x, zp=27.5, zpsys='ab')

        return flux[revert_args]
    
class ZwickyMean:
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
        self.redshift = redshift
        self.t0 = params[0]
        self.x0 = params[1]
        self.x1 = params[2]
        self.c = params[3]
        self.mwebv = 0.16
        self.mwr_v = 3.1

        dust = sncosmo.F99Dust(r_v=self.mwr_v)
        self.model = sncosmo.Model(source=self.templatename, effects=[dust], effect_names=['mw'], effect_frames=['obs'])
        self.model.set(z=self.redshift, t0=self.t0, x0=1.e-9*self.x0, x1=self.x1, c=self.c, mwebv=self.mwebv)

    def _reset(self, params):
        """
        Resets the mean function parameters.

        Args:
            params (list): List containing parameters [redshift, t0, x0, x1, c] or [t0, x0, x1, c].
        """
        self.params = params
        self.t0 = params[0]
        self.x0 = params[1]
        self.x1 = params[2]
        self.c = params[3]
        self.model.set(z=self.redshift, t0=self.t0, x0=1.e-9*self.x0, x1=self.x1, c=self.c, mwebv=self.mwebv)

    def mean(self, x, params=None, bands=None):
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

        flux = self.model.bandflux(reordered_bands, reordered_x, zp=27.5, zpsys='ab')

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

