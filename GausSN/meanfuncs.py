import numpy as np
import jax.numpy as jnp
import sncosmo

class UniformMean:
    def __init__(self, params):
        self.c = params[0]
        self.params = params
        self.scale = [1]
        
    def _reset(self, params):
        self.c = params[0]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        if params != None:
            self._reset(params)
        return self.c

class sncosmoMean:
    def __init__(self, templatename, params, redshift=None):
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

class SALTMean:
    def __init__(self, templatename, params, redshift=None):
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

    def mean(self, x, params=None, bands=None):
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
    def __init__(self, params):
        self.A = params[0]
        self.w = params[1]
        self.phi = params[2]
        self.params = params
        self.scale = [0.25, 3, 10]
        
    def _reset(self, params):
        self.A = params[0]
        self.w = params[1]
        self.phi = params[2]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        if params != None:
            self._reset(params)
        return self.A * np.sin((y*self.w) + self.phi)
    
class Gaussian:
    def __init__(self, params):
        self.A = params[0]
        self.mu = params[1]
        self.sigma = params[2]
        self.params = params
        self.scale = [0.25, 5, 0.5]
        
    def _reset(self, params):
        self.A = params[0]
        self.mu = params[1]
        self.sigma = params[2]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        if params != None:
            self._reset(params)
        exponent = -(y - self.mu)**2 / (2 * self.sigma**2)
        return self.A * jnp.exp(exponent) / (self.sigma * jnp.sqrt(2*jnp.pi))

class Bazin2009:
    """
    params = [A, B, t0, T_fall, T_rise]
    """
    def __init__(self, params):
        self.A = params[0]
        self.beta = params[1]
        self.t0 = params[2]
        self.Tfall = params[3]
        self.Trise = params[4]
        self.params = params
        self.scale = [0.25, 5, 10, 5, 5]
        
    def _reset(self, params):
        self.A = params[0]
        self.beta = params[1]
        self.t0 = params[2]
        self.Tfall = params[3]
        self.Trise = params[4]
        self.params = params
    
    def mean(self, y, params=None, bands=None):
        if params != None:
            self._reset(params)
        mu = np.zeros([len(y)])

        a = np.exp(-(y - self.t0)/self.Tfall)
        b = 1 + np.exp((y - self.t0)/self.Trise)
        mu = self.A * (a/b) + self.beta
        return mu

class Karpenka2012:
    """
    params = [A, B, t1, t0, T_fall, T_rise, offset]
    
    bounds (with squared exp kernel) = [(10, 10000), (10,300), (10,20*np.max(zband["fluxcal"])), (-np.max(zband["fluxcal"])/150,-1), t1_bound, t0_bound, (20, np.max(zband['mjd'])-np.min(zband['mjd'])), (1, 50)]
    """
    def __init__(self, params):
        self.A = np.exp(params[0])
        self.B = np.exp(params[1])
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
        self.scale = [0.25, 5, 10, 10, 5, 5]
        
    def _reset(self, params):
        self.A = np.exp(params[0])
        self.B = np.exp(params[1])
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
    
    def mean(self, y, params=None, bands=None):
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
    params = [A, beta, t1, t0, T_fall, T_rise]
    """
    
    def __init__(self, params):
        self.A = params[0]
        self.beta = params[1]
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
        self.scale = [0.25, 5, 10, 10, 5, 5]
        
    def _reset(self, params):
        self.A = params[0]
        self.beta = params[1]
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
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

class ExpFunction:
    """
    params = [A, tau]
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        self.scale = [0.25, 5]
        
    def mean(self, y, params=None, bands=None):
        if params != None:
            self._reset(params)
        mu = np.zeros([len(y)])
        mu = self.A * np.exp(y*self.tau)
        return mu
    
class QuarticFunction:
    """
    params = [a, b, c, d, e]
    """
    def __init__(self, params):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.e = params[4]
        self.params = params
        self.scale = [1, 1, 1, 1, 1]
        
    def _reset(self, params):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]
        self.e = params[4]
        self.params = params
        
    def mean(self, y, params=None, bands=None):
        if params != None:
            self._reset(params)
        mu = a*(y**4) + b*(y**3) + c*(y**2) + d*y + e
        return mu