import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

class UniformMean:
    def __init__(self, params):
        self.c = params[0]
        self.ndim = len(params)
        self.scale = [1]
        
    def mean(self, y, mean_params=None):
        if mean_params is None:
            c = self.c
        else:
            c = mean_params[0]
        return jnp.repeat(c, len(y))

class Sin:
    def __init__(self, params):
        self.A = params[0]
        self.w = params[1]
        self.phi = params[2]
        self.ndim = len(params)
        self.scale = [0.25, 3, 10]
        
    def mean(self, y, mean_params=None):
        if mean_params is None:
            A, w, phi = self.A, self.w, self.phi
        else:
            A, w, phi = mean_params[0], mean_params[1], mean_params[2]
        return self.A * np.sin((y*w) + phi)
    
class Gaussian:
    def __init__(self, params):
        self.A = params[0]
        self.mu = params[1]
        self.sigma = params[2]
        self.ndim = len(params)
        self.scale = [0.25, 5, 0.5]
        
    def mean(self, y, mean_params=None):
        if mean_params is None:
            A, mu, sigma = self.A, self.mu, self.sigma
        else:
            A, mu, sigma = mean_params[0], mean_params[1], mean_params[2]
        exponent = -(y - mu)**2 / (2 * sigma**2)
        return A * jnp.exp(exponent) / (sigma * jnp.sqrt(2*jnp.pi))

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
        self.ndim = len(params)
        self.scale = [0.25, 5, 10, 5, 5]
    
    def mean(self, y, mean_params=None):
        if mean_params is None:
            A, beta, t0, Tfall, Trise = self.A, self.beta, self.t0, self.Tfall, self.Trise
        else:
            A, beta, t0, Tfall, Trise = mean_params[0], mean_params[1], mean_params[2], mean_params[3], mean_params[4]
        
        mu = np.zeros([len(y)])
        a = np.exp(-(y - t0)/Tfall)
        b = 1 + np.exp((y - t0)/Trise)
        mu = A * (a/b) + beta
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
        self.ndim = len(params)
        self.scale = [0.25, 5, 10, 10, 5, 5]
    
    def mean(self, y, mean_params=None):
        if mean_params is None:
            A, B, t1, t0, Tfall, Trise = self.A, self.B, self.t1, self.t0, self.Tfall, self.Trise
        else:
            A, B, t1, t0, Tfall, Trise = mean_params[0], mean_params[1], mean_params[2], mean_params[3], mean_params[4], mean_params[5]

        mu = np.zeros([len(y)])
        a = 1 + (B * ((y - t1)**2))
        b = np.exp(-(y - t0)/Tfall)
        c = 1 + np.exp(-(y - t0)/Trise)
        mu = (A * a * (b/c))
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
        self.ndim = len(params)
        self.scale = [0.25, 5, 10, 10, 5, 5]
        
    def mean(self, y, mean_params=None):
        if mean_params is None:
            A, beta, t1, t0, Tfall, Trise = self.A, self.beta, self.t1, self.t0, self.Tfall, self.Trise
        else:
            A, beta, t1, t0, Tfall, Trise = mean_params[0], mean_params[1], mean_params[2], mean_params[3], mean_params[4], mean_params[5]

        mu = np.zeros([len(y)])
        denom = 1 + np.exp(-(y - t0)/Trise)
        constant = A + (beta * (t1 - t0))
    
        for i in range(len(y)):
            if y[i] < t1:
                a = A + (beta * (y[i] - t0))
                mu[i] = a
            else:
                b = np.exp(-(y[i] - t1)/Tfall)
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
        self.ndim = len(params)
        self.scale = [0.25, 5]
        
    def mean(self, y, mean_params=None):
        if mean_params is None:
            A, tau = self.A, self.tau
        else:
            A, tau = mean_params[0], mean_params[1]
        mu = np.zeros([len(y)])
        mu = A * np.exp(y*tau)
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
        self.ndim = len(params)
        self.scale = [1, 1, 1, 1, 1]
 
    def mean(self, y, mean_params=None):
        if mean_params is None:
            a, b, c, d, e = self.a, self.b, self.c, self.d, self.e
        else:
            a, b, c, d, e = mean_params[0], mean_params[1], mean_params[2], mean_params[3], mean_params[4]
        mu = a*(y**4) + b*(y**3) + c*(y**2) + d*y + e
        return mu