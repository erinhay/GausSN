import numpy as np

class UniformMean:
    def __init__(self, params):
        self.c = params[0]
        self.params = params
        
    def reset(self, params):
        self.c = params[0]
        self.params = params
        
    def mean(self, y):
        return np.repeat(self.c, len(y))

class Sin:
    def __init__(self, params):
        self.A = params[0]
        self.w = params[1]
        self.phi = params[2]
        self.params = params
        
    def reset(self, params):
        self.A = params[0]
        self.w = params[1]
        self.phi = params[2]
        self.params = params
        
    def mean(self, y):
        return self.A * np.sin((y*self.w) + self.phi)

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
        
    def reset(self, params):
        self.A = params[0]
        self.beta = params[1]
        self.t0 = params[2]
        self.Tfall = params[3]
        self.Trise = params[4]
        self.params = params
    
    def mean(self, y):
        mu = np.zeros([len(y)])

        a = np.exp(-(y - self.t0)/self.Tfall)
        b = 1 + np.exp((y - self.t0)/self.Trise)
        mu = self.A * (a/b) + self.beta
        return mu

class Karpenka2012:
    """
    params = [A, B, t1, t0, T_fall, T_rise]
    """
    def __init__(self, params):
        self.A = params[0]
        self.beta = params[1]
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
        
    def reset(self, params):
        self.A = params[0]
        self.beta = params[1]
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
    
    def mean(self, y):
        mu = np.zeros([len(y)])

        a = 1 + (self.beta*((y - self.t1)**2))
        b = np.exp(-(y - self.t0)/self.Tfall)
        c = 1 + np.exp(-(y - self.t0)/self.Trise)
        mu = self.A * a * (b/c)
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
        
    def reset(self, params):
        self.A = params[0]
        self.beta = params[1]
        self.t1 = params[2]
        self.t0 = params[3]
        self.Tfall = params[4]
        self.Trise = params[5]
        self.params = params
        
    def mean(self, y):
        mu = np.zeros([len(y)])
    
        for i in range(len(y)):
            if y < self.t1:
                a = self.A + (self.beta * (y[i] - self.t0))
                b = 1 + np.exp(-(y[i] - self.t0)/self.Trise)
                mu[i] = a/b
            else:
                a = self.A + (self.beta * (self.t1 - self.t0))
                b = 1 + np.exp(-(y[i] - self.t0)/self.Tfall)
                c = np.exp(-(y[i] - self.t1)/self.Trise)
                mu[i] = (a*c)/b
        return mu

class ExpFunction:
    """
    params = [A, tau]
    """
    def __init__(self, params):
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        
    def mean(self, y):
        mu = np.zeros([len(y)])
        mu = self.A * np.exp(y*self.tau)
        return mu