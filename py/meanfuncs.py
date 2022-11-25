import numpy as np

class UniformMean:
    def __init__(self, c):
        self.c = c
        
    def mean(self, y):
        return np.repeat(self.c, len(y))

class Sin:
    def __init__(self, A, w, phi):
        self.A = A
        self.w = w
        self.phi = phi
        
    def mean(self, y):
        return self.A * np.sin((y*self.w) + self.phi)

class Bazin2009:
    """
    params = [A, B, t0, T_fall, T_rise]
    """
    def __init__(self, A, beta, t0, T_fall, T_rise):
        self.A = A
        self.beta = beta
        self.t0 = t0
        self.Tfall = T_fall
        self.Trise = T_rise
    
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
    def __init__(self, A, beta, t1, t0, T_fall, T_rise):
        self.A = A
        self.beta = beta
        self.t1 = t1
        self.t0 = t0
        self.Tfall = T_fall
        self.Trise = T_rise
    
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
    
    def __init__(self, A, beta, t1, t0, T_fall, T_rise):
        self.A = A
        self.beta = beta
        self.t1 = t1
        self.t0 = t0
        self.Tfall = T_fall
        self.Trise = T_rise
        
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
    def __init__(self, A, tau):
        self.A = A
        self.tau = tau
        
    def mean(self, y):
        mu = np.zeros([len(y)])
        mu = self.A * np.exp(y*self.tau)
        return mu