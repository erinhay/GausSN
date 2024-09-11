import jax.numpy as jnp
import jax

class ExpSquaredKernel:
    """
    Exponentiated Squared Kernel for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Exponentiated Squared Kernel with given parameters.

        Args:
            params (list): List containing parameters [A, tau].
                A (float): Amplitude parameter of the kernel.
                tau (float): Length scale parameter of the kernel.
        """
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        """
        Resets the kernel parameters.

        Args:
            params (list): List containing parameters [A, tau].
                A (float): Amplitude parameter of the kernel.
                tau (float): Length scale parameter of the kernel.
        """
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Exponentiated Squared Kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params (list, optional): List containing parameters [A, tau] to reset kernel parameters.
                Defaults to None.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the Exponentiated Squared Kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime is None:
            x_prime = x

        K = self.A**2 * jnp.exp(-(x[:, None] - x_prime[None, :])**2/(2*self.tau**2))
        return K
    
class ExponentialKernel:
    """
    Exponential Kernel for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Exponential Kernel with given parameters.

        Args:
            params (list): List containing parameters [A, tau].
                A (float): Amplitude parameter of the kernel.
                tau (float): Length scale parameter of the kernel.
        """
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        """
        Resets the kernel parameters.

        Args:
            params (list): List containing parameters [A, tau].
                A (float): Amplitude parameter of the kernel.
                tau (float): Length scale parameter of the kernel.
        """
        self.A = params[0]
        self.tau = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Exponential Kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params (list, optional): List containing parameters [A, tau] to reset kernel parameters.
                Defaults to None.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the Exponential Kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        K = self.A**2 * jnp.exp(-vector_mag/self.tau)
        return K

class ConstantKernel:
    def __init__(self, params):
        """
        Initializes the Constant Kernel with given parameters.

        Args:
            params (list): List containing parameters [c].
                c (float): Constant value of the kernel.
        """
        self.c = params[0]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        """
        Resets the kernel parameters.

        Args:
            params (list): List containing parameters [c].
                c (float): Constant value of the kernel.
        """
        self.c = params[0]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Constant Kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params (list, optional): List containing parameters [c] to reset kernel parameters.
                Defaults to None.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the Constant Kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        K = jnp.ones([len(x), len(x_prime)]) * self.c
        return K

class DotProductKernel:
    """
    Dot Product Kernel for Gaussian processes.
    """
    def __init__(self):
        """
        Initializes the Dot Product Kernel.
        """
        self.c = 0
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self):
        """
        Resets the kernel parameters.
        """
        self.c = 0
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Dot Product Kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params: Not used in this kernel.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the Dot Product Kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        K = jnp.dot(x[:, None], x_prime[None, :])
        return K

class Matern32Kernel:
    """
    Matern 3/2 Kernel for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Matern 3/2 Kernel with given parameters.

        Args:
            params (list): List containing parameters [A, l].
                A (float): Amplitude parameter of the kernel.
                l (float): Length scale parameter of the kernel.
        """
        self.A = params[0]
        self.l = params[1]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        """
        Resets the kernel parameters.

        Args:
            params (list): List containing parameters [A, l].
                A (float): Amplitude parameter of the kernel.
                l (float): Length scale parameter of the kernel.
        """
        self.A = params[0]
        self.l = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Matern 3/2 Kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params (list, optional): List containing parameters [A, l] to reset kernel parameters.
                Defaults to None.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the Matern 3/2 Kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x
        
        r2 = (x[:, None] - x_prime[None, :])**2
        K = self.A**2 * ((1 + jnp.sqrt(3*r2)/self.l) * jnp.exp(-jnp.sqrt(3*r2)/self.l))
        return K

class Matern52Kernel:
    """
    Matern 5/2 Kernel for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Matern 5/2 Kernel with given parameters.

        Args:
            params (list): List containing parameters [A, l].
                A (float): Amplitude parameter of the kernel.
                l (float): Length scale parameter of the kernel.
        """
        self.A = params[0]
        self.l = params[1]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        """
        Resets the kernel parameters.

        Args:
            params (list): List containing parameters [A, l].
                A (float): Amplitude parameter of the kernel.
                l (float): Length scale parameter of the kernel.
        """
        self.A = params[0]
        self.l = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Matern 5/2 Kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params (list, optional): List containing parameters [A, l] to reset kernel parameters.
                Defaults to None.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the Matern 5/2 Kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x
        
        r2 = (x[:, None] - x_prime[None, :])**2
        a = 1 + (jnp.sqrt(5*r2)/self.l) + (5*r2/(3*(self.l**2)))
        b = -jnp.sqrt(5*r2)/self.l
        K = self.A**2 * a * jnp.exp(b)
        return K
    
class RationalQuadraticKernel:
    """
    Rational Quadratic Kernel for Gaussian processes.
    """
    def __init__(self, params):
        """
        Initializes the Rational Quadratic Kernel with given parameters.

        Args:
            params (list): List containing parameters [A, tau, scale_mixture].
                A (float): Amplitude parameter of the kernel.
                tau (float): Length scale parameter of the kernel.
                scale_mixture (float): Scale mixture parameter of the kernel.
        """
        self.A = params[0]
        self.tau = params[1]
        self.scale_mixture = params[2]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        """
        Resets the kernel parameters.

        Args:
            params (list): List containing parameters [A, tau, scale_mixture].
                A (float): Amplitude parameter of the kernel.
                tau (float): Length scale parameter of the kernel.
                scale_mixture (float): Scale mixture parameter of the kernel.
        """
        self.A = params[0]
        self.tau = params[1]
        self.scale_mixture = params[2]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Rational Quadratic Kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params (list, optional): List containing parameters [A, tau, scale_mixture] to reset kernel parameters.
                Defaults to None.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the Rational Quadratic Kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        K = self.A**2 * (1 + (x[:, None] - x_prime[None, :])**2/(2*self.scale_mixture*self.tau**2))
        return K
    
class GibbsKernel:
    """
    Gibbs Kernel for Gaussian processes.
    
    Defined as:
        K = A^2 * ( 2*tau(y|mu, sigma)*tau(y'|mu, sigma) / (tau(y|mu, sigma)**2 + tau(y'|mu, sigma)**2) )**(1/2) * exp( (y - y')**2 / (tau(y|mu, sigma)**2 + tau(y'|mu, sigma)**2) )
    where:
        tau(y|mu, sigma) = lambda * (1 - p * N(y|mu, sigma))
    and N(y|mu, sigma) is a normal distribution with mean mu and variance sigma**2.
    """
    def __init__(self, params):
        """
        Initializes the Gibbs Kernel with given parameters.

        Args:
            params (list): List containing parameters [A, lambda, p, mu, sigma].
                A (float): Amplitude parameter of the kernel.
                lambda (float): Lambda parameter of the kernel.
                p (float): P parameter of the kernel.
                mu (float): Mean of the normal distribution.
                sigma (float): Standard deviation of the normal distribution.
        """
        self.A = params[0]
        self.lamdba = params[1]
        self.p = params[2]
        self.mu = params[3]
        self.sigma = params[4]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        """
        Resets the kernel parameters.

        Args:
            params (list): List containing parameters [A, lambda, p, mu, sigma].
                A (float): Amplitude parameter of the kernel.
                lambda (float): Lambda parameter of the kernel.
                p (float): P parameter of the kernel.
                mu (float): Mean of the normal distribution.
                sigma (float): Standard deviation of the normal distribution.
        """
        self.A = params[0]
        self.lamdba = params[1]
        self.p = params[2]
        self.mu = params[3]
        self.sigma = params[4]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Gibbs Kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params (list, optional): List containing parameters [A, lambda, p, mu, sigma] to reset kernel parameters.
                Defaults to None.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the Gibbs Kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        normal_x = jnp.exp(-(x[:, None] - self.mu)**2 / (2*(self.sigma**2))) / (self.sigma * jnp.sqrt(2*jnp.pi))
        normal_xprime = jnp.exp(-(x_prime[None, :] - self.mu)**2 / (2*(self.sigma**2))) / (self.sigma * jnp.sqrt(2*jnp.pi))
        
        tau_x = self.lamdba * (1 - (self.p * normal_x))
        tau_xprime = self.lamdba * (1 - (self.p * normal_xprime))
        
        root_num = 2 * tau_x * tau_xprime
        exp_num = (x[:, None] - x_prime[None, :])**2
        denom = (tau_x**2) + (tau_xprime**2)
        K = self.A**2 * jnp.sqrt(root_num/denom) * jnp.exp(-exp_num/denom)
        return K
    
class OUKernel:
    def __init__(self, params):
        """
        Initializes the Ornstein-Uhlenbeck (OU) kernel with given parameters.

        Args:
            params (list): List containing parameters [A, l].
                A (float): Amplitude parameter of the OU kernel.
                l (float): Length scale parameter of the OU kernel.
        """
        self.A = params[0]
        self.l = params[1]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        """
        Resets the kernel parameters.

        Args:
            params (list): List containing parameters [A, l].
                A (float): Amplitude parameter of the OU kernel.
                l (float): Length scale parameter of the OU kernel.
        """
        self.A = params[0]
        self.l = params[1]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        """
        Computes the covariance matrix using the Ornstein-Uhlenbeck (OU) kernel.

        Args:
            x (jax.numpy.ndarray): Input data matrix of shape (n_samples, n_features).
            x_prime (jax.numpy.ndarray, optional): Second input data matrix of shape (n_samples, n_features).
                Defaults to None, which means x_prime is set to x.
            params (list, optional): List containing parameters [A, l] to reset kernel parameters.
                Defaults to None.

        Returns:
            jax.numpy.ndarray: Covariance matrix computed using the OU kernel.
        """
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        r2 = (x[:, None] - x_prime[None, :])**2
        K = self.A * jnp.exp(-jnp.sqrt(r2) / self.l)
        return K

class PeriodicKernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.p = params[2]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        self.p = params[2]
        self.params = params
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        sin = vector_mag / self.p
        exponent = -2 * ( jnp.sin(jnp.pi * sin) / self.l)**2
        K = self.A * jnp.exp(exponent)
        return K

class JJKernel:
    def __init__(self, params):
        self.A = params[0]
        self.l = params[1]
        self.m = params[2]
        self.b = params[3]
        self.params = params
        self.covariance = jax.jit(self._covariance)
        
    def _reset(self, params):
        self.A = params[0]
        self.l = params[1]
        self.m = params[2]
        self.b = params[3]
        self.params = params

    def _p(self, x):
        return self.m * x + self.b
        
    def _covariance(self, x, x_prime=None, params=None):
        if params != None:
            self._reset(params)

        if x_prime == None:
            x_prime = x

        vector_mag = jnp.sqrt((x[:, None] - x_prime[None, :])**2)
        sin = vector_mag / self._p(x)
        exponent = -2 * ( jnp.sin(jnp.pi * sin) / self.l)**2
        K = self.A * jnp.exp(exponent)
        return K
        
