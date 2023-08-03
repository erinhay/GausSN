import numpy as np
import jax.numpy as jnp

class LensingModel:
    """
    Basic lensing model for resolved light curves in flux space. If the true underlying light curve is given by f(t), then image n is given by f'(t) = beta * f(t - delta). Here, delta represents the time delay and beta represents the magnification.
    
    Initialize with 2 lensing parameters per image, the number of images, and the number of passbands.
    
    Lensing parameters should be inputted as [delta_image2, beta_image2, delta_image3, beta_image3, ... , delta_imageN, beta_imageN].
    """
    def __init__(self, lensing_params, n_images, n_bands, indices = None):
        self.lensing_params = lensing_params
        self.deltas = jnp.array([0] + self.lensing_params[::2])
        self.betas = jnp.array([1] + self.lensing_params[1::2])
        self.n_images = n_images
        self.n_bands = n_bands
        self.indices = indices
        
        self.scale = [5, 0.5]*(self.n_images-1)
        
    def reset(self, lensing_params):
        self.lensing_params = lensing_params
        self.deltas = jnp.array([0] + self.lensing_params[::2])
        self.betas = jnp.array([1] + self.lensing_params[1:][::2])
        
    def rescale_data(self, y, yerr):
        factor = jnp.max(y) - jnp.min(y)
        y_rescaled = y/factor
        yerr_rescaled = yerr/factor
        return y_rescaled, yerr_rescaled
    
    def constant(self, x, beta):
        return np.repeat(beta, len(x))

    def magnification_matrix(self, x):
        self.magnification_vector = np.ones(len(x))
        for pb in range(self.n_bands):
            for n in range(self.n_images):
                self.magnification_vector[self.indices[(self.n_images*pb)+n] : self.indices[(self.n_images*pb)+n+1]] = self.constant(x[self.indices[(self.n_images*pb)+n] : self.indices[(self.n_images*pb)+n+1]], self.betas[n])
        return jnp.diag(self.magnification_vector)

    def time_shift(self, x):
        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), jnp.array(self.indices[1:])-jnp.array(self.indices[:-1]))
        return x - delta_vector
    

class SigmoidMicrolensing_LensingModel:
    """
    Basic lensing model for resolved light curves in flux space. If the true underlying light curve is given by f(t), then image n is given by f'(t) = beta(t) * f(t - delta). Here, delta represents the time delay and beta represents the magnification due to lensing and microlensing.
    
    For the case of sigmoid microlensing, beta(t) = beta_0 + ( (beta_1 - beta_0) * (1 + exp(-r*(t - t_0)) )^(-1) ), where beta_0 is the macrolensing effect, beta_1 is the scale of the microlensing effect, r is the inverse rate of change of the microlensing effect, and t_0 is the location of the change in microlensing.
    
    Initialize with 5 lensing parameters per image, the number of images, and the number of passbands.
    
    Lensing parameters should be inputted as [delta_image2, beta0_image2, beta1_image2, r_image2, t0_image2, ... , delta_imageN, beta0_imageN, beta1_imageN, r_imageN, t0_imageN].
    """
    
    def __init__(self, lensing_params, n_images, n_bands, indices = None):
        self.lensing_params = lensing_params
        self.deltas = jnp.array([0] + self.lensing_params[::5])
        self.beta0s = jnp.array([1] + self.lensing_params[1::5])
        self.beta1s = jnp.array([1] + self.lensing_params[2::5])
        self.rs = jnp.array([0] + self.lensing_params[3::5])
        self.t0s = jnp.array([0] + self.lensing_params[4::5])
        self.n_images = n_images
        self.n_bands = n_bands
        self.indices = indices
        
        self.scale = [5, 0.5, 0.5, 0.5, 10]*(self.n_images-1)
        
    def reset(self, lensing_params):
        self.lensing_params = lensing_params
        self.deltas = jnp.array([0] + self.lensing_params[::5])
        self.beta0s = jnp.array([1] + self.lensing_params[1::5])
        self.beta1s = jnp.array([1] + self.lensing_params[2::5])
        self.rs = jnp.array([1] + self.lensing_params[3::5])
        self.t0s = jnp.array([1] + self.lensing_params[4::5])
        
    def rescale_data(self, y, yerr):
        factor = jnp.max(y) - jnp.min(y)
        y_rescaled = y/factor
        yerr_rescaled = yerr/factor
        return y_rescaled, yerr_rescaled
    
    def sigmoid(self, x, beta0, beta1, r, t0):
        num = beta1 - beta0
        denom = 1 + (np.e ** ( - r * (x - t0) ) )
        return beta0 + (num/denom)

    def magnification_matrix(self, x):
        self.magnification_vector = np.ones(len(x))
        for pb in range(self.n_bands):
            for n in range(self.n_images):
                self.magnification_vector[self.indices[(self.n_images*pb)+n] : self.indices[(self.n_images*pb)+n+1]] = self.sigmoid(x[self.indices[(self.n_images*pb)+n] : self.indices[(self.n_images*pb)+n+1]], self.beta0s[n], self.beta1s[n], self.rs[n], self.t0s[n])
        return jnp.diag(self.magnification_vector)
    
    def time_shift(self, x):
        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), jnp.array(self.indices[1:])-jnp.array(self.indices[:-1]))
        return x - delta_vector
    
    