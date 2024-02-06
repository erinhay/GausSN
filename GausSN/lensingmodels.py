import numpy as np
import jax.numpy as jnp
import jax

class NoLensing:
    def __init__(self):
        self.mask = 1

    def _lens(self, x, params=None):
        return x, 1

    def lensed_mean_covariance(self, x, yerr, kernel_params=None, meanfunc_params=None, lensing_params=None):
        mean_vector = self.meanfunc.mean(x, params=meanfunc_params, bands=self.bands)
        cov_matrix = self.kernel.covariance(x, params=kernel_params) + np.diag(yerr**2)
        return mean_vector, cov_matrix

    def import_from_gp(self, kernel, meanfunc, bands=None, images=None, indices=None):
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands

class ConstantLensing:
    def __init__(self, params):
        self.deltas = jnp.array([0] + params[0::2])
        self.betas = jnp.array([1] + params[1::2])
        self.params = params
        self.scale = [1]
        
    def _reset(self, params):
        self.deltas = jnp.array([0] + params[0::2])
        self.betas = jnp.array([1] + params[1::2])
        self.params = params
        
    def _make_mask(self):
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[(self.n_images+1)*pb]
            stop = self.indices[(self.n_images+1)*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask

    def _time_shift(self, x, delta):
        return x - delta

    def _magnify(self, x, beta):
        return beta

    def _lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta_vector = jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats)

        new_x = self._time_shift(x, delta_vector)
        b = self._magnify(new_x, beta_vector)

        return new_x, b

    def lensed_mean_covariance(self, x, yerr, kernel_params=None, meanfunc_params=None, lensing_params=None):

        #Create empty arrays which will contain the mean function, mean_vector, and the covariance matrix, K (which will later become cov_matrix).
        mean_vector = np.zeros(len(x))
        K = np.zeros((len(x), len(x)))

        #Reset lensing params if necessary.
        if lensing_params != None:
            self._reset(lensing_params)

        #Make a vector of deltas/betas that encodes the image information. The delta/beta for each image are repeated a number of times
        #corresponding to the number of observations of an image. ( len(self.deltas) = # of images )
        # NOTE: This is a temporary solution, which only works for N images + unresolved data in 1 band. "repeats" is a list: [# of observations
        # for image 1, # of observations for image 2, # of observations of unresolved data]. 
        # After making the delta/beta_vector for the resolved data ( jnp.repeat(jnp.tile(...), ...) ), I concatenate a bunch of 0s to the
        # delta/beta_vector. While this part of the vector will NOT be used (because the mean function/covariance for unresolved data depends
        # on all images, so I have to loop over each delta/beta for those elements), it is probably best to have delta/beta_vector be the same
        # length as x (especially when you get to more bands and you will need to pad delta_vector between resolved data to get the right indices
        # when you're accessing different bands).
        delta_vector = jnp.concatenate([jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats[:-1]), jnp.repeat(0, self.repeats[-1])])
        beta_vector = jnp.concatenate([jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats[:-1]), jnp.repeat(0, self.repeats[-1])])

        #Loop over the time data, x
        for i in range(len(x)):

            #For each time, x[i], get the time shifted time, x_i. The entry is shifted back by delta_vector[i], where delta_vector[i] is
            #specific to the image that the data point corresponds to.
            #Repeat the same process for getting the magnification factor, b_i, for time x[i] based on the image info,
            #which is encoded in beta_vector[i].
            x_i = self._time_shift(x[i], delta_vector[i])
            b_i = self._magnify(x_i, beta_vector[i])

            #If the data is resolved, the mean vector is just equal to one evaluation of the mean function at time x_i in band bands[i]
            #which is magnified by b_i.
            if 'image' in self.images[i]:
                mean_vector[i] = b_i * self.meanfunc.mean(x_i, params=meanfunc_params, bands=[self.bands[i]])

            #If the data is resolved, you have to loop over all images to get the contribution to the mean from each image.
            elif self.images[i] == 'unresolved':
                for n in range(self.n_images):
                    x_i = self._time_shift(x[i], self.deltas[n])
                    b_i = self._magnify(x_i, self.betas[n])

                    mean_vector[i] += b_i * self.meanfunc.mean(x_i, params=meanfunc_params, bands=[self.bands[i]])

            #Loop over the time data, x, a second time (for the cov matrix)
            for j in range(len(x)):

                x_j = self._time_shift(x[j], delta_vector[j])
                b_j = self._magnify(x_j, beta_vector[j])

                #If both data points are resolved, then the covariance element K[i,j] is one evaluation of the kernel function.
                if 'image' in self.images[i] and 'image' in self.images[j]:
                    K[i,j] = b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params)

                #If only one data point is resolved, then you have to loop over one image
                elif 'image' in self.images[i] and self.images[j] == 'unresolved':
                    for n in range(self.n_images):
                        x_j = x[j] - self.deltas[n]
                        b_j = self._magnify(x_j, self.betas[n])

                        K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params) 

                elif self.images[i] == 'unresolved' and 'image' in self.images[j]:
                    for n in range(self.n_images):
                        x_i = x[i] - self.deltas[n]
                        b_i = self._magnify(x_i, self.betas[n])

                        K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params) 

                #If both data points are resolved, then you have to loop over both images.
                elif self.images[i] == 'unresolved' and self.images[j] == 'unresolved':
                    for n in range(self.n_images):
                        for m in range(self.n_images):
                            x_i = x[i] - self.deltas[n]
                            x_j = x[j] - self.deltas[m]
                            b_i = self._magnify(x_i, self.betas[n])
                            b_j = self._magnify(x_j, self.betas[m])

                            K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params)

        #Multiply K by a mask which removes covariance between different bands and add measurement errors.
        cov_matrix = jnp.multiply(self.mask, K) + jnp.diag(yerr**2)
        return mean_vector, cov_matrix

    def import_from_gp(self, kernel, meanfunc, bands, images, indices):
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands
        self.n_bands = len(np.unique(bands))
        self.images = images
        self.n_images = len(np.unique(images))
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()
    
class SigmoidMicrolensing:
    def __init__(self, params):
        self.deltas = jnp.array([0] + params[5::5])
        self.beta0s = jnp.array(params[1::5])
        self.beta1s = jnp.array(params[2::5])
        self.rs = jnp.array(params[3::5])
        self.t0s = jnp.array(params[4::5])
        self.params = params
        
    def _reset(self, params):
        self.deltas = jnp.array([0] + params[5::5])
        self.beta0s = jnp.array(params[1::5])
        self.beta1s = jnp.array(params[2::5])
        self.rs = jnp.array(params[3::5])
        self.t0s = jnp.array(params[4::5])
        self.params = params

    def _make_mask(self):
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[self.n_images*pb]
            stop = self.indices[self.n_images*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask
        
    def _time_shift(self, x, delta):
        return x - delta

    def _magnify(self, x, beta0, beta1, r, t0):
        denom = 1 + jnp.exp(-r * (x-t0) )
        return beta0 + (beta1/denom)

    def _lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        r_vector = jnp.repeat(jnp.tile(self.rs, self.n_bands), self.repeats)
        t0_vector = jnp.repeat(jnp.tile(self.t0s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta0_vector, beta1_vector, r_vector, t0_vector)

        return x, b

    def lensed_mean_covariance(self, x, yerr, kernel_params=None, meanfunc_params=None, lensing_params=None):

        mean_vector = np.zeros(len(x))
        K = np.zeros((len(x), len(x)))

        if lensing_params != None:
            self._reset(lensing_params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats[:-1])
        beta_vector = jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats[:-1])

        for i in range(len(x)):

            x_i = self._time_shift(x[i], delta_vector[i])
            b_i = self._magnify(x_i, beta_vector[i])

            if 'image' in self.images[i]:

                mean_vector[i] = b_i * self.meanfunc.mean(x_i, params=meanfunc_params, bands=[self.bands[i]])

            elif self.images[i] == 'unresolved':
                for n in range(self.n_images):
                    x_i = self._time_shift(x[i], self.deltas[n])
                    b_i = self._magnify(x_i, self.betas[n])

                    mean_vector[i] += b_i * self.meanfunc.mean(x_i, params=meanfunc_params, bands=[self.bands[i]])

            for j in range(len(x)):

                x_j = self._time_shift(x[j], delta_vector[j])
                b_j = self._magnify(x_j, beta_vector[j])

                if 'image' in self.images[i] and 'image' in self.images[j]:
                    K[i,j] = b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params)

                elif 'image' in self.images[i] and self.images[j] == 'unresolved':
                    for n in range(self.n_images):
                        x_j = x[j] - self.deltas[n]
                        b_j = self._magnify(x_j, self.betas[n])

                        K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params) 

                elif self.images[i] == 'unresolved' and 'image' in self.images[j]:
                    for n in range(self.n_images):
                        x_i = x[i] - self.deltas[n]
                        b_i = self._magnify(x_i, self.betas[n])

                        K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params) 

                elif self.images[i] == 'unresolved' and self.images[j] == 'unresolved':
                    for n in range(self.n_images):
                        for m in range(self.n_images):
                            x_i = x[i] - self.deltas[n]
                            x_j = x[j] - self.deltas[m]
                            b_i = self._magnify(x_i, self.betas[n])
                            b_j = self._magnify(x_j, self.betas[m])

                            K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params)

        cov_matrix = jnp.multiply(self.mask, K) + jnp.diag(yerr**2)
        return mean_vector, cov_matrix

    def import_from_gp(self, kernel, meanfunc, bands, images, indices):
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands
        self.n_bands = len(np.unique(bands))
        self.images = images
        self.n_images = len(np.unique(images))
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class PolynomialLensingKernel:
    def __init__(self, params):
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.beta2s = jnp.array([0] + params[3::5])
        self.beta3s = jnp.array([0] + params[4::5])
        self.params = params
        self.scale = [0.5, 5]

    def _reset(self, params):
        self.deltas = jnp.array([0] + params[0::5])
        self.beta0s = jnp.array([1] + params[1::5])
        self.beta1s = jnp.array([0] + params[2::5])
        self.beta2s = jnp.array([0] + params[3::5])
        self.beta3s = jnp.array([0] + params[4::5])
        self.params = params

    def _make_mask(self):
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[self.n_images*pb]
            stop = self.indices[self.n_images*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask
        
    def _time_shift(self, x, delta):
        return x - delta

    def _magnify(self, x, beta0, beta1, beta2, beta3):
        return beta0 + (beta1*x) + (beta2 * (x**2)) + (beta3 * (x**3))

    def _lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta0_vector = jnp.repeat(jnp.tile(self.beta0s, self.n_bands), self.repeats)
        beta1_vector = jnp.repeat(jnp.tile(self.beta1s, self.n_bands), self.repeats)
        beta2_vector = jnp.repeat(jnp.tile(self.beta2s, self.n_bands), self.repeats)
        beta3_vector = jnp.repeat(jnp.tile(self.beta3s, self.n_bands), self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._magnify(x, beta0_vector, beta1_vector, beta2_vector, beta3_vector)
        return x, b

    def lensed_mean_covariance(self, x, yerr, kernel_params=None, meanfunc_params=None, lensing_params=None):

        mean_vector = np.zeros(len(x))
        K = np.zeros((len(x), len(x)))

        if lensing_params != None:
            self._reset(lensing_params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats[:-1])
        beta_vector = jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats[:-1])

        for i in range(len(x)):

            x_i = self._time_shift(x[i], delta_vector[i])
            b_i = self._magnify(x_i, beta_vector[i])

            if 'image' in self.images[i]:

                mean_vector[i] = b_i * self.meanfunc.mean(x_i, params=meanfunc_params, bands=[self.bands[i]])

            elif self.images[i] == 'unresolved':
                for n in range(self.n_images):
                    x_i = self._time_shift(x[i], self.deltas[n])
                    b_i = self._magnify(x_i, self.betas[n])

                    mean_vector[i] += b_i * self.meanfunc.mean(x_i, params=meanfunc_params, bands=[self.bands[i]])

            for j in range(len(x)):

                x_j = self._time_shift(x[j], delta_vector[j])
                b_j = self._magnify(x_j, beta_vector[j])

                if 'image' in self.images[i] and 'image' in self.images[j]:
                    K[i,j] = b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params)

                elif 'image' in self.images[i] and self.images[j] == 'unresolved':
                    for n in range(self.n_images):
                        x_j = x[j] - self.deltas[n]
                        b_j = self._magnify(x_j, self.betas[n])

                        K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params) 

                elif self.images[i] == 'unresolved' and 'image' in self.images[j]:
                    for n in range(self.n_images):
                        x_i = x[i] - self.deltas[n]
                        b_i = self._magnify(x_i, self.betas[n])

                        K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params) 

                elif self.images[i] == 'unresolved' and self.images[j] == 'unresolved':
                    for n in range(self.n_images):
                        for m in range(self.n_images):
                            x_i = x[i] - self.deltas[n]
                            x_j = x[j] - self.deltas[m]
                            b_i = self._magnify(x_i, self.betas[n])
                            b_j = self._magnify(x_j, self.betas[m])

                            K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params)

        cov_matrix = jnp.multiply(self.mask, K) + jnp.diag(yerr**2)
        return mean_vector, cov_matrix

    def import_from_gp(self, kernel, meanfunc, bands, images, indices):
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands
        self.n_bands = len(np.unique(bands))
        self.images = images
        self.n_images = len(np.unique(images))
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

class FlexibleDust_ConstantLensingKernel:
    def __init__(self, params, n_bands):
        self.deltas = jnp.array([0] + params[0::2])
        self.betas_mask = jnp.isin(jnp.arange(len(params)), jnp.concatenate([jnp.array([0,1]), jnp.array(list(range(*slice(2,None,n_bands+1).indices(len(params)))))]), invert=True)
        self.betas = jnp.concatenate([jnp.repeat(1, n_bands), jnp.array(params)[self.betas_mask]])
        self.params = params
        self.scale = [0.5, 5]
    
    def _reset(self, params):
        self.deltas = jnp.array([0] + params[0::self.n_bands+1])
        self.betas = jnp.concatenate([jnp.repeat(1, self.n_bands), jnp.array(params)[self.betas_mask]])
        self.params = params

    def _make_mask(self):
        mask = np.zeros((self.indices[-1], self.indices[-1]))
        for pb in range(self.n_bands):
            start = self.indices[self.n_images*pb]
            stop = self.indices[self.n_images*(pb+1)]
            mask[start:stop, start:stop] = 1
        return mask

    def _time_shift(self, x, delta):
        return x - delta

    def _magnify(self, x, beta):
        return beta

    def _lens(self, x, params=None):
        if params != None:
            self._reset(params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats)
        beta_vector = jnp.repeat(self.betas, self.repeats)

        x = self._time_shift(x, delta_vector)
        b = self._lens(x, beta_vector)

        return x, b

    def lensed_mean_covariance(self, x, yerr, kernel_params=None, meanfunc_params=None, lensing_params=None):

        mean_vector = np.zeros(len(x))
        K = np.zeros((len(x), len(x)))

        if lensing_params != None:
            self._reset(lensing_params)

        delta_vector = jnp.repeat(jnp.tile(self.deltas, self.n_bands), self.repeats[:-1])
        beta_vector = jnp.repeat(jnp.tile(self.betas, self.n_bands), self.repeats[:-1])

        for i in range(len(x)):

            x_i = self._time_shift(x[i], delta_vector[i])
            b_i = self._magnify(x_i, beta_vector[i])

            if 'image' in self.images[i]:

                mean_vector[i] = b_i * self.meanfunc.mean(x_i, params=meanfunc_params, bands=[self.bands[i]])

            elif self.images[i] == 'unresolved':
                for n in range(self.n_images):
                    x_i = self._time_shift(x[i], self.deltas[n])
                    b_i = self._magnify(x_i, self.betas[n])

                    mean_vector[i] += b_i * self.meanfunc.mean(x_i, params=meanfunc_params, bands=[self.bands[i]])

            for j in range(len(x)):

                x_j = self._time_shift(x[j], delta_vector[j])
                b_j = self._magnify(x_j, beta_vector[j])

                if 'image' in self.images[i] and 'image' in self.images[j]:
                    K[i,j] = b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params)

                elif 'image' in self.images[i] and self.images[j] == 'unresolved':
                    for n in range(self.n_images):
                        x_j = x[j] - self.deltas[n]
                        b_j = self._magnify(x_j, self.betas[n])

                        K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params) 

                elif self.images[i] == 'unresolved' and 'image' in self.images[j]:
                    for n in range(self.n_images):
                        x_i = x[i] - self.deltas[n]
                        b_i = self._magnify(x_i, self.betas[n])

                        K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params) 

                elif self.images[i] == 'unresolved' and self.images[j] == 'unresolved':
                    for n in range(self.n_images):
                        for m in range(self.n_images):
                            x_i = x[i] - self.deltas[n]
                            x_j = x[j] - self.deltas[m]
                            b_i = self._magnify(x_i, self.betas[n])
                            b_j = self._magnify(x_j, self.betas[m])

                            K[i,j] += b_i * b_j * self.kernel.covariance(x_i, x_j, params=kernel_params)

        cov_matrix = jnp.multiply(self.mask, K) + jnp.diag(yerr**2)
        return mean_vector, cov_matrix
    
    def import_from_gp(self, kernel, meanfunc, bands, images, indices):
        self.kernel = kernel
        self.meanfunc = meanfunc
        self.bands = bands
        self.n_bands = len(np.unique(bands))
        self.images = images
        self.n_images = len(np.unique(images))
        self.indices = indices
        self.repeats = self.indices[1:]-self.indices[:-1]
        self.mask = self._make_mask()

# Spline ML model (think about locations of knots, number of knots)
# Mixture of Sigmoids
# Look at special classes of polynomials
# Numerical Recipes (spline and interpolation chapters)
