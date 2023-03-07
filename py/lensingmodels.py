class LensingModel:
    """
    Initialize with lensing parameters
    """
    def __init__(self, lensing_params, n_images, n_bands):
        self.lensing_params = lensing_params
        self.delta = lensing_params[0]
        self.beta_naught = lensing_params[1]
        self.n_images = n_images
        self.n_bands = n_bands
        
    def reset(self, lensing_params):
        self.lensing_params = lensing_params
        self.delta = lensing_params[0]
        self.beta_naught = lensing_params[1]
        
    def rescale_data(self, y, yerr):
        factor = np.max(y)
        y_rescaled = y/factor
        yerr_rescaled = yerr/factor
        return y_rescaled, yerr_rescaled
    
    def prepare_magnification_matrix_bases(self, x):
        self.indices = [index for index in range(len(x)) if x[index] < x[index-1]]
        self.indices.append(len(x))
        
        temp = []
        for pb in range(self.n_bands):
            for n in range(self.n_images):
                
                vector = [0] * (self.indices[(self.n_images*pb)+n] - self.indices[0])
                vector = vector + ([1] * (self.indices[(self.n_images*pb)+1+n] - self.indices[(self.n_images*pb)+n]))
                vector = vector + ([0] * (self.indices[-1] - self.indices[(self.n_images*pb)+1+n]))
                temp.append(vector)
        
        self.base_matrices = {}
        for i in range(self.n_images):
            for j in range(self.n_images):
                for pb in range(self.n_bands):
                    if pb == 0:
                        self.base_matrices[i,j] = np.outer(temp[i+(self.n_images*pb)], temp[j+(self.n_images*pb)])
                    else:
                        self.base_matrices[i,j] = self.base_matrices[i,j] + np.outer(temp[i+(self.n_images*pb)], temp[j+(self.n_images*pb)])
        
    def magnification_matrix(self):
        try:
            betas = [1, self.beta_naught]
            for i in range(self.n_images):
                for j in range(self.n_images):
                    if i == 0 and j == 0:
                        output = self.base_matrices[i,j]
                    else:
                        output = output + (betas[i]*betas[j]*self.base_matrices[i,j])
        except AttributeError:
            raise Exception("You must initialize the magnification matrix by first running prepare_magnification_matrix_bases(xdata) to retrieve the magnification matrix basis vectors.")
                    
        return output

    def time_shift(self, x):
        for index in range(len(self.indices)-1):
            if index == 0:
                delta = np.repeat(0, self.indices[index+1]-self.indices[index])
            # ONLY WORKS FOR 2 IMAGES
            elif (index % 2) == 0:
                delta = np.concatenate([delta, np.repeat(0, self.indices[index+1]-self.indices[index])])
            else:
                delta = np.concatenate([delta, np.repeat(self.delta, self.indices[index+1]-self.indices[index])])
                
        return x - delta