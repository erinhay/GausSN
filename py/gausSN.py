import numpy as np

def ExpSquaredKernel(y, y_prime, params):
    K = np.zeros([len(y), len(y_prime)])
    for i in range(len(y)):
        for j in range(len(y_prime)):
            K[i,j] = params[0]**2 * np.exp(-(y[i] - y_prime[j])**2/(2*params[1]**2))
    return K

def UniformKernel(y, y_prime, params):
    K = np.zeros([len(y), len(y_prime)])
    for i in range(len(y)):
        for j in range(len(y_prime)):
            K[i,j] = params[0]
    return K

def DotProductKernel(y, y_prime, params):
    K = np.zeros([len(y), len(y_prime)])
    for i in range(len(y)):
        for j in range(len(y_prime)):
            K[i,j] = np.dot(y[i], y_prime[j])
    return K

def Matern32Kernel(y, y_prime, params):
    K = np.zeros([len(y), len(y_prime)])
    for i in range(len(y)):
        for j in range(len(y_prime)):
            r = np.linalg.norm(y[i]-y_prime[j])
            r2 = r**2
            K[i,j] = (1 + np.sqrt(3*r2)) * np.exp(-np.sqrt(3*r2))
    return K

def Matern52Kernel(y, y_prime, params):
    K = np.zeros([len(y), len(y_prime)])
    for i in range(len(y)):
        for j in range(len(y_prime)):
            r = np.linalg.norm(y[i]-y_prime[j])
            r2 = r**2
            K[i,j] = (1 + np.sqrt(5*r2) + (5*r2/3)) * np.exp(-np.sqrt(5*r2))
    return K

def UniformMean(y, params):
    return np.repeat(params[0], len(y))

def Sin(y, params):
    return params[0] * np.sin((y*params[1]) + params[2])

def Bezin2009(y, params):
    """
    params = [A, B, t0, tau_fall, tau_rise]
    """
    mu = np.zeros([len(y)])
    
    a = np.exp(-(y - params[2])/params[3])
    b = 1 + np.exp((y - params[2])/params[4])
    mu = params[0] * (a/b) + params[1]
    return mu

def Karpenka2012(y, params):
    """
    params = [A, B, t1, t0, T_fall, T_rise]
    """
    mu = np.zeros([len(y)])
    
    a = 1 + (params[1]*((y - params[2])**2))
    b = np.exp(-(y - params[3])/params[4])
    c = 1 + np.exp(-(y - params[3])/params[5])
    mu = params[0] * a * (b/c)
    return mu

def Villar2019(y, params):
    """
    params = [A, beta, t1, t0, T_fall, T_rise]
    """
    mu = np.zeros([len(y)])
    
    for i in range(len(y)):
        if y < params[2]:
            a = params[0] + (params[1] * (y[i] - params[3]))
            b = 1 + np.exp(-(y[i] - params[3])/params[5])
            mu[i] = a/b
        else:
            a = params[0] + (params[1] * (params[2] - params[3]))
            b = 1 + np.exp(-(y[i] - params[3])/params[5])
            c = np.exp(-(y[i] - params[2])/params[5])
            mu[i] = (a*c)/b
    return mu

def ExpFunction(y, params):
    """
    params = [A, tau]
    """
    mu = np.zeros([len(y)])
    
    mu = params[0] * np.exp(y*params[1])
    return mu

def GP(U, V_x, V_y, kernel, kernel_params, mean, mean_params):
    """
    expectation = mu_U + (cov_UV * cov_VV^-1) * (V - mu_V)
    variance = cov_UU - (cov_UV * cov_VV^-1 * cov_VU)
    
    U = new data
    V_x = observed data, x
    V_y = observed data, y
    """
    cov_UV = kernel(U, V_x, kernel_params)
    cov_VV = kernel(V_x, V_x, kernel_params)
    cov_UU = kernel(U, U, kernel_params)
    
    mu_U = mean(U, mean_params)
    mu_V = mean(V_x, mean_params)
    
    expectation = mu_U + (cov_UV @ np.linalg.inv(cov_VV) @ (V_y-mu_V))
    variance = cov_UU - (cov_UV @ np.linalg.inv(cov_VV) @ np.transpose(cov_UV))
    
    return expectation, variance
