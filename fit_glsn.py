#!/home/eeh55/.conda/envs/glsnEnv/bin/python3.10
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from astropy.io import ascii, fits
from astropy.table import Table, vstack
from dynesty import utils as dyfunc

from GausSN import gausSN, kernels, meanfuncs, lensingmodels

parser = argparse.ArgumentParser(description="GausSN Strongly Lensed Supernovae Light Curve Fitting Pipeline")
parser.add_argument("--method", default='dynesty', help="Sampling method. Choice of 'dynesty' or 'emcee'. Defaults to 'dynesty'.")
parser.add_argument("lcpath", help="Path to glSNe light curve file.")
parser.add_argument("savepath", help="Path to a folder where output files will be saved.")
args = parser.parse_args()
    
def run_gaussn(snid, data):
                                        
    print('Initializing GP')
    meanfunc_params = [0]
    meanfunc = meanfuncs.UniformMean(meanfunc_params)
    kernel = kernels.ExpSquaredKernel([0.8, 30])
    gp = gausSN.GP(kernel, meanfunc)

    image1 = data[data['image'] == 'image_1'].to_pandas().reset_index()
    image2 = data[data['image'] == 'image_2'].to_pandas().reset_index()
    peak_im1_loc = np.argmax(image1['flux'].rolling(3).mean())
    peak_im2_loc = np.argmax(image2['flux'].rolling(3).mean())
    init_delta = image2.loc[peak_im2_loc]['time'] - image1.loc[peak_im1_loc]['time']
    init_beta = image2.loc[peak_im2_loc]['flux'] / image1.loc[peak_im1_loc]['flux']
    
    lm = lensingmodels.SigmoidMicrolensing_LensingModel([init_delta, init_beta, init_beta, 0, np.mean(data['time'])])

    def ptform(u):
        prior = u
        #prior[0] = (u[0] * 1)
        #prior[1] = (u[1] * 10) + 20
        prior[0] = (u[0] * 650) - 275
        prior[1] = (u[1] * 52) + 0.1
        prior[2] = norm.ppf(u[2], loc=prior[1], scale=0.5)
        prior[3] = norm.ppf(u[3], loc=0, scale=0.5)

        delta_vector = np.repeat(np.tile([0, prior[0]], gp.n_bands), gp.indices[1:]-gp.indices[:-1])
        shifted_time = data['time'] - delta_vector
    
        prior[4] = (u[4] * (np.max(shifted_time) - np.min(shifted_time))) + np.min(shifted_time)

        return(prior)

    def log_prior(params):

        delta_vector = np.repeat(np.tile([0, params[0]], gp.n_bands), gp.indices[1:]-gp.indices[:-1])
        shifted_time = data['time'] - delta_vector

        if params[0] < -275 or params[0] > 375 or params[1] < 0.1 or params[1] > 52.1 or params[4] < np.min(shifted_time) or params[4] > np.max(shifted_time):
            return -np.inf

        beta1_r_mu = np.array([params[1], 0])
        beta1_r_sigma = np.diag(np.array([0.5, 0.5]))
        mnv_log_pdf = np.log(np.linalg.det(2 * np.pi * beta1_r_sigma)) + np.dot(np.transpose(np.array([params[2], params[3]]) - beta1_r_mu), np.linalg.solve(beta1_r_sigma, (np.array([params[2], params[3]]) - beta1_r_mu)))
        
        return -0.5 * mnv_log_pdf

    print('Optimizing parameters')
    if args.method == 'emcee':
        sampler = gp.optimize_parameters(x = data['time'], y = data['flux'], yerr = data['fluxerr'], band = data['band'], image = data['image'],
                                         method='emcee', logprior = log_prior, lensing_model = lm, fix_mean_params=True, fix_kernel_params=True)
    elif args.method == 'dynesty':
        sampler = gp.optimize_parameters(x = data['time'], y = data['flux'], yerr = data['fluxerr'], band = data['band'], image = data['image'],
                                         method='dynesty', ptform=ptform, sampler_kwargs = {'sample': 'rslice', 'nlive': 500},
                                         lensing_model = lm, fix_mean_params=True, fix_kernel_params=True)
    return sampler

if not os.path.exists(args.savepath) or not os.path.exists(args.lcpath):
    raise ValueError("Provided savepath and/or lcpath does not exist!")

param_names = ['delta', 'beta0', 'beta1', 'r', 't0']
output_dict = {}

snid = args.lcpath.split('/')[-1].split('.')[0]

print('Starting ', snid)
starttime = time.time()
    
hdul = fits.open(args.lcpath)
data1 = Table(hdul[1].data)
data2 = Table(hdul[2].data)
data = vstack([data1, data2])
data.sort(['band', 'image', 'time'])

sampler = run_gaussn(snid, data)

if args.method == 'dynesty':
    mean, cov = dyfunc.mean_and_cov(sampler.results.samples, sampler.results.importance_weights())
    pickle.dump(sampler.results, open(args.savepath+'chains/'+snid+'_chains.pkl', 'wb'))

    output_dict = {}
    for i, pn in enumerate(param_names):
        output_dict[pn] = mean[i]
        output_dict[pn+'_err'] = np.sqrt(np.diag(cov))[i]

elif args.method == 'emcee':
    flat_chains = sampler.get_chain(discard=500, flat=True)
    np.savetxt(args.savepath+'chains/'+snid+'_chains.dat', sampler.get_chain(flat=True))
        
    for i, pn in enumerate(param_names):
        output_dict[pn] = np.mean(flat_chains, axis=0)[i]
        output_dict[pn+'_err'] = np.std(flat_chains, axis=0)[i]

np.save(args.savepath+snid+'_output.npy', output_dict)
        
endtime = time.time()
print(f'Done {snid} in {(endtime-starttime)/60} mins') 
