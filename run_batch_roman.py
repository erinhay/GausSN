import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from astropy.table import Table, vstack
from dynesty import utils as dyfunc

from GausSN import gausSN, kernels, meanfuncs, lensingmodels

parser = argparse.ArgumentParser(description="GausSN Strongly Lensed Supernovae Light Curve Fitting Pipeline")
parser.add_argument("lcdir", help="Path to folder with glSNe light curve files.")
parser.add_argument("savepath", help="Path to a folder where output files will be saved.")
args = parser.parse_args()

def ptform(u):
    prior = u
    prior[0] = (u[0] * 1)
    prior[1] = (u[1] * 10) + 20
    prior[2] = (u[2] * 650) - 275
    prior[3] = (u[3] * 52) + 0.1
    return(prior)

def log_prior(params):

    delta_vector = np.repeat(np.tile([1, params[2]], len(pb_ids)), np.array(indices[1:])-np.array(indices[:-1]))
    shifted_xdat = xdat - delta_vector

    if params[0] < 0 or params[0] > 1 or params[1] < 20 or params[1] > 30 or params[2] < -275 or params[2] > 375 or params[3] < 0.1 or params[3] > 52.1 or params[6] < np.min(shifted_xdat) or params[6] > np.max(shifted_xdat):
        return -np.inf

    beta1_r_mu = np.array([params[3], 0])
    beta1_r_sigma = np.diag(np.array([0.1, 1]))
    mnv_log_pdf = np.log(np.linalg.det(2 * np.pi * beta1_r_sigma)) + np.dot(np.transpose(np.array([params[4], params[5]]) - beta1_r_mu), np.linalg.solve(beta1_r_sigma, (np.array([params[4], params[5]]) - beta1_r_mu)))
    
    return -0.5 * mnv_log_pdf
    
def run_gaussn(snid, data):
                                        
    print('Initializing GP')
    meanfunc_params = [0]
    meanfunc = meanfuncs.UniformMean(meanfunc_params)
    kernel = kernels.ExpSquaredKernel([0.5, 25])
    gp = gausSN.GP(kernel, meanfunc)

    init_delta = data[np.argmax(data[data['image'] == 'image_2']['flux'])]['time'] - data[np.argmax(data[data['image'] == 'image_1']['flux'])]['time']
    init_beta = np.max(data[data['image'] == 'image_2']['flux']) / np.max(data[data['image'] == 'image_1']['flux'])
    
    lm = lensingmodels.SigmoidMicrolensing_LensingModel([init_delta, init_beta, init_beta, 0, np.mean(xdat)], len(im_ids), len(pb_ids), indices)

    print('Optimizing parameters')
    sampler = gp.optimize_parameters(x = xdat, y = ydat, yerr = yerrdat, n_bands = len(pb_ids),
                                     method='zeus', logprior=log_prior,
                                     #method='MCMC', logprior=log_prior,
                                     #method='nested_sampling', ptform=ptform,
                                     #fix_kernel_params=True
                                     lensing_model = lm, fix_mean_params=True)
    return sampler

if not os.path.exists(args.savepath):
    raise ValueError("Provided savepath does not exist!")
    
try:
    output_table = Table.read(args.savepath+'summary_table_microlensing.dat', format='ascii')
    output_dict = np.load(args.savepath+'summary_dict_microlensing.npy', allow_pickle=True).item()
except:
    output_table = Table(names=['snid', 'mass_model', 'delta', 'delta_err', 'beta', 'beta_err'], dtype = ['str', 'str', 'float', 'float', 'float', 'float'])
    output_dict = {}
    output_dict[mass_model] = {}

param_names = ['A', 'tau', 'delta', 'beta0', 'beta1', 'r', 't0']

for fn in os.listdir(args.lcdir):
    snid = fn.split('.')[0]
    mass_model = args.lcdir.split('/')[-2]
    
    if np.isin(snid, output_table['snid']):
        continue
    
    print('Starting ', snid)
    starttime = time.time()
    
    try:
        hdul = fits.open(args.lcdir+fn)
        data1 = Table(hdul[1].data)
        data2 = Table(hdul[2].data)
        data = vstack([data1, data2])
    except:
        print(f'Fail: {snid}')
        continue

    im_ids = np.unique(data['image'])
    pb_ids = np.unique(data['band'])

    indices = [0]
    for j, pb_id in enumerate(pb_ids):
        band = data[data['band'] == pb_id]
        for i, im_id in enumerate(im_ids):
            if i == 0 and j == 0:
                xdat = band[band['image'] == im_id]['time']
                ydat = band[band['image'] == im_id]['flux']
                yerrdat = band[band['image'] == im_id]['fluxerr']
                indices.append(len(xdat))
            else:
                xdat = np.concatenate([xdat, band[band['image'] == im_id]['time'].value])
                ydat = np.concatenate([ydat, band[band['image'] == im_id]['flux'].value])
                yerrdat = np.concatenate([yerrdat, band[band['image'] == im_id]['fluxerr'].value])
                indices.append(len(xdat))

    try:
        sampler = run_gaussn(snid, data)
    except:
        print(f'Fail: {snid}')
        continue

    flat_chains = sampler.get_chain(discard=500, flat=True)
    np.savetxt(args.savepath+'chains/'+snid+'_chains.dat', sampler.get_chain(flat=True))
    
    try:
        output_table = Table.read(args.savepath+'summary_table_microlensing.dat', format='ascii')
        output_dict = np.load(args.savepath+'summary_dict_microlensing.npy', allow_pickle=True).item()
    except:
        output_table = Table(names=['snid', 'mass_model', 'delta', 'delta_err', 'beta', 'beta_err'],
                     dtype = ['str', 'str', 'float', 'float', 'float', 'float'])
        output_dict = {}
        output_dict[mass_model] = {}
        
    output_dict[mass_model][snid] = {}
    for i, pn in enumerate(param_names):
        output_dict[mass_model][snid][pn] = np.mean(flat_chains, axis=0)[i]
        output_dict[mass_model][snid][pn+'_err'] = np.std(flat_chains, axis=0)[i]
    np.save(args.savepath+'summary_dict_microlensing.npy', output_dict)
        
    new_row = [snid, mass_model, np.mean(flat_chains, axis=0)[2], np.std(flat_chains, axis=0)[2], np.mean(flat_chains, axis=0)[3], np.std(flat_chains, axis=0)[3]]
    output_table.add_row(new_row)
    output_table.write(args.savepath+'summary_table_microlensing.dat', format='ascii', overwrite=True)
    
    endtime = time.time()
    print(f'Done {snid} in {(endtime-starttime)/60} mins') 
