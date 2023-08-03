import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table, vstack
from dynesty import utils as dyfunc

from GausSN import gausSN, kernels, meanfuncs, lensingmodels

parser = argparse.ArgumentParser(description="GausSN Strongly Lensed Supernovae Light Curve Fitting Pipeline")
parser.add_argument("lcdir", help="Path to folder with glSNe light curve files.")
parser.add_argument("savepath", help="Path to a folder where output files will be saved.")
args = parser.parse_args()

def log_prior(params):
    if params[0] > 0 and params[0] < 1 and params[1] > 20 and params[1] < 30 and params[2] > 0 and params[2] < 100 and params[3] > 0.01 and params[3] < 15:
        return 0
    else:
        return -np.inf
    
def run_gaussn(snid, data):
    
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
    
    print('Initializing GP')
    meanfunc_params = [0]
    meanfunc = meanfuncs.UniformMean(meanfunc_params)
    kernel = kernels.ExpSquaredKernel([0.5, 25])
    gp = gausSN.GP(kernel, meanfunc)

    init_delta = data[np.argmax(data[data['image'] == 'image_2']['flux'])]['time'] - data[np.argmax(data[data['image'] == 'image_1']['flux'])]['time']
    init_beta = np.max(data[data['image'] == 'image_2']['flux']) / np.max(data[data['image'] == 'image_1']['flux'])
    
    lm = lensingmodels.LensingModel([init_delta, init_beta], len(im_ids), len(pb_ids), indices)

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
    output_table = Table.read(args.savepath+'summary_table.dat', format='ascii')
    output_dict = np.load(args.savepath+'summary_dict.npy', allow_pickle=True).item()
except:
    output_table = Table(names=['snid', 'delta', 'delta_err', 'beta', 'beta_err'], dtype = ['str', 'float', 'float', 'float', 'float'])
    output_dict = {}

param_names = ['A', 'tau', 'delta', 'beta']
    
for fn in os.listdir(args.lcdir):
    snid = fn.split('.')[0]
    
    if np.isin(snid, output_table['snid']):
        continue
    
    print('Starting ', snid)
    starttime = time.time()
    
    try:
        data = Table.read(args.lcdir+fn)
        sampler = run_gaussn(snid, data)
    except:
        print(f'Fail: {snid}')
        continue
    
    flat_chains = sampler.get_chain(discard=500, flat=True)
    np.savetxt(args.savepath+'chains/'+snid+'chains.dat', sampler.get_chain(flat=True))
    
    try:
        output_table = Table.read(args.savepath+'summary_table.dat', format='ascii')
        output_dict = np.load(args.savepath+'summary_dict.npy', allow_pickle=True).item()
    except:
        output_table = Table(names=['snid', 'delta', 'delta_err', 'beta', 'beta_err'],
                     dtype = ['str', 'float', 'float', 'float', 'float'])
        output_dict = {}
        
    output_dict[snid] = {}
    for i, pn in enumerate(param_names):
        output_dict[snid][pn] = np.mean(flat_chains, axis=0)[i]
        output_dict[snid][pn+'_err'] = np.std(flat_chains, axis=0)[i]
    np.save(args.savepath+'summary_dict.npy', output_dict)
        
    new_row = [snid, np.mean(flat_chains, axis=0)[2], np.std(flat_chains, axis=0)[2], np.mean(flat_chains, axis=0)[3], np.std(flat_chains, axis=0)[3]]
    output_table.add_row(new_row)
    output_table.write(args.savepath+'summary_table.dat', format='ascii', overwrite=True)
    
    endtime = time.time()
    print(f'Done {snid} in {(endtime-starttime)/60} mins')   
