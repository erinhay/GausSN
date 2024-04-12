import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot
from GausSN import gausSN, kernels, meanfuncs, lensingmodels
plt.style.use('/data/eeh55/Github/GausSN/ipynb/stylesheet/GausSN.mplstyle')

ordered = np.array(['B_CSP', 'V_CSP', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'roman::Z', 'lssty', 'roman::Y', 'roman::J', 'roman::H', 'F105W', 'F125W', 'F160W', 'EulerCAM', 'WFI'])

def plot_object(data, color_dict={'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'darkorange'}, title='Gravitationally Lensed Supernova'):

    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(8,3*len(np.unique(data['band']))), sharex=True)

    if len(np.unique(data['band'])) > 1:
        bands = ordered[np.isin(ordered, data['band'])]
        for b, pb_id in enumerate(bands):
            band = data[data['band'] == pb_id]

            for im_id in np.unique(data['image']):
                image = band[band['image'] == im_id]

                markers, caps, bars = ax[b].errorbar(image['time'], image['flux'], yerr=image['fluxerr'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
                [bar.set_alpha(0.5) for bar in bars]
            band_label = pb_id[-1] + ' band' if not np.isin(pb_id, ['F105W', 'F125W', 'F160W']) else pb_id
            ax[b].set_ylabel(band_label, fontsize=14)

        ax[0].legend()
        ax[-1].set_xlabel('Time [days]', fontsize=14)
        ax[0].set_title(title, fontsize=24)

    else:
        for im_id in np.unique(data['image']):
            image = data[data['image'] == im_id]

            markers, caps, bars = ax.errorbar(image['time'], image['flux'], yerr=image['fluxerr'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
            [bar.set_alpha(0.5) for bar in bars]
        band_label = image['band'][0][-1] + ' band' if not np.isin(image['band'][0], ['F105W', 'F125W', 'F160W']) else image['band'][0]
        ax.set_ylabel(band_label, fontsize=14)

        ax.legend()
        ax.set_xlabel('Time [days]', fontsize=14)
        ax.set_title(title, fontsize=24)
    
    fig.supylabel('Flux', fontsize=20, x=-0.06)
    plt.subplots_adjust(hspace=0)
    return fig, ax

def plot_fitted_object(data, results, kernel, meanfunc, lensingmodel, fix_kernel_params=False, fix_mean_params=False, fix_lensing_params=False, title=''):

    bands = ordered[np.isin(ordered, data['band'])]
    color_dict = {'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'tab:orange'}

    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(8,3*len(np.unique(bands))), sharex=True, sharey=True)

    for b, pb_id in enumerate(bands):
        band = data[data['band'] == pb_id]

        for im_id in np.unique(data['image']):
            image = band[band['image'] == im_id]

            ax[b].errorbar(image['time'], image['flux'], yerr=image['fluxerr'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1], zorder=1)
        ax[b].set_ylabel(pb_id[-1] + ' band', fontsize=16)

    color_dict = {'image_1': 'tab:blue', 'image_2': 'palevioletred', 'image_3': 'tab:green', 'image_4': 'darkorange'}
    samples = results.samples_equal()
    
    N_obs = 50
    predict_times = np.linspace(-30, 110, N_obs)

    for iter in np.random.choice(len(samples), 200):
        sample = samples[iter]

        if not fix_lensing_params and not fix_mean_params and not fix_kernel_params:
            kernel_params = [sample[i] for i in range(len(kernel.params))]
            meanfunc_params = [sample[i+len(kernel.params)] for i in range(len(meanfunc.params))]
            lensing_params = [sample[i+len(kernel.params)+len(meanfunc.params)] for i in range(len(lensingmodel.params))]
            kernel._reset(kernel_params)
            meanfunc._reset(meanfunc_params)
            lensingmodel._reset(lensing_params)
        elif not fix_mean_params and not fix_kernel_params:
            kernel_params = [sample[i] for i in range(len(kernel.params))]
            meanfunc_params = [sample[i+len(kernel.params)] for i in range(len(meanfunc.params))]
            kernel._reset(kernel_params)
            meanfunc._reset(meanfunc_params)
        elif not fix_mean_params and not fix_lensing_params:
            meanfunc_params = [sample[i] for i in range(len(meanfunc.params))]
            lensing_params = [sample[i+len(meanfunc.params)] for i in range(len(lensingmodel.params))]
            meanfunc._reset(meanfunc_params)
            lensingmodel._reset(lensing_params)
        elif not fix_kernel_params and not fix_lensing_params:
            kernel_params = [sample[i] for i in range(len(kernel.params))]
            lensing_params = [sample[i+len(kernel.params)] for i in range(len(lensingmodel.params))]
            kernel._reset(kernel_params)
            lensingmodel._reset(lensing_params)
        elif not fix_kernel_params:
            kernel_params = [sample[i] for i in range(len(kernel.params))]
            kernel._reset(kernel_params)
        elif not fix_mean_params:
            meanfunc_params = [sample[i] for i in range(len(meanfunc.params))]
            meanfunc._reset(meanfunc_params)
        elif not fix_lensing_params:
            lensing_params = [sample[i] for i in range(len(lensingmodel.params))]
            lensingmodel._reset(lensing_params)
        
        gp = gausSN.GP(kernel, meanfunc, lensingmodel)
        
        for b, pb_id in enumerate(bands):
            band = data[data['band'] == pb_id]

            repeats = np.array([len(band[band['image'] == pb_id]) for pb_id in np.unique(data['image'])])
            lensingmodel.repeats = repeats
            lensingmodel.n_bands = 1
            shifted_time_data, b_vector = lensingmodel._lens(jnp.array(band['time'].value))

            exp, cov = gp.predict(predict_times, shifted_time_data, band['flux']/b_vector, band['fluxerr']/b_vector, band = [pb_id])
            
            for i in range(1):
                y_vals = np.random.multivariate_normal(mean=exp, cov=cov, size=1)

                for m, im_id in enumerate(np.unique(data['image'])):
                    repeats = np.zeros(len(np.unique(data['image'])), dtype='int')
                    repeats[m] = int(len(predict_times))
                    lensingmodel.repeats = repeats
                    shifted_predict_times, b_vector_predict = lensingmodel._lens(jnp.array(predict_times))

                    ax[b].plot(predict_times + lensingmodel.deltas[m], y_vals[0] * b_vector_predict, color=color_dict[im_id], alpha=0.02, zorder=2)
            
    ax[0].legend(loc='upper right')
    ax[-1].set_xlabel('Time [days]', fontsize=16)
    ax[0].set_title(title, fontsize=24)

    ax[0].set_ylim(-0.5, 1.5)

    fig.supylabel('Flux', fontsize=20, y=0.494)
    return fig, ax

def make_traceplot(results, param_names=None, truths=None):
    plt.figure()
    fig, ax = dyplot.traceplot(results, show_titles=True, labels=param_names, truths=truths)
    fig.tight_layout()
    return fig, ax

def make_corner(results, param_names=None, truths=None):
    fig, ax = dyplot.cornerplot(results, show_titles=True, labels=param_names, truths=truths, label_kwargs={'fontsize': 16},
                                truth_color='crimson', truth_kwargs={'alpha': 1}, smooth=0.02, color='tab:blue',
                                hist_kwargs={'alpha': 0.6, 'histtype': 'stepfilled'}, hist2d_kwargs={'fill_contours': False},
                                quantiles=[0.025, 0.5, 0.975], quantiles_2d=[0.1, 0.4, 0.65, 0.85, 0.96])
    return fig, ax
