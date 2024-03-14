import numpy as np
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot
plt.style.use('/data/eeh55/Github/GausSN/ipynb/stylesheet/GausSN.mplstyle')

def plot_object(data, color_dict={'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'darkorange'}, title='Gravitationally Lensed Supernova'):

    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(6,2*len(np.unique(data['band']))), sharex=True)

    if len(np.unique(data['band'])) > 1:
        ordered = np.array(['B_CSP', 'V_CSP', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'roman::Z', 'lssty', 'roman::Y', 'roman::J', 'roman::H', 'F105W', 'F125W', 'F160W'])
        bands = ordered[np.isin(ordered, data['band'])]
        for b, pb_id in enumerate(bands):
            band = data[data['band'] == pb_id]

            for im_id in np.unique(data['image']):
                image = band[band['image'] == im_id]

                _, _, bars = ax[b].errorbar(image['time'], image['flux_micro'], yerr=image['fluxerr_micro'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
                [bar.set_alpha(0.5) for bar in bars]
            band_label = pb_id[-1] + ' band' if not np.isin(pb_id, ['F105W', 'F125W', 'F160W']) else pb_id
            ax[b].set_ylabel(band_label, fontsize=14)

        ax[0].legend()
        ax[-1].set_xlabel('Time [days]', fontsize=14)
        ax[0].set_title(title, fontsize=24)

    else:
        for im_id in np.unique(data['image']):
            image = data[data['image'] == im_id]

            _, _, bars = ax.errorbar(image['time'], image['flux_micro'], yerr=image['fluxerr_micro'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
            [bar.set_alpha(0.5) for bar in bars]
        band_label = image['band'][0][-1] + ' band' if not np.isin(image['band'][0], ['F105W', 'F125W', 'F160W']) else image['band'][0]
        ax.set_ylabel(band_label, fontsize=14)

        ax.legend()
        ax.set_xlabel('Time [days]', fontsize=14)
        ax.set_title(title, fontsize=24)
    
    fig.supylabel('Flux', fontsize=20, y=0.494)
    plt.subplots_adjust(hspace=0)
    return fig, ax

def plot_object_mag(data, color_dict={'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'darkorange'}, title='Gravitationally Lensed Supernova'):

    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(8,3*len(np.unique(data['band']))), sharex=True)

    if len(np.unique(data['band'])) > 1:
        ordered = np.array(['B_CSP', 'V_CSP', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'roman::Z', 'lssty', 'roman::Y', 'roman::J', 'roman::H', 'F125W', 'F160W'])
        bands = ordered[np.isin(ordered, data['band'])]
        for b, pb_id in enumerate(bands):
            band = data[data['band'] == pb_id]

            for im_id in np.unique(data['image']):
                image = band[band['image'] == im_id]

                ax[b].errorbar(image['time'], image['mag'], yerr=image['magerr'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
            ax[b].set_ylabel(pb_id, fontsize=14)
            ax[b].invert_yaxis()

        ax[0].legend()
        ax[-1].set_xlabel('MJD [days]', fontsize=14)
        ax[0].set_title(title, fontsize=24)

    else:
        for im_id in np.unique(data['image']):
            image = band[band['image'] == im_id]

            ax.errorbar(image['time'], image['mag'], yerr=image['magerr'], ls='None', marker='.', color=color_dict[im_id], label=im_id)
        ax.set_ylabel(data['band'][0])
        ax.invert_yaxis()

        ax.legend()
        ax.set_xlabel('MJD', fontsize=14)
        ax.set_title(title, fontsize=24)
    
    fig.supylabel('Magnitude', fontsize=20, x=-0.06)
    plt.subplots_adjust(hspace=0)
    return fig

def make_traceplot(results, param_names=None, truths=None):
    plt.figure()
    fig, ax = dyplot.traceplot(results, show_titles=True, labels=param_names, truths=truths)
    fig.tight_layout()
    return fig, ax

def make_corner(results, param_names=None, truths=None):
    fig, ax = dyplot.cornerplot(results, show_titles=True, labels=param_names, truths=truths, label_kwargs={'fontsize': 16},
                                truth_color='crimson', truth_kwargs={'alpha': 1}, smooth=0.02, color='tab:blue',
                                hist_kwargs={'alpha': 0.6, 'histtype': 'stepfilled'}, hist2d_kwargs={'fill_contours': False},
                                quantiles=[0.16, 0.5, 0.84], title_quantiles=[0.16, 0.5, 0.84], quantiles_2d=[0.1, 0.4, 0.65, 0.85, 0.96])
    return fig, ax
