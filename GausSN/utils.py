import numpy as np
import matplotlib.pyplot as plt

def plot_object(data, color_dict={'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'darkorange'}, title='Gravitationally Lensed Supernova'):

    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(5,2*len(np.unique(data['band']))), sharex=True, sharey=True)

    if len(np.unique(data['band'])) > 1:
        ordered = np.array(['B_CSP', 'V_CSP', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'roman::Z', 'lssty', 'roman::Y', 'roman::J', 'roman::H', 'F105W', 'F125W', 'F160W'])
        bands = ordered[np.isin(ordered, data['band'])]
        for b, pb_id in enumerate(bands):
            band = data[data['band'] == pb_id]

            for im_id in np.unique(data['image']):
                image = band[band['image'] == im_id]

                markers, caps, bars = ax[b].errorbar(image['time'], image['flux'], yerr=image['fluxerr'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
                [bar.set_alpha(0.5) for bar in bars]
            band_label = pb_id[-1] + ' band' if not np.isin(pb_id, ['F105W', 'F125W', 'F160W']) else pb_id
            ax[b].set_ylabel(band_label)

        ax[0].legend()
        ax[-1].set_xlabel('Time [days]')
        ax[0].set_title(title)

    else:
        for im_id in np.unique(data['image']):
            image = data[data['image'] == im_id]

            markers, caps, bars = ax.errorbar(image['time'], image['flux'], yerr=image['fluxerr'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
            [bar.set_alpha(0.5) for bar in bars]
        band_label = image['band'][0][-1] + ' band' if not np.isin(image['band'][0], ['F105W', 'F125W', 'F160W']) else image['band'][0]
        ax.set_ylabel(band_label)

        ax.legend()
        ax.set_xlabel('Time [days]')
        ax.set_title(title)
    
    fig.supylabel('Flux', fontsize=20, x=-0.1)
    plt.subplots_adjust(hspace=0)
    return fig

def plot_object_mag(data, color_dict={'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'darkorange'}, title='Gravitationally Lensed Supernova'):

    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(5,2*len(np.unique(data['band']))), sharex=True)

    if len(np.unique(data['band'])) > 1:
        ordered = np.array(['B_CSP', 'V_CSP', 'lsstu', 'lsstg', 'lsstr', 'lssti', 'lsstz', 'roman::Z', 'lssty', 'roman::Y', 'roman::J', 'roman::H', 'F125W', 'F160W'])
        bands = ordered[np.isin(ordered, data['band'])]
        for b, pb_id in enumerate(bands):
            band = data[data['band'] == pb_id]

            for im_id in np.unique(data['image']):
                image = band[band['image'] == im_id]

                ax[b].errorbar(image['time'], image['mag'], yerr=image['magerr'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
            ax[b].set_ylabel(pb_id)
            ax[b].invert_yaxis()

        ax[0].legend()
        ax[-1].set_xlabel('MJD [days]')
        ax[0].set_title(title)

    else:
        for im_id in np.unique(data['image']):
            image = band[band['image'] == im_id]

            ax.errorbar(image['time'], image['mag'], yerr=image['magerr'], ls='None', marker='.', color=color_dict[im_id], label=im_id)
        ax.set_ylabel(data['band'][0])
        ax.invert_yaxis()

        ax.legend()
        ax.set_xlabel('MJD')
        ax.set_title(title)
    
    fig.supylabel('Magnitude', fontsize=20, x=-0.1)
    plt.subplots_adjust(hspace=0)
    return fig
