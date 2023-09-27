import numpy as np
import matplotlib.pyplot as plt

def plot_object(data, color_dict={'image_1': 'navy', 'image_2': 'crimson', 'image_3': 'tab:green', 'image_4': 'darkorange'}, title='Gravitationally Lensed Supernova'):

    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(5,2*len(np.unique(data['band']))), sharex=True)

    if len(np.unique(data['band'])) > 1:
        for b, pb_id in enumerate(np.unique(data['band'])):
            band = data[data['band'] == pb_id]

            for im_id in np.unique(data['image']):
                image = band[band['image'] == im_id]

                ax[b].errorbar(image['time'], image['flux'], yerr=image['fluxerr'], ls='None', marker='.', color=color_dict[im_id], label='Image '+im_id[-1])
            ax[b].set_ylabel(pb_id[-1] + ' band', fontsize=14)

        ax[0].legend(fontsize=12)
        ax[-1].set_xlabel('MJD [days]', fontsize=18)
        ax[0].set_title(title)

    else:
        for b, pb_id in enumerate(np.unique(data['band'])):
            band = data[data['band'] == pb_id]

            for im_id in np.unique(data['image']):
                image = band[band['image'] == im_id]

                ax.errorbar(image['time'], image['flux'], yerr=image['fluxerr'], ls='None', marker='.', color=color_dict[im_id], label=im_id)
            ax.set_ylabel(pb_id, fontsize=14)

        ax.legend()
        ax.set_xlabel('MJD', fontsize=14)
        ax.set_title(title)
    
    fig.supylabel('Flux', fontsize=18, x=-0.075)
    plt.subplots_adjust(hspace=0)
    return fig

