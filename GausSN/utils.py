import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot

from GausSN import gausSN

plt.style.use('/data/eeh55/Github/GausSN/ipynb/stylesheet/GausSN.mplstyle')

# Array specifying the order of bands in increasing wavelength
ordered = np.array(['uvf475w', 'uvf625w', 'uvf814w', 'B_CSP', 'V_CSP', 'lsstu', 'lsstg', 'ztfg', 'ps1::g', 'IOOg', 'lsstr', 'ztfr', 'ps1::r', 'IOOr',
                    'lssti', 'ps1::i', 'IOOi', 'lsstz', 'IOOz', 'roman::Z', 'lssty', 'roman::Y', 'HAWKI_Y', 'roman::J', 'HAWKI_J',
                    'roman::H', 'HAWKI_H', 'HAWKI_K', 'f105w', 'f110w', 'f125w', 'f160w', 'f475w', 'EulerCAM', 'WFI'])

def plot_object(data, color_dict={'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'darkorange', 'unresolved': 'k'}, marker_dict={'image_1': 'o', 'image_2': 's', 'image_3': '>', 'image_4': '<', 'unresolved': '.'}, title='Gravitationally Lensed Supernova'):
    """
    Plots the glSN light curve data.

    Args:
        data (Table): Data containing flux measurements for different images and bands.
        color_dict (dict): Dictionary mapping image number (i.e. 'image_1') to colors.
        title (str): Title of the plot.

    Returns:
        tuple: The figure and axis of the plot.
    """
    if 'fluxcal' in data.columns:
        key = 'fluxcal'
    elif 'flux' in data.columns:
        key = 'flux'
    else:
        raise NameError("Please make sure you have a column labeled either flux or fluxcal in your data table!")

    # Create subplots based on the number of unique bands
    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(6, 2*len(np.unique(data['band']))), sharex=True)

    if len(np.unique(data['band'])) > 1:
        # If there are multiple bands, iterate over each band
        bands = ordered[np.isin(ordered, data['band'])]
        for b, pb_id in enumerate(bands):
            band = data[data['band'] == pb_id]

            try:
                color_dict_temp = color_dict[pb_id]
            except:
                color_dict_temp = color_dict

            try:
                marker_dict_temp = marker_dict[pb_id]
            except:
                marker_dict_temp = marker_dict

            # Plot flux for each image in the band
            for im_id in np.unique(data['image']):
                image = band[band['image'] == im_id]

                try:
                    color = color_dict_temp[im_id]
                except:
                    color = color_dict_temp

                try:
                    marker = marker_dict_temp[im_id]
                except:
                    marker = marker_dict_temp

                image_label = 'Image '+im_id[-1] if not im_id == 'unresolved' else im_id
                _, _, bars = ax[b].errorbar(image['time'], image[key], yerr=image[key+'err'], ls='None', marker=marker, color=color, label=image_label)
                [bar.set_alpha(0.5) for bar in bars]
            # Set ylabel for the band
            band_label = pb_id[-1] + ' band' if not np.isin(pb_id, ['f105w', 'f110w', 'f125w', 'f160w', 'f475w', 'uvf475w', 'uvf625w', 'uvf814w', 'WFI', 'EulerCAM']) else pb_id
            ax[b].set_ylabel(band_label, fontsize=14)

        # Add legend and xlabel
        ax[0].legend()
        ax[-1].set_xlabel('Time [days]', fontsize=14)
        ax[0].set_title(title, fontsize=24)

    else:
        # If there's only one band, use ax instead of indexing ax
        for im_id in np.unique(data['image']):
            image = data[data['image'] == im_id]

            image_label = 'Image '+im_id[-1] if not im_id == 'unresolved' else im_id
            _, _, bars = ax.errorbar(image['time'], image[key], yerr=image[key+'err'], ls='None', marker=marker_dict[im_id], color=color_dict[im_id], label=image_label)
            [bar.set_alpha(0.5) for bar in bars]
        # Set ylabel for the single band
        band_label = pb_id[-1] + ' band' if not np.isin(pb_id, ['f105w', 'f110w', 'f125w', 'f160w', 'uvf475w', 'uvf625w', 'uvf814w', 'WFI', 'EulerCAM']) else pb_id
        ax.set_ylabel(band_label, fontsize=14)

        # Add legend and xlabel
        ax.legend()
        ax.set_xlabel('Time [days]', fontsize=14)
        ax.set_title(title, fontsize=24)
    
    # Set ylabel for the flux and adjust subplot spacing
    fig.supylabel('Flux', fontsize=20, y=0.494)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, ax

def plot_fitted_object(data, results, kernel, meanfunc, lensingmodel, fix_kernel_params=False, fix_mean_params=False, fix_lensing_params=False, predict_times = np.linspace(-30, 110, 50), color_dict_data = {'image_1': 'darkblue', 'image_2': 'crimson', 'image_3': 'darkgreen', 'image_4': 'tab:orange'}, color_dict_fit = {'image_1': 'tab:blue', 'image_2': 'palevioletred', 'image_3': 'tab:green', 'image_4': 'darkorange'}, marker_dict={'image_1': 'o', 'image_2': 's', 'image_3': '>', 'image_4': '<'}, title=''):
    """
    Plots the fitted glSN with uncertainties.

    Args:
        data (Table): Data containing flux measurements.
        results (obj): Results of the fitting.
        kernel (obj): Kernel object.
        meanfunc (obj): Mean function object.
        lensingmodel (obj): Lensing model object.
        fix_kernel_params (bool): If True, fixes kernel parameters.
        fix_mean_params (bool): If True, fixes mean function parameters.
        fix_lensing_params (bool): If True, fixes lensing model parameters.
        color_dict_data (dict): Dictionary mapping image number (i.e. 'image_1') to colors for plotting data.
        color_dict_fit (dict): Dictionary mapping image number (i.e. 'image_1') to colors for plotting fitted light curves.
        title (str): Title of the plot.

    Returns:
        tuple: The figure and axis of the plot.
    """

    # Array specifying the order of bands
    bands = ordered[np.isin(ordered, data['band'])]

    # Create subplots based on the number of unique bands
    fig, ax = plt.subplots(len(np.unique(data['band'])), 1, figsize=(8, 3*len(np.unique(bands))), sharex=True)

    # Plot flux measurements for each band and image
    for b, pb_id in enumerate(bands):
        band = data[data['band'] == pb_id]

        try:
            color_dict_data_temp = color_dict_data[pb_id]
        except:
            color_dict_data_temp = color_dict_data

        try:
            marker_dict_temp = marker_dict[pb_id]
        except:
            marker_dict_temp = marker_dict

        for im_id in np.unique(data['image']):
            image = band[band['image'] == im_id]

            try:
                color = color_dict_data_temp[im_id]
            except:
                color = color_dict_data_temp

            try:
                marker = marker_dict_temp[im_id]
            except:
                marker = marker_dict_temp

            image_label = 'Image '+im_id[-1] if not im_id == 'unresolved' else im_id
            ax[b].errorbar(image['time'], image['flux'], yerr=image['fluxerr'], ls='None', marker=marker, color=color, label=image_label, zorder=1)
        band_label = pb_id[-1] + ' band' if not np.isin(pb_id, ['f105w', 'f110w', 'f125w', 'f160w', 'f475w', 'uvf475w', 'uvf625w', 'uvf814w', 'WFI', 'EulerCAM']) else pb_id
        ax[b].set_ylabel(band_label, fontsize=16)

    # Get equal-weighted samples from the results
    samples = results.samples_equal()

    # Iterate over random samples from the posterior
    for iter in np.random.choice(len(samples), 200):
        sample = samples[iter]

        # Reset parameters based on whether they are fixed
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
        
        # Create GP object
        gp = gausSN.GP(kernel, meanfunc, lensingmodel)
        
        # Predict flux for each band
        for b, pb_id in enumerate(bands):
            band = data[data['band'] == pb_id]

            try:
                color_dict_fit_temp = color_dict_fit[pb_id]
            except:
                color_dict_fit_temp = color_dict_fit

            repeats = np.array([len(band[band['image'] == pb_id]) for pb_id in np.unique(data['image'])])
            lensingmodel.repeats = repeats
            lensingmodel.n_bands = 1
            shifted_time_data, b_vector = lensingmodel._lens(jnp.array(band['time'].value))

            exp, cov = gp.predict(predict_times, shifted_time_data, band['flux']/b_vector, band['fluxerr']/b_vector, band = [pb_id])
            
            # Plot predicted flux for each image
            for i in range(1):
                y_vals = np.random.multivariate_normal(mean=exp, cov=cov, size=1)

                for m, im_id in enumerate(np.unique(data['image'])):
                    
                    try:
                        color = color_dict_fit_temp[im_id]
                    except:
                        color = color_dict_fit_temp

                    repeats = np.zeros(len(np.unique(data['image'])), dtype='int')
                    repeats[m] = int(len(predict_times))
                    lensingmodel.repeats = repeats
                    _, b_vector_predict = lensingmodel._lens(jnp.array(predict_times))

                    ax[b].plot(predict_times + lensingmodel.deltas[m], y_vals[0] * b_vector_predict, color=color, alpha=0.02, zorder=2)
    
    # Add legend, xlabel, title, and adjust plot limits
    ax[0].legend(loc='upper right')
    ax[-1].set_xlabel('Time [days]', fontsize=16)
    ax[0].set_title(title, fontsize=24)

    # Set ylabel for the flux and adjust subplot spacing
    fig.supylabel('Flux', fontsize=20, y=0.494)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, ax

def make_traceplot(results, param_names=None, truths=None, dyplot_trace_kwargs={}):
    """
    Creates a trace plot for the given results.

    Args:
        results: The results to plot.
        param_names (list): Names of parameters to include in the plot.
        truths (list): True values of the parameters.
        dyplot_trace_kwargs (dict): Additional keyword arguments for dyplot.traceplot.

    Returns:
        tuple: The figure and axis of the trace plot.
    """
    show_titles = dyplot_trace_kwargs.pop('show_titles', True)
    
    plt.figure()
    fig, ax = dyplot.traceplot(results, labels=param_names, truths=truths, show_titles=show_titles, **dyplot_trace_kwargs)
    fig.tight_layout()
    return fig, ax

def make_corner(results, param_names=None, truths=None, dyplot_corner_kwargs={}):
    """
    Creates a corner plot for the given results.

    Args:
        results: The results to plot.
        param_names (list): Names of parameters to include in the plot.
        truths (list): True values of the parameters.
        dyplot_corner_kwargs (dict): Additional keyword arguments for dyplot.cornerplot.

    Returns:
        tuple: The figure and axis of the corner plot.
    """
    show_titles = dyplot_corner_kwargs.pop('show_titles', True)
    label_kwargs = dyplot_corner_kwargs.pop('label_kwargs', {'fontsize': 16})
    truth_color = dyplot_corner_kwargs.pop('truth_color', 'crimson')
    truth_kwargs = dyplot_corner_kwargs.pop('truth_kwargs', {'alpha': 1})
    smooth = dyplot_corner_kwargs.pop('smooth', 0.02)
    color = dyplot_corner_kwargs.pop('color', 'tab:blue')
    hist_kwargs = dyplot_corner_kwargs.pop('hist_kwargs', {'alpha': 0.6, 'histtype': 'stepfilled'})
    hist2d_kwargs = dyplot_corner_kwargs.pop('hist2d_kwargs', {'fill_contours': False})
    quantiles = dyplot_corner_kwargs.pop('quantiles', [0.16, 0.5, 0.84])
    title_quantiles = dyplot_corner_kwargs.pop('title_quantiles', [0.16, 0.5, 0.84])
    quantiles_2d = dyplot_corner_kwargs.pop('quantiles_2d', [0.1, 0.4, 0.65, 0.85, 0.96])

    fig, ax = dyplot.cornerplot(results, labels=param_names, truths=truths, show_titles=show_titles, label_kwargs=label_kwargs, truth_color=truth_color,
                                truth_kwargs=truth_kwargs, smooth=smooth, color=color, hist_kwargs=hist_kwargs, hist2d_kwargs=hist2d_kwargs,
                                quantiles=quantiles, title_quantiles=title_quantiles, quantiles_2d=quantiles_2d, **dyplot_corner_kwargs)
    return fig, ax
