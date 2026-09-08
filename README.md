# GausSN / Glimpse

Welcome to GausSN/Glimpse – a package for fitting astronomical light curve data using Gaussian Processes (GPs), with specific functionality for fitting time-delays of multiply-imaged supernovae (SNe) and quasars (QSOs). GausSN and Glimpse are independent, but related models. GausSN models the underlying transient evolution as a GP, so no light curve template is required (useful for QSOs or peculiar SNe). A parametric model can be used to describe the relative time-varying magnification due to microlensing. Glimpse models the underlying transient evolution with a light curve template (useful for SNe Ia) and the GP models time-varying deviations from the template affecting each image independently due to microlensing. In each model, time-delay estimation happens concurrently with GP hyperparameters, light curve template parameters (if applicable), and/or parameterised microlensing parameters (if applicable). The GausSN model is hosted on the ``gaussn'' branch, whereas the Glimpse model is hosted on the ``glimpse'' branch. 

This package was developed by Erin Hayes (@erinhay), with vital help from Stephen Thorp (@stevet40), Nikki Arendse (@Nikki1510), Matthew Grayling (@mattgrayling), Suhail Dhawan (@sdhawan21), and Kaisey Mandel (@CambridgeAstroStat). If you find the GausSN code helpful to your work, please cite:

> Hayes E.E., et al., 2024, [MNRAS](https://ui.adsabs.harvard.edu/abs/2024MNRAS.530.3942H/abstract), [530, 4](https://doi.org/10.1093/mnras/stae1086),

and if you find the Glimpse code helpful to your work, please cite:

> Hayes E.E., et al., 2026, [MNRAS](https://ui.adsabs.harvard.edu/abs/2026MNRAS.546ag113H/abstract), [546, 3](https://doi.org/10.1093/mnras/stag113).

For an introduction to how to use GausSN/Glimpse, see the ``getting_started.ipynb'' notebook in the ipynb folder!

Thanks to Vidhi Lalchand (@vr308), Ben Boyd (@benboyd97), and Sam Ward (@sam-m-ward) for helpful discussion and Christian Kirkham (@astrochristian) for testing the code.
