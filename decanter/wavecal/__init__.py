"""Physical wavelength calibration on OH airglow and telluric H2O lines.

The period ThAr solution is kept as a fixed shape template; what moves during
a night is measured on the atmosphere, frame by frame, in velocity units:
``z_sky`` from the OH lines of the sky spectra (the instrument) and
``z_star`` from the telluric H2O lines of the object spectra (the
starlight). See :mod:`decanter.wavecal.model` for the method and its
validation. Needs the optional ``wavecal`` extra (JAX).

    import decanter, decanter.wavecal as wc
    series = decanter.reduce_many(pairs, calib)
    sol = wc.anchor_reduction(series, calib)     # night series, single frame or sparse epochs
    sol.z_star, sol.z_star_err                   # km/s per object frame
    u = sol.labels(163, lam, kind="star")        # ln(lambda_true / lambda_label) per pixel
    flux = wc.export_resample(flux, lam, u, lam) # the one resample

A WARP reduction directory works too: ``wc.anchor(wc.load_warp_night(root))``.
"""
from decanter.wavecal.model import *  # noqa: F403
from decanter.wavecal.model import __all__  # noqa: F401
