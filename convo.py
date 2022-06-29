import numpy as np
import emcee
import matplotlib.pyplot as plt

# import corner
from bemcee.corner_HDR import corner
from bemcee.lines_reading import read_acol_Ha_xdr, Sliding_Outlier_Removal
from bemcee.lines_radialv import delta_v
from bemcee.be_theory import W2oblat, hfrac2tms
from bemcee.utils import griddataBA, lineProf
from astropy.io import fits
from scipy.interpolate import interp1d
from konoha.hpd import hpd_grid
import pyhdust.spectools as spec
from get_results_from_chain import get_result

# #def gfunc(a, b, c, x):
#     return a * np.exp(-((x - b) ** 2) / (2.0 * (c ** 2)))
#
#
# def gaussfold(vl, fx, fwhm):
#
#     st_dev = fwhm / (2.0 * np.sqrt(2.0 * np.log(2)))
#     ampl = 1 / (np.sqrt(2 * np.pi * (st_dev ** 2)))
#     gg = gfunc(ampl, 0.0, st_dev, vl) + 1
#     # gg /= gg.sum()
#     fold = np.convolve(fx, gg, mode="same")
#     return fold


def gaussian1d(npix, fwhm, normalize=True):
    # Initialize Gaussian params
    cntrd = (npix + 1.0) * 0.5
    st_dev = 0.5 * fwhm / np.sqrt(2.0 * np.log(2))
    x = np.linspace(1, npix, npix)

    # Make Gaussian
    ampl = 1 / (np.sqrt(2 * np.pi * (st_dev ** 2)))
    expo = np.exp(-((x - cntrd) ** 2) / (2 * (st_dev ** 2)))
    gaussian = ampl * expo

    # Normalize
    if normalize:
        gaussian /= gaussian.sum()

    return gaussian


def gaussfold(lam, flux, fwhm):

    lammin = -1500
    lammax = 1500

    dlambda = fwhm / float(17)
    interlam = lammin + dlambda * np.arange(float((lammax - lammin) / dlambda + 1))
    x = interp1d(lam, flux, kind="linear", fill_value="extrapolate")
    interflux = x(interlam)

    fwhm_pix = fwhm / dlambda
    window = int(17 * fwhm_pix)
    if window > len(interlam):
        window = len(interlam)
    # print(fwhm, dlambda, fwhm_pix, window)

    # Get a 1D Gaussian Profile
    gauss = gaussian1d(window, fwhm_pix)
    # Convolve input spectrum with the Gaussian profile
    fold = np.convolve(interflux, gauss, mode="same")
    # print(len(interflux), len(interlam), len(gauss))

    y = interp1d(interlam, fold, kind="linear", fill_value="extrapolate")
    fluxfold = y(lam)

    return fluxfold


def gauss_conv(params, vmod, fmod):
    # v_h, v_e, fac_e = params
    v_e, fac_e = params
    notscat_flux = [i * (1 - fac_e) for i in fmod]
    scat_flux = [i * fac_e for i in fmod]
    # print(len(notscat_flux), len(scat_flux))
    # notscat_conv = gaussfold(vmod, notscat_flux, v_h)
    scat_conv = gaussfold(vmod, scat_flux, v_e)
    flux_conv = scat_conv + notscat_flux  # flux after convolution
    # print(len(notscat_conv), len(scat_conv), len(flux_conv))
    # plt.plot(vmod, flux_conv, "r")
    # plt.plot(vmod, fmod, "b")
    # plt.show()
    return flux_conv


def log_prob(params, vmod, flux, dflux, ranges, fmod):
    lp = log_prior(params, ranges, vmod, fmod)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(params, vmod, flux, dflux, fmod)


def log_prior(params, ranges, vmod, fmod):
    # v_h, v_e, fac_e = params
    v_e, fac_e = params
    if (
        # ranges[0, 0] < v_h < ranges[0, 1]
        ranges[0, 0] < v_e < ranges[0, 1]
        and ranges[1, 0] < fac_e < ranges[1, 1]
    ):
        # fconv = gauss_conv(params, vmod, fmod)
        # EW_mod = spec.EWcalc(vmod, fconv) / 10.0
        # chi2_ew = ((EW_data - EW_mod) / (0.1 * EW_data)) ** 2.0
        # print(chi2_ew)
        return 0.0
    return -np.inf


def log_like(params, vmod, flux, dflux, fmod):
    # v_h, v_e, fac_e = params
    fconv = gauss_conv(params, vmod, fmod)
    limits = np.logical_and(vmod > -600, vmod < 600)
    # blimits = np.logical_or(vmod < -75.0, vmod > 75.0)
    # limits = np.logical_and(alimits, blimits)

    chi2 = np.sum((flux[limits] - fconv[limits]) ** 2 / (dflux[limits]) ** 2.0)
    # plt.plot(vmod[limits], flux[limits], "r")
    # plt.plot(vmod[limits], fconv[limits], "b")
    # plt.show()
    N = len(flux)
    chi2_red = chi2 / N
    # print(chi2_red)
    return -0.5 * chi2_red


if __name__ == "__main__":
    # the model
    # ["M", "ob", "Hfrac", "sig0", "Rd", "mr", "cosi"]
    lista_obs = np.array(["UV+VIS+NIR+MIR+FIR+MICROW+RADIO", "Ha"])

    ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_acol_Ha_xdr(lista_obs)
    star = "HD37795"
    nchain = "22-06-24-183819Walkers_500_Nmcmc_5000_af_0.22_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy"
    pars, errs = get_result(nchain, star)
    print(pars)
    M, W, ttms, sig0, Rd, mr, i, dist, ebmv = [a[0] for a in pars]
    # M = 4.7727121
    # W = 0.85899054
    ob = W2oblat(W)
    # ob = 1.367282138
    # ttms = 0.9903534984
    Hfrac = hfrac2tms(ttms, inverse=True)
    # Hfrac = 0.1023499
    # sig0 = 11.86051131
    # Rd = 16.5923432784
    # mr = 2.355036862
    # i = 43.54239333115
    cosi = np.cos(np.deg2rad(i))
    # cosi = 0.820833

    mod_pars = [M, ob, Hfrac, sig0, Rd, mr, cosi]
    # u = np.where(lista_obs == "Ha")
    index = 0  # u[0][0]
    print(index)
    mod = griddataBA(minfo, models[index], mod_pars, listpar, dims)
    lbd_central = 0.6562801
    vmod, fmod = lineProf(lbdarr[index], mod, hwidth=5000.0, lbc=lbd_central)

    # the data
    fname = "../data/HD37795/spectra/1424476i.fits.gz"

    hdr_list = fits.open(fname)
    fits_data = hdr_list[0].data
    fits_header = hdr_list[0].header
    lbd = fits_data[0, :]
    ordem = lbd.argsort()
    lbd = lbd[ordem] * 10
    flux_norm = fits_data[1, ordem]
    vl, fx = lineProf(lbd, flux_norm, hwidth=5000.0, lbc=lbd_central * 1e4)
    vel, flux = Sliding_Outlier_Removal(vl, fx, 50, 8, 15)
    radv = delta_v(vel, flux, "Ha")
    print("RADIAL VELOCITY = {0}".format(radv))
    vel = vel - radv
    EW_data = spec.EWcalc(vel, flux) / 10.0
    # plt.plot(vel, flux)
    # plt.show()

    interpolator = interp1d(vel, flux)
    flux = interpolator(vmod)
    dflux = 0.04 * flux

    # plt.plot(vmod, fmod, "r")
    # plt.plot(vmod, flux, "b")
    # plt.show()
    # params: v_h, v_e, fac_e
    # fac_e = [0.2, 0.5] #Fraction of light scattered by electrons
    # v_e = [500.0, 650.0] #Speed of electron motion
    # v_h = [10.0, 20.0] #Sound speed of the disk

    # ranges = np.array([[5.0, 50.0], [100.0, 800.0], [0.1, 0.8]])
    ranges = np.array([[5.0, 800.0], [0.01, 0.99]])

    ndim = 2

    np.random.seed(42)

    nwalkers = 200
    p0 = p0 = [
        np.random.rand(ndim) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        for i in range(nwalkers)
    ]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=[vmod, flux, dflux, ranges, fmod]
    )

    state = sampler.run_mcmc(p0, 200, progress=True)
    sampler.reset()

    sampler.run_mcmc(state, 3000, progress=True)

    flat_samples = sampler.get_chain(discard=200, thin=15, flat=True)

    bpars = []
    berrs = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        bpars.append(mcmc[1])
        berrs.append([q[0], q[1]])
        # txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        # txt = txt.format(mcmc[1], q[0], q[1], labels[i])

    best_pars = []
    best_errs = []
    hpds = []

    for i in range(ndim):
        # print(samples[:,i])
        hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(flat_samples[:, i], alpha=0.32)
        # mode_val = mode1(np.round(samples[:,i], decimals=2))
        bpars = []
        epars = []
        hpds.append(hpd_mu)
        for (x0, x1) in hpd_mu:
            # qvalues = hpd(samples[:,i], alpha=0.32)
            cut = flat_samples[flat_samples[:, i] > x0, i]
            cut = cut[cut < x1]
            median_val = np.median(cut)

            bpars.append(median_val)
            epars.append([x1 - median_val, median_val - x0])

        best_errs.append(epars)
        best_pars.append(bpars)

    print(best_pars)

    # labels = [r"$v_h$", r"$v_e$", r"$\mathrm{fac}_e$"]
    labels = [r"$v_e$", r"$\mathrm{fac}_e$"]
    fig = corner(flat_samples, labels=labels, hdr=True, truths=best_pars)
    plt.savefig("convo_corner.png")
    plt.close()
    fig2 = plt.figure(2)
    plt.plot(vmod, flux, "o", alpha=0.5, label="EW = {:.2f}".format(EW_data))
    bflux = gauss_conv([best_pars[0][0], best_pars[1][0]], vmod, fmod)
    bEW = spec.EWcalc(vmod, bflux) / 10.0
    plt.plot(vmod, bflux, label="EW = {:.2f}".format(bEW))
    plt.savefig("convo_fit.png")
