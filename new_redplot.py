import numpy as np
import matplotlib.pylab as plt
from bemcee.corner_HDR import corner
import sys
import importlib
import organizer as info
from bemcee.be_theory import obl2W, hfrac2tms
from bemcee.hpd import hpd_grid
from PyAstronomy import pyasl
from bemcee.utils import griddataBAtlas, griddataBA, linfit, lineProf
from synphot import units, SourceSpectrum, SpectralElement, Observation

mod_name = sys.argv[1] + "_" + "user_settings"
flag = importlib.import_module(mod_name)

lines_dict = {"Ha": 0.6562801, "Hb": 0.4861363, "Hd": 0.410174, "Hg": 0.4340462}


def plot_residuals(par, npy, current_folder, fig_name):
    """
    Create residuals plot separated from the corner
    For the SED and the lines

    Usage:
    plot_residuals(par, npy, current_folder, fig_name)

    """
    # plt.rc("xtick", labelsize="x-large")
    # plt.rc("ytick", labelsize="x-large")

    chain = np.load(npy)
    par_list = []
    flat_samples = chain.reshape((-1, info.Ndim))
    inds = np.random.randint(len(flat_samples), size=300)
    for ind in inds:
        params = flat_samples[ind]
        par_list.append(params)

    if flag.SED:
        if flag.model == "aeri":
            dist = par[4][0]
            ebv = par[5][0]
            if flag.include_rv:
                rv = par[6][0]
            else:
                rv = 3.1
        elif flag.model == "acol" or flag.model == "aara":
            dist = par[7][0]
            ebv = par[8][0]
            if flag.include_rv:
                rv = par[9][0]
            else:
                rv = 3.1
        elif flag.model == "beatlas":
            dist = par[5][0]
            ebv = par[6][0]
            if flag.include_rv:
                rv = par[7][0]
            else:
                rv = 3.1
        # print(dist)
        u = np.where(info.lista_obs == "UV")
        index = u[0][0]

        # Observations
        logF_UV = info.logF[index]
        flux_UV = 10.0 ** logF_UV
        dlogF_UV = info.dlogF[index]
        lbd_UV = info.wave[index]

        dist = 1e3 / dist
        norma = (10.0 / dist) ** 2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
        uplim = info.dlogF[index] == 0.0
        keep = np.logical_not(uplim)
        dflux = dlogF_UV * flux_UV
        logF_list = np.zeros([len(par_list), len(logF_UV)])

        # par_list = chain[:, -1, :]
        # inds = np.random.randint(len(flat_samples), size=100)
        for i, params in enumerate(par_list):
            if flag.binary_star:
                logF_mod_UV_1_list = griddataBA(
                    info.minfo,
                    info.logF_grid[index],
                    params[: -info.lim],
                    info.listpar,
                    info.dims,
                )
                logF_mod_UV_2_list = griddataBA(
                    info.minfo,
                    info.logF_grid[index],
                    np.array([params[-1], 0.1, params[2], params[3]]),
                    info.listpar,
                    info.dims,
                )
                logF_list[i] = np.log10(
                    10.0 ** np.array(logF_mod_UV_1_list)
                    + 10.0 ** np.array(logF_mod_UV_2_list)
                )
            else:
                if flag.model != "beatlas":
                    logF_list[i] = griddataBA(
                        info.minfo,
                        info.logF_grid[index],
                        params[: -info.lim],
                        info.listpar,
                        info.dims,
                    )
                else:
                    logF_list[i] = griddataBAtlas(
                        info.minfo,
                        info.logF_grid[index],
                        params[: -info.lim],
                        info.listpar,
                        info.dims,
                        isig=info.dims["sig0"],
                    )
            # par_list[i] = params
        logF_list += np.log10(norma)

        bottom, left = 0.82, 0.51  # 0.80, 0.48  # 0.75, 0.48
        width, height = 0.96 - left, 0.97 - bottom
        ax1 = plt.axes([left, bottom, width, height])
        ax2 = plt.axes([left, bottom - 0.07, width, 0.06])
        ax1.get_xaxis().set_visible(False)

        # Plot Models
        for i in range(len(par_list)):
            if flag.model == "aeri":
                ebv_temp = par_list[i][5]
            elif flag.model == "beatlas":
                ebv_temp = par_list[i][6]
            else:
                ebv_temp = par_list[i][8]
            F_temp = pyasl.unred(
                lbd_UV * 1e4, 10 ** logF_list[i], ebv=-1 * ebv_temp, R_V=rv
            )
            ax1.plot(lbd_UV, F_temp, color="gray", alpha=0.1, lw=0.6)

        ax2.plot(lbd_UV, (flux_UV - F_temp) / dflux, "ks", ms=8, alpha=0.2)
        # ax2.set_ylim(-10,10)
        if flag.votable or flag.data_table:
            # ax2.set_xscale('log')
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax2.set_xscale("log")

        # Plot Data
        keep = np.where(flux_UV > 0)  # avoid plot zero flux
        ax1.errorbar(
            lbd_UV[keep],
            flux_UV[keep],
            yerr=dflux[keep],
            ls="",
            marker="o",
            alpha=0.5,
            ms=8,
            color="k",
            linewidth=1,
        )
        ax2.axhline(y=0.0, ls=(0, (5, 10)), lw=0.5, color="gray")
        ax2.set_xlabel(r"$\lambda\,\mathrm{[\mu m]}$", fontsize=18)
        ax1.set_ylabel(
            r"$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}\, \mu m^{-1}]}$",
            fontsize=18,
        )
        ax2.set_ylabel(r"$(F-F_\mathrm{m})/\sigma$", fontsize=18)
        ax2.sharex(ax1)
        ax1.tick_params(axis="both", which="major", labelsize=19)
        ax2.tick_params(axis="both", which="major", labelsize=19)

        if flag.Ha:
            line = "Ha"
            plot_line(line, par, par_list)

        return


def plot_line(line, par, par_list):
    """
    Plots residuals for the lines

    Usage:
    plot_line(line, par, par_list)
    """

    lbd_central = lines_dict[line]
    # Finding position
    u = np.where(info.lista_obs == line)
    index = u[0][0]

    # Observations
    # logF_line = logF[index]
    flux_line = info.logF[index]
    dflux_line = info.dlogF[index]
    # dflux_line = dlogF[index] * flux_line

    # keep = np.where(flux_line > 0)  # avoid plot zero flux
    lbd_line = info.wave[index]

    F_list = np.zeros([len(par_list), len(flux_line)])
    F_list_unnorm = np.zeros([len(par_list), len(flux_line)])
    # chi2 = np.zeros(len(F_list))
    for i, params in enumerate(par_list):
        if flag.binary_star:
            F_mod_line_1_list = griddataBA(
                info.minfo,
                info.logF_grid[index],
                params[: -info.lim],
                info.listpar,
                info.dims,
            )
            F_mod_line_2_list = griddataBA(
                info.minfo,
                info.logF_grid[index],
                np.array([params[-1], 0.1, params[2], params[3]]),
                info.listpar,
                info.dims,
            )
            F_list[i] = linfit(info.wave[index], F_mod_line_1_list + F_mod_line_2_list)
            # logF_list[i] = np.log(norm_spectra(lbd[index], F_list))
        else:
            F_list_unnorm[i] = griddataBA(
                info.minfo,
                info.logF_grid[index],
                params[: -info.lim],
                info.listpar,
                info.dims,
            )
            F_list[i] = linfit(info.wave[index], F_list_unnorm[i])

    # np.savetxt(current_folder + fig_name + '_new_residuals_' + line +'.dat', np.array([lbd_line, flux_mod_line]).T)

    # Plot
    bottom, left = 0.57, 0.62  # 0.80, 0.48  # 0.75, 0.48
    width, height = 0.96 - left, 0.12
    ax3 = plt.axes([left, bottom, width, height])
    ax4 = plt.axes([left, bottom - 0.07, width, 0.06])
    ax3.get_xaxis().set_visible(False)

    # Plot models
    for i in range(len(par_list)):
        vl, fx = lineProf(lbd_line, F_list[i], hwidth=5000.0, lbc=lbd_central)
        # ax1.plot(lbd_line, F_list[i], color='gray', alpha=0.1)
        ax3.plot(vl, fx, color="gray", alpha=0.1)
    vl, fxx = lineProf(lbd_line, flux_line, hwidth=5000.0, lbc=lbd_central)
    ax3.errorbar(
        vl,
        fxx,
        yerr=dflux_line,
        ls="",
        marker="o",
        alpha=0.5,
        ms=5,
        color="k",
        linewidth=1,
    )
    # ax1.errorbar(lbd_line, flux_line, yerr= dflux_line, ls='', marker='o', alpha=0.5, ms=5, color='k', linewidth=1)

    # Best fit
    # ax1.plot(lbd_line, flux_mod_line, color='red', ls='-', lw=3.5, alpha=0.4, label='Best fit \n chi2 = {0:.2f}'.format(chi2_line))

    ax3.set_ylabel("Normalized Flux", fontsize=18)
    # ax1.set_xlim(min(vl), max(vl))
    ax3.set_xlim(-700, +700)
    ax3.set_ylim(-0.05, 4)
    # ax3.set_title(line)

    ax4.plot(vl, (fxx - fx) / dflux_line, marker="o", color="k", alpha=0.5)

    ax4.set_ylabel(r"$(F-F_\mathrm{m})/\sigma$", fontsize=18)
    ax4.set_xlabel(r"Velocity [km/s]", fontsize=18)
    ax4.sharex(ax3)
    ax3.tick_params(axis="both", which="major", labelsize=19)
    ax4.tick_params(axis="both", which="major", labelsize=19)

    return


Nwalk = 500
nint_mcmc = 5000

# 22-03-23-183504Walkers_300_Nmcmc_5000_af_0.21_a_2.0+aeri_distPrior_inclPrior+iue.npy
# af = "0.27"
# date = "22-03-13-233220"
# tag = "+acol_SigmaClipData_vsiniPrior_distPrior+votable+iue"
# 22-04-28-030932Walkers_500_Nmcmc_5000_af_0.28_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha
# 22-05-04-145412Walkers_500_Nmcmc_5000_af_0.20_a_1.6+aara_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO.npy
# AARA
af = "0.20"
date = "22-05-04-145412"
tag = "+aara_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO"

#
# # BCMI
# # 22-05-12-011643Walkers_900_Nmcmc_5000_af_0.27_a_1.2+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
# af = "0.27"
# date = "22-05-12-011643"
# tag = "+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha"
#
# # ACOL
# # 22-04-28-030932Walkers_500_Nmcmc_5000_af_0.28_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
# af = "0.28"
# date = "22-04-28-030932"
# tag = "+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha"

current_folder = str(flag.folder_fig) + str(flag.stars) + "/"
fig_name = (
    "Walkers_"
    + str(Nwalk)
    + "_Nmcmc_"
    + str(nint_mcmc)
    + "_af_"
    + str(af)
    + "_a_"
    + str(flag.a_parameter)
    + tag
)
file_npy = (
    flag.folder_fig
    + str(flag.stars)
    + "/"
    + date
    + "Walkers_"
    + str(Nwalk)
    + "_Nmcmc_"
    + str(nint_mcmc)
    + "_af_"
    + str(af)
    + "_a_"
    + str(flag.a_parameter)
    + tag
    + ".npy"
)

chain = np.load(file_npy)

flatchain_1 = chain.reshape((-1, chain.shape[-1]))


samples = np.copy(flatchain_1)[-100000:]


for i in range(len(samples)):
    if flag.model == "acol" or flag.model == "aara":
        samples[i][1] = obl2W(samples[i][1])
        samples[i][2] = hfrac2tms(samples[i][2])
        samples[i][6] = (np.arccos(samples[i][6])) * (180.0 / np.pi)

    elif flag.model == "aeri":
        samples[i][3] = (np.arccos(samples[i][3])) * (180.0 / np.pi)

    elif flag.model == "beatlas":
        samples[i][1] = obl2W(samples[i][1])
        samples[i][4] = (np.arccos(samples[i][4])) * (180.0 / np.pi)

    if flag.model == "aara":
        samples[i][5] = samples[i][5] + 1.5

new_ranges = []
for i in range(info.Ndim):
    new_ranges.append(np.array([np.min(samples[:, i]), np.max(samples[:, i])]))

best_pars = []
best_errs = []
hpds = []

for i in range(info.Ndim):
    print("## " + info.labels[i])
    # print(samples[:,i])
    hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(samples[:, i], alpha=0.32)
    # mode_val = mode1(np.round(samples[:,i], decimals=2))
    bpars = []
    epars = []
    hpds.append(hpd_mu)
    for (x0, x1) in hpd_mu:
        # qvalues = hpd(samples[:,i], alpha=0.32)
        cut = samples[samples[:, i] > x0, i]
        cut = cut[cut < x1]
        median_val = np.median(cut)

        bpars.append(median_val)
        epars.append([x1 - median_val, median_val - x0])
        print(
            "{0:.3f} + {1:.3f} - {2:.3f}".format(
                median_val, x1 - median_val, median_val - x0
            )
        )

    best_errs.append(epars)
    best_pars.append(bpars)

print(best_pars)


fig_corner = corner(
    samples,
    labels=info.labels,
    labels2=info.labels2,
    range=new_ranges,
    quantiles=None,
    plot_contours=True,
    show_titles=False,
    title_kwargs={"fontsize": 15},
    label_kwargs={"fontsize": 19},
    truths=best_pars,
    hdr=True,
    truth_color=info.truth_color,
    color=info.color,
    color_hist=info.color_hist,
    color_dens=info.color_dens,
    smooth=1,
    plot_datapoints=False,
    fill_contours=True,
    combined=True,
)

# b = SpectralElement.from_filter("johnson_b")
# v = SpectralElement.from_filter("johnson_v")

plot_residuals(best_pars, file_npy, current_folder, fig_name)


plt.savefig(current_folder + fig_name + "REDONE.png", dpi=100)
