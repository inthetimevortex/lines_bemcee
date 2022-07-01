import numpy as np
import matplotlib.pylab as plt
from bemcee.corner_HDR import corner
import sys
import importlib
import organizer as info
from bemcee.be_theory import obl2W, hfrac2tms, W2oblat
from bemcee.hpd import hpd_grid
from PyAstronomy import pyasl
import pyhdust as hdt
from bemcee.utils import (
    griddataBAtlas,
    griddataBA,
    linfit,
    lineProf,
    geneva_interp_fast,
    geneva_interp,
    beta,
    oblat2w,
)
from bemcee.lines_reading import read_BAphot2_xdr

# from PyAstronomy import pyasl
import seaborn as sns
import matplotlib.ticker as ticker
from konoha.constants import Msun, Rsun, sigma, G, Lsun
from PT_makemag import convolution_Johnsons, convolution_JohnsonsZP
import os

# plt.rc("xtick", labelsize=7.5)
# plt.rc("ytick", labelsize=7)

# from synphot import units, SourceSpectrum, SpectralElement, Observation

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
    sns.set_style("ticks")
    sfmt = ticker.ScalarFormatter(useMathText=True)
    sfmt.set_scientific(True)
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

        ax2.plot(lbd_UV, (flux_UV - F_temp) / dflux, "ks", ms=6, alpha=0.2)
        # ax2.set_ylim(-10,10)
        if flag.votable or flag.data_table:
            # ax2.set_xscale('log')
            ax1.set_xscale("log")
            # ax1.set_yscale("log")
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
            ms=6,
            color="k",
            linewidth=1,
        )
        ax2.axhline(y=0.0, ls=(0, (5, 10)), lw=0.7, color="k")
        ax2.set_xlabel(r"$\lambda\,\mathrm{[\mu m]}$", fontsize=16)
        ax1.set_ylabel(
            r"$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}\, \mu m^{-1}]}$",
            fontsize=16,
        )
        ax2.set_ylabel(r"$(F-F_\mathrm{m})/\sigma$", fontsize=16)
        ax2.sharex(ax1)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-7, 0))
        # ax1.tick_params(axis="both", reset=True, which="major", labelsize=16)
        # ax2.tick_params(axis="both", reset=True, which="major", labelsize=16)

        # if flag.Ha:
        #        line = "Ha"
        #        plot_line(line, par, par_list)

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


def print_to_latex(params_fit, errors_fit, date):
    """
    Prints results in latex table format

    """
    # params_fit = []
    # errors_fit = []
    # for i in range(len(errs_fit)):
    #     errors_fit.append(errs_fit[i][0])
    #     params_fit.append(par_fit[i][0])
    fname = date + "gcas.txt"

    if flag.model == "aeri":
        names = ["Mstar", "W", "t/tms", "i", "Dist", "E(B-V)"]
        if flag.include_rv:
            names = names + ["RV"]
        if flag.binary_star:
            names = names + ["M2"]
    if flag.model == "acol":
        names = ["Mstar", "W", "t/tms", "logn0", "Rd", "n", "i", "Dist", "E(B-V)"]
        if flag.include_rv:
            names = names + ["RV"]
        if flag.binary_star:
            names = names + ["M2"]
    if flag.model == "beatlas":
        names = ["Mstar", "W", "Sig0", "n", "i", "Dist", "E(B-V)"]
        if flag.include_rv:
            names = names + ["RV"]
        if flag.binary_star:
            names = names + ["M2"]

    file1 = open(fname, "w")
    L = [
        r"\begin{table}" + " \n",
        "\centering \n",
        r"\begin{tabular}{lll}" + " \n",
        "\hline \n",
        "Parameter  & Value & Type \\\ \n",
        "\hline \n",
    ]
    file1.writelines(L)

    params_to_print = []
    # print(errors_fit[0][1])
    for i in range(len(params_fit)):
        params_to_print.append(
            names[i]
            + "= {0:.3f} +{1:.3f} -{2:.3f}".format(
                params_fit[i], errors_fit[i][0], errors_fit[i][1]
            )
        )
        file1.writelines(
            info.labels[i]
            + "& ${0:.3f}^{{+{1:.3f}}}_{{-{2:.3f}}}$ & Free \\\ \n".format(
                params_fit[i], errors_fit[i][0], errors_fit[i][1]
            )
        )

    # if len(hpds[0]) > 1:

    Mstar = params_fit[0]
    Mstar_range = [Mstar + errors_fit[0][0], Mstar - errors_fit[0][1]]

    W = params_fit[1]
    W_range = [W + errors_fit[1][0], W - errors_fit[1][1]]

    tms = params_fit[2]
    tms_range = [tms + errors_fit[2][0], tms - errors_fit[2][1]]

    incl = params_fit[3]
    incl_range = [incl + errors_fit[3][0], incl - errors_fit[3][1]]

    oblat = W2oblat(W)
    ob_max, ob_min = W2oblat(W_range[0]), W2oblat(W_range[1])
    oblat_range = [ob_max, ob_min]
    print(oblat_range, oblat)

    if tms <= 1.0:
        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr="014")
    else:
        Rpole, logL = geneva_interp(Mstar, oblat, tms, Zstr="014")
    # Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014')

    Rpole_range = [0.0, 100.0]
    logL_range = [0.0, 100000.0]

    for mm in Mstar_range:
        for oo in oblat_range:
            for tt in tms_range:
                if tt <= 1.0:
                    Rpolet, logLt, _ = geneva_interp_fast(mm, oo, tt, Zstr="014")
                else:
                    Rpolet, logLt = geneva_interp(mm, oo, tt, Zstr="014")
                # Rpolet, logLt, _ = geneva_interp_fast(mm, oo, tt, Zstr='014')
                if Rpolet > Rpole_range[0]:
                    Rpole_range[0] = Rpolet
                    # print('Rpole max is now = {}'.format(Rpole_range[0]))
                if Rpolet < Rpole_range[1]:
                    Rpole_range[1] = Rpolet
                    # print('Rpole min is now = {}'.format(Rpole_range[1]))
                if logLt > logL_range[0]:
                    logL_range[0] = logLt
                    # print('logL max is now = {}'.format(logL_range[0]))
                if logLt < logL_range[1]:
                    logL_range[1] = logLt
                    # print('logL min is now = {}'.format(logL_range[1]))

    # beta_range = [beta(oblat_range[0], is_ob=True), beta(oblat_range[1], is_ob=True)]
    #
    # beta_par = beta(oblat, is_ob=True)

    Req = oblat * Rpole
    Req_max, Req_min = oblat_range[0] * Rpole_range[0], oblat_range[1] * Rpole_range[1]

    omega = oblat2w(oblat)
    wcrit = np.sqrt(8.0 / 27.0 * G * Mstar * Msun / (Rpole * Rsun) ** 3)
    vsini = omega * wcrit * (Req * Rsun) * np.sin(np.deg2rad(incl)) * 1e-5

    A_roche = (
        4.0
        * np.pi
        * (Rpole * Rsun) ** 2
        * (
            1.0
            + 0.19444 * omega ** 2
            + 0.28053 * omega ** 4
            - 1.9014 * omega ** 6
            + 6.8298 * omega ** 8
            - 9.5002 * omega ** 10
            + 4.6631 * omega ** 12
        )
    )

    Teff = ((10.0 ** logL) * Lsun / sigma / A_roche) ** 0.25

    Teff_range = [0.0, 50000.0]
    vsini_range = [0.0, 10000.0]
    for mm in Mstar_range:
        for oo in oblat_range:
            for tt in tms_range:
                for ii in incl_range:
                    if tt <= 1.0:
                        rr, ll, _ = geneva_interp_fast(mm, oo, tt, Zstr="014")
                    else:
                        rr, ll = geneva_interp(mm, oo, tt, Zstr="014")
                    wcrit = np.sqrt(8.0 / 27.0 * G * mm * Msun / (rr * Rsun) ** 3)
                    # print(rr, oo)
                    omega_ = oblat2w(oo)
                    vsinit = (
                        omega_
                        * wcrit
                        * (oo * rr * Rsun)
                        * np.sin(np.deg2rad(ii))
                        * 1e-5
                    )
                    if vsinit > vsini_range[0]:
                        vsini_range[0] = vsinit
                        print("vsini max is now = {}".format(vsini_range[0]))

                    if vsinit < vsini_range[1]:
                        vsini_range[1] = vsinit
                        print("vsini min is now = {}".format(vsini_range[1]))
                    A_roche = (
                        4.0
                        * np.pi
                        * (rr * Rsun) ** 2
                        * (
                            1.0
                            + 0.19444 * omega_ ** 2
                            + 0.28053 * omega_ ** 4
                            - 1.9014 * omega_ ** 6
                            + 6.8298 * omega_ ** 8
                            - 9.5002 * omega_ ** 10
                            + 4.6631 * omega_ ** 12
                        )
                    )

                    Teff_ = ((10.0 ** ll) * Lsun / sigma / A_roche) ** 0.25
                    if Teff_ > Teff_range[0]:
                        Teff_range[0] = Teff_
                        # print('Teff max is now = {}'.format(Teff_range[0]))
                    if Teff_ < Teff_range[1]:
                        Teff_range[1] = Teff_
                        # print('Teff min is now = {}'.format(Teff_range[1]))

    file1.writelines(
        r"$R_{\rm eq}/R_{\rm p}$"
        + " & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n".format(
            oblat, oblat_range[0] - oblat, oblat - oblat_range[1]
        )
    )
    params_to_print.append(
        "Oblateness = {0:.2f} +{1:.2f} -{2:.2f}".format(
            oblat, oblat_range[0] - oblat, oblat - oblat_range[1]
        )
    )
    file1.writelines(
        r"$R_{\rm eq}\,[R_\odot]$"
        + " & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n".format(
            Req, Req_max - Req, Req - Req_min
        )
    )
    params_to_print.append(
        "Equatorial radius = {0:.2f} +{1:.2f} -{2:.2f}".format(
            Req, Req_max - Req, Req - Req_min
        )
    )
    file1.writelines(
        r"$\log(L)\,[L_\odot]$"
        + " & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n".format(
            logL, logL_range[0] - logL, logL - logL_range[1]
        )
    )
    params_to_print.append(
        "Log Luminosity  = {0:.2f} +{1:.2f} -{2:.2f}".format(
            logL, logL_range[0] - logL, logL - logL_range[1]
        )
    )
    # file1.writelines(
    #     r"$\beta$"
    #     + " & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived \\\ \n".format(
    #         beta_par, beta_range[1] - beta_par, beta_par - beta_range[0]
    #     )
    # )
    # params_to_print.append(
    #     "Beta  = {0:.2f} +{1:.2f} -{2:.2f}".format(
    #         beta_par, beta_range[1] - beta_par, beta_par - beta_range[0]
    #     )
    # )
    file1.writelines(
        r"$v \sin i\,\rm[km/s]$"
        + " & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n".format(
            vsini, vsini_range[0] - vsini, vsini - vsini_range[1]
        )
    )
    params_to_print.append(
        "vsini = {0:.2f} +{1:.2f} -{2:.2f}".format(
            vsini, vsini_range[0] - vsini, vsini - vsini_range[1]
        )
    )
    file1.writelines(
        r"$T_{\rm eff}$"
        + " & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n".format(
            Teff, Teff_range[0] - Teff, Teff - Teff_range[1]
        )
    )
    params_to_print.append(
        "Teff = {0:.2f} +{1:.2f} -{2:.2f}".format(
            Teff, Teff_range[0] - Teff, Teff - Teff_range[1]
        )
    )

    L = ["\hline \n", "\end{tabular} \n" "\end{table} \n"]

    file1.writelines(L)

    file1.close()

    params_print = " \n".join(map(str, params_to_print))

    return params_to_print


Nwalk = 300
nint_mcmc = 7000

# 22-03-23-183504Walkers_300_Nmcmc_5000_af_0.21_a_2.0+aeri_distPrior_inclPrior+iue.npy
# af = "0.27"
# date = "22-03-13-233220"
# tag = "+acol_SigmaClipData_vsiniPrior_distPrior+votable+iue"
# 22-04-28-030932Walkers_500_Nmcmc_5000_af_0.28_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha
# 22-05-04-145412Walkers_500_Nmcmc_5000_af_0.20_a_1.6+aara_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO.npy
# AARA
# 22-06-04-131513Walkers_1000_Nmcmc_7000_af_0.15_a_1.4+aara_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO.npy
af = "0.15"
date = "22-06-04-131513"
tag = "+aara_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO"

#
# # BCMI
# # 22-05-12-011643Walkers_900_Nmcmc_5000_af_0.27_a_1.2+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
# 22-05-25-142921Walkers_500_Nmcmc_5000_af_0.25_a_1.8+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
# 22-06-03-131900Walkers_600_Nmcmc_6000_af_0.29_a_1.8+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
# af = "0.29"
# date = "22-06-03-131900"
# tag = "+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha"
#
# # ACOL
# # 22-04-28-030932Walkers_500_Nmcmc_5000_af_0.28_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
# 22-05-25-122141Walkers_500_Nmcmc_5000_af_0.19_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
# 22-06-03-021252Walkers_700_Nmcmc_6000_af_0.19_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
# 22-06-15-194229Walkers_300_Nmcmc_5000_af_0.25_a_2.0+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha.npy
af = "0.25"
date = "22-06-15-194229"
tag = "+acol_vsiniPrior_distPriorUV+VIS+NIR+MIR+FIR+MICROW+RADIO+Ha"

# # GCAS
# 22-06-22-100634Walkers_300_Nmcmc_7000_af_0.30_a_2.0+aeri_vsiniPrior_distPrior_inclPriorUV.npy
# 22-06-25-072903Walkers_300_Nmcmc_7000_af_0.34_a_2.0+aeri_vsiniPrior_distPrior_inclPriorUV.npy
# 22-06-27-065631Walkers_300_Nmcmc_7000_af_0.28_a_2.0+aeri_vsiniPrior_distPriorUV.npy
# 22-06-29-082543Walkers_300_Nmcmc_7000_af_0.31_a_2.0+aeri_vsiniPrior_distPrior_inclPriorUV.npy
# 22-06-29-060044Walkers_300_Nmcmc_7000_af_0.29_a_2.0+aeri_distPrior_inclPriorUV.npy
af = "0.31"
date = "22-06-29-082543"
tag = "+aeri_vsiniPrior_distPrior_inclPriorUV"

nchain = (
    "22-06-29-060044Walkers_300_Nmcmc_7000_af_0.29_a_2.0+aeri_distPrior_inclPriorUV"
)

date = nchain.split("Walkers")[0]
Nwalk = nchain.split("Walkers")[1].split("_")[1]
Nmcmc = nchain.split("Walkers")[1].split("_")[3]
af = nchain.split("Walkers")[1].split("_")[5]
apar = nchain.split("Walkers")[1].split("+")[0][-3:]
tag = "+" + nchain.split("Walkers")[1].split("+")[-1]
model = nchain.split("Walkers")[1].split("+")[1].split("_")[0]

current_folder = str(flag.folder_fig) + str(flag.stars) + "/"
fig_name = (
    "Walkers_"
    + str(Nwalk)
    + "_Nmcmc_"
    + str(Nmcmc)
    + "_af_"
    + str(af)
    + "_a_"
    + str(apar)
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
    + str(Nmcmc)
    + "_af_"
    + str(af)
    + "_a_"
    + str(apar)
    + tag
    + ".npy"
)

chain = np.load(file_npy)

flatchain_1 = chain.reshape((-1, chain.shape[-1]))


samples = np.copy(flatchain_1)[-100000:]


for i in range(len(samples)):
    if model == "acol" or model == "aara":
        samples[i][1] = obl2W(samples[i][1])
        samples[i][2] = hfrac2tms(samples[i][2])
        samples[i][6] = (np.arccos(samples[i][6])) * (180.0 / np.pi)

    elif model == "aeri":
        samples[i][3] = (np.arccos(samples[i][3])) * (180.0 / np.pi)

    elif model == "beatlas":
        samples[i][1] = obl2W(samples[i][1])
        samples[i][4] = (np.arccos(samples[i][4])) * (180.0 / np.pi)

    if model == "aara":
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
    hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(samples[:, i], alpha=0.32, roundto=3)
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
            "{0:.5f} + {1:.5f} - {2:.5f}".format(
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

plot_residuals(best_pars, file_npy, current_folder, fig_name)

params_to_print = print_to_latex(
    [a[0] for a in best_pars], [a[0] for a in best_errs], date
)

plt.savefig(current_folder + fig_name + "REDONE.png", dpi=100)

Mass = best_pars[0][0]
W = best_pars[1][0]
ttms = best_pars[2][0]
incl = best_pars[3][0]
dist = best_pars[4][0]
ebmv = best_pars[5][0]

oblat = W2oblat(W)
omega = oblat2w(oblat)
if ttms <= 1.0:
    Rpole, logL, age = geneva_interp_fast(Mass, oblat, ttms, Zstr="014")
else:
    Rpole, logL = geneva_interp(Mass, oblat, ttms, Zstr="014")
# Beta = beta(oblat, is_ob=True)

print(oblat, Rpole, logL)
wcrit = np.sqrt(8.0 / 27.0 * G * Mass * Msun / (Rpole * Rsun) ** 3)

cosi = np.cos(incl * np.pi / 180)
vsini = omega * wcrit * (Rpole * Rsun * oblat) * np.sin(np.deg2rad(incl)) * 1e-5
print("the vsini")
print(vsini)

params = [Mass, W, ttms, np.cos(np.deg2rad(incl))]

ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_BAphot2_xdr(["UV+VIS"])
mod = griddataBA(minfo, models[0], params, listpar, dims)  # erg/s/cm2/micron
lbdarr = lbdarr[0]
dista = 1e3 / dist
norma = (10 / dista) ** 2
mod = mod * norma
flux_mod = pyasl.unred(lbdarr * 1e4, mod, ebv=-1 * ebmv, R_V=3.1)

vega_path = os.path.join(hdt.hdtpath(), "refs/stars/", "vega.dat")
vega = np.loadtxt(vega_path)
wv_vg = vega[:, 0]  # will be in \AA
fx_vg = vega[:, 1]  # erg s-1 cm-2 \AA-1
fU, fB, fV, fR, fI = convolution_JohnsonsZP(wv_vg, fx_vg)  # zero points magnitudes.

mU, mB, mV, mR, mI = convolution_Johnsons(
    lbdarr * 1e4, flux_mod * 1e-4, fU, fB, fV, fR, fI
)
print(mU, mB, mV, mR, mI)

# tcs = pyasl.TransmissionCurves()
# wvl = np.linspace(3000, 10000, 10000)
# Ufilt = tcs.getTransCurve("Johnson U")
# Uband = Ufilt(wvl)
# Ubs = tcs.convolveWith(lbdarr * 1e4, flux_mod * 1e-4, "Johnson U")
# Uval = np.trapz(Ubs)
#
# Bfilt = tcs.getTransCurve("Johnson B")
# Bband = Bfilt(wvl)
# Bbs = tcs.convolveWith(lbdarr * 1e4, flux_mod * 1e-4, "Johnson B")
# Bval = np.trapz(Bbs)
#
#
# Vfilt = tcs.getTransCurve("Johnson V")
# Vband = Vfilt(wvl)
# Vbs = tcs.convolveWith(lbdarr * 1e4, flux_mod * 1e-4, "Johnson V")
# Vval = np.trapz(Vbs)
