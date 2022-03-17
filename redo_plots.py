#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  redo_plots.py
#
#  Copyright 2020 Amanda Rubio <amanda@Pakkun>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#


from PyAstronomy import pyasl
import numpy as np
import matplotlib.pylab as plt
from bemcee.be_theory import hfrac2tms, oblat2w, obl2W
from bemcee.utils import (
    beta,
    geneva_interp_fast,
    griddataBAtlas,
    griddataBA,
    lineProf,
    linfit,
)
from bemcee.lines_reading import (
    create_list,
    read_BAphot2_xdr,
    read_observables,
    create_tag,
    read_acol_Ha_xdr,
)
import corner
from bemcee.corner_HDR import corner, hist2d
from bemcee.constants import G, Msun, Rsun, Lsun, sigma
import seaborn as sns

# from pymc3.stats import hpd
# import user_settings as flag
from bemcee.hpd import hpd_grid
import sys
import importlib
from operator import is_not
from functools import partial

mod_name = sys.argv[1] + "_" + "user_settings"
# print(sys.argv[1])
flag = importlib.import_module(mod_name)


# ==============================================================================
def print_to_latex(par_fit, errs_fit, current_folder, fig_name, hpds, labels):
    """
    Prints results in latex table format

    Usage:
    params_to_print = print_to_latex(params_fit, errors_fit, current_folder, fig_name, labels, hpds)
    """
    params_fit = []
    errors_fit = []
    for i in range(len(errs_fit)):
        errors_fit.append(errs_fit[i][0])
        params_fit.append(par_fit[i][0])
    fname = current_folder + fig_name + ".txt"

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
            + "= {0:.2f} +{1:.2f} -{2:.2f}".format(
                params_fit[i], errors_fit[i][0], errors_fit[i][1]
            )
        )
        file1.writelines(
            labels[i]
            + "& ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Free \\\ \n".format(
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

    cosi = params_fit[3]
    cosi_range = [cosi + errors_fit[3][0], cosi - errors_fit[3][1]]

    oblat = 1 + 0.5 * (W ** 2)  # Rimulo 2017
    ob_max, ob_min = 1 + 0.5 * (W_range[0] ** 2), 1 + 0.5 * (W_range[1] ** 2)
    oblat_range = [ob_max, ob_min]

    Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr="014")

    Rpole_range = [0.0, 100.0]
    logL_range = [0.0, 100000.0]

    for mm in Mstar_range:
        for oo in oblat_range:
            for tt in tms_range:
                Rpolet, logLt, _ = geneva_interp_fast(mm, oo, tt, Zstr="014")
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

    beta_range = [beta(oblat_range[0], is_ob=True), beta(oblat_range[1], is_ob=True)]

    beta_par = beta(oblat, is_ob=True)

    Req = oblat * Rpole
    Req_max, Req_min = oblat_range[0] * Rpole_range[0], oblat_range[1] * Rpole_range[1]

    wcrit = np.sqrt(8.0 / 27.0 * G * Mstar * Msun / (Rpole * Rsun) ** 3)
    vsini = W * wcrit * (Req * Rsun) * np.sin(cosi * np.pi / 180.0) * 1e-5

    w_ = oblat2w(oblat)
    A_roche = (
        4.0
        * np.pi
        * (Rpole * Rsun) ** 2
        * (
            1.0
            + 0.19444 * w_ ** 2
            + 0.28053 * w_ ** 4
            - 1.9014 * w_ ** 6
            + 6.8298 * w_ ** 8
            - 9.5002 * w_ ** 10
            + 4.6631 * w_ ** 12
        )
    )

    Teff = ((10.0 ** logL) * Lsun / sigma / A_roche) ** 0.25

    Teff_range = [0.0, 50000.0]
    for oo in oblat_range:
        for rr in Rpole_range:
            for ll in logL_range:

                w_ = oblat2w(oo)
                A_roche = (
                    4.0
                    * np.pi
                    * (rr * Rsun) ** 2
                    * (
                        1.0
                        + 0.19444 * w_ ** 2
                        + 0.28053 * w_ ** 4
                        - 1.9014 * w_ ** 6
                        + 6.8298 * w_ ** 8
                        - 9.5002 * w_ ** 10
                        + 4.6631 * w_ ** 12
                    )
                )

                Teff_ = ((10.0 ** ll) * Lsun / sigma / A_roche) ** 0.25
                if Teff_ > Teff_range[0]:
                    Teff_range[0] = Teff_
                    # print('Teff max is now = {}'.format(Teff_range[0]))
                if Teff_ < Teff_range[1]:
                    Teff_range[1] = Teff_
                    # print('Teff min is now = {}'.format(Teff_range[1]))

    vsini_range = [0.0, 10000.0]
    for mm in Mstar_range:
        for rr in Rpole_range:
            for oo in oblat_range:
                for ww in W_range:
                    for ii in cosi_range:
                        wcrit = np.sqrt(8.0 / 27.0 * G * mm * Msun / (rr * Rsun) ** 3)
                        # print(wcrit)
                        vsinit = (
                            ww
                            * wcrit
                            * (oo * rr * Rsun)
                            * np.sin(ii * np.pi / 180.0)
                            * 1e-5
                        )
                        if vsinit > vsini_range[0]:
                            vsini_range[0] = vsinit
                            # print('vsini max is now = {}'.format(vsini_range[0]))

                        if vsinit < vsini_range[1]:
                            vsini_range[1] = vsinit
                            # print('vsini min is now = {}'.format(vsini_range[1]))

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
    file1.writelines(
        r"$\beta$"
        + " & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived \\\ \n".format(
            beta_par, beta_range[1] - beta_par, beta_par - beta_range[0]
        )
    )
    params_to_print.append(
        "Beta  = {0:.2f} +{1:.2f} -{2:.2f}".format(
            beta_par, beta_range[1] - beta_par, beta_par - beta_range[0]
        )
    )
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


def find_lim():
    """ Defines the value of "lim", to only use the model params in the
    interpolation

    Usage:
    lim = find_lim()

    """
    if flag.SED:
        if flag.include_rv and not flag.binary_star:
            lim = 3
        elif flag.include_rv and flag.binary_star:
            lim = 4
        elif flag.binary_star and not flag.include_rv:
            lim = 3
        else:
            lim = 2
        # if flag.Ha and flag.model=='acol':
        #    lim = lim+2
    else:
        if flag.binary_star:
            lim = 1
        else:
            if flag.model == "aeri":
                lim = -4
            else:
                lim = -7
        # if flag.Ha and flag.model=='acol':
        #    lim = 2
    return lim


def read_stars(stars_table):
    """ Reads info in the star.txt file in data/star folder

        Usage
        stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs,
        list_pre_ebmv, incl0, sig_incl0 = read_stars(star)
    """
    typ = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    file_data = flag.folder_data + stars_table

    a = np.genfromtxt(
        file_data,
        usecols=typ,
        unpack=True,
        comments="#",
        dtype={
            "names": (
                "star",
                "plx",
                "sig_plx",
                "vsini",
                "sig_vsini",
                "pre_ebmv",
                "inc",
                "sinc",
                "lbd_range",
            ),
            "formats": ("S9", "f2", "f2", "f4", "f4", "f4", "f4", "f4", "U40"),
        },
    )

    (
        stars,
        list_plx,
        list_sig_plx,
        list_vsini_obs,
        list_sig_vsini_obs,
        list_pre_ebmv,
        incl0,
        sig_incl0,
        lbd_range,
    ) = (
        a["star"],
        a["plx"],
        a["sig_plx"],
        a["vsini"],
        a["sig_vsini"],
        a["pre_ebmv"],
        a["inc"],
        a["sinc"],
        a["lbd_range"],
    )

    if np.size(stars) == 1:
        stars = stars.astype("str")
    else:
        for i in range(len(stars)):
            stars[i] = stars[i].astype("str")

    return (
        list_plx,
        list_sig_plx,
        list_vsini_obs,
        list_sig_vsini_obs,
        list_pre_ebmv,
        incl0,
        sig_incl0,
    )


def set_ranges(star, lista_obs, listpar):
    """ Defines the ranges and Ndim

    Usage:
    ranges, Ndim = set_ranges(star, lista_obs, listpar)

    """
    print(75 * "=")

    print("\nRunning star: %s\n" % star)
    print(75 * "=")

    if flag.SED or flag.normal_spectra is False:
        if flag.include_rv:
            ebmv, rv = [[0.0, 0.3], [1.0, 2.5]]
        else:
            rv = 3.1
            ebmv, rv = [[0.0, 0.4], None]

        dist_min = file_plx - flag.Nsigma_dis * file_dplx
        dist_max = file_plx + flag.Nsigma_dis * file_dplx

        addlistpar = [ebmv, [dist_min, dist_max], rv]
        addlistpar = list(filter(partial(is_not, None), addlistpar))

        if flag.model == "aeri" or flag.model == "befavor":

            ranges = np.array(
                [
                    [listpar[0][0], listpar[0][-1]],
                    [listpar[1][0], listpar[1][-1]],
                    [listpar[2][0], listpar[2][-1]],
                    [listpar[3][0], listpar[3][-1]],
                    [dist_min, dist_max],
                    [ebmv[0], ebmv[-1]],
                ]
            )

        elif flag.model == "aara" or flag.model == "acol":

            ranges = np.array(
                [
                    [listpar[0][0], listpar[0][-1]],
                    [listpar[1][0], listpar[1][-1]],
                    [listpar[2][0], listpar[2][-1]],
                    [listpar[3][0], listpar[3][-1]],
                    [listpar[4][0], listpar[4][-1]],
                    [listpar[5][0], listpar[5][-1]],
                    [listpar[6][0], listpar[6][-1]],
                    [dist_min, dist_max],
                    [ebmv[0], ebmv[-1]],
                ]
            )

        elif flag.model == "beatlas":

            ranges = np.array(
                [
                    [listpar[0][0], listpar[0][-1]],
                    [listpar[1][0], listpar[1][-1]],
                    [listpar[2][0], listpar[2][-1]],
                    [listpar[3][0], listpar[3][-1]],
                    [listpar[4][0], listpar[4][-1]],
                    [dist_min, dist_max],
                    [ebmv[0], ebmv[-1]],
                ]
            )

        if flag.include_rv:
            ranges = np.concatenate([ranges, [rv]])

        if flag.binary_star:
            M2 = [listpar[0][0], listpar[0][-1]]
            ranges = np.concatenate([ranges, [M2]])

        # if flag.Ha and flag.model == 'acol':
        #     fac_e = [0.1, 0.5] #Fraction of light scattered by electrons
        #     v_e = [400.0, 900.0] #Speed of electron motion
        #     #v_h = [10.0, 20.0] #Sound speed of the disk
        #     ranges = np.concatenate([ranges, [fac_e]])
        #     ranges = np.concatenate([ranges, [v_e]])
        #     #ranges = np.concatenate([ranges, [v_h]])

    else:
        if flag.model == "aeri" or flag.model == "befavor":

            ranges = np.array(
                [
                    [listpar[0][0], listpar[0][-1]],
                    [listpar[1][0], listpar[1][-1]],
                    [listpar[2][0], listpar[2][-1]],
                    [listpar[3][0], listpar[3][-1]],
                ]
            )

        elif flag.model == "acol" or flag.model == "aara" or flag.model == "pol":

            ranges = np.array(
                [
                    [listpar[0][0], listpar[0][-1]],
                    [listpar[1][0], listpar[1][-1]],
                    [listpar[2][0], listpar[2][-1]],
                    [listpar[3][0], listpar[3][-1]],
                    [listpar[4][0], listpar[4][-1]],
                    [listpar[5][0], listpar[5][-1]],
                    [listpar[6][0], listpar[6][-1]],
                ]
            )

        elif flag.model == "beatlas":

            ranges = np.array(
                [
                    [listpar[0][0], listpar[0][-1]],
                    [listpar[1][0], listpar[1][-1]],
                    [listpar[2][0], listpar[2][-1]],
                    [listpar[3][0], listpar[3][-1]],
                    [listpar[4][0], listpar[4][-1]],
                ]
            )

        if flag.binary_star:
            M2 = [listpar[0][0], listpar[0][-1]]
            ranges = np.concatenate([ranges, [M2]])

        # if flag.Ha and flag.model == 'acol':
        #     fac_e = [0.0, 0.5] #Fraction of light scattered by electrons
        #     v_e = [100.0, 600.0] #Speed of electron motion
        #     #v_h = [10.0, 20.0] #Sound speed of the disk
        #     ranges = np.concatenate([ranges, [fac_e]])
        #     ranges = np.concatenate([ranges, [v_e]])
        #     #ranges = np.concatenate([ranges, [v_h]])

    if flag.box_W:
        if flag.box_W_max == "max":
            ranges[1][0] = flag.box_W_min
        elif flag.box_W_min == "min":
            ranges[1][1] = flag.box_W_max
        else:
            ranges[1][0], ranges[1][1] = flag.box_W_min, flag.box_W_max

        if flag.box_i:
            if flag.model == "aeri":
                indx = 3
            elif flag.model == "acol":
                indx = 6
            if flag.box_i_max == "max":
                ranges[indx][0] = flag.box_i_min
            elif flag.box_i_min == "min":
                ranges[indx][1] = flag.box_i_max
            else:
                ranges[indx][0], ranges[indx][1] = flag.box_i_min, flag.box_i_max

    Ndim = len(ranges)

    return ranges, Ndim


lines_dict = {"Ha": 6562.801, "Hb": 4861.363, "Hd": 4101.74, "Hg": 4340.462}

sns.set_style("white", {"xtick.major.direction": "in", "ytick.major.direction": "in"})


lista_obs = create_list()
tag = create_tag()
list_of_stars = flag.stars + "/" + flag.stars + ".txt"
star = np.copy(flag.stars)

if flag.model == "acol":
    ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_acol_Ha_xdr(lista_obs)
else:
    ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_BAphot2_xdr(lista_obs)

(
    dist_pc,
    sig_dist_pc,
    vsin_obs,
    sig_vsin_obs,
    file_ebmv,
    file_incl,
    file_dincl,
) = read_stars(list_of_stars)
file_plx, file_dplx, file_vsini, file_dvsini = (
    dist_pc,
    sig_dist_pc,
    vsin_obs,
    sig_vsin_obs,
)
ranges, Ndim = set_ranges(star, lista_obs, listpar)

# ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs, Ndim = read_star_info(star, lista_obs, listpar)


logF, dlogF, logF_grid, wave, box_lim = read_observables(models, lbdarr, lista_obs)


# Walkers_100_Nmcmc_10000_af_0.26_a_2.0+aeri_SigmaClipData_distPrior_boxW+votable+iue+Ha
# 21-04-30-125218Walkers_300_Nmcmc_5000_af_0.34_a_2.0+aeri_SigmaClipData_vsiniPrior_distPrior_boxW+iue.png
# 21-04-30-125218Walkers_300_Nmcmc_5000_af_0.34_a_2.0+aeri_SigmaClipData_vsiniPrior_distPrior_boxW+iue.npy
# 21-04-30-190346Walkers_300_Nmcmc_5000_af_0.24
# 21-04-30-190346Walkers_300_Nmcmc_5000_af_0.24_a_2.0+aeri_SigmaClipData_vsiniPrior_distPrior_boxW_inclPrior+iue+Ha
# 21-05-31-123058Walkers_1000_Nmcmc_10000_af_0.28_a_3.0+aeri_removePartHa_SigmaClipData+iue+Ha.npy
# 21-06-04-120833Walkers_70_Nmcmc_300_af_0.24_a_3.0+aeri_removePartHa_SigmaClipData+iue+Ha.npy
# 21-06-16-130248Walkers_700_Nmcmc_2000_af_0.20_a_3.0+aeri_SigmaClipData_vsiniPrior_distPrior_boxW+iue+Ha.npy
# 21-06-15-163115Walkers_700_Nmcmc_2000_af_0.21_a_3.0+aeri_SigmaClipData_vsiniPrior_distPrior_boxW+iue+Ha.npy
# 21-07-08-200903Walkers_200_Nmcmc_2500_af_0.27_a_2.0+aeri_SigmaClipData_distPrior+iue.npy
# 21-07-09-213943Walkers_200_Nmcmc_2500_af_0.14_a_2.0+aeri_SigmaClipData_distPrior+iue.npy
# 21-07-27-164753Walkers_400_Nmcmc_2500_af_0.23_a_1.6+aeri_SigmaClipData_distPrior+iue+Ha.npy
# 21-07-29-152022Walkers_400_Nmcmc_3000_af_0.30_a_2.0+aeri_SigmaClipData_distPrior+Ha.npy
# 21-08-31-215021Walkers_300_Nmcmc_3000_af_0.27_a_2.0+acol_SigmaClipData_vsiniPrior_distPrior+votable+iue+Ha.npy
# 21-09-09-163657Walkers_400_Nmcmc_10000_af_0.27_a_2.0+acol_SigmaClipData_vsiniPrior_distPrior+votable+iue+Ha.npy

Nwalk = 500
# Nwalk = 70
nint_mcmc = 5000
# nint_mcmc = 300
# af = '0.24'
af = "0.27"
date = "22-03-13-233220"
# date = '21-08-31-215021'

current_folder = str(flag.folder_fig) + str(flag.stars) + "/"
fig_name = (
    "Walkers_"
    + np.str(Nwalk)
    + "_Nmcmc_"
    + np.str(nint_mcmc)
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

flatchain_1 = chain.reshape((-1, Ndim))
samples = np.copy(flatchain_1)


for i in range(len(samples)):
    if flag.model == "acol":
        samples[i][1] = obl2W(samples[i][1])
        samples[i][2] = hfrac2tms(samples[i][2])
        samples[i][6] = (np.arccos(samples[i][6])) * (180.0 / np.pi)

    if flag.model == "aeri":
        samples[i][3] = (np.arccos(samples[i][3])) * (180.0 / np.pi)

    if flag.model == "beatlas":
        samples[i][1] = obl2W(samples[i][1])
        samples[i][4] = (np.arccos(samples[i][4])) * (180.0 / np.pi)

# plot corner


new_ranges = np.copy(ranges)

if flag.model == "aeri":
    new_ranges[3] = (np.arccos(ranges[3])) * (180.0 / np.pi)
    new_ranges[3] = np.array([new_ranges[3][1], new_ranges[3][0]])
if flag.model == "acol":
    new_ranges[1] = obl2W(ranges[1])
    new_ranges[2][0] = hfrac2tms(ranges[2][1])
    new_ranges[2][1] = hfrac2tms(ranges[2][0])
    new_ranges[6] = (np.arccos(ranges[6])) * (180.0 / np.pi)
    new_ranges[6] = np.array([new_ranges[6][1], new_ranges[6][0]])
if flag.model == "beatlas":
    new_ranges[1] = obl2W(ranges[1])
    new_ranges[4] = (np.arccos(ranges[4])) * (180.0 / np.pi)
    new_ranges[4] = np.array([new_ranges[4][1], new_ranges[4][0]])

best_pars = []
best_errs = []
hpds = []

for i in range(Ndim):
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

    best_errs.append(epars)
    best_pars.append(bpars)

print(best_pars)

if flag.model == "aeri":
    if flag.SED:
        labels = [
            r"$M\,[M_\odot]$",
            r"$W$",
            r"$t/t_\mathrm{ms}$",
            r"$i[\mathrm{^o}]$",
            r"$\pi\,[mas]$",
            r"E(B-V)",
        ]
        if flag.include_rv is True:
            labels = labels + [r"$R_\mathrm{V}$"]

    else:
        labels = [r"$M\,[M_\odot]$", r"$W$", r"$t/t_\mathrm{ms}$", r"$i[\mathrm{^o}]$"]
    labels2 = labels

elif flag.model == "acol" or flag.model == "pol":
    if flag.SED:
        labels = [
            r"$M\,[\mathrm{M_\odot}]$",
            r"$W$",
            r"$t/t_\mathrm{ms}$",
            r"$\log \, n_0 \, [\mathrm{cm^{-3}}]$",
            r"$R_\mathrm{D}\, [R_\star]$",
            r"$n$",
            r"$i[\mathrm{^o}]$",
            r"$\pi\,[\mathrm{pc}]$",
            r"E(B-V)",
        ]
        labels2 = [
            r"$M$",
            r"$W$",
            r"$t/t_\mathrm{ms}$",
            r"$\log \, n_0 $",
            r"$R_\mathrm{D}$",
            r"$n$",
            r"$i$",
            r"$\pi$",
            r"E(B-V)",
        ]
        if flag.include_rv is True:
            labels = labels + [r"$R_\mathrm{V}$"]
            labels2 = labels2 + [r"$R_\mathrm{V}$"]
    else:
        labels = [
            r"$M\,[\mathrm{M_\odot}]$",
            r"$W$",
            r"$t/t_\mathrm{ms}$",
            r"$\log \, n_0 \, [\mathrm{cm^{-3}}]$",
            r"$R_\mathrm{D}\, [R_\star]$",
            r"$n$",
            r"$i[\mathrm{^o}]$",
        ]
        labels2 = [
            r"$M$",
            r"$W$",
            r"$t/t_\mathrm{ms}$",
            r"$\log \, n_0 $",
            r"$R_\mathrm{D}$",
            r"$n$",
            r"$i$",
        ]

    # if flag.Ha:
    #    labels = labels + [r'$F_e$', r'$v_e \,[km/s]$']
    #    labels2 = labels2 + [r'$F_e$', r'$v_e$']

elif flag.model == "beatlas":
    labels = [
        r"$M\,[\mathrm{M_\odot}]$",
        r"$W$",
        r"$\Sigma_0 \, [\mathrm{g/cm^{-2}}]$",
        r"$n$",
        r"$i[\mathrm{^o}]$",
        r"$\pi\,[\mathrm{pc}]$",
        r"E(B-V)",
    ]
    labels2 = [
        r"$M$",
        r"$W$",
        r"$\\Sigma_0 $",
        r"$R_\mathrm{D}$",
        r"$n$",
        r"$i$",
        r"$\pi$",
        r"E(B-V)",
    ]
    if flag.include_rv is True:
        labels = labels + [r"$R_\mathrm{V}$"]
        labels2 = labels2 + [r"$R_\mathrm{V}$"]

if flag.binary_star:
    labels = labels + [r"$M2\,[M_\odot]$"]
    labels2 = labels2 + [r"$M2\,[M_\odot]$"]


def plot_residuals_new(
    par,
    lbd,
    logF,
    dlogF,
    minfo,
    listpar,
    lbdarr,
    logF_grid,
    isig,
    dims,
    Nwalk,
    Nmcmc,
    npy,
    box_lim,
    lista_obs,
    current_folder,
    fig_name,
):
    """
    Create residuals plot separated from the corner

    """

    chain = np.load(npy)
    par_list = []
    flat_samples = chain.reshape((-1, Ndim))
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        params = flat_samples[ind]
        par_list.append(params)

    # oblat = 1 + 0.5*(W**2) # Rimulo 2017
    # Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014')

    # print(cosi)
    # cosi = np.cos(cosi * np.pi/180.)
    # par[3] = np.cos(par[3] * np.pi/180.)
    # print(cosi)

    # ***
    # chain = np.load(npy)
    # par_list = chain[:, -1, :]

    if flag.SED:
        if flag.model == "aeri":
            dist = par[4][0]
            ebv = par[5][0]
            if flag.include_rv:
                rv = par[6][0]
            else:
                rv = 3.1
        elif flag.model == "acol":
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
        lim = find_lim()
        # Finding position
        u = np.where(lista_obs == "UV")
        index = u[0][0]
        # Finding the corresponding flag.model (interpolation)
        # logF_mod_UV = griddataBA(minfo, logF_grid[index], par[:-lim],
        #                         listpar, dims)
        # Observations
        logF_UV = logF[index]
        flux_UV = 10.0 ** logF_UV
        dlogF_UV = dlogF[index]
        lbd_UV = lbd[index]

        dist = 1e3 / dist
        norma = (10.0 / dist) ** 2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
        uplim = dlogF[index] == 0
        keep = np.logical_not(uplim)

        # convert to physical units
        # logF_mod_UV += np.log10(norma)

        # flux_mod_UV = 10.**logF_mod_UV
        dflux = dlogF_UV * flux_UV

        # flux_mod_UV = pyasl.unred(lbd_UV * 1e4, flux_mod_UV, ebv=-1 * ebv, R_V=rv)

        # chi2_UV = np.sum((flux_UV - flux_mod_UV)**2. / dflux**2.)
        N_UV = len(logF_UV)
        # chi2_UV = chi2_UV/N_UV
        logF_list = np.zeros([len(par_list), len(logF_UV)])

        # chi2 = np.zeros(len(logF_list))
        # for i in range(len(par_list)):
        #    logF_list[i] = griddataBA(minfo, logF_grid[index], par_list[i, :-lim],
        #                              listpar, dims)
        for i, params in enumerate(par_list):
            if flag.binary_star:
                logF_mod_UV_1_list = griddataBA(
                    minfo, logF_grid[index], params[:-lim], listpar, dims
                )
                logF_mod_UV_2_list = griddataBA(
                    minfo,
                    logF_grid[index],
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
                        minfo, logF_grid[index], params[:-lim], listpar, dims
                    )
                else:
                    logF_list[i] = griddataBAtlas(
                        minfo,
                        logF_grid[index],
                        params[:-lim],
                        listpar,
                        dims,
                        isig=dims["sig0"],
                    )
            # par_list[i] = params
        # logF_list += np.log10(norma)
        # Plot
        # fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})

        logF_list += np.log10(norma)

        bottom, left = 0.84, 0.51  # 0.80, 0.48  # 0.75, 0.48
        width, height = 0.96 - left, 0.97 - bottom
        ax1 = plt.axes([left, bottom, width, height])

        # for j in range(len(logF_list)):
        #    chi2[j] = np.sum((logF_UV[keep] - logF_list[j][keep])**2 / (dlogF_UV[keep])**2)

        # Plot Models
        for i in range(len(par_list)):
            if flag.model == "aeri":
                ebv_temp = par_list[i][5]
            elif flag.model == "beatlas":
                ebv_temp = par_list[i][6]
            else:
                ebv_temp = par_list[i][8]
            # ebv_temp = np.copy(ebv)
            F_temp = pyasl.unred(
                lbd_UV * 1e4, 10 ** logF_list[i], ebv=-1 * ebv_temp, R_V=rv
            )
            plt.plot(lbd_UV, F_temp, color="gray", alpha=0.1)
            # Residuals  --- desta forma plota os residuos de todos os modelos, mas acho que nao eh o que quero
            # ax2.plot(lbd_UV, (flux_UV - F_temp) / dflux, 'bs', alpha=0.2)

        # Applying reddening to the best model

        # Best fit
        # ax1.plot(lbd_UV, flux_mod_UV, color='red', ls='-', lw=3.5, alpha=0.4,
        #         label=r'Best Fit' '\n' '$\chi^2$ = {0:.2f}'.format(chi2_UV))
        # ax2.plot(lbd_UV, (flux_UV - flux_mod_UV) / dflux, 'bs', alpha=0.2)
        # ax2.set_ylim(-10,10)
        # Plot Data
        keep = np.where(flux_UV > 0)  # avoid plot zero flux
        ax1.errorbar(
            lbd_UV[keep],
            flux_UV[keep],
            yerr=dflux[keep],
            ls="",
            marker="o",
            alpha=0.5,
            ms=4,
            color="k",
            linewidth=1,
        )

        ax1.set_xlabel("$\lambda\,\mathrm{[\mu m]}$", fontsize=14)
        ax1.set_ylabel(
            r"$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2} \mu m^{-1}}$", fontsize=14
        )
        # ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=14)
        # plt.tick_params(labelbottom='off')
        ax1.set_xlim(min(lbd_UV), max(lbd_UV))
        # ax2.set_xlim(min(lbd_UV), max(lbd_UV))
        # plt.tick_params(direction='in', length=6, width=2, colors='gray',
        #    which='both')
        # ax1.legend(loc='upper right', fontsize=13)
        ax1.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        # plt.tight_layout()
        # plt.savefig(current_folder + fig_name + '_new_residuals-UV-REDONE' + '.png', dpi=100)
        # plt.close()

    if flag.Ha:
        # Finding position
        line = "Ha"
        plot_line(
            line,
            lista_obs,
            minfo,
            logF_grid,
            par,
            listpar,
            dims,
            logF,
            dlogF,
            lbd,
            par_list,
            current_folder,
            fig_name,
        )
    if flag.Hb:
        line = "Hb"
        plot_line(
            line,
            lista_obs,
            minfo,
            logF_grid,
            par,
            listpar,
            dims,
            logF,
            dlogF,
            lbd,
            par_list,
            current_folder,
            fig_name,
        )

    if flag.Hd:
        line = "Hd"
        plot_line(
            line,
            lista_obs,
            minfo,
            logF_grid,
            par,
            listpar,
            dims,
            logF,
            dlogF,
            lbd,
            par_list,
            current_folder,
            fig_name,
        )
    if flag.Hg:
        line = "Hg"
        plot_line(
            line,
            lista_obs,
            minfo,
            logF_grid,
            par,
            listpar,
            dims,
            logF,
            dlogF,
            lbd,
            par_list,
            current_folder,
            fig_name,
        )


def plot_line(
    line,
    lista_obs,
    minfo,
    F_grid,
    par,
    listpar,
    dims,
    flux,
    errors,
    lbd,
    par_list,
    current_folder,
    fig_name,
):
    # Finding position
    u = np.where(lista_obs == line)
    index = u[0][0]
    # Finding the corresponding flag.model (interpolation)
    lim = find_lim()

    # if check_list(lista_obs, 'UV'):
    #    logF_mod_line = griddataBA(minfo, logF_grid[index], par[:-lim] ,listpar, dims)
    # else:
    #    logF_mod_line = griddataBA(minfo, logF_grid[index], par ,listpar, dims)

    # if flag.binary_star:
    #     F_mod_line_1 = griddataBA(minfo, F_grid[index], par[:-lim], listpar, dims)
    #     F_mod_line_2 = griddataBA(minfo, F_grid[index], np.array([par[-1], 0.1, par[2], par[3]]), listpar, dims)
    #     flux_mod_line = linfit(lbd[index], F_mod_line_1 + F_mod_line_2)
    # logF_mod_line = np.log10(F_mod_line)
    # logF_mod_Ha = np.log(norm_spectra(lbd[index], F_mod_Ha_unnormed))
    # else:
    #     F_mod_line_unnorm = griddataBA(minfo, F_grid[index], par[:-lim], listpar, dims)
    #     flux_mod_line = linfit(lbd[index], F_mod_line_unnorm)
    # logF_mod_Ha = np.log10(F_mod_Ha)
    # logF_mod_line = np.log(norm_spectra(lbd[index], F_mod_line_unnormed))
    # logF_mod_line = np.log10(10.**logF_mod_line_1 + 10.**logF_mod_line_2)

    # logF_mod_line = griddataBA(minfo, logF_grid[index], par[:-lim],listpar, dims)
    # flux_mod_line = 10.**logF_mod_line
    # Observations
    # logF_line = logF[index]
    flux_line = flux[index]
    dflux_line = errors[index]
    # dflux_line = dlogF[index] * flux_line

    keep = np.where(flux_line > 0)  # avoid plot zero flux
    lbd_line = lbd[index]

    F_list = np.zeros([len(par_list), len(flux_line)])
    F_list_unnorm = np.zeros([len(par_list), len(flux_line)])
    chi2 = np.zeros(len(F_list))
    # for i in range(len(par_list)):
    for i, params in enumerate(par_list):

        if flag.binary_star:
            F_mod_line_1_list = griddataBA(
                minfo, F_grid[index], params[:-lim], listpar, dims
            )
            F_mod_line_2_list = griddataBA(
                minfo,
                F_grid[index],
                np.array([params[-1], 0.1, par_list[i, 2], par_list[i, 3]]),
                listpar,
                dims,
            )
            F_list[i] = linfit(lbd[index], F_mod_line_1_list + F_mod_line_2_list)
            # logF_list[i] = np.log(norm_spectra(lbd[index], F_list))
        else:
            F_list_unnorm[i] = griddataBA(
                minfo, F_grid[index], params[:-lim], listpar, dims
            )
            F_list[i] = linfit(lbd[index], F_list_unnorm[i])

    # logF_list[i]= np.log10(flux_mod_line_list)
    # chi2_line = np.sum((flux_line[keep] - flux_mod_line[keep])**2 / (dflux_line[keep])**2.)
    # N_line = len(flux_line[keep])
    # chi2_line = chi2_line/N_line
    # np.savetxt(current_folder + fig_name + '_new_residuals_' + line +'.dat', np.array([lbd_line, flux_mod_line]).T)

    bottom, left = 0.65, 0.51  # 0.80, 0.48  # 0.75, 0.48
    width, height = 0.96 - left, 0.78 - bottom
    ax2 = plt.axes([left, bottom, width, height])
    lbc = lines_dict[line]
    print(lbc)
    # Plot
    # fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
    # Plot models
    # ax2 = plt.subplot(211)
    vel, fx = lineProf(lbd_line, flux_line, hwidth=2500.0, lbc=lbc * 1e-4)
    for i in range(len(par_list)):
        ax2.plot(vel, F_list[i], color="gray", alpha=0.1)
    ax2.errorbar(
        vel[keep],
        flux_line[keep],
        yerr=dflux_line[keep],
        ls="",
        marker="o",
        alpha=0.5,
        ms=4,
        color="blue",
        linewidth=1,
    )

    # Best fit
    # ax2.plot(vel, flux_mod_line, color='red', ls='-', lw=3.5, alpha=0.4, label=r'Best Fit' '\n' '$\chi^2$ = {0:.2f}'.format(chi2_line))

    ax2.set_ylabel("Normalized Flux", fontsize=14)
    ax2.set_xlim(-2300, 2300)
    # ax2.legend(loc='lower right', fontsize=13)
    ax2.set_title(line)
    ax2.set_xlabel("Vel [km/s]", fontsize=14)
    # Residuals
    # ax2.plot(lbd_line[keep], (flux_line[keep] - flux_mod_line[keep])/dflux_line[keep], marker='o', alpha=0.5)

    # ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=14)
    # ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=14)
    # ax2.set_xlim(min(lbd_line), max(lbd_line))
    # plt.tight_layout()
    # plt.savefig(current_folder + fig_name + '_new_residuals_REDONE' + line +'.png', dpi=100)
    # plt.close()


if flag.corner_color == "blue":
    truth_color = "xkcd:cobalt"
    color = "xkcd:cornflower"
    color_hist = "xkcd:powder blue"
    color_dens = "xkcd:clear blue"

elif flag.corner_color == "dark blue":
    truth_color = "xkcd:deep teal"
    color = "xkcd:dark teal"
    color_hist = "xkcd:pale sky blue"
    color_dens = "xkcd:ocean"

elif flag.corner_color == "teal":
    truth_color = "xkcd:charcoal"
    color = "xkcd:dark sea green"
    color_hist = "xkcd:pale aqua"
    color_dens = "xkcd:seafoam blue"

elif flag.corner_color == "green":
    truth_color = "xkcd:forest"
    color = "xkcd:forest green"
    color_hist = "xkcd:light grey green"
    color_dens = "xkcd:grass green"

elif flag.corner_color == "yellow":
    truth_color = "xkcd:mud brown"
    color = "xkcd:sandstone"
    color_hist = "xkcd:pale gold"
    color_dens = "xkcd:sunflower"

elif flag.corner_color == "orange":
    truth_color = "xkcd:chocolate"
    color = "xkcd:cinnamon"
    color_hist = "xkcd:light peach"
    color_dens = "xkcd:bright orange"

elif flag.corner_color == "red":
    truth_color = "xkcd:mahogany"
    color = "xkcd:deep red"
    color_hist = "xkcd:salmon"
    color_dens = "xkcd:reddish"

elif flag.corner_color == "purple":
    truth_color = "xkcd:deep purple"
    color = "xkcd:medium purple"
    color_hist = "xkcd:soft purple"
    color_dens = "xkcd:plum purple"

elif flag.corner_color == "violet":
    truth_color = "xkcd:royal purple"
    color = "xkcd:purpley"
    color_hist = "xkcd:pale violet"
    color_dens = "xkcd:blue violet"

elif flag.corner_color == "pink":
    truth_color = "xkcd:wine"
    color = "xkcd:pinky"
    color_hist = "xkcd:light pink"
    color_dens = "xkcd:pink red"

truth_color = "k"

for i, bp in enumerate(best_pars):
    if len(bp) > 1:
        value = input("More than 1 result for param " + labels[i])
        best_pars[i] = [bp[int(value)]]
# beeps = [len(bp) > 1 for bp in best_pars]


corner(
    samples,
    labels=labels,
    labels2=labels,
    range=new_ranges,
    quantiles=None,
    plot_contours=True,
    show_titles=False,
    title_kwargs={"fontsize": 15},
    label_kwargs={"fontsize": 19},
    truths=best_pars,
    hdr=True,
    truth_color=truth_color,
    color=color,
    color_hist=color_hist,
    color_dens=color_dens,
    smooth=1,
    plot_datapoints=False,
    fill_contours=True,
    combined=True,
)

add_res_plots = True

if add_res_plots:
    plot_residuals_new(
        best_pars,
        wave,
        logF,
        dlogF,
        minfo,
        listpar,
        lbdarr,
        logF_grid,
        isig,
        dims,
        Nwalk,
        nint_mcmc,
        file_npy,
        box_lim,
        lista_obs,
        current_folder,
        fig_name,
    )

plt.savefig(current_folder + fig_name + "_REDONE.png", dpi=100)
plt.close()


try:
    print_to_latex(best_pars, best_errs, current_folder, fig_name, hpds, labels)
except:
    print("Error in print_to_latex")

# exit()

# for i in range(len(hpds)):
#     if len(hpds[i]) > 1:
#         value = input(labels[i]+' has more than 1 solution. To choose this parameters, type ' + str(i)+'\n')
#         if value != '':
#             break


value = int(value)

# value=0

orange = ["xkcd:light peach", "xkcd:bright orange"]
violet = ["xkcd:pale violet", "xkcd:blue violet"]
green = ["xkcd:light grey green", "xkcd:grass green"]

color_list = [orange, violet, green]
plt.ioff()

fig, ax = corner(
    samples,
    labels=labels,
    range=ranges,
    quantiles=None,
    truths=None,
    hist_kwargs={
        "lw": 2,
        "alpha": 0.0,
        "fill": False,
        "color": None,
        "edgecolor": None,
    },
    label_kwargs={"fontsize": 19},
    plot_contours=False,
    plot_density=False,
    no_fill_contours=True,
    fill_contours=False,
    color=None,
    smooth=1,
    plot_datapoints=False,
    alpha=0.0,
    levels=None,
    combined=True,
)

for hh, (x0, x1) in enumerate(hpds[value]):
    newsamps = samples[np.where((samples[:, 0] < x1) & (samples[:, 0] > x0))[0]]
    for i in range(len(ax)):
        h1 = ax[i, i].hist(
            newsamps[:, i],
            histtype="step",
            bins=20,
            stacked=True,
            fill=True,
            color=color_list[hh][1],
            label=labels,
            edgecolor=None,
            zorder=-1,
            lw=2,
            alpha=0.6,
            range=ranges[i],
        )

        ### 2-d histograms
        if i <= len(ax):
            for j in np.arange(i + 1, len(ax.T)):

                # contour levels
                levels = 1.0 - np.exp(-0.5 * np.arange(1.0, 2.1, 0.5) ** 2)

                # poly
                hist2d(
                    newsamps[:, i],
                    newsamps[:, j],
                    smooth=1,
                    range=[ranges[i], ranges[j]],
                    plot_datapoints=False,
                    plot_contours=True,
                    plot_density=False,
                    no_fill_contours=True,
                    ax=ax[j, i],
                    alpha=0.4,
                    fill_contours=True,
                    color=color_list[hh][1],
                    color_dens=color_list[hh][0],
                    levels=levels,
                    zorder=-1,
                )

plt.savefig(current_folder + fig_name + "_REDONE_SEPARATE.png", dpi=100)

# corner_HDR.corner(newsamps, labels=labels, labels2=labels, range=ranges, quantiles=None, plot_contours=True, show_titles=False,
#            title_kwargs={'fontsize': 15}, label_kwargs={'fontsize': 19}, truths = best_pars, hdr=True,
#            truth_color=color_list[i][0], color=color_list[i][1], color_hist=color_list[i][2], color_dens=color_list[i][3],
#            smooth=1, plot_datapoints=False, fill_contours=True, combined=True)
