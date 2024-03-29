import numpy as np
import math
import matplotlib.pyplot as plt
from .utils import (
    bin_data,
    find_nearest,
    jy2cgs,
    readBAsed,
    readXDRsed,
    lineProf,
    kde_scipy,
)
from scipy.interpolate import griddata
import atpy
import os
from astropy.io import fits
from pyhdust import spectools as spec
from astropy.stats import median_absolute_deviation as MAD
from scipy.signal import detrend
from .lines_radialv import delta_v
import importlib
import pandas as pd
from scipy.interpolate import interp1d
from __init__ import mod_name
from icecream import ic

flag = importlib.import_module(mod_name)


lines_dict = {"Ha": 6562.801, "Hb": 4861.363, "Hd": 4101.74, "Hg": 4340.462}


# ==============================================================================
def find_nearest2(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# =======================================================================
# Read the xdr combining continuum and lines
def read_BAphot2_xdr(lista_obs):

    dims = ["M", "W", "tms", "cosi"]
    dims = dict(zip(dims, range(len(dims))))
    isig = None  # photospheric flag.model is None

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    # xdrPL = flag.folder_models + 'BAphot__UV2M_Ha_Hb_Hg_Hd.xdr'
    xdrPL = flag.folder_models + "BeAtlas2021_phot_Puxadinho.xdr"
    ninfo, ranges, lbdarr, minfo, models = readXDRsed(xdrPL, quiet=False)
    models = 1e-8 * models  # erg/s/cm2/micron

    # print(len(models))
    # print(len(minfo))

    # Correction for negative parameter values (in cosi for instance)
    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.0

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0 and (j != 0 or j != len(models[i][j]) - 1):
                models[i][j] = (models[i][j - 1] + models[i][j + 1]) / 2.0

    # Combining models and lbdarr
    models_combined = []
    lbd_combined = []

    if flag.SED:
        if flag.votable or flag.data_table:
            lbd_UV, models_UV = xdr_remove_lines(lbdarr, models)
        # UV is from lbdarr[0] to lbdarr[224]
        else:
            j = 0
            shape = (6600, 225)
            models_UV = np.zeros(shape)
            while j < 6600:
                models_UV[j] = models[j][:225]
                j += 1

            lbd_UV = lbdarr[:225]

        # plt.plot(lbd_UV, models_UV[30])
        # plt.xscale('log')
        # plt.yscale('log')
        models_combined.append(models_UV)
        lbd_combined.append(lbd_UV)

    if flag.Ha:
        lbdc = 0.656
        models_combined, lbd_combined = select_xdr_part(
            lbdarr, models, models_combined, lbd_combined, lbdc
        )

    if flag.Hb:
        lbdc = 0.486
        models_combined, lbd_combined = select_xdr_part(
            lbdarr, models, models_combined, lbd_combined, lbdc
        )

    if flag.Hd:
        lbdc = 0.410
        models_combined, lbd_combined = select_xdr_part(
            lbdarr, models, models_combined, lbd_combined, lbdc
        )

    if flag.Hg:
        lbdc = 0.434
        models_combined, lbd_combined = select_xdr_part(
            lbdarr, models, models_combined, lbd_combined, lbdc
        )

    listpar = [
        np.unique(minfo[:, 0]),
        np.unique(minfo[:, 1]),
        np.unique(minfo[:, 2]),
        np.unique(minfo[:, 3]),
    ]
    # print(np.unique(minfo[:,2]))
    return ctrlarr, minfo, models_combined, lbd_combined, listpar, dims, isig


# =============================================================================
def select_xdr_part(lbdarr, models, models_combined, lbd_combined, lbdc):
    line_peak = find_nearest2(lbdarr, lbdc)
    keep_a = find_nearest2(lbdarr, 0.6400)
    keep_b = find_nearest2(lbdarr, 0.6700)
    lbd_line = lbdarr[keep_a:keep_b]
    # print(lbd_line)
    models_line = models[:, keep_a:keep_b]
    # print(len(lbd_line), len(models_line[0][1:-1]))
    # lbdarr_line = lbd_line*1e4
    vl_, fx_ = lineProf(lbd_line, models_line[0], hwidth=5000.0, lbc=lbdc, ssize=0.5)
    new_models = np.zeros([len(models_line), len(fx_)])
    for i, mm in enumerate(models_line):
        # print(len(mm))
        vl, fx = lineProf(lbd_line, mm, hwidth=5000.0, lbc=lbdc, ssize=0.5)
        # print(len(vl), len(fx))
        # print(len(new_models[0]))
        new_models[i] = fx
    c = 299792.458
    wl = c * lbdc / (c - vl) * 1e4
    lbdarr_line = wl / (1.0 + 2.735182e-4 + 131.4182 / wl ** 2 + 2.76249e8 / wl ** 4)
    models_combined.append(new_models)
    lbd_combined.append(lbdarr_line * 1e-4)
    # print(vl)
    # print(lbd_combined[1])
    return models_combined, lbd_combined


# =============================================================================
def xdr_remove_lines(lbdarr, models):
    for line in lines_dict:
        keep_a = find_nearest2(lbdarr, lines_dict[line] - 0.007)
        keep_b = find_nearest2(lbdarr, lines_dict[line] + 0.007)
        lbdarr1 = lbdarr[:keep_a]
        lbdarr2 = lbdarr[keep_b:]
        lbdarr = np.concatenate([lbdarr1, lbdarr2])
        novo_models1 = models[:, :keep_a]
        novo_models2 = models[:, keep_b:]
        novo_models = np.hstack((novo_models1, novo_models2))

    return lbdarr, novo_models


# ==============================================================================
def read_aara_xdr():

    dims = ["M", "ob", "Hfrac", "sig0", "Rd", "mr", "cosi"]
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    xdrPL = flag.folder_models + "aara_sed.xdr"  #

    listpar, lbdarr, minfo, models = readBAsed(xdrPL, quiet=False)

    # F(lbd)] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.0

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0.0 or models[i][j] == 0.0:
                models[i][j] = (models[i][j + 1] + models[i][j - 1]) / 2.0

    # n0 to logn0
    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    minfo[:, 4] = np.log10(minfo[:, 4])

    if True:
        mask = []
        tmp, idx = find_nearest(lbdarr, 1000)
        for i in range(len(models)):
            if models[i][idx] > 2.21834e-10:
                mask.append(i)
                # print(i)
                # plt.plot(lbdarr, models[i], alpha=0.1)
        tmp, idx = find_nearest(lbdarr, 80)
        for i in range(len(models)):
            if models[i][idx] > 2e-8:
                mask.append(i)
                # print(i)
                # # plt.plot(lbdarr, models[i], alpha=0.1)
        tmp, idx = find_nearest(lbdarr, 850)
        for i in range(len(models)):
            if models[i][idx] > 7e-11:
                mask.append(i)
        #         print(i)
        #         plt.plot(lbdarr, models[i], alpha=0.1)
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()

        new_models = np.delete(models, mask, axis=0)
        new_minfo = np.delete(minfo, mask, axis=0)

        models = np.copy(new_models)
        minfo = np.copy(new_minfo)

    # delete columns of fixed par
    cols2keep = [0, 1, 3, 4, 5, 7, 8]
    cols2delete = [2, 6]
    listpar = [listpar[i] for i in cols2keep]
    minfo = np.delete(minfo, cols2delete, axis=1)
    listpar[3].sort()

    return ctrlarr, minfo, [models], [lbdarr], listpar, dims, isig


# ==============================================================================
def read_befavor_xdr():

    dims = ["M", "ob", "Hfrac", "sig0", "Rd", "mr", "cosi"]
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, 0.014, np.NaN, 0.0, 50.0, 60.0, 3.5, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    xdrPL = flag.folder_models + "BeFaVOr.xdr"
    listpar, lbdarr, minfo, models = readBAsed(xdrPL, quiet=False)
    # [models] = [F(lbd)]] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.0

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0 and (j != 0 or j != len(models[i][j]) - 1):
                models[i][j] = (models[i][j - 1] + models[i][j + 1]) / 2.0

    # delete columns of fixed par
    cols2keep = [0, 1, 3, 8]
    cols2delete = [2, 4, 5, 6, 7]
    listpar = [listpar[i] for i in cols2keep]
    minfo = np.delete(minfo, cols2delete, axis=1)
    listpar[3].sort()
    listpar[3][0] = 0.0

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# ==============================================================================
def read_beatlas_xdr(lista_obs):

    dims = ["M", "ob", "sig0", "mr", "cosi"]
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]
    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    xdrPL = flag.folder_models + "disk_flx.xdr"  # 'PL.xdr'

    listpar, lbdarr, minfo, models = readBAsed(xdrPL, quiet=False)

    # F(lbd)] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.0

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0.0 or models[i][j] == 0.0:
                models[i][j] = (models[i][j + 1] + models[i][j - 1]) / 2.0

    listpar[-1][0] = 0.0

    listpar[2] = np.round(listpar[2], decimals=3)
    print(listpar)
    return ctrlarr, minfo, [models], [lbdarr], listpar, dims, isig


# ===============================================================================


def read_acol_Ha_xdr(lista_obs):

    # print(params_tmp)
    dims = ["M", "ob", "Hfrac", "sig0", "Rd", "mr", "cosi"]
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.

    xdrPL = flag.folder_models + "acol_Ha.xdr"

    listpar, lbdarr, minfo, models = readBAsed(xdrPL, quiet=False)

    # Filter (removing bad models)
    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.0

    for i in range(len(listpar)):
        for j in range(len(listpar[i])):
            if listpar[i][j] < 0:
                listpar[i][j] = 0.0

    mask = np.ones(
        len(minfo[0]), dtype=bool
    )  # isso provavelmente remove os parâmetros com só 1 valor
    mask[[2, 6]] = False
    result = []
    for i in range(len(minfo)):
        result.append(minfo[i][mask])
    minfo = np.copy(result)

    for i in range(np.shape(minfo)[0]):
        minfo[i][3] = np.log10(minfo[i][3])

    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    listpar = list(
        [
            listpar[0],
            listpar[1],
            listpar[3],
            listpar[4],
            listpar[5],
            listpar[7],
            listpar[8],
        ]
    )

    # Combining models and lbdarr
    models_combined = []
    lbd_combined = []

    if flag.SED:
        lbd_UV, models_UV = xdr_remove_lines(lbdarr, models)

        models_combined.append(models_UV)
        lbd_combined.append(lbd_UV)

    if flag.Ha:
        lbdc = 0.6563
        models_combined, lbd_combined = select_xdr_part(
            lbdarr, models, models_combined, lbd_combined, lbdc
        )

    return ctrlarr, minfo, models_combined, lbd_combined, listpar, dims, isig


# ==============================================================================


def read_acol_pol_xdr():

    dims = ["M", "ob", "Hfrac", "sig0", "Rd", "mr", "cosi"]
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    xdrPL = flag.folder_models + "acol_pol_Ha.xdr"

    listpar, lbdarr, minfo, models = readBAsed(xdrPL, quiet=False)
    models = np.array(models) * 100.0  # in percentage
    # minfo, models = minfo[3600:], models[3600:]

    #    new_models = []
    #    idx_lbd = np.where(lbdarr < 1.0)
    #
    #    for i in range(len(models)):
    #        tmp = np.copy(models[i])
    #        new_models.append(tmp[idx_lbd])
    #    lbdarr = lbdarr[idx_lbd]
    #    models = np.copy(new_models)
    #
    #    idx_lbd = np.where(lbdarr > 0.3)
    #    new_models = []
    #    for i in range(len(models)):
    #        tmp = np.copy(models[i])
    #        new_models.append(tmp[idx_lbd])
    #    lbdarr = lbdarr[idx_lbd]
    #    models = np.copy(new_models)

    for i in range(np.shape(listpar)[0]):
        for j in range(len(listpar[i])):
            if listpar[i][j] < 0:
                listpar[i][j] = 0.0

    mask = np.ones(len(minfo[0]), dtype=bool)
    mask[[2, 6]] = False
    result = []
    for i in range(len(minfo)):
        result.append(minfo[i][mask])
    minfo = np.copy(result)

    for i in range(np.shape(minfo)[0]):
        minfo[i][3] = np.log10(minfo[i][3])

    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    listpar = list(
        [
            listpar[0],
            listpar[1],
            listpar[3],
            listpar[4],
            listpar[5],
            listpar[7],
            listpar[8],
        ]
    )

    tab = flag.folder_tables + "/qe_ixon.csv"
    a = np.loadtxt(tab, delimiter=",")
    temp_lbd = []
    for i in range(len(a)):
        temp_lbd.append(a[i][0])
    temp_lbd = np.array(temp_lbd)
    temp_lbd = temp_lbd * 1e-3

    qe = []
    for i in range(len(a)):
        qe.append(a[i][1])
    qe = np.array(qe) / 100.0

    lbd_temp = np.array([0.3656, 0.4353, 0.5477, 0.6349, 0.8797])
    new_models = []
    # print(len(models))
    # THIS TOOK WAY TOO LONG, AND WAS ALWAYS THE SAME FOR ANY SIMULATION,
    # SO I SAVED THE RESULT IN THE MODELS/ FOLDER
    # for i in range(len(models)):
    #     print(i)
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[0])
    #     qet = qe[idx]
    #     Uf = doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[i], filt='U')
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[1])
    #     qet = qe[idx]
    #     Bf = doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[i], filt='B')
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[2])
    #     qet = qe[idx]
    #     Vf = doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[i], filt='V')
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[3])
    #     qet = qe[idx]
    #     Rf = doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[i], filt='R')
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[4])
    #     qet = qe[idx]
    #     If = doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[i], filt='I')
    #
    #     new_models.append(np.array([Uf, Bf, Vf, Rf, If]))
    #
    # np.save(flag.folder_models + 'acol_pol_models.npy', new_models)
    new_models = np.load(flag.folder_models + "acol_pol_models.npy")
    # models = np.copy(new_models)

    # new_models = np.load('models/' + 'models_pol_bcmi.npy')
    lbdarr = np.array([0.3656, 0.4353, 0.5477, 0.6349, 0.8797])
    # models = np.copy(new_models)

    return ctrlarr, minfo, new_models, lbdarr, listpar, dims, isig


# =====================================================================
def read_aara_pol():

    dims = ["M", "ob", "Hfrac", "sig0", "Rd", "mr", "cosi"]
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # xdrPL = flag.folder_models + "aara_pol_comp.xdr"
    xdrPL = flag.folder_models + "aara_pol.xdr"
    xdrPL2 = flag.folder_models + "aara_pol_comp.xdr"

    listpar, lbdarr, minfo, models = readBAsed(xdrPL, quiet=False)
    listpar2, lbdarr2, minfo2, models2 = readBAsed(xdrPL2, quiet=False)
    # print(listpar)
    # print(listpar2)
    listpar[0] = np.append(listpar[0], listpar2[0][-1])
    # print(listpar)

    minfo = np.concatenate((minfo, minfo2), axis=0)
    models = np.concatenate((models, models2), axis=0)

    # listpar, lbdarr, minfo, models = readBAsed(xdrPL, quiet=False)

    models = np.array(models) * 100.0  # in percentage
    # print(minfo)
    # minfo, models = minfo[3600:], models[3600:]

    # new_models = []
    # idx_lbd = np.where(lbdarr < 1.0)
    #
    # for i in range(len(models)):
    #     tmp = np.copy(models[i])
    #     new_models.append(tmp[idx_lbd])
    # lbdarr = lbdarr[idx_lbd]
    # models = np.copy(new_models)

    # idx_lbd = np.where(lbdarr > 0.3)
    # new_models = []
    # for i in range(len(models)):
    #     tmp = np.copy(models[i])
    #     new_models.append(tmp[idx_lbd])
    # lbdarr = lbdarr[idx_lbd]
    # models = np.copy(new_models)

    for i in range(np.shape(listpar)[0]):
        for j in range(len(listpar[i])):
            if listpar[i][j] < 0:
                listpar[i][j] = 0.0

    mask = np.ones(len(minfo[0]), dtype=bool)
    mask[[2, 6]] = False
    result = []
    for i in range(len(minfo)):
        result.append(minfo[i][mask])
    minfo = np.copy(result)

    for i in range(np.shape(minfo)[0]):
        minfo[i][3] = np.log10(minfo[i][3])

    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    listpar = list(
        [
            listpar[0],
            listpar[1],
            listpar[3],
            listpar[4],
            listpar[5],
            listpar[7],
            listpar[8],
        ]
    )

    # if doconvpol is True:
    # tab = 'tables/qe_ixon.csv'
    # a = np.loadtxt(tab, delimiter=',')
    # temp_lbd = []
    # for i in range(len(a)):
    #     temp_lbd.append(a[i][0])
    # temp_lbd = np.array(temp_lbd)
    # temp_lbd = temp_lbd * 1e-3
    #
    # qe = []
    # for i in range(len(a)):
    #     qe.append(a[i][1])
    # qe = np.array(qe) / 100.

    # lbd_temp = np.array([0.3656, 0.4353, 0.5477, 0.6349, 0.8797])
    # new_models = []
    # for j in range(len(models)):
    #     # print(j)
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[0])
    #     qet = qe[idx]
    #     # print(len(models[j]), models[j])
    #     Uf = hdt.doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[j], filt='U')
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[1])
    #     qet = qe[idx]
    #     Bf = hdt.doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[j], filt='B')
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[2])
    #     qet = qe[idx]
    #     Vf = hdt.doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[j], filt='V')
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[3])
    #     qet = qe[idx]
    #     Rf = hdt.doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[j], filt='R')
    #     temp, idx = find_nearest(temp_lbd, lbd_temp[4])
    #     qet = qe[idx]
    #     If = hdt.doFilterConvPol(x0=lbdarr * 1e4, intens=qet,
    #                              pol=models[j], filt='I')
    #     new_models.append(np.array([Uf, Bf, Vf, Rf, If]))
    # models = np.copy(new_models)
    # else:
    models = np.load(flag.folder_models + "NEWmodels_pol_aara.npy")
    lbdarr = np.array([0.3656, 0.4353, 0.5477, 0.6349, 0.8797])
    # models = np.copy(new_models)
    # if doconvpol is True:
    #    np.save(file='models/' + 'models_pol_aara.npy', arr=models)

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# =======================================================================


def read_stellar_prior():
    if flag.stellar_prior is True:
        chain = np.load(flag.folder_fig + flag.stars + "/" + flag.npy_star)
        Ndim = np.shape(chain)[-1]
        flatchain = chain.reshape((-1, Ndim))

        mas = flatchain[:, 0]
        obl = flatchain[:, 1]
        age = flatchain[:, 2]
        # dis = flatchain[:, -2]
        # ebv = flatchain[:, -1]

        # grid_mas = np.linspace(np.min(mas), np.max(mas), 100)
        # grid_obl = np.linspace(np.min(obl), np.max(obl), 100)
        # grid_age = np.linspace(np.min(age), np.max(age), 100)
        # grid_ebv = np.linspace(np.min(ebv), np.max(ebv), 100)

        grid_mas = np.linspace(3.0, 7.0, 100)
        grid_obl = np.linspace(1.00, 1.5, 100)
        grid_age = np.linspace(0.01, 0.99, 100)
        # grid_dis = np.linspace(0.00, 140, 100)
        # grid_ebv = np.linspace(0.00, 0.10, 100)

        pdf_mas = kde_scipy(x=mas, x_grid=grid_mas, bandwidth=0.005)
        pdf_obl = kde_scipy(x=obl, x_grid=grid_obl, bandwidth=0.005)
        pdf_age = kde_scipy(x=age, x_grid=grid_age, bandwidth=0.01)
        # pdf_dis = kde_scipy(x=dis, x_grid=grid_dis, bandwidth=0.01)
        # pdf_ebv = kde_scipy(x=ebv, x_grid=grid_ebv, bandwidth=0.0005)

    else:
        grid_mas = 0
        grid_obl = 0
        grid_age = 0
        # grid_dis = 0
        # grid_ebv = 0

        pdf_mas = 0
        pdf_obl = 0
        pdf_age = 0
        # pdf_dis = 0
        # pdf_ebv = 0

    grid_priors = [grid_mas, grid_obl, grid_age]
    pdf_priors = [pdf_mas, pdf_obl, pdf_age]

    return grid_priors, pdf_priors


def read_espadons(fname):

    # read fits
    if str(flag.stars) == "HD37795":
        hdr_list = fits.open(fname)
        fits_data = hdr_list[0].data
        fits_header = hdr_list[0].header
        # read MJD
        MJD = fits_header["MJDATE"]
        lat = fits_header["LATITUDE"]
        lon = fits_header["LONGITUD"]
        lbd = fits_data[0, :]
        ordem = lbd.argsort()
        lbd = lbd[ordem] * 10
        flux_norm = fits_data[1, ordem]
    # vel, flux = spt.lineProf(lbd, flux_norm, lbc=lbd0)
    else:
        print(fname)
        lbd, flux_norm, MJD, dateobs, datereduc, fitsfile = spec.loadfits(fname)
    #    plt.plot (lbd, flux_norm)
    #    plt.show()
    return lbd, flux_norm, MJD


# ==================================================================================
# READS LINES FROM DATA
# ==================================================================================
def read_line_spectra(models, lbdarr, linename):

    table = flag.folder_data + str(flag.stars) + "/" + "spectra/" + "list_spectra.txt"

    if os.path.isfile(table) is False or os.path.isfile(table) is True:
        os.system(
            "ls "
            + flag.folder_data
            + str(flag.stars)
            + "/spectra"
            + "/* | xargs -n1 basename >"
            + flag.folder_data
            + "/"
            + "list_spectra.txt"
        )
        spectra_list = np.genfromtxt(table, comments="#", dtype="str")
        file_name = np.copy(spectra_list)

    fluxes, waves, errors = [], [], []

    if file_name.tolist()[-3:] == "csv" or file_name.tolist()[-3:] == "txt":
        file_ha = str(flag.folder_data) + str(flag.stars) + "/spectra/" + str(file_name)
        try:
            wl, normal_flux = np.loadtxt(file_ha, delimiter=" ").T
        except:
            wl, normal_flux = np.loadtxt(file_ha)
    elif file_name.tolist()[-4:] == "fits" or file_name.tolist()[-7:] == "fits.gz":
        file_ha = str(flag.folder_data) + str(flag.stars) + "/spectra/" + str(file_name)
        # print(file_ha)
        wl, normal_flux, MJD = read_espadons(file_ha)
    else:
        file_ha = str(flag.folder_data) + str(flag.stars) + "/spectra/" + str(file_name)
        wl = np.load(file_ha)["lbd"]
        normal_flux = np.load(file_ha)["flux"]

    # --------- Selecting an approximate wavelength interval
    c = 299792.458  # km/s
    lbd_central = lines_dict[linename]

    if flag.Sigma_Clip:
        # gives the line profiles in velocity space
        vl, fx = lineProf(wl, normal_flux, hwidth=5000.0, lbc=lbd_central)
        vel, fluxes = Sliding_Outlier_Removal(vl, fx, 50, 8, 15)
        # wl = c * lbd_central / (c - vel)
    else:
        vel = (wl - lbd_central) * c / lbd_central
        largura = 6000  # km/s    # 6000 is compatible with sigmaclip
        lim_esq1 = np.where(vel < -largura / 2.0)
        lim_dir1 = np.where(vel > largura / 2.0)

        inf = lim_esq1[0][-1]
        sup = lim_dir1[0][0]

        # wl = wl[inf:sup]
        fluxes = normal_flux[inf:sup]

    if flag.stars == "MT91-213":
        radv = 2.8
    else:
        radv = delta_v(vel, fluxes, "Ha")
    # AMANDA_VOLTAR: o que fazer quando o
    # radv = 2.8
    print("RADIAL VELOCITY = {0}".format(radv))
    vel = vel - radv
    wl = c * lbd_central / (c - vel)

    # Changing the units
    waves = wl * 1e-4  # mum
    # nao tenho informacao sobre os erros ainda

    interpolator = interp1d(waves, fluxes)
    lbdarr = lbdarr[1:-1]
    fluxes = interpolator(lbdarr)
    errors = np.ones(len(fluxes)) * 0.04

    # TESTING NEW OPTION
    # # Bin the spectra data to the lbdarr (model)
    # # Creating box limits array
    box_lim = np.zeros(len(lbdarr) - 1)
    # for i in range(len(lbdarr) - 1):
    #     box_lim[i] = (lbdarr[i] + lbdarr[i + 1]) / 2.0
    #
    # # Binned flux
    # bin_flux = np.zeros(len(box_lim) - 1)
    #
    # lbdarr = lbdarr[1:-1]
    # errors = np.zeros(len(lbdarr))
    # # sigma_new.fill(0.017656218)
    # # AMANDA_VOLTAR: erros devem ser estimados numa rotina de precondicionamento
    # # HD6226: 0.04
    # # MT91-213: 0.01
    # if flag.stars == "HD6226":
    #     erval = 0.04
    # else:
    #     erval = 0.04
    # for i in range(len(box_lim) - 1):
    #     # lbd observations inside the box
    #     index = np.argwhere((waves > box_lim[i]) & (waves < box_lim[i + 1]))
    #     if len(index) == 0:
    #         bin_flux[i] = -np.inf
    #     elif len(index) == 1:
    #         bin_flux[i] = fluxes[index[0][0]]
    #         errors[i] = erval
    #     else:
    #         # Calculating the mean flux in the box
    #         bin_flux[i] = np.sum(fluxes[index[0][0] : index[-1][0]]) / len(
    #             fluxes[index[0][0] : index[-1][0]]
    #         )
    #         errors[i] = erval / np.sqrt(len(index))
    #
    # flx = bin_flux

    # que estou interpolando sao as observacoes nos modelos, entao nao eh models_new que tenho que fazer. o models ta pronto
    # log space
    # logF = np.log10(obs_new)
    # mask = np.where(obs_new == -np.inf)
    # logF[mask] = -np.inf # so that dlogF == 0 in these points and the chi2 will not be computed
    # dlogF = sigma_new/obs_new

    # lbdarr = lbdarr[np.isfinite(flx)]
    # flux = flx[np.isfinite(flx)]
    # errors = errors[np.isfinite(flx)]

    # cutwave = waves[np.logical_and(waves > np.min(lbdarr), waves < np.max(lbdarr))]
    # fluxes = fluxes[np.logical_and(waves > np.min(lbdarr), waves < np.max(lbdarr))]
    # errors = np.ones(len(cutwave)) * 0.04

    novo_models = np.zeros((len(models), len(lbdarr)))
    for i in range(len(models)):
        mm = models[i][1:-1][np.isfinite(fluxes)]
        novo_models[i] = mm
    # for i, umodel in enumerate(models):
    #     interpolator = interp1d(lbdarr, umodel)
    #     mm = interpolator(cutwave)
    #     novo_models[i] = mm

    # if linename == 'Ha':
    #    if flag.remove_partHa:
    #        lbdarr1 = lbdarr[:149]
    #        lbdarr2 = lbdarr[194:]
    #        lbdarr = np.append(lbdarr1,lbdarr2)
    #        novo_models = np.zeros((5500,191))
    #        obs_new1 = flux[:149]
    #        obs_new2 = flux[194:]
    #        flux = np.append(obs_new1,obs_new2)
    #        #logF = np.log10(obs_new)
    #        errors = np.append(errors[:149], errors[194:])
    #        #dlogF = sigma_new2 / obs_new
    #        i = 0
    #        while i < len(models):
    #            novo_models[i] = np.append(novo_models[i][:149],novo_models[i][194:])
    #            i+=1
    #        #logF_grid = np.log10(novo_models)

    if flag.only_wings:
        # plt.plot(waves, fluxes)
        # point_a, point_b = plt.ginput(2)
        # plt.close()
        # H_peak = find_nearest2(lbdarr, lines_dict[linename])
        # keep_a = find_nearest2(lbdarr, point_a[0])
        # keep_b = find_nearest2(lbdarr, point_b[0])
        keep_a = find_nearest2(lbdarr, lines_dict[linename] - 0.0009)
        keep_b = find_nearest2(lbdarr, lines_dict[linename] + 0.0009)
        lbdarr1 = lbdarr[:keep_a]
        lbdarr2 = lbdarr[keep_b:]
        lbdarr = np.concatenate([lbdarr1, lbdarr2])
        obs_new1 = fluxes[:keep_a]
        obs_new2 = fluxes[keep_b:]
        flux = np.concatenate([obs_new1, obs_new2])
        # logF = np.log10(obs_neww)
        sigma_new1 = errors[:keep_a]
        sigma_new2 = errors[keep_b:]
        errors = np.concatenate([sigma_new1, sigma_new2])
        # dlogF = sigma_new/obs_neww
        novo_models1 = novo_models[:, :keep_a]
        novo_models2 = novo_models[:, keep_b:]
        novo_models = np.hstack((novo_models1, novo_models2))
        # logF_grid = np.log10(novo_models)

    if flag.only_centerline:
        plt.plot(waves, fluxes)
        point_a, point_b = plt.ginput(2)
        plt.close()
        # H_peak = find_nearest2(lbdarr, lines_dict[linename])
        keep_a = find_nearest2(lbdarr, point_a[0])
        keep_b = find_nearest2(lbdarr, point_b[0])
        lbdarr1 = lbdarr[keep_a:]
        lbdarr2 = lbdarr[:keep_b]
        lbdarr = np.concatenate([lbdarr1, lbdarr2])
        obs_new1 = fluxes[keep_a:]
        obs_new2 = fluxes[:keep_b]
        flux = np.concatenate([obs_new1, obs_new2])
        # logF = np.log10(obs_neww)
        sigma_new1 = errors[keep_a:]
        sigma_new2 = errors[:keep_b]
        errors = np.concatenate([sigma_new1, sigma_new2])
        # dlogF = sigma_new/obs_neww
        novo_models1 = novo_models[:, keep_a:]
        novo_models2 = novo_models[:, :keep_b]
        novo_models = np.hstack((novo_models1, novo_models2))
        # logF_grid = np.log10(novo_models)

    if flag.remove_partHa:
        # if not flag.acrux:
        #    plt.plot(waves, fluxes)
        #    point_a, point_b = plt.ginput(2)
        #    plt.close()
        # else:
        point_a = [0.6573]
        point_b = [0.6585]

        # H_peak = find_nearest2(lbdarr, lines_dict[linename])
        keep_a = find_nearest2(lbdarr, point_a[0])
        keep_b = find_nearest2(lbdarr, point_b[0])
        lbdarr1 = lbdarr[:keep_a]
        lbdarr2 = lbdarr[keep_b:]
        lbdarr = np.concatenate([lbdarr1, lbdarr2])
        obs_new1 = fluxes[:keep_a]
        obs_new2 = fluxes[keep_b:]
        flux = np.concatenate([obs_new1, obs_new2])
        # logF = np.log10(obs_neww)
        sigma_new1 = errors[:keep_a]
        sigma_new2 = errors[keep_b:]
        errors = np.concatenate([sigma_new1, sigma_new2])
        # dlogF = sigma_new/obs_neww
        novo_models1 = novo_models[:, :keep_a]
        novo_models2 = novo_models[:, keep_b:]
        novo_models = np.hstack((novo_models1, novo_models2))

    # limits = np.logical_and(lbdarr > 0.6541, lbdarr < 0.6585)

    # print(lbdarr)
    # print(len(flux), len(lbdarr), len(novo_models[0]))
    return fluxes, errors, novo_models, lbdarr, box_lim  # lbdarr, box_lim


# ==============================================================================
def read_iue(models, lbdarr):

    table = flag.folder_data + flag.stars + "/" + "list_iue.txt"

    # os.chdir(flag.folder_data + str(flag.stars) + '/')
    if os.path.isfile(table) is False:
        os.system(
            "ls "
            + flag.folder_data
            + flag.stars
            + "/*.FITS | xargs -n1 basename >"
            + flag.folder_data
            + flag.stars
            + "/"
            + "list_iue.txt"
        )

    iue_list = np.genfromtxt(table, comments="#", dtype="str")
    # print(iue_list.size)
    if iue_list.size == 1:
        # print(file_name)
        file_name = str(iue_list)
    else:
        file_name = np.copy(iue_list)

    fluxes, waves, errors = [], [], []
    print(file_name)

    if file_name[0][-3:] == "csv":
        file_iue = str(flag.folder_data) + flag.stars + "/" + str(file_name)
        wave, flux, sigma = np.loadtxt(str(file_iue), delimiter=",").T
        fluxes = np.concatenate((fluxes, flux * 1e4), axis=0)
        waves = np.concatenate((waves, wave * 1e-4), axis=0)
        errors = np.concatenate((errors, sigma * 1e4), axis=0)

    else:
        # Combines the observations from all files in the folder, taking the good quality ones
        for fname in file_name:
            file_iue = str(flag.folder_data) + flag.stars + "/" + str(fname)
            # print(file_iue)
            hdulist = fits.open(file_iue)
            tbdata = hdulist[1].data
            wave = tbdata.field("WAVELENGTH") * 1e-4  # mum
            flux = tbdata.field("FLUX") * 1e4  # erg/cm2/s/A -> erg/cm2/s/mum
            sigma = tbdata.field("SIGMA") * 1e4  # erg/cm2/s/A -> erg/cm2/s/mum

            # Filter of bad data: '0' is good data
            qualy = tbdata.field("QUALITY")
            idx = np.where((qualy == 0))
            wave = wave[idx]
            sigma = sigma[idx]
            flux = flux[idx]

            idx = np.where((flux > 0.0))
            wave = wave[idx]
            sigma = sigma[idx]
            flux = flux[idx]

            fluxes = np.concatenate((fluxes, flux), axis=0)
            waves = np.concatenate((waves, wave), axis=0)
            errors = np.concatenate((errors, sigma), axis=0)

    if os.path.isdir(flag.folder_fig + flag.stars) is False:
        os.mkdir(flag.folder_fig + flag.stars)

    # ------------------------------------------------------------------------------
    # Would you like to cut the spectrum?
    # if flag.cut_iue_regions is True:
    #    wave_lim_min_iue = 0.13
    #    wave_lim_max_iue = 0.30
    #
    #    # Do you want to select a range to middle UV? (2200 bump region)
    #    wave_lim_min_bump_iue = 0.20  # 0.200 #0.195  #0.210 / 0.185
    #    wave_lim_max_bump_iue = 0.30  # 0.300 #0.230  #0.300 / 0.335
    #
    #    indx = np.where(((waves >= wave_lim_min_iue) &
    #                     (waves <= wave_lim_max_iue)))
    #    indx2 = np.where(((waves >= wave_lim_min_bump_iue) &
    #                      (waves <= wave_lim_max_bump_iue)))
    #    indx3 = np.concatenate((indx, indx2), axis=1)[0]
    #    waves, fluxes, errors = waves[indx3], fluxes[indx3], errors[indx3]

    # else: # remove observations outside the range
    wave_lim_min_iue = min(waves)
    wave_lim_max_iue = 0.290
    indx = np.where(((waves >= wave_lim_min_iue) & (waves <= wave_lim_max_iue)))
    waves, fluxes, errors = waves[indx], fluxes[indx], errors[indx]

    # sort the combined observations in all files
    new_wave, new_flux, new_sigma = zip(*sorted(zip(waves, fluxes, errors)))

    nbins = 200
    xbin, ybin, dybin = bin_data(new_wave, new_flux, nbins, exclude_empty=True)

    # just to make sure that everything is in order
    ordem = xbin.argsort()
    wave = xbin[ordem]
    flux = ybin[ordem]
    sigma = dybin[ordem]

    for i in range(len(sigma)):
        if sigma[i] < flux[i] * 0.01:
            sigma[i] = flux[i] * 0.01

    # ic(sigma / flux)

    return wave, flux, sigma


# ======================================================================


def combine_sed(wave, flux, sigma, models, lbdarr):
    """Combines SED parts into 1 array
    """
    if flag.lbd_range == "UV":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 0.3  # mum
    if flag.lbd_range == "UV+VIS":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 0.8  # mum
    if flag.lbd_range == "UV+VIS+NIR":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 5.0  # mum
    if flag.lbd_range == "UV+VIS+NIR+MIR":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 40.0  # mum
    if flag.lbd_range == "UV+VIS+NIR+MIR+FIR":
        wave_lim_min = 0.13  # mum
        wave_lim_max = 350.0  # mum
    if flag.lbd_range == "UV+VIS+NIR+MIR+FIR+MICROW+RADIO":
        wave_lim_min = 0.13  # mum
        wave_lim_max = np.max(wave)  # mum
    if flag.lbd_range == "VIS+NIR+MIR+FIR+MICROW+RADIO":
        wave_lim_min = 0.39  # mum
        wave_lim_max = np.max(wave)  # mum
    if flag.lbd_range == "NIR+MIR+FIR+MICROW+RADIO":
        wave_lim_min = 0.7  # mum
        wave_lim_max = np.max(wave)  # mum
    if flag.lbd_range == "MIR+FIR+MICROW+RADIO":
        wave_lim_min = 5.0  # mum
        wave_lim_max = np.max(wave)  # mum
    if flag.lbd_range == "FIR+MICROW+RADIO":
        wave_lim_min = 40.0  # mum
        wave_lim_max = np.max(wave)  # mum
    if flag.lbd_range == "MICROW+RADIO":
        wave_lim_min = 1e3  # mum
        wave_lim_max = np.max(wave)  # mum
    if flag.lbd_range == "RADIO":
        wave_lim_min = 1e6  # mum
        wave_lim_max = np.max(wave)  # mum

    band = np.copy(flag.lbd_range)
    # print(wave_lim_max)
    ordem = wave.argsort()
    wave = wave[ordem]
    flux = flux[ordem]
    sigma = sigma[ordem]
    # print(sigma)
    # ------------------------------------------------------------------------------
    # select lbdarr to coincide with lbd
    # models_new = np.zeros([len(models), len(wave)])

    idx = np.where((wave >= wave_lim_min) & (wave <= wave_lim_max))
    wave = wave[idx]
    flux = flux[idx]
    sigma = sigma[idx]
    models_new = np.zeros([len(models), len(wave)])

    for i in range(len(models)):  # A interpolacao
        models_new[i, :] = 10.0 ** griddata(
            np.log10(lbdarr), np.log10(models[i]), np.log10(wave), method="linear"
        )
    # to log space
    logF = np.log10(flux)
    dlogF = sigma / flux
    logF_grid = np.log10(models_new)

    return logF, dlogF, logF_grid, wave


# ==============================================================================
def read_opd_pol(models):
    """
    Reads polarization data_pos

    Usage:
    wave, flux, sigma = read_opd_pol

    """

    table = flag.folder_data + str(flag.stars) + "/" + "pol/" + "list_pol.txt"

    if os.path.isfile(table) is False or os.path.isfile(table) is True:
        os.system(
            "ls "
            + flag.folder_data
            + str(flag.stars)
            + "/pol"
            + "/* | xargs -n1 basename >"
            + flag.folder_data
            + "/"
            + "list_pol.txt"
        )
        pol_list = np.genfromtxt(table, comments="#", dtype="str")
        table_csv = np.copy(pol_list)

    # Reading opd data from csv (beacon site)
    if table_csv != "hpol.npy":
        csv_file = flag.folder_data + str(flag.stars) + "/pol/" + str(table_csv)
        df = pd.read_csv(csv_file)
        JD = df["#MJD"] + 2400000
        Filter = df["filt"]
        # flag = df['flag']
        lbd = np.array([0.3656, 0.4353, 0.5477, 0.6349, 0.8797])
        P = np.array(df["P"])  # * 100  # Units of percentage
        SIGMA = np.array(df["sigP"])

        # Plot P vs lambda
        Pol, error, wave = [], [], []
        nu, nb, nv, nr, ni = 0.0, 0.0, 0.0, 0.0, 0.0
        wu, wb, wv, wr, wi = 0.0, 0.0, 0.0, 0.0, 0.0
        eu, eb, ev, er, ei = 0.0, 0.0, 0.0, 0.0, 0.0

        filtros = np.unique(Filter)
        for h in range(len(JD)):
            for filt in filtros:
                if Filter[h] == filt:  # and flag[h] is not 'W':
                    if filt == "u":
                        wave.append(lbd[0])
                        Pol.append(P[h])
                        error.append(SIGMA[h])
                        wu = wu + P[h]  # * 100.
                        nu = nu + 1.0
                        eu = eu + (SIGMA[h]) ** 2.0
                    if filt == "b":
                        wave.append(lbd[1])
                        Pol.append(P[h])
                        error.append(SIGMA[h])
                        wb = wb + P[h]  # * 100.
                        nb = nb + 1.0
                        eb = eb + (SIGMA[h]) ** 2.0
                    if filt == "v":
                        wave.append(lbd[2])
                        Pol.append(P[h])
                        error.append(SIGMA[h])
                        wv = wv + P[h]  # * 100.
                        nv = nv + 1.0
                        ev = ev + (SIGMA[h]) ** 2.0
                    if filt == "r":
                        wave.append(lbd[3])
                        Pol.append(P[h])  # 100. * P[i])
                        error.append(SIGMA[h])
                        wr = wr + P[h]  # * 100.
                        nr = nr + 1.0
                        er = er + (SIGMA[h]) ** 2.0
                    if filt == "i":
                        wave.append(lbd[4])
                        Pol.append(P[h])
                        error.append(SIGMA[h])
                        wi = wi + P[h]  # * 100.
                        ni = ni + 1.0
                        ei = ei + (SIGMA[h]) ** 2.0
        try:
            eu = np.sqrt(eu / nu)
            eb = np.sqrt(eb / nb)
            ev = np.sqrt(ev / nv)
            er = np.sqrt(er / nr)
            ei = np.sqrt(ei / ni)
        except:
            # eu = np.sqrt(eu / nu)
            eb = np.sqrt(eb / nb)
            ev = np.sqrt(ev / nv)
            er = np.sqrt(er / nr)
            ei = np.sqrt(ei / ni)

        # eu = np.sqrt(eu)
        # eb = np.sqrt(eb)
        # ev = np.sqrt(ev)
        # er = np.sqrt(er)
        # ei = np.sqrt(ei)
        try:
            sigma = np.array([eu, eb, ev, er, ei])  # * 100
        except:
            sigma = np.array([eb, ev, er, ei])  # * 100

        try:
            mean_u = wu / nu
        except:
            mean_u = 0.0
            print("u sem dado")

        try:
            mean_b = wb / nb
        except:
            mean_b = 0.0
            print("b sem dado")

        try:
            mean_v = wv / nv
        except:
            mean_v = 0.0
            print("v sem dado")

        try:
            mean_r = wr / nr
        except:
            mean_r = 0.0
            print("r sem dado")

        try:
            mean_i = wi / ni
        except:
            mean_i = 0.0
            print("i sem dado")

        try:
            flux = np.array([mean_u, mean_b, mean_v, mean_r, mean_i])
        except:
            flux = np.array([mean_b, mean_v, mean_r, mean_i])
        # flux = np.array([mean_b, mean_v, mean_r, mean_i])
        wave = np.copy(lbd)
    else:
        # print(table_csv, star, folder_data)
        wave, flux, sigma = np.load(
            flag.folder_data + str(flag.stars) + "/" + str(table_csv)
        )

    # logF = np.log10(flux)
    # dlogF = sigma / flux

    logF_grid = models
    box_lim = 0.0

    return flux, sigma, logF_grid, wave, box_lim


# ============================================================================


def read_votable():

    table = flag.folder_data + str(flag.stars) + "/" + "list.txt"

    # os.chdir(folder_data + str(star) + '/')
    # if os.path.isfile(table) is False or os.path.isfile(table) is True:
    # os.system('ls ' + flag.folder_data + str(flag.stars) +
    #            '/*.xml | xargs -n1 basename >' +
    #            flag.folder_data + str(flag.stars) + '/' + 'list.txt')
    vo_list = np.genfromtxt(table, comments="#", dtype="str")
    table_name = np.copy(vo_list)
    vo_file = flag.folder_data + str(flag.stars) + "/" + str(table_name)
    # thefile = 'data/HD37795/alfCol.sed.dat' #folder_data + str(star) + '/' + str(table_name)
    # table = np.genfromtxt(thefile, usecols=(1, 2, 3))
    # wave, flux, sigma = table[:,0], table[:,1], table[:,2]

    try:
        t1 = atpy.Table(vo_file, pedantic=False)
        wave = t1["Wavelength"][:]  # Angstrom
        flux = t1["Flux"][:]  # erg/cm2/s/A
        sigma = t1["Error"][:]  # erg/cm2/s/A
    except:
        t1 = atpy.Table(vo_file, tid=1)
        wave = t1["SpectralAxis0"][:]  # Angstrom
        flux = t1["Flux0"][:]  # erg/cm2/s/A
        sigma = [0.0] * len(flux)  # erg/cm2/s/A

    new_wave, new_flux, new_sigma = zip(*sorted(zip(wave, flux, sigma)))

    new_wave = list(new_wave)
    new_flux = list(new_flux)
    new_sigma = list(new_sigma)

    # # Filtering null sigmas
    # for h in range(len(new_sigma)):
    #     if new_sigma[h] == 0.0:
    #         new_sigma[h] = 0.002 * new_flux[h]

    wave = np.copy(new_wave) * 1e-4
    flux = np.copy(new_flux) * 1e4
    sigma = np.copy(new_sigma) * 1e4

    for i in range(len(sigma)):
        if sigma[i] != 0:
            if sigma[i] < flux[i] * 0.1:
                # print("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                # print(wave[i], flux[i])
                sigma[i] = flux[i] * 0.1

    keep = wave > 0.34
    wave, flux, sigma = wave[keep], flux[keep], sigma[keep]

    if flag.stars == "HD37795":
        fname = flag.folder_data + "/HD37795/alfCol.txt"
        data = np.loadtxt(
            fname,
            dtype={
                "names": ("lbd", "flux", "dflux", "source"),
                "formats": (np.float, np.float, np.float, "|S20"),
            },
        )
        wave = np.hstack([wave, data["lbd"]])
        flux = np.hstack([flux, jy2cgs(1e-3 * data["flux"], data["lbd"])])
        sigma = np.hstack([sigma, jy2cgs(1e-3 * data["dflux"], data["lbd"])])

    if flag.stars == "HD58715":
        fname = flag.folder_data + "/HD58715/bcmi_radio.dat"
        data = np.loadtxt(
            fname,
            dtype={
                "names": ("lbd", "flux", "dflux", "source"),
                "formats": (np.float, np.float, np.float, "|S20"),
            },
        )
        wave = np.hstack([wave, data["lbd"]])
        flux = np.hstack([flux, jy2cgs(data["flux"], data["lbd"])])
        sigma = np.hstack([sigma, jy2cgs(data["dflux"], data["lbd"])])

    return wave, flux, sigma


# ==============================================================================
# Calls the xdr reading function
def read_models(lista_obs):

    if flag.model == "aeri":
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_BAphot2_xdr(
            lista_obs
        )
    if flag.model == "befavor":
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_befavor_xdr()
    if flag.model == "aara":
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_aara_xdr()
    if flag.model == "beatlas":
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_beatlas_xdr(
            lista_obs
        )
    if flag.model == "acol":
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_acol_Ha_xdr(
            lista_obs
        )
    if flag.model == "pol":
        if flag.stars == "HD158427":
            ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_aara_pol()
            print(minfo[:, 0])
        else:
            ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_acol_pol_xdr()

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# =======================================================================
# Sigmaclip routine by Jonathan Labadie-Bartz


def Sliding_Outlier_Removal(x, y, window_size, sigma=3.0, iterate=1):
    # remove NANs from the data
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    # make sure that the arrays are in order according to the x-axis
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]

    # tells you the difference between the last and first x-value
    x_span = x.max() - x.min()
    i = 0
    x_final = x
    y_final = y
    while i < iterate:
        i += 1
        x = x_final
        y = y_final

        # empty arrays that I will append not-clipped data points to
        x_good_ = np.array([])
        y_good_ = np.array([])

        # Creates an array with all_entries = True. index where you want to remove outliers are set to False
        tf_ar = np.full((len(x),), True, dtype=bool)
        ar_of_index_of_bad_pts = np.array([])  # not used anymore

        # this is how many days (or rather, whatever units x is in) to slide the window center when finding the outliers
        slide_by = window_size / 5.0

        # calculates the total number of windows that will be evaluated
        Nbins = int((int(x.max() + 1) - int(x.min())) / slide_by)

        for j in range(Nbins + 1):
            # find the minimum time in this bin, and the maximum time in this bin
            x_bin_min = x.min() + j * (slide_by) - 0.5 * window_size
            x_bin_max = x.min() + j * (slide_by) + 0.5 * window_size

            # gives you just the data points in the window
            x_in_window = x[(x > x_bin_min) & (x < x_bin_max)]
            y_in_window = y[(x > x_bin_min) & (x < x_bin_max)]

            # if there are less than 5 points in the window, do not try to remove outliers.
            if len(y_in_window) > 5:

                # Removes a linear trend from the y-data that is in the window.
                y_detrended = detrend(y_in_window, type="linear")
                y_in_window = y_detrended
                # print(np.median(m_in_window_))
                y_med = np.median(y_in_window)

                # finds the Median Absolute Deviation of the y-pts in the window
                y_MAD = MAD(y_in_window)

                # This mask returns the not-clipped data points.
                # Maybe it is better to only keep track of the data points that should be clipped...
                mask_a = (y_in_window < y_med + y_MAD * sigma) & (
                    y_in_window > y_med - y_MAD * sigma
                )
                # print(str(np.sum(mask_a)) + '   :   ' + str(len(m_in_window)))
                y_good = y_in_window[mask_a]
                x_good = x_in_window[mask_a]

                y_bad = y_in_window[~mask_a]
                x_bad = x_in_window[~mask_a]

                # keep track of the index --IN THE ORIGINAL FULL DATA ARRAY-- of pts to be clipped out
                try:
                    clipped_index = np.where([x == z for z in x_bad])[1]
                    tf_ar[clipped_index] = False
                    ar_of_index_of_bad_pts = np.concatenate(
                        [ar_of_index_of_bad_pts, clipped_index]
                    )
                except IndexError:
                    # print('no data between {0} - {1}'.format(x_in_window.min(), x_in_window.max()))
                    pass
            # puts the 'good' not-clipped data points into an array to be saved

            # x_good_= np.concatenate([x_good_, x_good])
            # y_good_= np.concatenate([y_good_, y_good])

            # print(len(mask_a))
            # print(len(m
            # print(m_MAD)

        ##multiple data points will be repeated! We don't want this, so only keep unique values.
        # x_uniq, x_u_indexs = np.unique(x_good_, return_index=True)
        # y_uniq = y_good_[x_u_indexs]

        ar_of_index_of_bad_pts = np.unique(ar_of_index_of_bad_pts)
        # print('step {0}: remove {1} points'.format(i, len(ar_of_index_of_bad_pts)))
        # print(ar_of_index_of_bad_pts)

        # x_bad = x[ar_of_index_of_bad_pts]
        # y_bad = y[ar_of_index_of_bad_pts]
        # x_final = x[

        x_final = x[tf_ar]
        y_final = y[tf_ar]
    return (x_final, y_final)


# =======================================================================
# Creates file tag


def create_tag():

    # def create_tag(model,Ha, Hb, Hd, Hg):
    tag = "+" + flag.model

    if flag.only_wings:
        tag = tag + "_onlyWings"
    if flag.only_centerline:
        tag = tag + "_onlyCenterLine"
    if flag.remove_partHa:
        tag = tag + "_removePartHa"
    if flag.vsini_prior:
        tag = tag + "_vsiniPrior"
    if flag.dist_prior:
        tag = tag + "_distPrior"
    if flag.box_W:
        tag = tag + "_boxW"
    if flag.box_i:
        tag = tag + "_boxi"
    if flag.incl_prior:
        tag = tag + "_inclPrior"
    tag = tag + flag.lbd_range
    if flag.Ha:
        tag = tag + "+Ha"
    if flag.Hb:
        tag = tag + "+Hb"
    if flag.Hd:
        tag = tag + "+Hd"
    if flag.Hg:
        tag = tag + "+Hg"

    return tag


# =======================================================================
# Creates list of observables


def create_list():

    lista = np.array([])

    if flag.SED:
        lista = np.append(lista, "UV")
    if flag.Ha:
        lista = np.append(lista, "Ha")
    if flag.Hb:
        lista = np.append(lista, "Hb")
    if flag.Hd:
        lista = np.append(lista, "Hd")
    if flag.Hg:
        lista = np.append(lista, "Hg")

    return lista


# ========================================================================


def read_table():
    table = flag.folder_data + str(flag.stars) + "/sed_data/" + "list_sed.txt"

    # os.chdir(folder_data + str(star) + '/')
    # if os.path.isfile(table) is False or os.path.isfile(table) is True:
    os.system(
        "ls "
        + flag.folder_data
        + str(flag.stars)
        + "/sed_data/*.dat /*.csv /*.txt | xargs -n1 basename >"
        + flag.folder_data
        + str(flag.stars)
        + "/sed_data/"
        + "list_sed.txt"
    )
    vo_list = np.genfromtxt(table, comments="#", dtype="str")
    table_name = np.copy(vo_list)
    table = flag.folder_data + str(flag.stars) + "/sed_data/" + str(table_name)

    typ = (0, 1, 2, 3)

    a = np.genfromtxt(
        table,
        usecols=typ,
        unpack=True,
        delimiter="\t",
        comments="#",
        dtype={"names": ("B", "V", "R", "I"), "formats": ("f4", "f4", "f4", "f4")},
    )

    B_mag, V_mag, R_mag, I_mag = a["B"], a["V"], a["R"], a["I"]

    # lbd array (center of bands)
    wave = np.array([0.4361, 0.5448, 0.6407, 0.7980])

    c = 299792458e6  # um/s

    # Change observed magnitude to flux
    # Zero magnitude flux 10^-20.erg.s^-1.cm^2.Hz^-1  ---> convert to erg.s^-1.cm^2.um^-1
    B_flux0 = 4.26 * 10 ** (-20)
    V_flux0 = 3.64 * 10 ** (-20)
    R_flux0 = 3.08 * 10 ** (-20)
    I_flux0 = 2.55 * 10 ** (-20)

    B_flux = B_flux0 * 10.0 ** (-B_mag / 2.5)
    V_flux = V_flux0 * 10.0 ** (-V_mag / 2.5)
    R_flux = R_flux0 * 10.0 ** (-R_mag / 2.5)
    I_flux = I_flux0 * 10.0 ** (-I_mag / 2.5)

    # ---> convert to erg.s^-1.cm^2.um^-1
    B_flux = B_flux * c / (wave[0] ** 2)
    V_flux = V_flux * c / (wave[1] ** 2)
    R_flux = R_flux * c / (wave[2] ** 2)
    I_flux = I_flux * c / (wave[3] ** 2)

    flux = np.array([B_flux, V_flux, R_flux, I_flux])
    logF = np.log10(np.array([B_flux, V_flux, R_flux, I_flux]))
    # Uncertainty (?)
    # dlogF = 0.01*logF
    sigma = 0.01 * flux

    return wave, flux, sigma


def read_table_additional():
    table = flag.folder_data + str(flag.stars) + "/sed_data/" + "list_sed.txt"

    # os.chdir(folder_data + str(star) + '/')
    # if os.path.isfile(table) is False or os.path.isfile(table) is True:
    os.system(
        "ls "
        + flag.folder_data
        + str(flag.stars)
        + "/sed_data/*.dat /*.csv /*.txt | xargs -n1 basename >"
        + flag.folder_data
        + str(flag.stars)
        + "/sed_data/"
        + "list_sed.txt"
    )
    vo_list = np.genfromtxt(table, comments="#", dtype="str")
    table_name = np.copy(vo_list)
    table = flag.folder_data + str(flag.stars) + "/sed_data/" + str(table_name)

    try:
        lbds, flxs, sigs = np.loadtxt(table).T
    except:
        lbds, flxs = np.loadtxt(table).T
        sigs = flxs * 0.03

    return lbds, flxs, sigs


# =======================================================================
# Read the data files


def read_observables(models, lbdarr, lista_obs):

    logF_combined = []
    dlogF_combined = []
    logF_grid_combined = []
    wave_combined = []
    box_lim_combined = []

    if flag.SED:

        u = np.where(lista_obs == "UV")
        index = u[0][0]

        if flag.iue:
            wave, flux, sigma = read_iue(models[index], lbdarr[index])
        else:
            wave, flux, sigma = [], [], []

        if flag.votable:
            wave0, flux0, sigma0 = read_votable()
        elif flag.data_table:
            if flag.stars == "piAqr" or flag.stars == "gammaCas":
                wave0, flux0, sigma0 = read_table_additional()
            else:
                wave0, flux0, sigma0 = read_table()
        else:
            wave0, flux0, sigma0 = [], [], []

        wave = np.hstack([wave, wave0])
        flux = np.hstack([flux, flux0])
        sigma = np.hstack([sigma, sigma0])
        logF_UV, dlogF_UV, logF_grid_UV, wave_UV = combine_sed(
            wave, flux, sigma / 2.0, models[index], lbdarr[index]
        )
        # print(dlogF_UV)

        logF_combined.append(logF_UV)
        dlogF_combined.append(dlogF_UV)
        logF_grid_combined.append(logF_grid_UV)
        wave_combined.append(wave_UV)

    if flag.Ha:

        u = np.where(lista_obs == "Ha")
        index = u[0][0]
        logF_Ha, dlogF_Ha, logF_grid_Ha, wave_Ha, box_lim_Ha = read_line_spectra(
            models[index], lbdarr[index], "Ha"
        )

        logF_combined.append(logF_Ha)
        dlogF_combined.append(dlogF_Ha)
        logF_grid_combined.append(logF_grid_Ha)
        wave_combined.append(wave_Ha)
        box_lim_combined.append(box_lim_Ha)

    if flag.Hb:

        u = np.where(lista_obs == "Hb")
        index = u[0][0]
        logF_Hb, dlogF_Hb, logF_grid_Hb, wave_Hb, box_lim_Hb = read_line_spectra(
            models[index], lbdarr[index], "Hb"
        )

        logF_combined.append(logF_Hb)
        dlogF_combined.append(dlogF_Hb)
        logF_grid_combined.append(logF_grid_Hb)
        wave_combined.append(wave_Hb)
        box_lim_combined.append(box_lim_Hb)

    if flag.Hd:

        u = np.where(lista_obs == "Hd")
        index = u[0][0]
        logF_Hd, dlogF_Hd, logF_grid_Hd, wave_Hd, box_lim_Hd = read_line_spectra(
            models[index], lbdarr[index], "Hd"
        )

        logF_combined.append(logF_Hd)
        dlogF_combined.append(dlogF_Hd)
        logF_grid_combined.append(logF_grid_Hd)
        wave_combined.append(wave_Hd)
        box_lim_combined.append(box_lim_Hd)

    if flag.Hg:

        u = np.where(lista_obs == "Hg")
        index = u[0][0]
        logF_Hg, dlogF_Hg, logF_grid_Hg, wave_Hg, box_lim_Hg = read_line_spectra(
            models[index], lbdarr[index], "Hg"
        )

        logF_combined.append(logF_Hg)
        dlogF_combined.append(dlogF_Hg)
        logF_grid_combined.append(logF_grid_Hg)
        wave_combined.append(wave_Hg)
        box_lim_combined.append(box_lim_Hg)

    if flag.pol:

        logF_pol, dlogF_pol, logF_grid_pol, wave_pol, box_lim_pol = read_opd_pol(models)
        print(logF_pol)

        logF_combined.append(logF_pol)
        dlogF_combined.append(dlogF_pol)
        logF_grid_combined.append(logF_grid_pol)
        wave_combined.append(wave_pol)
        box_lim_combined.append(box_lim_pol)

    return (
        logF_combined,
        dlogF_combined,
        logF_grid_combined,
        wave_combined,
        box_lim_combined,
    )
