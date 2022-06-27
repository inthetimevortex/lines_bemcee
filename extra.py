import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from konoha.utils import readBAsed


def read_acol_Ha_xdr():

    # print(params_tmp)
    dims = ["M", "ob", "Hfrac", "sig0", "Rd", "mr", "cosi"]
    dims = dict(zip(dims, range(len(dims))))
    # isig = dims["sig0"]

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

    xdrPL = "../models/acol_Ha.xdr"

    listpar, lbdarr, minfo, models = readBAsed(xdrPL, quiet=False)

    # Filter (removing bad models)
    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.0

    for i in range(np.shape(listpar)[0]):
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
    return ctrlarr, minfo, models, lbdarr, listpar, dims


ctrlarr, minfo, models, lbdarr, listpar, dims = read_acol_Ha_xdr()

# incls = np.arccos(listpar[-1]) * 180/np.pi

for i in range(len(lbdarr)):
    #for incl in listpar[-1]:
        incl = listpar[-1][2]
        minfs = minfo[0, :-1]
        cond = [np.all(a[:-1] == minfs) for a in minfo]
        mods = models[cond]
        cond2 = [np.all(a == np.append(minfs, incl)) for a in minfo]
        mod_incl = models[cond2][0]
        for m in mods:
            plt.plot(lbdarr[i], m[i] / mod_incl[i])
        coeffs = np.polyfit(listpar[-1], , deg=1)
        # plt.xscale("log")
