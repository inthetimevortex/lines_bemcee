import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from konoha.utils import readBAsed
from alive_progress import alive_bar
from joblib import Parallel, delayed


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


def extrapol(minfs, minfo, incl_, cosi, i):
    minfs = [mm, oo, hh, lnn, rr, nn]
    cond = [np.all(a[:-1] == minfs) for a in minfo]
    mods = models[cond]
    # cond2 = [np.all(a == np.append(minfs, incl)) for a in minfo]
    # mod_incl = models[cond2][0]
    # for m in mods:
    # plt.plot(lbdarr[i], m[i] / mod_incl[i])

    coeffs = np.polyfit(incl_, mods[:, i], deg=1)
    poly = np.poly1d(coeffs)
    new_val = poly(cosi)
    # new_flux.append(new_val)
    return new_val


ctrlarr, minfo, models, lbdarr, listpar, dims = read_acol_Ha_xdr()

mass_ = listpar[0]
oblat_ = listpar[1]
hfrac_ = listpar[2]
ln0_ = listpar[3]
rd_ = listpar[4]
n_ = listpar[5]
incl_ = listpar[6]  # these are cosine values!

cosi = np.cos(60.0 * np.pi / 180)

print("  M  # Oblt # Hfrc # logn0 # Rdisk #  n  ")
for mm in mass_:
    for oo in oblat_:
        for hh in hfrac_:
            for lnn in ln0_:
                for rr in rd_:
                    for nn in n_:
                        print(
                            "{0:.2f} # {1:.2f} # {2:.2f} # {3:.2f} # {4:.2f} # {5:.2f}".format(
                                mm, oo, hh, lnn, rr, nn
                            )
                        )
                        # with alive_bar(100, bar="filling") as bar:
                        minfs = [mm, oo, hh, lnn, rr, nn]
                        new_flux = Parallel(n_jobs=4)(
                            delayed(extrapol)(minfs, minfo, incl_, cosi, i)
                            for i in range(len(lbdarr))
                        )
                        # bar()
                        # for i in range(len(lbdarr)):
                        #     # for ii in incl_:
                        #     # incl = listpar[-1][2]
                        #     minfs = [mm, oo, hh, lnn, rr, nn]
                        #     cond = [np.all(a[:-1] == minfs) for a in minfo]
                        #     mods = models[cond]
                        #     # cond2 = [np.all(a == np.append(minfs, incl)) for a in minfo]
                        #     # mod_incl = models[cond2][0]
                        #     # for m in mods:
                        #     # plt.plot(lbdarr[i], m[i] / mod_incl[i])
                        #
                        #     coeffs = np.polyfit(incl_, mods[:, i], deg=1)
                        #     poly = np.poly1d(coeffs)
                        #     new_val = poly(cosi)
                        #     new_flux.append(new_val)
                        # print(new_flux)
                        # bar()
                        # print(m[i])
                        # plt.plot(incl_, mods[:, i], "o")
                        # plt.plot(cosi, new_val, "*")
                        # for mod in mods:
                        #     plt.plot(lbdarr, mod)
                        # plt.plot(lbdarr, new_flux, "k", lw=2)
                        # plt.xscale("log")
                        # plt.yscale("log")
                        # # plt.title(
                        # #     "{0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}, {5:.2f}".format(
                        # #         mm, oo, hh, lnn, rr, nn
                        # #     )
                        # # )
                        # plt.show()
    # plt.xscale("log")
