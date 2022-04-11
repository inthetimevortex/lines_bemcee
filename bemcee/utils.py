from __future__ import print_function
import os as _os
import re as _re
import numpy as np
from scipy.interpolate import griddata
import pyhdust.phc as phc
from scipy.stats import gaussian_kde
import struct as struct
import tarfile as _tarfile
import bemcee.constants as const
import warnings as _warn
from scipy.interpolate import UnivariateSpline


def check_list(lista_obs, x):
    a = np.where(lista_obs == x)
    if len(a[0]) == 0:
        return False
    else:
        return True


# def norm_spectra(wl_c, flx_con):
#    '''normaliza espectro
#    '''
#    spl_fit = UnivariateSpline(wl_c, flx_con, w=spl_weight, k=3)
#    flx_normalized = flx_con - spl_fit(wl_c)
#
#    return flx_normalized


def jy2cgs(flux, lbd, inverse=False):
    """
    Converts from Jy units to erg/s/cm2/micron, and vice-versa

    [lbd] = micron

    Usage:
    flux_cgs = jy2cgs(flux, lbd, inverse=False)
    """
    if not inverse:
        flux_new = 3e-9 * flux / lbd ** 2
    else:
        flux_new = lbd ** 2 * flux / 3e-9

    return flux_new


def geneva_closest(Mstar, oblat, t, Zstr="014", tar=None, silent=True):
    """
    Interpolate models between rotation rates, at closest Mstar.

    Usage:
    Rpole, logL = geneva_closest(Mstar, oblat, t, Zstr='014', tar=None,
        silent=True)

    where t is given in tMS, and tar is the open tar file. The chosen
    metallicity is according to the input tar file. If tar=None, the
    code will take Zstr='014' by default.
    """
    # oblat to Omega/Omega_c
    w = oblat2w(oblat)

    # grid
    if Mstar <= 20.0:
        Mlist = np.array([1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0, 20.0])
        Mstr = np.array(
            [
                "1p700",
                "2p000",
                "2p500",
                "3p000",
                "4p000",
                "5p000",
                "7p000",
                "9p000",
                "12p00",
                "15p00",
                "20p00",
            ]
        )
        Vlist = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        Vstr = np.array(
            [
                "00000",
                "10000",
                "30000",
                "50000",
                "60000",
                "70000",
                "80000",
                "90000",
                "95000",
            ]
        )
    else:
        Mlist = np.array([20.0, 25.0, 32.0, 40.0, 60.0, 85.0, 120.0])
        Mstr = np.array(["20p00", "25p00", "32p00", "40p00", "60p00", "85p00", "120p0"])
        Vlist = np.array([0.0, 0.568])
        Vstr = np.array(["00000", "56800"])

    # read tar file
    if tar is None:
        # dir0 = '{0}/refs/geneva_models/'.format(_hdtpath())
        dir0 = "../defs/geneve_models/"
        fmod = "Z{:}.tar.gz".format(Zstr)
        tar = _tarfile.open(dir0 + fmod, "r:gz")
    else:
        Zstr = tar.getnames()[0][7:10]

    # find closest Mstar
    iM = np.where(np.abs(Mstar - Mlist) == np.min(np.abs(Mstar - Mlist)))[0][0]

    # find values at selected age
    nw = len(Vlist)
    wlist = np.zeros(nw)
    Rplist = np.zeros(nw)
    logLlist = np.zeros(nw)
    agelist = np.zeros(nw)
    for iw, vs in enumerate(Vstr):
        fname = "M{:}Z{:}00V{:}.dat".format(Mstr[iM], Zstr, vs)
        age1, _, logL1, _, Hfrac1, _, _, w1, Rpole1 = geneva_read(
            fname, Zstr="014", tar=None
        )
        # t1 = age1 / age1[np.where(Hfrac1 == 0.)[0][0]-1]
        t1 = age1 / age1[np.where(Hfrac1 < 1.0e-3)[0][0] - 1]
        if t > t1.max() and not silent:
            print(
                "[geneva_closest] Warning: requested age not available, "
                "taking t/tMS={:.2f} instead of t/tMS={:.2f}.".format(t1.max(), t)
            )
        it = np.where(np.abs(t - t1) == np.min(np.abs(t - t1)))[0][0]
        wlist[iw] = w1[it]
        Rplist[iw] = Rpole1[it]
        logLlist[iw] = logL1[it]
        agelist[iw] = age1[it] / 1e6
    # interpolate between rotation rates
    if w <= wlist.max():
        Rpole = griddata(wlist, Rplist, [w], method="linear")[0]
        logL = griddata(wlist, logLlist, [w], method="linear")[0]
        age = griddata(wlist, agelist, [w], method="linear")[0]
    else:
        if not silent:
            print(
                "[geneva_closest] Warning: no model rotating this fast at "
                "this age, taking closest model instead. (omega={:.2f} "
                "instead of omega={:.2f})".format(wlist.max(), w)
            )
        iwmax = np.where(wlist == wlist.max())[0][0]
        Rpole = Rplist[iwmax]
        logL = logLlist[iwmax]
        age = agelist[iwmax]

    return Rpole, logL, age


def geneva_read(fname, Zstr="014", tar=None):
    """
    Reads Geneva model file

    Usage:
    age, Mstar, logL, logTeff, Hfrac, Hefrac, oblat, w, Rpole =
        geneva_read(fname, tar=None)

    where tar is the read tar(.gz) opened file.
    """
    # read tar file
    if tar is None:
        # dir0 = '{0}/refs/geneva_models/'.format(_hdtpath())
        dir0 = "../defs/geneve_models/"
        fmod = "Z{:}.tar.gz".format(Zstr)
        tar = _tarfile.open(dir0 + fmod, "r:gz")
    else:
        Zstr = tar.getnames()[0][7:10]

    m = tar.getmember(fname)
    fname = tar.extractfile(m)

    # (age, M, logL, logTeff, Hfrac, Hefrac, oblat, w, Rpole)

    cols = (1, 2, 3, 4, 21, 22, 34, 39, 44)
    t = np.loadtxt(fname, usecols=cols, skiprows=2)
    age = t[:, 0]
    Mstar = t[:, 1]
    logL = t[:, 2]
    logTeff = t[:, 3]
    Hfrac = t[:, 4]
    Hefrac = t[:, 5]
    oblat = 1.0 / t[:, 6]
    w = t[:, 7]
    Rpole = t[:, 8]

    return age, Mstar, logL, logTeff, Hfrac, Hefrac, oblat, w, Rpole
    # return logL, logTeff


def geneva_interp(Mstar, oblat, t, Zstr="014", tar=None, silent=True):
    """
    Interpolates Geneva stellar models.

    Usage:
    Rpole, logL, age = geneva_interp(Mstar, oblat, t, tar=None, silent=True)

    where t is given in tMS, and tar is the open tar file. The chosen
    metallicity is according to the input tar file. If tar=None, the
    code will take Zstr='014' by default.
    """
    # oblat to Omega/Omega_c
    # w = oblat2w(oblat)

    # grid
    if Mstar <= 20.0:
        Mlist = np.array([1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0, 20.0])
    else:
        Mlist = np.array([20.0, 25.0, 32.0, 40.0, 60.0, 85.0, 120.0])

    # read tar file
    if tar is None:
        # dir0 = '{0}/refs/geneva_models/'.format(_hdtpath())
        dir0 = "../defs/geneve_models/"
        fmod = "Z{:}.tar.gz".format(Zstr)
        tar = _tarfile.open(dir0 + fmod, "r:gz")
    else:
        Zstr = tar.getnames()[0][7:10]

    # interpolation

    # creation of lists for polar radius and log, for extrapolation fit
    # and the ttms list used in the linear fit originally

    ttms = [0, 0.40, 0.65, 0.85, 1.00]
    L_log = []
    Rp = []

    # for ages inside the original grid, nothing happens
    if (t <= 1.0) * (t >= 0.0):
        Rpole, logL, age = geneva_interp_fast(Mstar, oblat, t, Zstr=Zstr, silent=silent)

        return Rpole, logL, age

    if t > 1.0:
        for time in ttms:
            Rpole, logL, age = geneva_interp_fast(
                Mstar, oblat, time, Zstr=Zstr, silent=silent
            )

            Rp.append(Rpole)
            L_log.append(logL)

        coeffs = np.polyfit(np.log10(ttms[-4:]), np.log10(Rp[-4:]), deg=1)
        poly = np.poly1d(coeffs)

        coeffs2 = np.polyfit(ttms[-4:], L_log[-4:], deg=1)
        poly2 = np.poly1d(coeffs2)

        Rpole = 10 ** (poly(np.log10(t)))
        logL = poly2(t)

        # in this case, no age. no physical meaning!
        return Rpole, logL


def beta(par, is_ob=False):
    r""" Calculate the :math:`\beta` value from Espinosa-Lara for a given
    rotation rate :math:`w_{\rm frac} = \Omega/\Omega_c`

    If ``is_ob == True``, it consider the param as ob (instead of
    :math:`w_{\rm frac}`). """

    # Ekstrom et al. 2008, Eq. 9
    if is_ob:
        wfrac = (1.5 ** 1.5) * np.sqrt(2.0 * (par - 1.0) / par ** 3)
    else:
        wfrac = par

    # avoid exceptions
    if wfrac == 0:
        return 0.25
    elif wfrac == 1:
        return 0.13535
    elif wfrac < 0 or wfrac > 1:
        _warn.warn("Invalid value of wfrac.")
        return 0.0

    # Espinosa-Lara VLTI-School 2013 lecture, slide 18...
    delt = 1.0
    omega1 = 0.0
    omega = wfrac
    while delt >= 1e-5:
        f = (3.0 / (2.0 + omega ** 2)) ** 3 * omega ** 2 - wfrac ** 2
        df = -108.0 * omega * (omega ** 2 - 1.0) / (omega ** 2 + 2.0) ** 4
        omega1 = omega - f / df
        delt = np.abs(omega1 - omega) / omega
        omega = omega1

    nthe = 100
    theta = np.linspace(0, np.pi / 2, nthe + 1)[1:]
    grav = np.zeros(nthe)
    teff = np.zeros(nthe)
    corr = np.zeros(nthe)
    beta = 0.0

    for ithe in range(nthe):

        delt = 1.0
        r1 = 0.0
        r = 1.0
        while delt >= 1e-5:
            f = (
                omega ** 2 * r ** 3 * np.sin(theta[ithe]) ** 2
                - (2.0 + omega ** 2) * r
                + 2.0
            )
            df = 3.0 * omega ** 2 * r ** 2 * np.sin(theta[ithe]) ** 2 - (
                2.0 + omega ** 2
            )
            r1 = r - f / df
            delt = np.abs(r1 - r) / r
            r = r1

        delt = 1.0
        n1 = 0.0
        ftheta = (
            1.0 / 3.0 * omega ** 2 * r ** 3 * np.cos(theta[ithe]) ** 3
            + np.cos(theta[ithe])
            + np.log(np.tan(theta[ithe] / 2.0))
        )
        n = theta[ithe]
        while delt >= 1e-5:
            f = np.cos(n) + np.log(np.tan(n / 2.0)) - ftheta
            df = -np.sin(n) + 1.0 / np.sin(n)
            n1 = n - f / df
            delt = abs(n1 - n) / n
            n = n1

        grav[ithe] = np.sqrt(
            1.0 / r ** 4
            + omega ** 4 * r ** 2 * np.sin(theta[ithe]) ** 2
            - 2.0 * omega ** 2 * np.sin(theta[ithe]) ** 2 / r
        )

        corr[ithe] = np.sqrt(np.tan(n) / np.tan(theta[ithe]))
        teff[ithe] = corr[ithe] * grav[ithe] ** 0.25

    u = ~np.isnan(teff)
    coef = np.polyfit(np.log(grav[u]), np.log(teff[u]), 1)
    beta = coef[0]

    return beta


def linfit(x, y, ssize=0.05, yerr=np.empty(0)):
    """
    linfit() - retorna um array (y) normalizado, em posicoes de x

    x eh importante, pois y pode ser nao igualmente amostrado.
    x e y devem estar em ordem crescente.

    ssize = % do tamanho de y; numero de pontos usados nas extremidades
    para a media do contínuo. 'ssize' de .5 à 0 (exclusive).

    OUTPUT: y, yerr (if given)

    .. code:: python

        #Example:
        import numpy as np
        import matplotlib.pyplot as plt
        import pyhdust.phc as phc
        import pyhdust.spectools as spt

        wv = np.linspace(6500, 6600, 101)
        flx = (np.arange(101)[::-1])/100.+1+phc.normgauss(4, x=wv,
        xc=6562.79)*5

        plt.plot(wv, flx)
        normflx = spt.linfit(wv, flx)
        plt.plot(wv, normflx, ls='--')

        plt.xlabel(r'$\lambda$ ($\AA$)')
        plt.ylabel('Flux (arb. unit)')

    .. image:: _static/spt_linfit.png
        :align: center
        :width: 500
    """
    ny = np.array(y)[:]
    if ssize < 0 or ssize > 0.5:
        _warn.warn("Invalid ssize value...", stacklevel=2)
        ssize = 0
    ssize = int(ssize * len(y))
    if ssize == 0:
        ssize = 1
    medx0, medx1 = np.average(x[:ssize]), np.average(x[-ssize:])
    if ssize > 9:
        medy0, medy1 = np.median(ny[:ssize]), np.median(ny[-ssize:])
    else:
        medy0, medy1 = np.average(ny[:ssize]), np.average(ny[-ssize:])
    new_y = medy0 + (medy1 - medy0) * (x - medx0) / (medx1 - medx0)
    idx = np.where(new_y != 0)
    ny[idx] = ny[idx] / new_y[idx]
    if len(yerr) == 0.0:
        return ny
    else:
        yerr = yerr / np.average(new_y)
        return ny, yerr


def lineProf(x, flx, lbc, flxerr=np.empty(0), hwidth=1000.0, ssize=0.05):
    """
    lineProf() - retorna um array (flx) normalizado e um array x em
    VELOCIDADES. `lbc` deve fornecido em mesma unidade de x para conversão
    lambda -> vel. Se vetor x jah esta em vel., usar funcao linfit().

    x eh importante, pois y pode ser nao igualmente amostrado.
    x e y devem estar em ordem crescente.

    ssize = % do tamanho de y; numero de pontos usados nas extremidades
    para a media do contínuo. 'ssize' de .5 à 0 (exclusive).

    OUTPUT: vel (array), flx (array)
    """
    x = (x - lbc) / lbc * const.c * 1e-5  # km/s
    idx = np.where(np.abs(x) <= 1.001 * hwidth)
    if len(flxerr) == 0:
        flux = linfit(x[idx], flx[idx], ssize=ssize)  # yerr=flxerr,
        if len(x[idx]) == 0:
            warn.warn("Wrong `lbc` in the lineProf function")
        return x[idx], flux
    else:
        flux, flxerr = linfit(x[idx], flx[idx], yerr=flxerr[idx], ssize=ssize)
        if len(x[idx]) == 0:
            warn.warn("Wrong `lbc` in the lineProf function")
        return x[idx], flux, flxerr


def readpck(n, tp, ixdr, f):
    """ Read XDR

    - n: length
    - tp: type ('i', 'l', 'f', 'd')
    - ixdr: counter
    - f: file-object

    :returns: ixdr (counter), np.array
    """
    sz = dict(zip(["i", "l", "f", "d"], [4, 4, 4, 8]))
    s = sz[tp]
    upck = ">{0}{1}".format(n, tp)
    return ixdr + n * s, np.array(struct.unpack(upck, f[ixdr : ixdr + n * s]))


def readXDRsed(xdrpath, quiet=False):
    """  Read a XDR with a set of models.

    The models' parameters (as well as their units) are defined at XDR
    creation.

    INPUT: xdrpath

    OUTPUT: ninfo, intervals, lbdarr, minfo, models

    (xdr dimensions, params limits, lambda array (um), mods params, mods flux)
    """
    ixdr = 0
    f = open(xdrpath, "rb").read()
    ixdr, ninfo = readpck(3, "l", ixdr, f)
    nq, nlbd, nm = ninfo
    ixdr, intervals = readpck(nq * 2, "f", ixdr, f)
    ixdr, lbdarr = readpck(nlbd, "f", ixdr, f)
    ixdr, listpar = readpck(nq * nm, "f", ixdr, f)
    ixdr, models = readpck(nlbd * nm, "f", ixdr, f)
    #
    if ixdr == len(f):
        if not quiet:
            print("# XDR {0} completely read!".format(xdrpath))
    else:
        _warn.warn(
            "# XDR {0} not completely read!\n# length "
            "difference is {1} /4".format(xdrpath),
            (len(f) - ixdr),
        )
    #
    return (
        ninfo,
        intervals.reshape((nq, 2)),
        lbdarr,
        listpar.reshape((nm, nq)),
        models.reshape((nm, nlbd)),
    )


def oblat2w(oblat):
    w = (1.5 ** 1.5) * np.sqrt(2.0 * (oblat - 1.0) / oblat ** 3.0)
    return w


def readBAsed(xdrpath, quiet=False):
    """ Read **only** the BeAtlas SED release.

    | Definitions:
    | -photospheric models: sig0 (and other quantities) == 0.00
    | -Parametric disk model default (`param` == True)
    | -VDD-ST models: n excluded (alpha and R0 fixed. Confirm?)
    | -The models flux are given in ergs/s/cm2/um. If ignorelum==True in the
    |   XDR creation, F_lbda/F_bol unit will be given.

    INPUT: xdrpath

    | OUTPUT: listpar, lbdarr, minfo, models
    | (list of mods parameters, lambda array (um), mods index, mods flux)
    """
    f = open(xdrpath, "rb").read()
    ixdr = 0
    #
    npxs = 3
    upck = ">{0}l".format(npxs)
    header = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
    ixdr += npxs * 4
    nq, nlb, nm = header
    #
    npxs = nq
    upck = ">{0}l".format(npxs)
    header = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
    ixdr += npxs * 4
    #
    listpar = [[] for i in range(nq)]
    for i in range(nq):
        npxs = header[i]
        upck = ">{0}f".format(npxs)
        listpar[i] = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
        ixdr += npxs * 4
    #
    npxs = nlb
    upck = ">{0}f".format(npxs)
    lbdarr = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
    ixdr += npxs * 4
    #
    npxs = nm * (nq + nlb)
    upck = ">{0}f".format(npxs)
    models = np.array(struct.unpack(upck, f[ixdr : ixdr + npxs * 4]))
    ixdr += npxs * 4
    models = models.reshape((nm, -1))
    # this will check if the XDR is finished.
    if ixdr == len(f):
        if not quiet:
            print("# XDR {0} completely read!".format(xdrpath))
    else:
        _warn.warn(
            "# XDR {0} not completely read!\n# length "
            "difference is {1}".format(xdrpath, (len(f) - ixdr) / 4)
        )
    #
    return listpar, lbdarr, models[:, 0:nq], models[:, nq:]


# ==============================================================================
def kde_scipy(x, x_grid, bandwidth=0.2):
    """Kernel Density Estimation with Scipy"""

    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
    return kde.evaluate(x_grid)


def quantile(x, q, weights=None):
    """
    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    if weights is None:
        return np.percentile(x, [100.0 * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()


# ==============================================================================
# BIN DATA
def bin_data(x, y, nbins, xran=None, exclude_empty=True):
    """
    Bins data

    Usage:
    xbin, ybin, dybin = bin_data(x, y, nbins, xran=None, exclude_empty=True)

    where dybin is the standard deviation inside the bins.
    """
    # make sure it is a numpy array
    x = np.array([x]).reshape((-1))
    y = np.array([y]).reshape((-1))
    # make sure it is in increasing order
    ordem = x.argsort()
    x = x[ordem]
    y = y[ordem]

    if xran is None:
        xmin, xmax = x.min(), x.max()
    else:
        xmin, xmax = xran[0], xran[1]

    xborders = np.linspace(xmin, xmax, nbins + 1)
    xbin = 0.5 * (xborders[:-1] + xborders[1:])

    ybin = np.zeros(nbins)
    dybin = np.zeros(nbins)
    for i in range(nbins):
        aux = (x > xborders[i]) * (x < xborders[i + 1])
        if np.array([aux]).any():
            ybin[i] = np.mean(y[aux])
            dybin[i] = np.std(y[aux])
        else:
            ybin[i] = np.nan
            dybin[i] = np.nan

    if exclude_empty:
        keep = np.logical_not(np.isnan(ybin))
        xbin, ybin, dybin = xbin[keep], ybin[keep], dybin[keep]

    return xbin, ybin, dybin


# ==============================================================================
def find_nearest(array, value):
    """
    Find the nearest value inside an array
    """

    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# ==============================================================================
def find_neighbours(par, par_grid, ranges):
    """
    Finds neighbours' positions of par in par_grid.

    Usage:
    keep, out, inside_ranges, par_new, par_grid_new = \
        find_neighbours(par, par_grid, ranges):

    where redundant columns in 'new' values are excluded,
    but length is preserved (i.e., par_grid[keep] in griddata call).
    """
    # check if inside ranges

    if len(par) == 4:
        ranges = ranges[0:4]
    if len(par) == 3:
        ranges = ranges[0:3]
    # print(par, len(ranges))
    # print(par, ranges)
    count = 0
    inside_ranges = True
    while (inside_ranges is True) * (count < len(par)):
        inside_ranges = (par[count] >= ranges[count, 0]) * (
            par[count] <= ranges[count, 1]
        )
        count += 1

    # find neighbours
    keep = np.array(len(par_grid) * [True])
    out = []

    if inside_ranges:
        for i in range(len(par)):
            # coincidence
            if (par[i] == par_grid[:, i]).any():
                keep *= par[i] == par_grid[:, i]
                out.append(i)
            # is inside
            else:
                # list of values
                par_list = np.array(list(set(par_grid[:, i])))
                # nearest value at left
                par_left = par_list[par_list < par[i]]
                par_left = par_left[np.abs(par_left - par[i]).argmin()]
                # nearest value at right
                par_right = par_list[par_list > par[i]]
                par_right = par_right[np.abs(par_right - par[i]).argmin()]
                # select rows
                kl = par_grid[:, i] == par_left
                kr = par_grid[:, i] == par_right
                keep *= kl + kr
        # delete coincidences
        par_new = np.delete(par, out)
        par_grid_new = np.delete(par_grid, out, axis=1)
    else:
        print("Warning: parameter outside ranges.")
        par_new = par
        par_grid_new = par_grid

    return keep, out, inside_ranges, par_new, par_grid_new


def geneva_interp_fast(Mstar, oblat, t, Zstr="014", silent=True):
    """
    Interpolates Geneva stellar models, from grid of
    pre-computed interpolations.

    Usage:
    Rpole, logL, age = geneva_interp_fast(Mstar, oblat, t, Zstr='014')

    where t is given in tMS, and tar is the open tar file. For now, only
    Zstr='014' is available.
    """
    # read grid
    # dir0 = '{0}/refs/geneva_models/'.format(_hdtpath())
    dir0 = "../defs/geneve_models/"
    if Mstar <= 20.0:
        fname = "geneva_interp_Z{:}.npz".format(Zstr)
    else:
        fname = "geneva_interp_Z{:}_highM.npz".format(Zstr)
    data = np.load(dir0 + fname)
    Mstar_arr = data["Mstar_arr"]
    oblat_arr = data["oblat_arr"]
    t_arr = data["t_arr"]
    Rpole_grid = data["Rpole_grid"]
    logL_grid = data["logL_grid"]
    age_grid = data["age_grid"]

    # build grid of parameters
    par_grid = []
    for M in Mstar_arr:
        for ob in oblat_arr:
            for tt in t_arr:
                par_grid.append([M, ob, tt])
    par_grid = np.array(par_grid)

    # set input/output parameters
    par = np.array([Mstar, oblat, t])

    # set ranges
    ranges = np.array(
        [[par_grid[:, i].min(), par_grid[:, i].max()] for i in range(len(par))]
    )

    # find neighbours
    keep, out, inside_ranges, par, par_grid = find_neighbours(par, par_grid, ranges)

    # interpolation method
    if inside_ranges:
        interp_method = "linear"
    else:
        if not silent:
            print(
                "[geneva_interp_fast] Warning: parameters out of available range, taking closest model"
            )
        interp_method = "nearest"

    if len(keep[keep]) == 1:
        # coincidence
        Rpole = Rpole_grid.flatten()[keep][0]
        logL = logL_grid.flatten()[keep][0]
        age = age_grid.flatten()[keep][0]
    else:
        # interpolation
        Rpole = griddata(
            par_grid[keep],
            Rpole_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]
        logL = griddata(
            par_grid[keep],
            logL_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]
        age = griddata(
            par_grid[keep],
            age_grid.flatten()[keep],
            par,
            method=interp_method,
            rescale=True,
        )[0]

    return Rpole, logL, age


# ==============================================================================
# def griddataBA(minfo, models, params, listpar, dims):
#     '''
#     Moser's routine to interpolate BeAtlas models
#     obs: last argument ('listpar') had to be included here
#     '''

#     print(params)
#     idx = np.arange(len(minfo))
#     lim_vals = len(params) * [[], ]
#     for i in range(len(params)):
#         # print(i, listpar[i], params[i], minfo[:, i])
#         lim_vals[i] = [
#             phc.find_nearest(listpar[i], params[i], bigger=False),
#             phc.find_nearest(listpar[i], params[i], bigger=True)]
#         tmp = np.where((minfo[:, i] == lim_vals[i][0]) |
#                        (minfo[:, i] == lim_vals[i][1]))
#         idx = np.intersect1d(idx, tmp[0])

#     out_interp = griddata(minfo[idx], models[idx], params)[0]

#     if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0):

#         mdist = np.zeros(np.shape(minfo))
#         ichk = range(len(params))
#         for i in ichk:
#             mdist[:, i] = np.abs(minfo[:, i] - params[i]) /\
#                 (np.max(listpar[i]) - np.min(listpar[i]))
#         idx = np.where(np.sum(mdist, axis=1) == np.min(np.sum(mdist, axis=1)))
#         if len(idx[0]) != 1:
#             out_interp = griddata(minfo[idx], models[idx], params)[0]
#         else:
#             out_interp = models[idx][0]

#     # if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0) or\
#     #    bool(np.isnan(np.sum(out_interp))) is True:
#     #     print("# Houve um problema na grade e eu nao consegui arrumar...")

#     return out_interp


def griddataBA(minfo, models, params, listpar, dims):
    """
    Moser's routine to interpolate BeAtlas models
    obs: last argument ('listpar') had to be included here
    """

    # print(params[0])
    idx = np.arange(len(minfo))
    lim_vals = len(params) * [
        [],
    ]
    for i in range(len(params)):
        # print(i, listpar[i], params[i], minfo[:, i])
        lim_vals[i] = [
            phc.find_nearest(listpar[i], params[i], bigger=False),
            phc.find_nearest(listpar[i], params[i], bigger=True),
        ]

        tmp = np.where(
            (minfo[:, i] == lim_vals[i][0]) | (minfo[:, i] == lim_vals[i][1])
        )

        idx = np.intersect1d(idx, tmp[0])

    # print(idx)
    out_interp = griddata(minfo[idx], models[idx], params)[0]

    if np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0:

        mdist = np.zeros(np.shape(minfo))
        ichk = range(len(params))
        for i in ichk:
            mdist[:, i] = np.abs(minfo[:, i] - params[i]) / (
                np.max(listpar[i]) - np.min(listpar[i])
            )
        idx = np.where(np.sum(mdist, axis=1) == np.min(np.sum(mdist, axis=1)))
        if len(idx[0]) != 1:
            out_interp = griddata(minfo[idx], models[idx], params)[0]
        else:
            out_interp = models[idx][0]

    # if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0) or\
    #    bool(np.isnan(np.sum(out_interp))) is True:
    #     print("# Houve um problema na grade e eu nao consegui arrumar...")

    return out_interp


# ==============================================================================
# def hfrac2tms(Hfrac, inverse=False):
#    '''
#    Converts nuclear hydrogen fraction into fractional time in the main-sequence,
#    (and vice-versa) based on the polynomial fit of the average of this relation
#    for all B spectral types and rotational velocities.
#
#    Usage:
#    t = hfrac2tms(Hfrac, inverse=False)
#    or
#    Hfrac = hfrac2tms(t, inverse=True)
#    '''
#    if not inverse:
#        coef = np.array([-0.57245754, -0.8041484 , -0.51897195,  1.00130795])
#        tms = coef.dot(np.array([Hfrac**3, Hfrac**2, Hfrac**1, Hfrac**0]))
#    else:
#        # interchanged parameter names
#        coef = np.array([-0.74740597,  0.98208541, -0.64318363, -0.29771094,  0.71507214])
#        tms = coef.dot(np.array([Hfrac**4, Hfrac**3, Hfrac**2, Hfrac**1, Hfrac**0]))
#
#    return tms


# ==============================================================================
def griddataBAtlas(minfo, models, params, listpar, dims, isig):
    idx = range(len(minfo))
    lim_vals = len(params) * [
        [],
    ]
    for i in [i for i in range(len(params)) if i != isig]:
        lim_vals[i] = [
            phc.find_nearest(listpar[i], params[i], bigger=False),
            phc.find_nearest(listpar[i], params[i], bigger=True),
        ]
        tmp = np.where(
            (minfo[:, i] == lim_vals[i][0]) | (minfo[:, i] == lim_vals[i][1])
        )
        idx = np.intersect1d(idx, tmp)
        #
    out_interp = griddata(minfo[idx], models[idx], params)[0]
    #
    if np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0:
        print("# Houve um problema na grade. Tentando arrumadar...")
        idx = np.arange(len(minfo))
        for i in [i for i in range(len(params)) if i != dims["sig0"]]:
            imin = lim_vals[i][0]
            if lim_vals[i][0] != np.min(listpar[i]):
                imin = phc.find_nearest(listpar[i], lim_vals[i][0], bigger=False)
            imax = lim_vals[i][1]
            if lim_vals[i][1] != np.max(listpar[i]):
                imax = phc.find_nearest(listpar[i], lim_vals[i][1], bigger=True)
            lim_vals[i] = [imin, imax]
            tmp = np.where(
                (minfo[:, i] >= lim_vals[i][0]) & (minfo[:, i] <= lim_vals[i][1])
            )
            idx = np.intersect1d(idx, phc.flatten(tmp))
        out_interp = griddata(minfo[idx], models[idx], params)[0]
    #
    if np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0:
        print("# Houve um problema na grade e eu nao conseguir arrumar...")
    #
    return out_interp


# ==============================================================================
def griddataBA_new(minfo, models, params, isig, silent=True):
    """
    Interpolates model grid

    Usage:
    model_interp = griddata(minfo, models, params, isig, silent=True)

    where
    minfo = grid of parameters
    models = grid of models
    params = parameters,
    isig = (normalized) sigma0 index

    Ex:
    # read grid
    xdrpath = 'beatlas/disk_flx.xdr'
    listpar, lbdarr, minfo, models = bat.readBAsed(xdrpath, quiet=True)
    # find isig
    dims = ['M', 'ob', 'sig0', 'nr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]
    # interpolation
    params = [12.4, 1.44, 0.9, 4.4, 0.1]
    model_interp = np.exp(griddataBA(minfo, np.log(models), params, isig))

    If photospheric models are interpolated, let isig=None. For spectra,
    it is recommended to enter the log of the grid of spectra as input,
    as shown in the example above.
    """
    # ranges
    ranges = np.array([[parr.min(), parr.max()] for parr in minfo.T])

    # find neighbours, delete coincidences
    if phc.is_inside_ranges(isig, [0, len(params) - 1]):
        # exclude sig0 dimension, to take all their entries for interpolation
        keep, out, inside_ranges, params1, minfo1 = phc.find_neighbours(
            np.delete(params, isig),
            np.delete(minfo, isig, axis=1),
            np.delete(ranges.T, isig, axis=1).T,
            silent=silent,
        )
        params = np.hstack([params1, params[isig]])
        minfo = np.vstack([minfo1.T, minfo[:, isig]]).T
    else:
        keep, out, inside_ranges, params, minfo = phc.find_neighbours(
            params, minfo, ranges, silent=silent
        )

    # interpolation
    model_interp = griddata(minfo[keep], models[keep], params, method="linear")[0]

    if np.isnan(model_interp).any() or np.sum(model_interp) == 0.0:
        if not silent:
            print(
                "[griddataBA] Warning: linear interpolation didnt work, taking closest model"
            )
        model_interp = griddata(minfo[keep], models[keep], params, method="nearest")[0]

    return model_interp
