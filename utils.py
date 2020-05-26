import numpy as np
from scipy.interpolate import griddata
import pyhdust.phc as phc
from scipy.stats import gaussian_kde
import user_settings as flag

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
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()

# ==============================================================================
# BIN DATA
def bin_data(x, y, nbins, xran=None, exclude_empty=True):
    '''
    Bins data

    Usage:
    xbin, ybin, dybin = bin_data(x, y, nbins, xran=None, exclude_empty=True)

    where dybin is the standard deviation inside the bins.
    '''
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
    '''
    Find the nearest value inside an array
    '''

    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# ==============================================================================
def find_neighbours(par, par_grid, ranges):
    '''
    Finds neighbours' positions of par in par_grid.

    Usage:
    keep, out, inside_ranges, par_new, par_grid_new = \
        find_neighbours(par, par_grid, ranges):

    where redundant columns in 'new' values are excluded,
    but length is preserved (i.e., par_grid[keep] in griddata call).
    '''
    # check if inside ranges

    if len(par) == 4:
        ranges = ranges[0: 4]
    if len(par) == 3:
        ranges = ranges[0: 3]
    # print(par, len(ranges))
    # print(par, ranges)
    count = 0
    inside_ranges = True
    while (inside_ranges is True) * (count < len(par)):
        inside_ranges = (par[count] >= ranges[count, 0]) *\
            (par[count] <= ranges[count, 1])
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
                keep *= (kl + kr)
        # delete coincidences
        par_new = np.delete(par, out)
        par_grid_new = np.delete(par_grid, out, axis=1)
    else:
        print('Warning: parameter outside ranges.')
        par_new = par
        par_grid_new = par_grid

    return keep, out, inside_ranges, par_new, par_grid_new


# ==============================================================================
def geneva_interp_fast(Par, oblat, t, neighbours_only=True, isRpole=False):
    '''
    Interpolates Geneva stellar models, from grid of
    pre-computed interpolations.

    Usage:
    Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
                                     neighbours_only=True, isRpole=False)
    or
    Mstar, logL = geneva_interp_fast(Rpole, oblat, t,
                                     neighbours_only=True, isRpole=True)
    (in this case, the option 'neighbours_only' will be set to 'False')

    where t is given in tMS, and tar is the open tar file. For now, only
    Z=0.014 is available.
    '''
    # from my_routines import find_neighbours
    from scipy.interpolate import griddata

    # read grid
    dir0 = flag.folder_defs +'/geneve_models/'
    fname = 'geneva_interp_Z014.npz'
    data = np.load(dir0 + fname)
    Mstar_arr = data['Mstar_arr']
    oblat_arr = data['oblat_arr']
    t_arr = data['t_arr']
    Rpole_grid = data['Rpole_grid']
    logL_grid = data['logL_grid']

    # build grid of parameters
    par_grid = []
    for M in Mstar_arr:
        for ob in oblat_arr:
            for tt in t_arr:
                par_grid.append([M, ob, tt])
    par_grid = np.array(par_grid)

    # set input/output parameters
    if isRpole:
        Rpole = Par
        par = np.array([Rpole, oblat, t])
        Mstar_arr = par_grid[:, 0].copy()
        par_grid[:, 0] = Rpole_grid.flatten()
        neighbours_only = False
    else:
        Mstar = Par
        par = np.array([Mstar, oblat, t])
    # print(par)

    # set ranges
    ranges = np.array([[par_grid[:, i].min(),
                        par_grid[:, i].max()] for i in range(len(par))])

    # find neighbours
    if neighbours_only:
        keep, out, inside_ranges, par, par_grid = \
            find_neighbours(par, par_grid, ranges)
    else:
        keep = np.array(len(par_grid) * [True])
        # out = []
        # check if inside ranges
        count = 0
        inside_ranges = True
        while (inside_ranges is True) * (count < len(par)):
            inside_ranges = (par[count] >= ranges[count, 0]) *\
                (par[count] <= ranges[count, 1])
            count += 1

    # interpolation method
    if inside_ranges:
        interp_method = 'linear'
    else:
        print('Warning: parameters out of available range,' +
              ' taking closest model.')
        interp_method = 'nearest'

    if len(keep[keep]) == 1:
        # coincidence
        if isRpole:
            Mstar = Mstar_arr[keep][0]
            Par_out = Mstar
        else:
            Rpole = Rpole_grid.flatten()[keep][0]
            Par_out = Rpole
        logL = logL_grid.flatten()[keep][0]
    else:
        # interpolation
        if isRpole:
            Mstar = griddata(par_grid[keep], Mstar_arr[keep], par,
                             method=interp_method, rescale=True)[0]
            Par_out = Mstar
        else:
            Rpole = griddata(par_grid[keep], Rpole_grid.flatten()[keep],
                             par, method=interp_method, rescale=True)[0]
            Par_out = Rpole
        logL = griddata(par_grid[keep], logL_grid.flatten()[keep],
                        par, method=interp_method, rescale=True)[0]

    return Par_out, logL


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
    '''
    Moser's routine to interpolate BeAtlas models
    obs: last argument ('listpar') had to be included here
    '''

    # print(params[0])
    idx = np.arange(len(minfo))
    lim_vals = len(params) * [[], ]
    for i in range(len(params)):
        # print(i, listpar[i], params[i], minfo[:, i])
        lim_vals[i] = [
            phc.find_nearest(listpar[i], params[i], bigger=False),
            phc.find_nearest(listpar[i], params[i], bigger=True)]
        tmp = np.where((minfo[:, i] == lim_vals[i][0]) |
                       (minfo[:, i] == lim_vals[i][1]))
        idx = np.intersect1d(idx, tmp[0])

    out_interp = griddata(minfo[idx], models[idx], params)[0]

    if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0):

        mdist = np.zeros(np.shape(minfo))
        ichk = range(len(params))
        for i in ichk:
            mdist[:, i] = np.abs(minfo[:, i] - params[i]) /\
                (np.max(listpar[i]) - np.min(listpar[i]))
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
def hfrac2tms(Hfrac, inverse=False):
    '''
    Converts nuclear hydrogen fraction into fractional time in the main-sequence,
    (and vice-versa) based on the polynomial fit of the average of this relation
    for all B spectral types and rotational velocities.

    Usage:
    t = hfrac2tms(Hfrac, inverse=False)
    or
    Hfrac = hfrac2tms(t, inverse=True)
    '''
    if not inverse:
        coef = np.array([-0.57245754, -0.8041484 , -0.51897195,  1.00130795])
        tms = coef.dot(np.array([Hfrac**3, Hfrac**2, Hfrac**1, Hfrac**0]))
    else:
        # interchanged parameter names
        coef = np.array([-0.74740597,  0.98208541, -0.64318363, -0.29771094,  0.71507214])
        tms = coef.dot(np.array([Hfrac**4, Hfrac**3, Hfrac**2, Hfrac**1, Hfrac**0]))

    return tms


# ==============================================================================
def griddataBAtlas(minfo, models, params, listpar, dims, isig):
    idx = range(len(minfo))
    lim_vals = len(params)*[ [], ]
    for i in [i for i in range(len(params)) if i != isig]:
        lim_vals[i] = [
            phc.find_nearest(listpar[i], params[i], bigger=False), 
            phc.find_nearest(listpar[i], params[i], bigger=True)]
        tmp = np.where((minfo[:, i] == lim_vals[i][0]) | 
                (minfo[:, i] == lim_vals[i][1]))
        idx = np.intersect1d(idx, tmp)
        #
    out_interp = griddata(minfo[idx], models[idx], params)[0]
    #
    if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0):
        print("# Houve um problema na grade. Tentando arrumadar...")
        idx = np.arange(len(minfo))
        for i in [i for i in range(len(params)) if i != dims["sig0"]]:
            imin = lim_vals[i][0]
            if lim_vals[i][0] != np.min(listpar[i]):
                imin = phc.find_nearest(listpar[i], lim_vals[i][0], 
                    bigger=False)
            imax = lim_vals[i][1]
            if lim_vals[i][1] != np.max(listpar[i]):
                imax = phc.find_nearest(listpar[i], lim_vals[i][1], 
                    bigger=True)
            lim_vals[i] = [imin, imax]
            tmp = np.where((minfo[:, i] >= lim_vals[i][0]) & 
                (minfo[:, i] <= lim_vals[i][1]))
            idx = np.intersect1d(idx, phc.flatten(tmp))
        out_interp = griddata(minfo[idx], models[idx], params)[0]
    #
    if (np.sum(out_interp) == 0 or np.sum(np.isnan(out_interp)) > 0):
        print("# Houve um problema na grade e eu nao conseguir arrumar...")
    #
    return out_interp


# ==============================================================================
def griddataBA_new(minfo, models, params, isig, silent=True):
    '''
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
    '''
    # ranges
    ranges = np.array([[parr.min(), parr.max()] for parr in minfo.T])

    # find neighbours, delete coincidences
    if phc.is_inside_ranges(isig, [0, len(params)-1]):
        # exclude sig0 dimension, to take all their entries for interpolation
        keep, out, inside_ranges, params1, minfo1 = phc.find_neighbours(np.delete(params, isig), \
                                                                    np.delete(minfo, isig, axis=1), \
                                                                    np.delete(ranges.T, isig, axis=1).T, \
                                                                    silent=silent)
        params = np.hstack([params1, params[isig]])
        minfo = np.vstack([minfo1.T, minfo[:, isig]]).T
    else:
        keep, out, inside_ranges, params, minfo = phc.find_neighbours(params, minfo, ranges, silent=silent)

    # interpolation
    model_interp = griddata(minfo[keep], models[keep], params, method='linear')[0]

    if np.isnan(model_interp).any() or np.sum(model_interp) == 0.:
        if not silent:
            print('[griddataBA] Warning: linear interpolation didnt work, taking closest model')
        model_interp = griddata(minfo[keep], models[keep], params, method='nearest')[0]

    return model_interp
