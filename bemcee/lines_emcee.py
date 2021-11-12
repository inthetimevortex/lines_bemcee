import numpy as np
import time
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from .constants import G, Msun, Rsun
from .be_theory import oblat2w, t_tms_from_Xc, obl2W, hfrac2tms
import emcee
from scipy.interpolate import griddata
from .corner_HDR import corner
import matplotlib as mpl
from matplotlib import ticker
from matplotlib import *
from .utils import find_nearest,griddataBAtlas, griddataBA, kde_scipy, quantile, \
        geneva_interp_fast, linfit, jy2cgs,check_list
import bemcee.corner_HDR
from .hpd import hpd_grid
from .lines_plot import print_output, par_errors, plot_residuals, print_output_means, print_to_latex
from .lines_convergence import plot_convergence
from .lines_gauss import gaussconv
from astropy.stats import SigmaClip
import seaborn as sns
import datetime
from scipy.special import erf
import importlib
from icecream import ic
from __init__ import mod_name
flag = importlib.import_module(mod_name)
import organizer as info

sns.set_style("white", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})

# ==============================================================================
def get_line_chi2(line, lname, logF_mod):
    '''Get the chi2 for the lines

    Usage:
    chi2 = get_line_chi2(line)
    '''
    if line:
        u = np.where(info.lista_obs == lname)
        index = u[0][0]

        #this is not actually log!
        logF_Ha = info.logF[index]
        dlogF_Ha = info.dlogF[index]
        logF_mod_Ha = logF_mod[index]
        
        wl = info.wave[index]

        limits = np.logical_and(wl > 0.6541, wl < 0.6585)
        
        uplim = dlogF_Ha == 0
        kp = np.logical_not(uplim)

        keep = np.logical_and(limits, kp)

        chi2_Ha = np.sum(((logF_Ha[keep] - logF_mod_Ha[keep])**2 / (dlogF_Ha[keep])**2.))

        N_Ha = len(logF_Ha[keep])
        chi2_Ha_red = chi2_Ha/N_Ha


    else:
        chi2_Ha_red = 0.
        N_Ha = 0.

    return chi2_Ha_red, N_Ha


# ==============================================================================
def lnlike(params, logF_mod):

    """
    Returns the likelihood probability function (-0.5 * chi2).

    Usage
    prob = lnlike(params, logF_mod)
    """


    if flag.SED:
        u = np.where(info.lista_obs == 'UV')
        index = u[0][0]

        lbd_UV = info.wave[index]
        logF_UV = info.logF[index]
        dlogF_UV = info.dlogF[index]

        if flag.model == 'befavor' or flag.model == 'aeri':
            dist = params[4]
            ebmv = params[5]
        if flag.model == 'aara' or flag.model == 'acol':
            dist = params[7]
            ebmv = params[8]
        if flag.model == 'beatlas':
            dist = params[5]
            ebmv = params[6]

        if flag.include_rv and not flag.binary_star:
            RV = params[-1]
        elif flag.binary_star:
            RV = params[-2]
        else:
            RV = 3.1

        #print(logF_UV)
        dist = 1e3 / dist
        norma = (10 / dist)**2
        uplim = dlogF_UV == 0.0

        keep = np.logical_not(uplim)

        logF_mod[index] += np.log10(norma)
        tmp_flux = 10**logF_mod[index]


        flux_mod = pyasl.unred(info.wave[index] * 1e4, tmp_flux, ebv=-1 * ebmv, R_V=RV)
        logF_mod_UV = np.log10(flux_mod)

        rms = np.array([jy2cgs(1e-3*0.1, 20000), jy2cgs(1e-3*0.1, 35000), jy2cgs(1e-3*0.1, 63000)])
        rms = np.array([1e-3*0.1, 1e-3*0.1, 1e-3*0.1])
        rms = np.array([1e-3*0.1, 1e-3*0.1])
        rms = np.array([1e-3*0.1])
        #

        upper_lim = jy2cgs(10**logF_UV[uplim], lbd_UV[uplim], inverse=True)
        mod_upper = jy2cgs(10**logF_mod_UV[uplim], lbd_UV[uplim], inverse=True)
        #ic(rms)
        #ic(10**logF_UV[uplim])

        #a parte dos uplims não é em log!
        chi2_UV = np.sum(((logF_UV[keep] - logF_mod_UV[keep])**2. / (dlogF_UV[keep])**2.))
        #TESTES DOS UPLIMS
        #if flag.model == 'acol':
        #    chi2_uplim = - 2. * np.sum( np.log( (np.pi/2.)**0.5 * rms * (1. + erf(((upper_lim - mod_upper)/((2**0.5)*rms))))))
        #    N_uplim = len(rms)
        #    chi2_uplim_red = chi2_uplim/N_uplim
            #if uplim.any():
            #    if (logF_mod_UV[uplim] > logF_UV[uplim]).any():
            #        chi2_uplim_red = np.inf
            #        N_uplim=0.
            #    else:
            #        chi2_uplim_red = 0.
            #        N_uplim = 0.
        #else: 
        #    chi2_uplim= 0.
        #    N_uplim = 0.
        #    chi2_uplim_red = 0.
        #chi2_UV = chi2_UV + chi2_uplim
        #N_UV = len(logF_UV)

        #print((mod_upper > upper_lim))
        #ic(chi2_uplim)
        N_UV = len(logF_UV[keep])
        chi2_UV_red = chi2_UV/N_UV
        #N_uplim = 3.
        #chi2_uplim_red = 0.
        #chi2_uplim_red = chi2_uplim/N_uplim
        #print('chi2_UV = {:.2f}'.format(chi2_UV))
        #print(upper_lim-mod_upper)
        #print('chi2_uplim = {:.2f}'.format(chi2_uplim))
    else:
        chi2_UV_red = 0.
        N_UV = 0.
        chi2_uplim_red = 0.
        N_uplim = 0.

    #if flag.Ha:
    chi2_Ha_red, N_Ha = get_line_chi2(flag.Ha, 'Ha', logF_mod)

    #if flag.Hb:
    chi2_Hb_red, N_Hb = get_line_chi2(flag.Hb, 'Hb', logF_mod)

    #if flag.Hd:
    chi2_Hd_red, N_Hd = get_line_chi2(flag.Hd, 'Hd', logF_mod)

    #if flag.Hg:
    chi2_Hg_red, N_Hg = get_line_chi2(flag.Hg, 'Hg', logF_mod)

    if flag.pol:
        chi2 = np.sum((info.logF[0] - logF_mod)**2 / (info.dlogF[0])**2.)
    else:
        chi2 = (chi2_UV_red + chi2_Ha_red + chi2_Hb_red+ chi2_Hd_red+ chi2_Hg_red)\
        *(N_UV + N_Ha + N_Hb + N_Hd+ N_Hg)


    #ic(chi2)
    #ic(chi2_UV_red)
    #ic(chi2_uplim_red)
    if chi2 is np.nan:
        chi2 = np.inf

    return -0.5 * chi2


# ==============================================================================
def lnprior(params):
    ''' Calculates the chi2 for the priors set in user_settings

    Usage:
    chi2_prior = lnprior(params)
    '''

    if flag.model == 'aeri':
        if flag.normal_spectra is False or flag.SED is True:
            Mstar, W, tms, cosi, dist, ebv = params[0], params[1],\
                params[2], params[3], params[4], params[5]
        else:
            Mstar, W, tms, cosi = params[0], params[1], params[2], params[3]
    if flag.model == 'befavor':
        Mstar, oblat, Hfrac, cosi, dist, ebv = params[0], params[1],\
            params[2], params[3], params[4], params[5]
    if flag.model == 'aara' or flag.model == 'acol' or flag.model=='pol':
        if flag.normal_spectra is False or flag.SED is True:
            Mstar, oblat, Hfrac, cosi, dist, ebv = params[0], params[1],\
                params[2], params[6], params[7], params[8]
        else:
            Mstar, oblat, Hfrac, cosi = params[0], params[1],\
                params[2], params[6]
    if flag.model == 'beatlas':
        Mstar, oblat, Hfrac, cosi, dist, ebv = params[0], params[1],\
            0.3, params[4], params[5], params[6]


    # Reading Stellar Priors
    if flag.stellar_prior is True:
        temp, idx_mas = find_nearest(flag.grid_prior[0], value=Mstar)
        temp, idx_obl = find_nearest(flag.grid_prior[1], value=oblat)
        temp, idx_age = find_nearest(flag.grid_prior[2], value=Hfrac)
        temp, idx_dis = find_nearest(flag.grid_prior[3], value=dist)
        temp, idx_ebv = find_nearest(flag.grid_prior[4], value=ebv)
        chi2_stellar_prior = Mstar * flag.pdf_prior[0][idx_mas] +\
            oblat * flag.pdf_prior[1][idx_obl] + \
            Hfrac * flag.pdf_prior[2][idx_age] + \
            dist * flag.pdf_prior[3][idx_dis] + \
            ebv * flag.pdf_prior[4][idx_ebv]
    else:
        chi2_stellar_prior = 0.0


    if flag.model == 'aeri':
        oblat = 1 + 0.5*(W**2) # Rimulo 2017

    else:
        tms = np.max(np.array([hfrac2tms(Hfrac), 0.]))

    # Vsini prior
    if flag.vsini_prior:
        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014')
        wcrit = np.sqrt(8. / 27. * G * Mstar * Msun / (Rpole * Rsun)**3)
        vsini = oblat2w(oblat) * wcrit * (Rpole * Rsun * oblat) *\
                                        np.sin(np.arccos(cosi)) * 1e-5

        chi2_vsi = (info.file_vsini - vsini)**2 / info.file_dvsini**2.

    else:
        chi2_vsi = 0

    # Distance prior
    if flag.normal_spectra is False or flag.SED is True:
        if flag.dist_prior:
            chi2_dis = (info.file_plx - dist)**2 / info.file_dplx**2.

        else:
            chi2_dis = 0
    else:
        chi2_dis = 0
        chi2_vsi = 0

    # Inclination prior
    if flag.incl_prior:
        incl = np.arccos(cosi)*180./np.pi # obtaining inclination from cosi
        chi2_incl = (info.file_incl - incl)**2 / info.file_dincl**2.
    else:
        chi2_incl = 0.


    chi2_prior =  chi2_vsi + chi2_dis + chi2_stellar_prior +  chi2_incl

    if chi2_prior is np.nan:
        chi2_prior = -np.inf

    return -0.5 * chi2_prior


# ==============================================================================
def lnprob(params):
    '''
    Calculates lnprob (lnpriot + lnlike)

    Usage:
    lprob = lnprob(params)
    '''

    count = 0
    inside_ranges = True
    while inside_ranges * (count < len(params)):
        inside_ranges = (params[count] >= info.ranges[count, 0]) *\
            (params[count] <= info.ranges[count, 1])
        count += 1

    if inside_ranges:


        logF_mod = []
        if flag.SED:
            u = np.where(info.lista_obs == 'UV')
            index = u[0][0]

            if flag.binary_star:
                M2 = params[-1]
                logF_mod_UV_1 = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
                logF_mod_UV_2 = griddataBA(info.minfo, info.logF_grid[index], np.array([M2, 0.1, params[2], params[3]]), info.listpar, info.dims)
                logF_mod_UV = np.log10(10.**logF_mod_UV_1 + 10.**logF_mod_UV_2)
            else:
                if flag.model != 'beatlas':
                    logF_mod_UV = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
                else:
                    #logF_mod_UV = griddataBA_new(info.minfo, info.logF_grid[index], params[:-info.lim], isig = info.dims["sig0"], silent=True)
                    logF_mod_UV = griddataBAtlas(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims, isig = info.dims["sig0"])

            logF_mod.append(logF_mod_UV)

        if flag.Ha:
            u = np.where(info.lista_obs == 'Ha')
            index = u[0][0]

            #if flag.SED:
            if flag.binary_star:
                M2 = params[-1]
                logF_mod_Ha_1 = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
                logF_mod_Ha_2 = griddataBA(info.minfo, info.logF_grid[index], np.array([M2, 0.1, params[2], params[3]]), info.listpar, info.dims)
                F_mod_Ha = linfit(info.wave[index], logF_mod_Ha_1 + logF_mod_Ha_2)
            else:
                #print(info.dims)
                #print(info.lim)
                #print(params[:-info.lim])
                logF_mod_Ha_unnorm = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
                F_mod_Ha = linfit(info.wave[index], logF_mod_Ha_unnorm)

            #if flag.model == 'acol':
            #    fac_e, v_e = params[-1], params[-3]
            #    wave_conv, flx_conv = gaussconv(fac_e, v_e, F_mod_Ha, info.wave[index])
            #    F_mod_Ha = griddata(wave_conv, flx_conv, info.wave[index], method='linear',fill_value=1)

            logF_mod.append(F_mod_Ha)

        if flag.Hb:
            u = np.where(info.lista_obs == 'Hb')
            index = u[0][0]
            if flag.SED:
                logF_mod_Hb = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
            else:
                logF_mod_Hb = griddataBA(info.minfo, info.logF_grid[index], params, info.listpar, info.dims)
            logF_mod.append(logF_mod_Hb)

        if flag.Hd:
            u = np.where(info.lista_obs == 'Hd')
            index = u[0][0]
            if flag.SED:
                logF_mod_Hd = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
            else:
                logF_mod_Hd = griddataBA(info.minfo, info.logF_grid[index], params, info.listpar, info.dims)
            logF_mod.append(logF_mod_Hd)

        if flag.Hg:
            u = np.where(info.lista_obs == 'Hg')
            index = u[0][0]
            if flag.SED:
                logF_mod_Hg = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
            else:
                logF_mod_Hg = griddataBA(info.minfo, info.logF_grid[index], params, info.listpar, info.dims)
            logF_mod.append(logF_mod_Hg)

        if flag.pol:
            logF_mod = griddataBA(info.minfo, info.logF_grid[0], params, info.listpar, info.dims)

        lp = lnprior(params)

        lk = lnlike(params, logF_mod)

        lpost = lp + lk

        if not np.isfinite(lpost):
            return -np.inf
        else:
            return lpost
    else:
        return -np.inf


# ==============================================================================
def run_emcee(p0, sampler, file_name):
    '''
    Calls emcee and does the MCMC calculations.

    Usage:
    sampler, params_fit, errors_fit, maxprob_index, minprob_index, af, file_npy_burnin =
    run_emcee(p0, sampler, file_name)
    '''
    print('\n')
    print(75 * '=')
    print('\n')
    print("Burning-in ...")
    start_time = time.time()
    pos, prob, state = sampler.run_mcmc(p0, flag.Sburn, progress=True)

    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    af = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction (BI):", af)

    # Saving the burn-in phase #########
    chain = sampler.chain
    date = datetime.datetime.today().strftime('%y-%m-%d-%H%M%S')
    file_npy_burnin = flag.folder_fig + str(file_name) + '/' + date + 'Burning_'+\
                      str(flag.Sburn)+"steps_Nwalkers_"+str(flag.Nwalk)+"normal-spectra_"+\
                      str(flag.normal_spectra)+"af_"+str(af)+".npy"
    np.save(file_npy_burnin, chain)
    ###########

    sampler.reset()

# ------------------------------------------------------------------------------
    print('\n')
    print(75 * '=')
    print("Running MCMC ...")
    pos, prob, state = sampler.run_mcmc(pos, flag.Smcmc, rstate0=state, progress=True)

    # Print out the mean acceptance fraction.
    af = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction:", af)

    # median and errors
    flatchain = sampler.flatchain
    par, errors = par_errors(flatchain)

    # best fit parameters
    maxprob_index = np.argmax(prob)
    minprob_index = np.argmax(prob)

    # Get the best parameters and their respective errors
    params_fit = pos[maxprob_index]
    errors_fit = [sampler.flatchain[:, i].std() for i in range(info.Ndim)]

# ------------------------------------------------------------------------------
    params_fit = []
    errors_fit = []
    pa = []

    for j in range(info.Ndim):
        for i in range(len(pos)):
            pa.append(pos[i][j])

    pa = np.reshape(pa, (info.Ndim, len(pos)))

    for j in range(info.Ndim):
        p = quantile(pa[j], [0.16, 0.5, 0.84])
        params_fit.append(p[1])
        errors_fit.append((p[0], p[2]))

    params_fit = np.array(params_fit)
    errors_fit = np.array(errors_fit)

# ------------------------------------------------------------------------------
    # Print the output
    print_output(params_fit, errors_fit)
    # Turn it True if you want to see the parameters' sample histograms

    return sampler, params_fit, errors_fit, maxprob_index, minprob_index, af, file_npy_burnin


# ==============================================================================
def emcee_inference(pool):
    '''
    Main MCMC function. Calls all others and saves results.

    Usage:
    new_emcee_inference(pool)
    '''

    p0 = [np.random.rand(info.Ndim) * (info.ranges[:, 1] - info.ranges[:, 0]) +
          info.ranges[:, 0] for i in range(flag.Nwalk)]

    start_time = time.time()


    if flag.acrux is True:
        sampler = emcee.EnsembleSampler(flag.Nwalk, info.Ndim, lnprob, pool=pool, \
           #moves=[(emcee.moves.KDEMove())])
            moves=[(emcee.moves.StretchMove(flag.a_parameter))])
    else:
        sampler = emcee.EnsembleSampler(flag.Nwalk, info.Ndim, lnprob, \
            moves=[(emcee.moves.StretchMove(flag.a_parameter))])

    sampler_tmp = run_emcee(p0, sampler, file_name=flag.stars)
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    sampler, params_fit, errors_fit, maxprob_index,\
        minprob_index, af, file_npy_burnin = sampler_tmp

    chain = sampler.chain

    if flag.af_filter is True:
        acceptance_fractions = sampler.acceptance_fraction
        chain = chain[(acceptance_fractions >= 0.20) &
                      (acceptance_fractions <= 0.50)]
        af = acceptance_fractions[(acceptance_fractions >= 0.20) &
                                  (acceptance_fractions <= 0.50)]
        af = np.mean(af)

    af = str('{0:.2f}'.format(af))

    date = datetime.datetime.today().strftime('%y-%m-%d-%H%M%S')
    # Saving first sample
    file_npy = flag.folder_fig + str(flag.stars) + '/' + date + 'Walkers_' +\
        str(flag.Nwalk) + '_Nmcmc_' + str(flag.Smcmc) +\
        '_af_' + str(af) + '_a_' + str(flag.a_parameter) +\
        info.tags + ".npy"
    np.save(file_npy, chain)



    flatchain_1 = chain.reshape((-1, info.Ndim)) # cada walker em cada step em uma lista só


    samples = np.copy(flatchain_1)


    for i in range(len(samples)):


        if flag.model == 'acol':
            samples[i][1] = obl2W(samples[i][1])
            samples[i][2] = hfrac2tms(samples[i][2])
            samples[i][6] = (np.arccos(samples[i][6])) * (180. / np.pi)

        if flag.model == 'aeri':
            samples[i][3] = (np.arccos(samples[i][3])) * (180. / np.pi)

        if flag.model == 'beatlas':
            samples[i][1] = obl2W(samples[i][1])
            samples[i][4] = (np.arccos(samples[i][4])) * (180. / np.pi)

    # plot corner


    new_ranges = np.copy(info.ranges)

    if flag.model == 'aeri':
        new_ranges[3] = (np.arccos(info.ranges[3])) * (180. / np.pi)
        new_ranges[3] = np.array([new_ranges[3][1], new_ranges[3][0]])
    if flag.model == 'acol':
        new_ranges[1] = obl2W(info.ranges[1])
        new_ranges[2][0] = hfrac2tms(info.ranges[2][1])
        new_ranges[2][1] = hfrac2tms(info.ranges[2][0])
        new_ranges[6] = (np.arccos(info.ranges[6])) * (180. / np.pi)
        new_ranges[6] = np.array([new_ranges[6][1], new_ranges[6][0]])
    if flag.model == 'beatlas':
        new_ranges[1] = obl2W(info.ranges[1])
        new_ranges[4] = (np.arccos(info.ranges[4])) * (180. / np.pi)
        new_ranges[4] = np.array([new_ranges[4][1], new_ranges[4][0]])



    best_pars = []
    best_errs = []
    hpds = []

    for i in range(info.Ndim):
        #print(samples[:,i])
        hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(samples[:,i], alpha=0.32)
        #mode_val = mode1(np.round(samples[:,i], decimals=2))
        bpars = []
        epars = []
        hpds.append(hpd_mu)
        for (x0, x1) in hpd_mu:
            #qvalues = hpd(samples[:,i], alpha=0.32)
            cut = samples[samples[:,i] > x0, i]
            cut = cut[cut < x1]
            median_val = np.median(cut)

            bpars.append(median_val)
            epars.append([x1- median_val, median_val - x0])

        best_errs.append(epars)
        best_pars.append(bpars)

    print(best_pars)


    fig_corner = corner(samples, labels=info.labels, labels2=info.labels2, \
            range=new_ranges, quantiles=None, plot_contours=True, \
            show_titles=False, title_kwargs={'fontsize': 15}, \
            label_kwargs={'fontsize': 19}, truths = best_pars, hdr=True,
            truth_color=info.truth_color, color=info.color, \
            color_hist=info.color_hist, color_dens=info.color_dens, \
            smooth=1, plot_datapoints=False, fill_contours=True, combined=True)


    #-----------Options for the corner plot-----------------
    #if flag.compare_results:
    #    # reference values (Domiciano de Souza et al. 2014)
    #    value1 = [6.1,0.84,0.6,60.6] # i don't have t/tms reference value
    #
    #    # Extract the axes
    #    ndim = 4
    #    axes = np.array(fig_corner.axes).reshape((ndim, ndim))
    #
    #    # Loop over the diagonal
    #    for i in range(ndim):
    #        if i != 2:
    #            ax = axes[i, i]
    #            ax.axvline(value1[i], color="g")
    #
    #    # Loop over the histograms
    #    for yi in range(ndim):
    #        for xi in range(yi):
    #            if xi != 2 and yi != 2:
    #                ax = axes[yi, xi]
    #                ax.axvline(value1[xi], color="g")
    #                ax.axhline(value1[yi], color="g")
    #                ax.plot(value1[xi], value1[yi], "sg")




    current_folder = str(flag.folder_fig) + str(flag.stars) + '/'

    fig_name = date + 'Walkers_' + np.str(flag.Nwalk) + '_Nmcmc_' +\
        np.str(flag.Smcmc) + '_af_' + str(af) + '_a_' +\
        str(flag.a_parameter) + info.tags

    try:
        params_to_print = print_to_latex(best_pars, best_errs, current_folder, fig_name, info.labels, hpds)
    except:
        print('Error in params_to_print')

    print_output_means(samples)

    plt.savefig(current_folder + fig_name + '.png', dpi=100)

    #print(params_fit)
    #print(best_pars)
    plt.close()
    plot_residuals(best_pars,file_npy,current_folder, fig_name)


    plot_convergence(file_npy, file_npy[:-4] + '_convergence',
                     file_npy_burnin,new_ranges,info.labels2)

    end = time.time()
    serial_data_time = end - start_time
    print("Serial took {0:.1f} seconds".format(serial_data_time))
    #chord_plot(folder=current_folder, file=file_npy[16:-4])
    return
