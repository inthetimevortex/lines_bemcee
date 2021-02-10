import numpy as np
import time
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from constants import G, Msun, Rsun
from .be_theory import oblat2w, t_tms_from_Xc, obl2W, hfrac2tms
import emcee
from .corner_HDR import corner
import matplotlib as mpl
from matplotlib import ticker
from matplotlib import *
from .utils import find_nearest,griddataBAtlas, griddataBA, kde_scipy, quantile, \
                    geneva_interp_fast, linfit, jy2cgs,check_list
import lines_bemcee.corner_HDR
from .hpd import hpd_grid
from .lines_plot import print_output, par_errors, plot_residuals_new, print_output_means, print_to_latex
from .lines_convergence import plot_convergence
from astropy.stats import SigmaClip
import seaborn as sns
import datetime
from scipy.special import erf
import importlib
from __init__ import mod_name
flag = importlib.import_module(mod_name)
import organizer as info

sns.set_style("white", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})


def get_line_chi2(line):
    '''Get the chi2 for the lines
    
    Usage:
    chi2 = get_line_chi2(line)
    '''
    if line:
        u = np.where(info.lista_obs == str(line).split('.')[-1])
        index = u[0][0]
        
        logF_Ha = info.logF[index]
        dlogF_Ha = info.dlogF[index] 
        logF_mod_Ha = logF_mod[index]           
        
        uplim = dlogF_Ha == 0
        keep = np.logical_not(uplim)
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
        
        #if flag.binary_star:
        #    M2, Lfrac = params[-2], params[-1]
        #    logF_mod_UV_1, logF_mod_UV_2 = logF_mod[index]            
        #    F_mod_UV_2 = Lfrac * 10**logF_mod_UV_2
        #    F_mod_UV_1 = (1. - Lfrac) * 10**logF_mod_UV_1
        #    logF_mod_UV_comb = np.log(F_mod_UV_1 + F_mod_UV_2)
        #
        #    logF_mod_UV_comb += np.log10(norma)
        #    tmp_flux = 10**logF_mod_UV_comb
        #    
        #
        #else:
        
        logF_mod[index] += np.log10(norma)
        tmp_flux = 10**logF_mod[index]
        
        
        flux_mod = pyasl.unred(info.wave[index] * 1e4, tmp_flux, ebv=-1 * ebmv, R_V=RV)
        logF_mod_UV = np.log10(flux_mod)
            
        rms = np.array([jy2cgs(1e-3*0.1, 20000), jy2cgs(1e-3*0.1, 35000), jy2cgs(1e-3*0.1, 63000)])
        #rms = np.array([1e-3*0.1, 1e-3*0.1, 1e-3*0.1])
        #
        upper_lim = jy2cgs(10**logF_UV[uplim], lbd_UV[uplim], inverse=True)
        mod_upper = jy2cgs(10**logF_mod_UV[uplim], lbd_UV[uplim], inverse=True)
        
        
        #a parte dos uplims não é em log!
        chi2_UV = np.sum(((logF_UV[keep] - logF_mod_UV[keep])**2. / (dlogF_UV[keep])**2.)) 
        #chi2_uplim = - 2. * np.sum( np.log( (np.pi/2.)**0.5 * rms * (1. + erf(((upper_lim - mod_upper)/((2**0.5)*rms))))))
        #chi2_UV = chi2_UV + chi2_uplim
        N_UV = len(logF_UV[keep])
        chi2_UV_red = chi2_UV/N_UV
        #print('chi2_UV = {:.2f}'.format(chi2_UV))
        #print(upper_lim-mod_upper)
        #print('chi2_uplim = {:.2f}'.format(chi2_uplim))
    else:
        chi2_UV_red = 0.
        N_UV = 0.
        
    if flag.Ha:
        chi2_Ha_red, N_Ha = get_line_chi2(flag.Ha)
    
    if flag.Hb:
        chi2_Hb_red, N_Hb = get_line_chi2(flag.Hb)

    if flag.Hd:
        chi2_Hd_red, N_Hd = get_line_chi2(flag.Hd)
    
    if flag.Hg:
        chi2_Hg_red, N_Hg = get_line_chi2(flag.Hg)
    
    #    u = np.where(info.lista_obs == 'Hg')
    #    index = u[0][0]
	#
    #    logF_Hg = info.logF[index]
    #    dlogF_Hg = info.dlogF[index] 
    #    logF_mod_Hg = logF_mod[index]           
    #    
    #    uplim = dlogF_Hg == 0
    #    keep = np.logical_not(uplim)
    #    chi2_Hg = np.sum(((logF_Hg[keep] - logF_mod_Hg[keep])**2 / (dlogF_Hg[keep])**2.))
    #    N_Hg = len(logF_Hg[keep])
    #    chi2_Hg_red = chi2_Hg/N_Hg
    #else:
    #    chi2_Hg_red = 0.
    #    N_Hg = 0. 

    chi2 = (chi2_UV_red + chi2_Ha_red + chi2_Hb_red+ chi2_Hd_red+ chi2_Hg_red)*(N_UV + N_Ha + N_Hb + N_Hd+ N_Hg)
      
		
    #print(chi2)
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
    if flag.model == 'aara' or flag.model == 'acol':
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
        
        #Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014') 

        #wcrit = np.sqrt(8. / 27. * G * Mstar * Msun / (Rpole * Rsun)**3)

        #vsini = W * wcrit * (Rpole * Rsun * oblat) *\
        #    np.sin(np.arccos(cosi)) * 1e-5

    else:
        tms = np.max(np.array([hfrac2tms(Hfrac), 0.]))

    Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014') 

    wcrit = np.sqrt(8. / 27. * G * Mstar * Msun / (Rpole * Rsun)**3)

    vsini = oblat2w(oblat) * wcrit * (Rpole * Rsun * oblat) *\
            np.sin(np.arccos(cosi)) * 1e-5
    
    # Vsini prior
    if flag.vsini_prior:
        chi2_vsi = (info.file_vsini - vsini)**2 / info.file_dvsin**2.

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
#def lnprob(params, lbd, logF, dlogF, info.minfo, listpar, logF_grid,
#           vsin_obs, sig_vsin_obs, dist_pc, sig_dist_pc, isig,
#           ranges, info.dims, pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv, grid_mas,
#           grid_obl, grid_age, grid_dis, grid_ebv,  box_lim, info.lista_obs, raw_waves, raw_flux):
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
                #print(params[:-lim])
                #print(len(params[:-lim]))
                if flag.model != 'beatlas':
                    logF_mod_UV = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
                else:
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
                #logF_mod_Ha = np.log10(F_mod_Ha)
                #logF_mod_Ha = np.log(norm_spectra(lbd[index], F_mod_Ha_unnormed))
            else:
                logF_mod_Ha_unnorm = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
                F_mod_Ha = linfit(info.wave[index], logF_mod_Ha_unnorm)
                #logF_mod_Ha = np.log10(F_mod_Ha)
            #else:
            #    logF_mod_Ha = griddataBA(info.minfo, logF_grid[index], params, listpar, info.dims)
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
    
    
        lp = lnprior(params)
    
        lk = lnlike(params, logF_mod)
        
        lpost = lp + lk
    
        #lp = lnprior(params, vsin_obs, sig_vsin_obs, dist_pc,
        #             sig_dist_pc, ranges, pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv,
        #             grid_mas, grid_obl, grid_age, grid_dis, grid_ebv)
        #
        #lk = lnlike(params, lbd, logF, dlogF, logF_mod, ranges,
        #            box_lim, info.lista_obs)
        #lpost = lp + lk
    
        # print('{0:.2f} , {1:.2f}, {2:.2f}'.format(lp, lk, lpost))
    
        if not np.isfinite(lpost):
            return -np.inf
        else:
            return lpost
    else:
        return -np.inf


# ==============================================================================
def run_emcee(p0, sampler, nib, nimc, Ndim, Nwalk, file_name):

    print('\n')
    print(75 * '=')
    print('\n')
    print("Burning-in ...")
    start_time = time.time()
    pos, prob, state = sampler.run_mcmc(p0, nib, progress=True)

    print("--- %s minutes ---" % ((time.time() - start_time) / 60))

    af = np.mean(sampler.acceptance_fraction)
    print("Mean acceptance fraction (BI):", af)

    # Saving the burn-in phase #########
    chain = sampler.chain
    date = datetime.datetime.today().strftime('%y-%m-%d-%H%M%S')
    file_npy_burnin = flag.folder_fig + str(file_name) + '/' + date + 'Burning_'+\
                      str(nib)+"steps_Nwalkers_"+str(Nwalk)+"normal-spectra_"+\
                      str(flag.normal_spectra)+"af_"+str(af)+".npy"
    np.save(file_npy_burnin, chain)
    ###########
    
    sampler.reset()

# ------------------------------------------------------------------------------
    print('\n')
    print(75 * '=')
    print("Running MCMC ...")
    pos, prob, state = sampler.run_mcmc(pos, nimc, rstate0=state, progress=True)

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
    errors_fit = [sampler.flatchain[:, i].std() for i in range(Ndim)]

# ------------------------------------------------------------------------------
    params_fit = []
    errors_fit = []
    pa = []

    for j in range(Ndim):
        for i in range(len(pos)):
            pa.append(pos[i][j])

    pa = np.reshape(pa, (Ndim, len(pos)))

    for j in range(Ndim):
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

# ================================================================================


# ==============================================================================
def new_emcee_inference(pool):
#def new_emcee_inference(star, Ndim, ranges, lbdarr, wave, logF, dlogF, info.minfo,
#                    listpar, logF_grid, vsin_obs, sig_vsin_obs, dist_pc,
#                    sig_dist_pc, isig, info.dims, tag, 
#                    pool, pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv,
#                    grid_mas, grid_obl, grid_age, grid_dis, grid_ebv,
#                    box_lim, info.lista_obs, models):

# emcee inference for new stellar grid 
        #1#if flag.long_process is True:
        #1#    Nwalk = 100  # 200  # 500
        #1#    nint_burnin = 700  # 50
        #1#    nint_mcmc = 10000  # 500  # 1000
        #1#else:
        #1#    Nwalk = 100
        #1#    nint_burnin = 25
        #1#    nint_mcmc = 100
        #1#
        ##p0 = np.mean(info.ranges, axis=1) + 1e-3 * np.random.randn(Nwalk, len(info.ranges))
        p0 = [np.random.rand(info.Ndim) * (info.ranges[:, 1] - info.ranges[:, 0]) +
              info.ranges[:, 0] for i in range(info.Nwalk)]
        #1#
        start_time = time.time()
        

        
        
        if flag.acrux is True:
            sampler = emcee.EnsembleSampler(info.Nwalk, info.Ndim, lnprob, pool=pool, moves=[(emcee.moves.StretchMove(flag.a_parameter))])
        else:
            sampler = emcee.EnsembleSampler(info.Nwalk, info.Ndim, lnprob, moves=[(emcee.moves.StretchMove(flag.a_parameter))])

        sampler_tmp = run_emcee(p0, sampler, info.nint_burnin, info.nint_mcmc,
                                info.Ndim, info.Nwalk, file_name=flag.stars)
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
            str(info.Nwalk) + '_Nmcmc_' + str(info.nint_mcmc) +\
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
        quantiles = [0.16, 0.5, 0.84]
        
        #1#if flag.model == 'aeri':
        #1#    if flag.SED:
        #1#        info.labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
        #1#              r'$i[\mathrm{^o}]$', r'$\pi\,[mas]$', r'E(B-V)']
        #1#        if flag.include_rv is True:
        #1#            info.labels = info.labels + [r'$R_\mathrm{V}$']
        #1#
        #1#    else:
        #1#        info.labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
        #1#              r'$i[\mathrm{^o}]$']
        #1#    info.labels2 = info.labels
        #1#    
        #1#if flag.model == 'acol':
        #1#    if flag.SED:
        #1#        info.labels = [r'$M\,[\mathrm{M_\odot}]$', r'$W$',
        #1#                    r"$t/t_\mathrm{ms}$",
        #1#                    r'$\log \, n_0 \, [\mathrm{cm^{-3}}]$',
        #1#                    r'$R_\mathrm{D}\, [R_\star]$',
        #1#                    r'$n$', r'$i[\mathrm{^o}]$', r'$\pi\,[\mathrm{pc}]$',
        #1#                    r'E(B-V)']
        #1#        info.labels2 = [r'$M$', r'$W$',
        #1#                    r"$t/t_\mathrm{ms}$",
        #1#                    r'$\log \, n_0 $',
        #1#                    r'$R_\mathrm{D}$',
        #1#                    r'$n$', r'$i$', r'$\pi$',
        #1#                    r'E(B-V)']                
        #1#        if flag.include_rv is True:
        #1#                info.labels = info.labels + [r'$R_\mathrm{V}$']
        #1#                info.labels2 = info.labels2 + [r'$R_\mathrm{V}$']
        #1#    else:
        #1#        info.labels = [r'$M\,[\mathrm{M_\odot}]$', r'$W$',
        #1#                    r"$t/t_\mathrm{ms}$",
        #1#                    r'$\log \, n_0 \, [\mathrm{cm^{-3}}]$',
        #1#                    r'$R_\mathrm{D}\, [R_\star]$',
        #1#                    r'$n$', r'$i[\mathrm{^o}]$']
        #1#        info.labels2 = [r'$M$', r'$W$',
        #1#                    r"$t/t_\mathrm{ms}$",
        #1#                    r'$\log \, n_0 $',
        #1#                    r'$R_\mathrm{D}$',
        #1#                    r'$n$', r'$i$']
        #1#if flag.model == 'beatlas':
        #1#    info.labels = [r'$M\,[\mathrm{M_\odot}]$', r'$W$',
        #1#                r'$\Sigma_0 \, [\mathrm{g/cm^{-2}}]$',
        #1#                r'$n$', r'$i[\mathrm{^o}]$', r'$\pi\,[\mathrm{pc}]$',
        #1#                r'E(B-V)']
        #1#    info.labels2 = [r'$M$', r'$W$',
        #1#                r'$\\Sigma_0 $',
        #1#                r'$R_\mathrm{D}$',
        #1#                r'$n$', r'$i$', r'$\pi$',
        #1#                r'E(B-V)']                
        #1#    if flag.include_rv is True:
        #1#            info.labels = info.labels + [r'$R_\mathrm{V}$']
        #1#            info.labels2 = info.labels2 + [r'$R_\mathrm{V}$']
        #1#
        #1#
        #1#
        #1#        
        #1#if flag.binary_star:
        #1#        info.labels = info.labels + [r'$M2\,[M_\odot]$']
        #1#        info.labels2 = info.labels2 + [r'$M2\,[M_\odot]$']
        #1#
        #1#if flag.corner_color == 'blue':
        #1#    truth_color='xkcd:cobalt'
        #1#    color='xkcd:cornflower'
        #1#    color_hist='xkcd:powder blue'
        #1#    color_dens='xkcd:clear blue'
        #1#    
        #1#elif flag.corner_color == 'dark blue':
        #1#    truth_color='xkcd:deep teal'
        #1#    color='xkcd:dark teal'
        #1#    color_hist='xkcd:pale sky blue'
        #1#    color_dens='xkcd:ocean'
        #1#    
        #1#elif flag.corner_color == 'teal':
        #1#    truth_color='xkcd:charcoal'
        #1#    color='xkcd:dark sea green'
        #1#    color_hist='xkcd:pale aqua'
        #1#    color_dens='xkcd:seafoam blue'
        #1#    
        #1#elif flag.corner_color == 'green':
        #1#    truth_color='xkcd:forest'
        #1#    color='xkcd:forest green'
        #1#    color_hist='xkcd:light grey green'
        #1#    color_dens='xkcd:grass green'
        #1#    
        #1#elif flag.corner_color == 'yellow':
        #1#    truth_color='xkcd:mud brown'
        #1#    color='xkcd:sandstone'
        #1#    color_hist='xkcd:pale gold'
        #1#    color_dens='xkcd:sunflower'
        #1#    
        #1#elif flag.corner_color == 'orange':
        #1#    truth_color='xkcd:chocolate'
        #1#    color='xkcd:cinnamon'
        #1#    color_hist='xkcd:light peach'
        #1#    color_dens='xkcd:bright orange'
        #1#    
        #1#elif flag.corner_color == 'red':
        #1#    truth_color='xkcd:mahogany'
        #1#    color='xkcd:deep red'
        #1#    color_hist='xkcd:salmon'
        #1#    color_dens='xkcd:reddish'
        #1#    
        #1#elif flag.corner_color == 'purple':
        #1#    truth_color='xkcd:deep purple'
        #1#    color='xkcd:medium purple'
        #1#    color_hist='xkcd:soft purple'
        #1#    color_dens='xkcd:plum purple'
        #1#    
        #1#elif flag.corner_color == 'violet':
        #1#    truth_color='xkcd:royal purple'
        #1#    color='xkcd:purpley'
        #1#    color_hist='xkcd:pale violet'
        #1#    color_dens='xkcd:blue violet'
        #1#    
        #1#elif flag.corner_color == 'pink':
        #1#    truth_color='xkcd:wine'
        #1#    color='xkcd:pinky'
        #1#    color_hist='xkcd:light pink'
        #1#    color_dens='xkcd:pink red'

        
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
        #for i in range(info.Ndim):
        #    #mode_val = mode1(np.round(samples[:,i], decimals=2))
        #    qvalues = hpd(samples[:,i], alpha=0.32)
        #    cut = samples[samples[:,i] > qvalues[0], i]
        #    cut = cut[cut < qvalues[1]]
        #    median_val = np.median(cut)
        #    
        #
        #    best_errs.append([qvalues[1] - median_val, median_val - qvalues[0]])
        #    best_pars.append(median_val)
        for i in range(info.Ndim):
            print(samples[:,i])
            hpd_mu, x_mu, y_mu, modes_mu = hpd_grid(samples[:,i], alpha=0.32)
            #mode_val = mode1(np.round(samples[:,i], decimals=2))
            bpars = []
            epars = []
            hpds.append(hpd_mu)
            #print(i, hpd_mu)
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
        

        fig_corner = corner(samples, labels=info.labels, labels2=info.labels2, range=new_ranges, quantiles=None, plot_contours=True, show_titles=False, 
                title_kwargs={'fontsize': 15}, label_kwargs={'fontsize': 19}, truths = best_pars, hdr=True,
                truth_color=info.truth_color, color=info.color, color_hist=info.color_hist, color_dens=info.color_dens, 
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
        
        fig_name = date + 'Walkers_' + np.str(info.Nwalk) + '_Nmcmc_' +\
            np.str(info.nint_mcmc) + '_af_' + str(af) + '_a_' +\
            str(flag.a_parameter) + info.tags
        
        #params_to_print = print_to_latex(best_pars, best_errs, current_folder, fig_name, info.labels, hpds)

        print_output_means(samples)    
        
        plt.savefig(current_folder + fig_name + '.png', dpi=100)

        #print(params_fit)
        #print(best_pars)
        plt.close()
        plot_residuals_new(best_pars,file_npy,current_folder, fig_name)

        
        plot_convergence(file_npy, file_npy[:-4] + '_convergence', 
                         file_npy_burnin,new_ranges,info.labels)
                       
        end = time.time()
        serial_data_time = end - start_time
        print("Serial took {0:.1f} seconds".format(serial_data_time))
        #chord_plot(folder=current_folder, file=file_npy[16:-4])
        return
# ==============================================================================


# ==============================================================================
#def run(input_params):
#
#
#
#    if np.size(flag.stars) == 1:
#        star= np.copy(flag.stars)
#        ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
#            Ndim = read_star_info(star, gv.info.lista_obs, listpar)
#
#
#        logF, dlogF, logF_grid, wave, box_lim =\
#                read_observables( models, lbdarr, info.lista_obs)
#                                
#
#        
#        if flag.model == 'aeri' or flag.model == 'acol':
#            new_emcee_inference(star, Ndim, ranges, lbdarr, wave, logF,
#                                dlogF, info.minfo, listpar, logF_grid, vsin_obs,
#                                sig_vsin_obs, dist_pc, sig_dist_pc, isig,
#                                info.dims, tags, pool, pdf_mas, pdf_obl, pdf_age,
#                                pdf_dis, pdf_ebv, grid_mas, grid_obl,
#                                grid_age, grid_dis, grid_ebv,
#                                box_lim, info.lista_obs, models)
#
#        else:
#            emcee_inference( star, Ndim, ranges, lbdarr, wave, logF, dlogF,
#                        info.minfo, listpar, logF_grid, vsin_obs, sig_vsin_obs,
#                        dist_pc, sig_dist_pc, isig, info.dims,  tag, pool, pdf_mas, pdf_obl, pdf_age,
#                        pdf_dis, pdf_ebv, grid_mas, grid_obl,
#                        grid_age, grid_dis, grid_ebv)
#    else:
#        for i in range(np.size(flag.stars)):
#            star=np.copy(flag.stars[i])
#            ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
#                Ndim = read_star_info(star,info.lista_obs, listpar)
#
#            wave0, flux0, sigma0 = read_votable(star)
#
#            logF, dlogF, logF_grid, wave =\
#                read_iue(models, lbdarr, wave0, flux0, sigma0, star, cut_iue_regions)
#
#            if flag.model == 'aeri':
#                new_emcee_inference(star, Ndim, ranges, lbdarr, wave, logF,
#                            dlogF, info.minfo, listpar, logF_grid, vsin_obs,
#                            sig_vsin_obs, dist_pc, sig_dist_pc, isig,
#                            info.dims, tag, pool, pdf_mas, pdf_obl, pdf_age,
#                            pdf_dis, pdf_ebv, grid_mas, grid_obl,
#                            grid_age, grid_dis, grid_ebv)
#            else:
#                emcee_inference(star, Ndim, ranges, lbdarr, wave, logF,
#                            dlogF, info.minfo, listpar, logF_grid, vsin_obs,
#                            sig_vsin_obs, dist_pc, sig_dist_pc, isig,
#                            info.dims, tag, pool, pdf_mas, pdf_obl, pdf_age,
#                            pdf_dis, pdf_ebv, grid_mas, grid_obl,
#                            grid_age, grid_dis, grid_ebv)
#    return

