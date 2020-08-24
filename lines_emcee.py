import numpy as np
import time
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from constants import G, Msun, Rsun
from be_theory import oblat2w, t_tms_from_Xc, obl2W
import emcee
import matplotlib as mpl
from matplotlib import ticker
from matplotlib import *
from utils import find_nearest,griddataBAtlas, griddataBA, kde_scipy, quantile, geneva_interp_fast, linfit
from be_theory import hfrac2tms
import corner_HDR
from pymc3.stats import hpd
from lines_plot import print_output, par_errors, plot_residuals_new, print_output_means, print_to_latex
from lines_reading import read_iue, read_votable, read_star_info, read_line_spectra, read_observables, check_list, Sliding_Outlier_Removal, find_lim
from lines_convergence import plot_convergence
from astropy.stats import SigmaClip
import seaborn as sns
import user_settings as flag

sns.set_style("white", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})




# ==============================================================================
def lnlike(params, logF_mod):
#def lnlike(params, lbd, logF, dlogF, logF_mod, ranges, box_lim, flag.lista_obs):

    """
    Returns the likelihood probability function.

    # p0 = mass
    # p1 = oblat
    # p2 = age
    # p3 = inclination
    # p4 = ebmv
    # p5 = distance
    
    For photospheric lines:
    
    # p0 = mass
    # p1 = oblat
    # p2 = age
    # p3 = inclination
    # p4 = normal flux modification (if True)
    """
    

    if check_list(flag.lista_obs, 'UV'):
        u = np.where(flag.lista_obs == 'UV')
        index = u[0][0]
        
        logF_UV = flag.logF[index]
        dlogF_UV = flag.dlogF[index]
		
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
        uplim = dlogF_UV == 0
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
        
        
        flux_mod = pyasl.unred(flag.wave[index] * 1e4, tmp_flux, ebv=-1 * ebmv, R_V=RV)
        logF_mod_UV = np.log10(flux_mod)
            
        
        chi2_UV = np.sum(((logF_UV[keep] - logF_mod_UV[keep])**2. / (dlogF_UV[keep])**2.))
        N_UV = len(logF_UV[keep])
        chi2_UV_red = chi2_UV/N_UV
        
    else:
        chi2_UV_red = 0.
        N_UV = 0.
        
    if check_list(flag.lista_obs, 'Ha'):
        u = np.where(flag.lista_obs == 'Ha')
        index = u[0][0]
		
        logF_Ha = flag.logF[index]
        dlogF_Ha = flag.dlogF[index] 
        logF_mod_Ha = logF_mod[index]           
        
        uplim = dlogF_Ha == 0
        keep = np.logical_not(uplim)
        chi2_Ha = np.sum(((logF_Ha[keep] - logF_mod_Ha[keep])**2 / (dlogF_Ha[keep])**2.))
        N_Ha = len(logF_Ha[keep])
        chi2_Ha_red = chi2_Ha/N_Ha
    
    else:
        chi2_Ha_red = 0.
        N_Ha = 0. 
    
    if check_list(flag.lista_obs, 'Hb'):
        u = np.where(flag.lista_obs == 'Hb')
        index = u[0][0]

		
        logF_Hb = flag.logF[index]
        dlogF_Hb = flag.dlogF[index] 
        logF_mod_Hb = logF_mod[index]           
        
        uplim = dlogF_Hb == 0
        keep = np.logical_not(uplim)
        chi2_Hb = np.sum(((logF_Hb[keep] - logF_mod_Hb[keep])**2 / (dlogF_Hb[keep])**2.))
        N_Hb = len(logF_Hb[keep])
        chi2_Hb_red = chi2_Hb/N_Hb
    else:
        chi2_Hb_red = 0.
        N_Hb = 0. 
    if check_list(flag.lista_obs, 'Hd'):
        u = np.where(flag.lista_obs == 'Hd')
        index = u[0][0]

        logF_Hd = flag.logF[index]
        dlogF_Hd = flag.dlogF[index] 
        logF_mod_Hd = logF_mod[index]           
        
        uplim = dlogF_Hd == 0
        keep = np.logical_not(uplim)
        chi2_Hd = np.sum(((logF_Hd[keep] - logF_mod_Hd[keep])**2 / (dlogF_Hd[keep])**2.))
        N_Hd = len(logF_Hd[keep])
        chi2_Hd_red = chi2_Hd/N_Hd
    else:
        chi2_Hd_red = 0.
        N_Hd = 0. 
    if check_list(flag.lista_obs, 'Hg'):
        u = np.where(flag.lista_obs == 'Hg')
        index = u[0][0]
	
        logF_Hg = flag.logF[index]
        dlogF_Hg = flag.dlogF[index] 
        logF_mod_Hg = logF_mod[index]           
        
        uplim = dlogF_Hg == 0
        keep = np.logical_not(uplim)
        chi2_Hg = np.sum(((logF_Hg[keep] - logF_mod_Hg[keep])**2 / (dlogF_Hg[keep])**2.))
        N_Hg = len(logF_Hg[keep])
        chi2_Hg_red = chi2_Hg/N_Hg
    else:
        chi2_Hg_red = 0.
        N_Hg = 0. 

    chi2 = (chi2_UV_red + chi2_Ha_red + chi2_Hb_red+ chi2_Hd_red+ chi2_Hg_red)*(N_UV + N_Ha + N_Hb + N_Hd+ N_Hg)
      
		
    #print(chi2)
    if chi2 is np.nan:
        chi2 = np.inf

    return -0.5 * chi2


# ==============================================================================
def lnprior(params):

    if flag.model == 'aeri':
        if flag.normal_spectra is False or flag.UV is True:
            Mstar, W, tms, cosi, dist, ebv = params[0], params[1],\
                params[2], params[3], params[4], params[5]
        else:
            Mstar, W, tms, cosi = params[0], params[1], params[2], params[3]			
    if flag.model == 'befavor':
        Mstar, oblat, Hfrac, cosi, dist, ebv = params[0], params[1],\
            params[2], params[3], params[4], params[5]
    if flag.model == 'aara' or flag.model == 'acol':
        if flag.normal_spectra is False or flag.UV is True:
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
        
        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014') 

        wcrit = np.sqrt(8. / 27. * G * Mstar * Msun / (Rpole * Rsun)**3)

        vsini = W * wcrit * (Rpole * Rsun * oblat) *\
            np.sin(np.arccos(cosi)) * 1e-5

    else:
        t = np.max(np.array([hfrac2tms(Hfrac), 0.]))

        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, t, Zstr='014') 

        wcrit = np.sqrt(8. / 27. * G * Mstar * Msun / (Rpole * Rsun)**3)

        vsini = oblat2w(oblat) * wcrit * (Rpole * Rsun * oblat) *\
            np.sin(np.arccos(cosi)) * 1e-5
    
    # Vsini prior
    if flag.vsini_prior:
        chi2_vsi = (flag.vsin_obs - vsini)**2 / flag.sig_vsin_obs**2.

    else:
        chi2_vsi = 0
    
    # Distance prior
    if flag.normal_spectra is False or flag.UV is True:
        if flag.dist_prior:
            chi2_dis = (flag.dist_pc - dist)**2 / flag.sig_dist_pc**2.

        else:
            chi2_dis = 0
    else:
        chi2_dis = 0
        chi2_vsi = 0

    # Inclination prior
    if flag.incl_prior:
        incl = np.arccos(cosi)*180./np.pi # obtaining inclination from cosi 
        chi2_incl = (flag.incl0 - incl)**2 / flag.sig_incl0**2.
    else:
        chi2_incl = 0.

       
    chi2_prior = chi2_vsi + chi2_dis + chi2_stellar_prior +  chi2_incl

    if chi2_prior is np.nan:
        chi2_prior = np.inf

    return -0.5 * chi2_prior




# ==============================================================================
def lnprob(params):
#def lnprob(params, lbd, logF, dlogF, flag.minfo, listpar, logF_grid,
#           vsin_obs, sig_vsin_obs, dist_pc, sig_dist_pc, isig,
#           ranges, flag.dims, pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv, grid_mas,
#           grid_obl, grid_age, grid_dis, grid_ebv,  box_lim, flag.lista_obs, raw_waves, raw_flux):
    count = 0
    inside_ranges = True
    while inside_ranges * (count < len(params)):
        inside_ranges = (params[count] >= flag.ranges[count, 0]) *\
            (params[count] <= flag.ranges[count, 1])
        count += 1

    if inside_ranges:
        
        lim = find_lim()

    
        logF_mod = []
        if check_list(flag.lista_obs, 'UV'):
            u = np.where(flag.lista_obs == 'UV')
            index = u[0][0]
            
            if flag.binary_star:
                M2 = params[-1]
                logF_mod_UV_1 = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
                logF_mod_UV_2 = griddataBA(flag.minfo, flag.logF_grid[index], np.array([M2, 0.1, params[2], params[3]]), flag.listpar, flag.dims)
                logF_mod_UV = np.log10(10.**logF_mod_UV_1 + 10.**logF_mod_UV_2)
            else:
                logF_mod_UV = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
                
            logF_mod.append(logF_mod_UV)                    
        
        if check_list(flag.lista_obs, 'Ha'):
            u = np.where(flag.lista_obs == 'Ha')
            index = u[0][0]
            
            #if check_list(flag.lista_obs, 'UV'):
            if flag.binary_star:
                M2 = params[-1]
                logF_mod_Ha_1 = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
                logF_mod_Ha_2 = griddataBA(flag.minfo, flag.logF_grid[index], np.array([M2, 0.1, params[2], params[3]]), flag.listpar, flag.dims)
                F_mod_Ha = linfit(flag.wave[index], logF_mod_Ha_1 + logF_mod_Ha_2)
                #logF_mod_Ha = np.log10(F_mod_Ha)
                #logF_mod_Ha = np.log(norm_spectra(lbd[index], F_mod_Ha_unnormed))
            else:
                logF_mod_Ha_unnorm = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
                F_mod_Ha = linfit(flag.wave[index], logF_mod_Ha_unnorm)
                #logF_mod_Ha = np.log10(F_mod_Ha)
            #else:
            #    logF_mod_Ha = griddataBA(flag.minfo, logF_grid[index], params, listpar, flag.dims)
            logF_mod.append(F_mod_Ha)
        if check_list(flag.lista_obs, 'Hb'):
            u = np.where(flag.lista_obs == 'Hb')
            index = u[0][0]
            if check_list(flag.lista_obs, 'UV'):
                logF_mod_Hb = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
            else:
                logF_mod_Hb = griddataBA(flag.minfo, flag.logF_grid[index], params, flag.listpar, flag.dims)
            logF_mod.append(logF_mod_Hb)
        if check_list(flag.lista_obs, 'Hd'):
            u = np.where(flag.lista_obs == 'Hd')
            index = u[0][0]
            if check_list(flag.lista_obs, 'UV'):
                logF_mod_Hd = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
            else:
                logF_mod_Hd = griddataBA(flag.minfo, flag.logF_grid[index], params, flag.listpar, flag.dims)
            logF_mod.append(logF_mod_Hd)
        if check_list(flag.lista_obs, 'Hg'):
            u = np.where(flag.lista_obs == 'Hg')
            index = u[0][0]
            if check_list(flag.lista_obs, 'UV'):
                logF_mod_Hg = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
            else:
                logF_mod_Hg = griddataBA(flag.minfo, flag.logF_grid[index], params, flag.listpar, flag.dims)
            logF_mod.append(logF_mod_Hg)


        lp = lnprior(params)

        lk = lnlike(params, logF_mod)


        #lp = lnprior(params, vsin_obs, sig_vsin_obs, dist_pc,
        #             sig_dist_pc, ranges, pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv,
        #             grid_mas, grid_obl, grid_age, grid_dis, grid_ebv)
        #
        #lk = lnlike(params, lbd, logF, dlogF, logF_mod, ranges,
        #            box_lim, flag.lista_obs)
        lpost = lp + lk

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
    file_npy_burnin = flag.folder_fig + str(file_name) + '/' + 'Burning_'+\
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
#def new_emcee_inference(star, Ndim, ranges, lbdarr, wave, logF, dlogF, flag.minfo,
#                    listpar, logF_grid, vsin_obs, sig_vsin_obs, dist_pc,
#                    sig_dist_pc, isig, flag.dims, tag, 
#                    pool, pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv,
#                    grid_mas, grid_obl, grid_age, grid_dis, grid_ebv,
#                    box_lim, flag.lista_obs, models):

# emcee inference for new stellar grid 
        if flag.long_process is True:
            Nwalk = 500  # 200  # 500
            nint_burnin = 600  # 50
            nint_mcmc = 2000  # 500  # 1000
        else:
            Nwalk = 20
            nint_burnin = 30
            nint_mcmc = 80
            
        
            
        p0 = [np.random.rand(flag.Ndim) * (flag.ranges[:, 1] - flag.ranges[:, 0]) +
              flag.ranges[:, 0] for i in range(Nwalk)]
        
        start_time = time.time()
        

        
        
        if flag.acrux is True:
            sampler = emcee.EnsembleSampler(Nwalk, flag.Ndim, lnprob, pool=pool, moves=[(emcee.moves.StretchMove(flag.a_parameter))])
        else:
            sampler = emcee.EnsembleSampler(Nwalk, flag.Ndim, lnprob, moves=[(emcee.moves.StretchMove(flag.a_parameter))])

        sampler_tmp = run_emcee(p0, sampler, nint_burnin, nint_mcmc,
                                flag.Ndim, Nwalk, file_name=flag.stars)
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

        # Saving first sample
        file_npy = flag.folder_fig + str(flag.stars) + '/' + 'Walkers_' +\
            str(Nwalk) + '_Nmcmc_' + str(nint_mcmc) +\
            '_af_' + str(af) + '_a_' + str(flag.a_parameter) +\
            flag.tags + ".npy"
        np.save(file_npy, chain)



        flatchain_1 = chain.reshape((-1, flag.Ndim)) # cada walker em cada step em uma lista só

        
        samples = np.copy(flatchain_1)
       


        for i in range(len(samples)):
           

            if flag.model == 'acol':
                samples[i][1] = obl2W(samples[i][1])
                samples[i][2] = hfrac2tms(samples[i][2])
                samples[i][6] = (np.arccos(samples[i][6])) * (180. / np.pi)

            if flag.model == 'aeri':
                # Converting angles to degrees
                samples[i][3] = (np.arccos(samples[i][3])) * (180. / np.pi)


        # plot corner
        quantiles = [0.16, 0.5, 0.84]
        
        if flag.model == 'aeri':
            if check_list(flag.lista_obs, 'UV'):
                labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
                      r'$i[\mathrm{^o}]$', r'$\pi\,[mas]$', r'E(B-V)']
                if flag.include_rv is True:
                    labels = labels + [r'$R_\mathrm{V}$']

            else:
                labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
                      r'$i[\mathrm{^o}]$']
            labels2 = labels
            
        if flag.model == 'acol':
            if check_list(flag.lista_obs, 'UV'):
                labels = [r'$M\,[\mathrm{M_\odot}]$', r'$W$',
                            r"$t/t_\mathrm{ms}$",
                            r'$\log \, n_0 \, [\mathrm{cm^{-3}}]$',
                            r'$R_\mathrm{D}\, [R_\star]$',
                            r'$n$', r'$i[\mathrm{^o}]$', r'$\pi\,[\mathrm{pc}]$',
                            r'E(B-V)']
                labels2 = [r'$M$', r'$W$',
                            r"$t/t_\mathrm{ms}$",
                            r'$\log \, n_0 $',
                            r'$R_\mathrm{D}$',
                            r'$n$', r'$i$', r'$\pi$',
                            r'E(B-V)']                
                if flag.include_rv is True:
                        labels = labels + [r'$R_\mathrm{V}$']
                        labels2 = labels2 + [r'$R_\mathrm{V}$']
            else:
                labels = [r'$M\,[\mathrm{M_\odot}]$', r'$W$',
                            r"$t/t_\mathrm{ms}$",
                            r'$\log \, n_0 \, [\mathrm{cm^{-3}}]$',
                            r'$R_\mathrm{D}\, [R_\star]$',
                            r'$n$', r'$i[\mathrm{^o}]$']
                labels2 = [r'$M$', r'$W$',
                            r"$t/t_\mathrm{ms}$",
                            r'$\log \, n_0 $',
                            r'$R_\mathrm{D}$',
                            r'$n$', r'$i$']
                
                
        if flag.binary_star:
                labels = labels + [r'$M2\,[M_\odot]$']
                labels2 = labels2 + [r'$M2\,[M_\odot]$']
        
        if flag.corner_color == 'blue':
            truth_color='xkcd:cobalt'
            color='xkcd:cornflower'
            color_hist='xkcd:powder blue'
            color_dens='xkcd:clear blue'
            
        elif flag.corner_color == 'dark blue':
            truth_color='xkcd:deep teal'
            color='xkcd:dark teal'
            color_hist='xkcd:pale sky blue'
            color_dens='xkcd:ocean'
            
        elif flag.corner_color == 'teal':
            truth_color='xkcd:charcoal'
            color='xkcd:dark sea green'
            color_hist='xkcd:pale aqua'
            color_dens='xkcd:seafoam blue'
            
        elif flag.corner_color == 'green':
            truth_color='xkcd:forest'
            color='xkcd:forest green'
            color_hist='xkcd:light grey green'
            color_dens='xkcd:grass green'
            
        elif flag.corner_color == 'yellow':
            truth_color='xkcd:mud brown'
            color='xkcd:sandstone'
            color_hist='xkcd:pale gold'
            color_dens='xkcd:sunflower'
            
        elif flag.corner_color == 'orange':
            truth_color='xkcd:chocolate'
            color='xkcd:cinnamon'
            color_hist='xkcd:light peach'
            color_dens='xkcd:bright orange'
            
        elif flag.corner_color == 'red':
            truth_color='xkcd:mahogany'
            color='xkcd:deep red'
            color_hist='xkcd:salmon'
            color_dens='xkcd:reddish'
            
        elif flag.corner_color == 'purple':
            truth_color='xkcd:deep purple'
            color='xkcd:medium purple'
            color_hist='xkcd:soft purple'
            color_dens='xkcd:plum purple'
            
        elif flag.corner_color == 'violet':
            truth_color='xkcd:royal purple'
            color='xkcd:purpley'
            color_hist='xkcd:pale violet'
            color_dens='xkcd:blue violet'
            
        elif flag.corner_color == 'pink':
            truth_color='xkcd:wine'
            color='xkcd:pinky'
            color_hist='xkcd:light pink'
            color_dens='xkcd:pink red'

        
        new_ranges = np.copy(flag.ranges)
        
        if flag.model == 'aeri':

            new_ranges[3] = (np.arccos(flag.ranges[3])) * (180. / np.pi)
            new_ranges[3] = np.array([flag.ranges[3][1], flag.ranges[3][0]])
        if flag.model == 'acol':
            new_ranges[1] = obl2W(flag.ranges[1])            
            new_ranges[2][0] = hfrac2tms(flag.ranges[2][1])
            new_ranges[2][1] = hfrac2tms(flag.ranges[2][0])
            new_ranges[6] = (np.arccos(flag.ranges[6])) * (180. / np.pi)
            new_ranges[6] = np.array([new_ranges[6][1], new_ranges[6][0]])



        best_pars = []
        best_errs = []
        for i in range(flag.Ndim):
            #mode_val = mode1(np.round(samples[:,i], decimals=2))
            qvalues = hpd(samples[:,i], alpha=0.32)
            cut = samples[samples[:,i] > qvalues[0], i]
            cut = cut[cut < qvalues[1]]
            median_val = np.median(cut)
            

            best_errs.append([qvalues[1] - median_val, median_val - qvalues[0]])
            best_pars.append(median_val)
        

        
        truth_color='k'

        fig_corner = corner_HDR.corner(samples, labels=labels, labels2=labels2, range=new_ranges, quantiles=None, plot_contours=True, show_titles=True, 
                title_kwargs={'fontsize': 15}, label_kwargs={'fontsize': 19}, truths = best_pars, hdr=True,
                truth_color=truth_color, color=color, color_hist=color_hist, color_dens=color_dens, 
                smooth=1, plot_datapoints=False, fill_contours=True, combined=True)
        

        #-----------Options for the corner plot-----------------
        if flag.compare_results:
            # reference values (Domiciano de Souza et al. 2014)
            value1 = [6.1,0.84,0.6,60.6] # i don't have t/tms reference value
            
            # Extract the axes
            ndim = 4
            axes = np.array(fig_corner.axes).reshape((ndim, ndim))
            
            # Loop over the diagonal 
            for i in range(ndim):
                if i != 2:
                    ax = axes[i, i]
                    ax.axvline(value1[i], color="g")
                
            # Loop over the histograms
            for yi in range(ndim):
                for xi in range(yi):
                    if xi != 2 and yi != 2:
                        ax = axes[yi, xi]
                        ax.axvline(value1[xi], color="g")
                        ax.axhline(value1[yi], color="g")
                        ax.plot(value1[xi], value1[yi], "sg")


        


        current_folder = str(flag.folder_fig) + str(flag.stars) + '/'
        fig_name = 'Walkers_' + np.str(Nwalk) + '_Nmcmc_' +\
            np.str(nint_mcmc) + '_af_' + str(af) + '_a_' +\
            str(flag.a_parameter) + flag.tags
        
        params_to_print = print_to_latex(best_pars, best_errs, current_folder, fig_name, labels)

        print_output_means(samples)    
        
        plt.savefig(current_folder + fig_name + '.png', dpi=100)


        plt.close()
        plot_residuals_new(params_fit,Nwalk,nint_mcmc,
                           file_npy,current_folder, fig_name)

        
        plot_convergence(file_npy, file_npy[:-4] + '_convergence', 
                         file_npy_burnin,new_ranges,labels)
                         
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
#            Ndim = read_star_info(star, gv.flag.lista_obs, listpar)
#
#
#        logF, dlogF, logF_grid, wave, box_lim =\
#                read_observables( models, lbdarr, flag.lista_obs)
#                                
#
#        
#        if flag.model == 'aeri' or flag.model == 'acol':
#            new_emcee_inference(star, Ndim, ranges, lbdarr, wave, logF,
#                                dlogF, flag.minfo, listpar, logF_grid, vsin_obs,
#                                sig_vsin_obs, dist_pc, sig_dist_pc, isig,
#                                flag.dims, tags, pool, pdf_mas, pdf_obl, pdf_age,
#                                pdf_dis, pdf_ebv, grid_mas, grid_obl,
#                                grid_age, grid_dis, grid_ebv,
#                                box_lim, flag.lista_obs, models)
#
#        else:
#            emcee_inference( star, Ndim, ranges, lbdarr, wave, logF, dlogF,
#                        flag.minfo, listpar, logF_grid, vsin_obs, sig_vsin_obs,
#                        dist_pc, sig_dist_pc, isig, flag.dims,  tag, pool, pdf_mas, pdf_obl, pdf_age,
#                        pdf_dis, pdf_ebv, grid_mas, grid_obl,
#                        grid_age, grid_dis, grid_ebv)
#    else:
#        for i in range(np.size(flag.stars)):
#            star=np.copy(flag.stars[i])
#            ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
#                Ndim = read_star_info(star,flag.lista_obs, listpar)
#
#            wave0, flux0, sigma0 = read_votable(star)
#
#            logF, dlogF, logF_grid, wave =\
#                read_iue(models, lbdarr, wave0, flux0, sigma0, star, cut_iue_regions)
#
#            if flag.model == 'aeri':
#                new_emcee_inference(star, Ndim, ranges, lbdarr, wave, logF,
#                            dlogF, flag.minfo, listpar, logF_grid, vsin_obs,
#                            sig_vsin_obs, dist_pc, sig_dist_pc, isig,
#                            flag.dims, tag, pool, pdf_mas, pdf_obl, pdf_age,
#                            pdf_dis, pdf_ebv, grid_mas, grid_obl,
#                            grid_age, grid_dis, grid_ebv)
#            else:
#                emcee_inference(star, Ndim, ranges, lbdarr, wave, logF,
#                            dlogF, flag.minfo, listpar, logF_grid, vsin_obs,
#                            sig_vsin_obs, dist_pc, sig_dist_pc, isig,
#                            flag.dims, tag, pool, pdf_mas, pdf_obl, pdf_age,
#                            pdf_dis, pdf_ebv, grid_mas, grid_obl,
#                            grid_age, grid_dis, grid_ebv)
#    return

