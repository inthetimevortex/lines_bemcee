from PyAstronomy import pyasl
import numpy as np
import matplotlib.pylab as plt
from be_theory import hfrac2tms
from pyhdust.rotstars import beta, geneva_interp_fast
from utils import griddataBAtlas, griddataBA
from lines_reading import check_list
import corner
import seaborn as sns
import user_settings as flag

sns.set_style("white", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})


# ==============================================================================
def par_errors(flatchain):
    '''
    Most likely parameters and respective asymmetric errors

    Usage:
    par, errors = par_errors(flatchain)
    '''
    quantile = list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                    zip(*np.percentile(flatchain,
                        [16, 50, 84], axis=0))))
    quantile = np.array(quantile)
    par, errors = quantile[:, 0], quantile[:, 1:].reshape((-1))

    return par, errors


# ==============================================================================
def print_output(params_fit, errors_fit):
    """ TBD """
    print(75 * '-')
    print('Output')

    for i in range(len(params_fit)):
        print(params_fit[i], ' +/- ', errors_fit[i])
    
    print(75 * '-')
    print('Derived parameters')
    
    if flag.model == 'aeri':
        Mstar = params_fit[0]
        W = params_fit[1]
        tms = params_fit[2]
        oblat = 1 + 0.5*(W**2) # Rimulo 2017
        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014')
        beta_par = beta(oblat, is_ob=True)
        
        print('Oblateness: ', oblat, ' +/- ', errors_fit[1]*W)
        print('Equatorial radius: ', oblat*Rpole, ' +/- ', errors_fit[1]*W*Rpole)
        print('Log Luminosity: ', logL) 
        print('Beta: ', beta_par)

    return

# ==============================================================================
def print_output_means(samples):
    """ Print the mean values obtained with the samples """
    
    if flag.model == 'aeri':
        lista = ['Mass ', 'W ', 't/tms', 'i']
        print(60 * '-')
        print('Output - mean values')
        for i in range(4):
           param = samples[:,i]
           param_mean = corner.quantile(param, 0.5, weights=None)
           param_sup = corner.quantile(param, 0.84, weights=None)
           param_inf = corner.quantile(param, 0.16, weights=None)
           print(lista[i], param_mean, ' +/- ', param_sup, param_inf)
           if i == 0:
               mass = param_mean
           if i == 1:
               oblat = 1 + 0.5*(param_mean**2) # Rimulo 2017
               oblat_sup = 1 + 0.5*(param_sup**2)
               oblat_inf = 1 + 0.5*(param_inf**2)
               print('Derived oblateness ', oblat, ' +/- ', oblat_sup, oblat_inf)
           if i == 2:
               tms = param_mean
    
        Rpole, logL, _ = geneva_interp_fast(mass, oblat, tms, Zstr='014')
        beta_par = beta(oblat, is_ob=True)
        
        print('Equatorial radius: ', oblat*Rpole)
        print('Log Luminosity: ', logL) 
        print('Beta: ', beta_par)

    return

# ==============================================================================
#def plot_fit(par, lbd, logF, dlogF, minfo, listpar, lbdarr, logF_grid,
#             isig, dims, Nwalk, Nmcmc, par_list, ranges, npy, f):
#    '''
#    Plots best model fit over data
#
#    Usage:
#    plot_fit(par, lbd, logF, dlogF, minfo, logF_grid, isig, Nwalk, Nmcmc)
#    where
#    par = np.array([Mstar, oblat, Sig0, Rd, n, cosi, dist])
#    '''
#    # flag.model parameters
#    if flag.include_rv is True:
#        Mstar, oblat, Hfrac, cosi, dist, ebv, rv = par
#        lim = 3
#        lim2 = 2
#    else:
#        Mstar, oblat, Hfrac, cosi, dist, ebv = par
#        lim = 2
#        lim2 = 1
#        rv = 3.1
#
#    # Rpole, Lstar, Teff = vkg.geneve_par(Mstar, oblat, Hfrac, folder_tables)
#    # t = np.max(np.array([hfrac2tms(hfrac), 0.]))
#    t = np.max(np.array([hfrac2tms(Hfrac), 0.]))
#    Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
#                                     neighbours_only=True, isRpole=False)
#    norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
#    uplim = dlogF == 0
#    keep = np.logical_not(uplim)
#    # chain = np.load(npy)
#
#    # interpolate models
#    logF_mod = griddataBA(minfo, logF_grid, par[:-lim], listpar, dims)
#    logF_list = np.zeros([len(par_list), len(logF_mod)])
#    chi2 = np.zeros(len(logF_list))
#    for i in range(len(par_list)):
#        logF_list[i] = griddataBA(minfo, logF_grid, par_list[i, :-lim],
#                                  listpar, dims)
#    # convert to physical units
#    logF_mod += np.log10(norma)
#    logF_list += np.log10(norma)
#    for j in range(len(logF_list)):
#        chi2[j] = np.sum((logF[keep] -
#                          logF_list[j][keep])**2 / (dlogF[keep])**2)
#
#    # chib = chi2[np.argsort(chi2)[-30:]] / max(chi2)
#
#    flux = 10.**logF
#    dflux = dlogF * flux
#    flux_mod = 10.**logF_mod
#
#    flux_mod = pyasl.unred(lbd * 1e4, flux_mod, ebv=-1 * ebv, R_V=rv)
#
#    # Plot definitions
#    bottom, left = 0.75, 0.48
#    width, height = 0.96 - left, 0.97 - bottom
#    plt.axes([left, bottom, width, height])
#
#    # plot fit
#    if flag.plot_in_log_scale is True:
#        for i in range(len(par_list)):
#            if i % 80 == 0:
#                ebv_temp = np.copy(par_list[i][-lim2])
#                F_temp = pyasl.unred(lbd * 1e4, 10**logF_list[i],
#                                     ebv=-1 * ebv_temp, R_V=rv)
#                plt.plot(lbd, lbd * F_temp, color='gray', alpha=0.1)
#
#        plt.errorbar(lbd, lbd * flux, yerr=lbd * dflux, ls='', marker='o',
#                     alpha=0.5, ms=10, color='blue', linewidth=2)
#        plt.plot(lbd, lbd * flux_mod, color='red', ls='-', lw=3.5, alpha=0.4,
#                 label='$\mathrm{Best\, fit}$')
#
#        plt.xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
#        plt.ylabel(r'$\lambda F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}]}$',
#                   fontsize=20)
#        plt.yscale('log')
#    else:
#        for i in range(len(par_list)):
#            if i % 80 == 0:
#                ebv_temp = np.copy(par_list[i][-lim2])
#                F_temp = pyasl.unred(lbd * 1e4, 10**logF_list[i],
#                                     ebv=-1 * ebv_temp, R_V=rv)
#                plt.plot(lbd, F_temp, color='gray', alpha=0.1)
#
#        plt.errorbar(lbd, flux, yerr=lbd * dflux, ls='', marker='o',
#                     alpha=0.5, ms=10, color='blue', linewidth=2)
#        plt.plot(lbd, flux_mod, color='red', ls='-', lw=3.5, alpha=0.4,
#                 label='$\mathrm{Best\, fit}$')
#        plt.xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
#        plt.ylabel(r'$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2} \mu m]}$',
#                   fontsize=20)
#
#    plt.xlim(min(lbd), max(lbd))
#    plt.tick_params(direction='in', length=6, width=2, colors='gray',
#                    which='both')
#    plt.legend(loc='upper right')
#
#    return
#
#
## ==============================================================================
#def plot_fit_last(par, lbd, logF, dlogF, minfo, listpar, lbdarr, logF_grid,
#                  isig, dims, Nwalk, Nmcmc, ranges, npy,  box_lim):
#    '''
#    Plots best flag.model fit over data
#
#    Usage:
#    plot_fit(par, lbd, logF, dlogF, minfo, logF_grid, isig, Nwalk, Nmcmc)
#    where
#    par = np.array([Mstar, oblat, Sig0, Rd, n, cosi, dist])
#    '''
#    # flag.model parameters
#    if flag.include_rv is True:
#        Mstar, W, tms, cosi, dist, ebv, rv = par
#        lim = 3
#        lim2 = 2
#    else:
#        if flag.model == 'aeri':
#            if flag.normal_spectra is False:
#                Mstar, W, tms, cosi, dist, ebv = par
#        if flag.model == 'befavor':
#            Mstar, oblat, Hfrac, cosi, dist, ebv = par
#        if flag.model == 'aara' or flag.model == 'acol' or flag.model == 'bcmi':
#            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, dist, ebv = par
#        if flag.model == 'beatlas':
#            Mstar, oblat, Sig0, n, cosi, dist, ebv = par
#            Hfrac = 0.30
#
#        lim = 2
#        lim2 = 1
#        rv = 3.1
#
#    # Rpole, Lstar, Teff = vkg.geneve_par(Mstar, oblat, Hfrac, folder_tables)
#    # t = np.max(np.array([hfrac2tms(hfrac), 0.]))
#    if flag.model == 'aeri':
#        oblat = 1 + 0.5*(W**2) # Rimulo 2017
#        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014') #rotina do pyhdust.rotflag.stars
#        
#    else:
#        t = np.max(np.array([hfrac2tms(Hfrac), 0.]))
#        Rpole, logL = geneva_interp_fast(Mstar, oblat, t,
#                                         neighbours_only=True, isRpole=False) # rotinas do utils.py
#    if flag.normal_spectra is False:
#        norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
#        
#
#        
#        # ***
#    chain = np.load(npy)
#    par_list = chain[:, -1, :]
#        
#        
#    # Finding position
#    u = np.where(lista_obs == 'UV')
#    index = u[0][0]
#    # Finding the corresponding flag.model (interpolation)
#    logF_mod_UV = griddataBA(minfo, logF_grid[index], par[:-lim],
#                                listpar, dims)  
#    # Observations
#    logF_UV = logF[index]
#    flux_UV = 10.**logF_UV
#    dlogF_UV = dlogF[index]
#    lbd_UV = lbd[index]
#    
#    
#    logF_list = np.zeros([len(par_list), len(logF_mod_UV)])
#    chi2 = np.zeros(len(logF_list))
#    for i in range(len(par_list)):
#        logF_list[i] = griddataBA(minfo, logF_grid[index], par_list[i, :-lim],
#                                    listpar, dims)
#    # Plot
#    fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
#    
#    norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
#    uplim = dlogF[index] == 0
#    keep = np.logical_not(uplim)  
#    
#    
#    # convert to physical units
#    logF_mod_UV += np.log10(norma)
#    logF_list += np.log10(norma)
#    flux_mod_UV = 10.**logF_mod_UV
#    
#    for j in range(len(logF_list)):
#        chi2[j] = np.sum((logF_UV[keep] -
#        logF_list[j][keep])**2 / (dlogF_UV[keep])**2)
#    
#    dflux = dlogF_UV * flux_UV
#
#    # Plot definitions
#    bottom, left = 0.80, 0.48  # 0.75, 0.48
#    width, height = 0.96 - left, 0.97 - bottom
#    plt.axes([left, bottom, width, height])
#
#
#    for i in range(len(par_list)):
#
#        ebv_temp = np.copy(ebv)
#        F_temp = pyasl.unred(lbd_UV * 1e4, 10**logF_list[i],
#                             ebv=-1 * ebv_temp, R_V=rv)
#        plt.plot(lbd_UV, lbd_UV * F_temp, color='gray', alpha=0.1)
#    
#    flux_mod_UV = pyasl.unred(lbd_UV * 1e4, flux_mod_UV, ebv=-1 * ebv, R_V=rv)
#    plt.errorbar(lbd_UV, lbd_UV * flux_UV, yerr=lbd_UV * dflux, ls='', marker='o',
#                 alpha=0.5, ms=10, color='blue', linewidth=2)
#    plt.plot(lbd_UV, lbd_UV * flux_mod_UV, color='red', ls='-', lw=3.5, alpha=0.4,
#             label='$\mathrm{Best\, fit}$')
#
#    plt.ylabel(r'$\lambda F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}]}$',
#               fontsize=20)
#    plt.yscale('log')
#    plt.tick_params(labelbottom='off')
#    # plt.setp(upperplot.get_xticklabels(), visible=False)
#
#
#    plt.tick_params(direction='in', length=6, width=2, colors='gray',
#                    which='both')
#
#
#    return


# ====================================================================================

def plot_residuals_new(par, lbd, logF, dlogF, minfo, listpar, lbdarr, logF_grid,
                       isig, dims, Nwalk, Nmcmc, ranges, npy, box_lim, lista_obs,
                       current_folder, fig_name):
    '''
    Create residuals plot separated from the corner 

    '''
    if flag.include_rv and check_list(lista_obs, 'UV'):
        Mstar, W, tms, cosi, dist, ebv, rv = par
        lim = 3
        lim2 = 2
    elif check_list(lista_obs, 'UV'):
        Mstar, W, tms, cosi, dist, ebv = par
        lim=2
        rv=3.1
    else:
        Mstar, W, tms, cosi = par
        rv=3.1
    if flag.model == 'aeri':

                
        oblat = 1 + 0.5*(W**2) # Rimulo 2017
        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014')
        
        # ***
        chain = np.load(npy)
        par_list = chain[:, -1, :]
        
        if check_list(lista_obs, 'UV'):
			# Finding position
            u = np.where(lista_obs == 'UV')
            index = u[0][0]
            # Finding the corresponding flag.model (interpolation)
            logF_mod_UV = griddataBA(minfo, logF_grid[index], par[:-lim],
                                     listpar, dims)  
            # Observations
            logF_UV = logF[index]
            flux_UV = 10.**logF_UV
            dlogF_UV = dlogF[index]
            lbd_UV = lbd[index]
            
            
            logF_list = np.zeros([len(par_list), len(logF_mod_UV)])
            chi2 = np.zeros(len(logF_list))
            for i in range(len(par_list)):
                logF_list[i] = griddataBA(minfo, logF_grid[index], par_list[i, :-lim],
                                          listpar, dims)
            # Plot
            fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
            
            norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
            uplim = dlogF[index] == 0
            keep = np.logical_not(uplim)  
            
            
            # convert to physical units
            logF_mod_UV += np.log10(norma)
            logF_list += np.log10(norma)
            flux_mod_UV = 10.**logF_mod_UV
            
            for j in range(len(logF_list)):
                chi2[j] = np.sum((logF_UV[keep] -
                logF_list[j][keep])**2 / (dlogF_UV[keep])**2)
            
            dflux = dlogF_UV * flux_UV

            # Plot Models
            for i in range(len(par_list)):
                
                ebv_temp = np.copy(ebv)
                F_temp = pyasl.unred(lbd_UV * 1e4, 10**logF_list[i],
                                     ebv=-1 * ebv_temp, R_V=rv)
                ax1.plot(lbd_UV, F_temp, color='gray', alpha=0.1)
                # Residuals  --- desta forma plota os residuos de todos os modelos, mas acho que nao eh o que quero  
                #ax2.plot(lbd_UV, (flux_UV - F_temp) / dflux, 'bs', alpha=0.2)
            
            
            # Applying reddening to the best model
            flux_mod_UV = pyasl.unred(lbd_UV * 1e4, flux_mod_UV, ebv=-1 * ebv, R_V=rv)
 
            # Best fit
            ax1.plot(lbd_UV, flux_mod_UV, color='red', ls='-', lw=3.5, alpha=0.4,
                     label='$\mathrm{Best\, fit}$')
            ax2.plot(lbd_UV, (flux_UV - flux_mod_UV) / dflux, 'bs', alpha=0.2)
            ax2.set_ylim(-10,10)
            # Plot Data
            keep = np.where(flux_UV > 0) # avoid plot zero flux
            ax1.errorbar(lbd_UV[keep], flux_UV[keep], yerr=dflux[keep], ls='', marker='o',
                         alpha=0.5, ms=5, color='blue', linewidth=1)

                         
            ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=16)
            ax1.set_ylabel(r'$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}]}$',
                       fontsize=20)	
            ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=16)
            #plt.tick_params(labelbottom='off')
            ax1.set_xlim(min(lbd_UV), max(lbd_UV))
            ax2.set_xlim(min(lbd_UV), max(lbd_UV))
            #plt.tick_params(direction='in', length=6, width=2, colors='gray',
            #    which='both')
            ax1.legend(loc='upper right')
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.tight_layout()
            plt.savefig(current_folder + fig_name + '_new_residuals-UV' + '.png', dpi=100)
            plt.close()
        

        if check_list(lista_obs, 'Ha'):
            # Finding position
            line = 'Ha'
            plot_line(line, lista_obs, minfo, logF_grid, par, listpar, dims, logF, dlogF, lbd, par_list, current_folder, fig_name)
        if check_list(lista_obs, 'Hb'):
            line = 'Hb'
            plot_line(line, lista_obs, minfo, logF_grid, par, listpar, dims, logF, dlogF, lbd, par_list, current_folder, fig_name)               
                           
        if check_list(lista_obs, 'Hd'):
            line = 'Hd'
            plot_line(line, lista_obs, minfo, logF_grid, par, listpar, dims, logF, dlogF, lbd, par_list, current_folder, fig_name)               
        if check_list(lista_obs, 'Hg'):
            line = 'Hg'
            plot_line(line, lista_obs, minfo, logF_grid, par, listpar, dims, logF, dlogF, lbd, par_list, current_folder, fig_name)               
                           
        
        #    if flag.normal_spectra is False:
        #        if flag.include_rv:
        #            Mstar, W, tms, cosi, dist, ebv, rv = par
        #        else:
        #            Mstar, W, tms, cosi, dist, ebv = par
        #        
        #
        #
        #    if flag.Halpha:
        #        oblat = 1 + 0.5*(W**2) # Rimulo 2017
        #        Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014') 
        #        uplim = dlogF == 0
        #        keep = np.logical_not(uplim)
        #
        #        # ***
        #        chain = np.load(npy)
        #        par_list = chain[:, -1, :]
        #
        #      
        #
        #
        #        flux = 10.**logF
        #        dflux = dlogF * flux
        #        flux_mod = 10.**logF_mod
        #
        #        logF_list = np.zeros([len(par_list), len(logF_mod)])
        #        chi2 = np.zeros(len(logF_list))
        #       
        #
        #
        #        fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
        #
        #        # Plot models
        #        for i in range(len(par_list)):
        #            ax1.plot(lbd, 10**logF_list[i], color='gray', alpha=0.1)
        #
        #        # Data
        #        keep = np.where(flux > 0) # avoid plot zero flux
        #        ax1.errorbar(lbd[keep], flux[keep], yerr= dflux[keep], ls='', marker='o', alpha=0.5, ms=5, color='blue', linewidth=1) 
        #
        #
        #            # Best fit
        #        ax1.plot(lbd, flux_mod, color='red', ls='-', lw=3.5, alpha=0.4, label='$\mathrm{Best\, fit}$')
        #        #ax1.plot(lbd[keep], flux_mod[keep], color='red', ls='', marker='o', alpha=0.4, label='$\mathrm{Best\, fit}$') #usei esse pra testar
        #
        #        #ax1.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
        #        ax1.set_ylabel('Normalized Flux',fontsize=18)
        #        ax1.set_xlim(min(lbd), max(lbd))
        #        ax1.legend(loc='lower right')
        #
        #        # Residuals
        #        ax2.plot(lbd[keep], (flux[keep] - flux_mod[keep])/dflux[keep], marker='o', alpha=0.5)
        #        #ax2.plot(lbd, (flux[keep] - 10.**logF_list[i])/dflux, 'bs', alpha=0.2) # esse aparecem mais pontos, não sei pq
        #
        #        ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=16)
        #        ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=16)
        #        ax2.set_xlim(min(lbd), max(lbd))
        #        #ax2.set_ylim(-20,10)
        #    #elif -- Hbeta
        #    else:
		#		
		#		# ***
        #        chain = np.load(npy)
        #        par_list = chain[:, -1, :]
        #        
        #        # Finding the corresponding flag.model (interpolation)
        #        logF_mod_UV = griddataBA(minfo, logF_grid, par[:-lim],
        #                                 listpar, dims)  
        #        # Observations
        #        logF_UV = logF
        #        flux_UV = 10.**logF_UV
        #        dlogF_UV = dlogF
        #        lbd_UV = lbd
        #        logF_list = np.zeros([len(par_list), len(logF_mod_UV)])
        #        chi2 = np.zeros(len(logF_list))
        #        for i in range(len(par_list)):
        #            logF_list[i] = griddataBA(minfo, logF_grid, par_list[i, :-lim],
        #                                      listpar, dims)
        #        # Plot
        #        fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
        #        
        #        norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
        #        uplim = dlogF == 0
        #        keep = np.logical_not(uplim)  
        #        
        #        
        #        # convert to physical units
        #        logF_mod_UV += np.log10(norma)
        #        logF_list += np.log10(norma)
        #        flux_mod_UV = 10.**logF_mod_UV
        #        
        #        for j in range(len(logF_list)):
        #            chi2[j] = np.sum((logF_UV[keep] -
        #            logF_list[j][keep])**2 / (dlogF_UV[keep])**2)
        #        
        #        dflux = dlogF_UV * flux_UV
        #        
        #        # Plot Models
        #        for i in range(len(par_list)):
        #            ebv_temp = np.copy(par_list[i][-1])
        #            F_temp = pyasl.unred(lbd_UV * 1e4, 10**logF_list[i],
        #                                 ebv=-1 * ebv_temp, R_V=rv)
        #            ax1.plot(lbd_UV, F_temp, color='gray', alpha=0.1)
        #            # Residuals  --- desta forma plota os residuos de todos os modelos, mas acho que nao eh o que quero  
        #            #ax2.plot(lbd_UV, (flux_UV - F_temp) / dflux, 'bs', alpha=0.2)
        #        
        #       
        #         
        #        # Applying reddening to the best model
        #        flux_mod_UV = pyasl.unred(lbd_UV * 1e4, flux_mod_UV, ebv=-1 * ebv, R_V=rv)
        #
        #        # Best fit
        #        ax1.plot(lbd_UV, flux_mod_UV, color='red', ls='-', lw=3.5, alpha=0.4,
        #                 label='$\mathrm{Best\, fit}$')
        #
        #        ax2.plot(lbd_UV, (flux_UV - flux_mod_UV) / dflux, 'bs', alpha=0.2)
        #        
        #        # Plot Data
        #        keep = np.where(flux_UV > 0) # avoid plot zero flux
        #        ax1.errorbar(lbd_UV[keep], flux_UV[keep], yerr=dflux[keep], ls='', marker='o',
        #                     alpha=0.5, ms=5, color='blue', linewidth=1) #antes estava linewidth = 2 , ms = 10
        #                     
        #        ax1.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
        #        ax1.set_ylabel(r'$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}]}$',
        #                   fontsize=20)	
        #        #plt.tick_params(labelbottom='off')
        #        ax1.set_xlim(min(lbd_UV), max(lbd_UV))
        #        #plt.tick_params(direction='in', length=6, width=2, colors='gray',
        #        #    which='both')
        #        ax1.legend(loc='upper right')
        #        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                
                #plt.savefig(current_folder + fig_name + '_new_residuals-UV' + '.png', dpi=100)
                #plt.tight_layout()
                #plt.close()        


#def plot_residuals(par, lbd, logF, dlogF, minfo, listpar, lbdarr, logF_grid,
#                   isig, dims, Nwalk, Nmcmc, ranges,
#                   npy):
#    '''
#    Plots best model fit over data
#
#    Usage:
#    plot_fit(par, lbd, logF, dlogF, minfo, logF_grid, isig, Nwalk, Nmcmc)
#    where
#    par = np.array([Mstar, oblat, Sig0, Rd, n, cosi, dist])
#    '''
#    # flag.model parameters
#    if flag.include_rv is True:
#        Mstar, oblat, Hfrac, cosi, dist, ebv, rv = par
#        lim = 3
#        lim2 = 2
#    else:
#        if flag.model == 'aeri':
#            if flag.normal_spectra is False:
#                Mstar, W, tms, cosi, dist, ebv = par
#            else:
#                if mod_normalflux:
#                    Mstar, W, tms, cosi, deltaFlux = par
#                else:
#                    if vrad:
#                        Mstar, W, tms, cosi, Vrad = par
#                    else:
#                        Mstar, W, tms, cosi = par
#        if flag.model == 'befavor':
#            Mstar, oblat, Hfrac, cosi, dist, ebv = par
#        if flag.model == 'aara' or flag.model == 'acol' or flag.model == 'bcmi':
#            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, dist, ebv = par
#        if flag.model == 'beatlas':
#            Mstar, oblat, Sig0, n, cosi, dist, ebv = par
#            Hfrac = 0.3
#
#        lim = 2
#        lim2 = 1
#        rv = 3.1
#
#    # Rpole, Lstar, Teff = vkg.geneve_par(Mstar, oblat, Hfrac, folder_tables)
#    # t = np.max(np.array([hfrac2tms(hfrac), 0.]))
#    #t = np.max(np.array([hfrac2tms(Hfrac), 0.]))
#    oblat = 1 + 0.5*(W**2) # Rimulo 2017
#    Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014') 
#    if flag.normal_spectra is False:
#        norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
#    uplim = dlogF == 0
#    keep = np.logical_not(uplim)
#
#    # ***
#    chain = np.load(npy)
#    par_list = chain[:, -1, :]
#
#    # interpolate models
#    if flag.model == 'beatlas':
#        logF_mod = griddataBAtlas(minfo, logF_grid, par[:-lim],
#                                  listpar, dims, isig)
#    else:
#        if flag.normal_spectra is False:
#            logF_mod = griddataBA(minfo, logF_grid, par[:-lim],
#                                  listpar, dims)
#
#                                  
#    logF_list = np.zeros([len(par_list), len(logF_mod)])
#    chi2 = np.zeros(len(logF_list))
#    for i in range(len(par_list)):
#        if flag.model == 'beatlas':
#            logF_list[i] = griddataBAtlas(minfo, logF_grid, par_list[i, :-lim],
#                                          listpar, dims, isig)
#        else:
#            if flag.normal_spectra is False:
#                logF_list[i] = griddataBA(minfo, logF_grid, par_list[i, :-lim],
#                                          listpar, dims)
#
#                                          
#    if flag.normal_spectra is False:
#        # convert to physical units
#        logF_mod += np.log10(norma)
#        logF_list += np.log10(norma)
#        for j in range(len(logF_list)):
#            chi2[j] = np.sum((logF[keep] -
#                              logF_list[j][keep])**2 / (dlogF[keep])**2)
#
#    # chib = chi2[np.argsort(chi2)[-30:]] / max(chi2)
#
#    flux = 10.**logF
#    dflux = dlogF * flux
#    flux_mod = 10.**logF_mod
#    
#    if flag.normal_spectra is False:
#        flux_mod = pyasl.unred(lbd * 1e4, flux_mod,
#                               ebv=-1 * ebv, R_V=rv)
#    # alphas = (1. - chi2 / max(chi2)) / 50.
#
#    # Plot definitions
#    bottom, left = 0.71, 0.48
#    width, height = 0.96 - left, 0.785 - bottom
#    plt.axes([left, bottom, width, height])
#
#    # plot fit
#    if flag.normal_spectra is False:
#        for i in range(len(par_list)):
#            ebv_temp = np.copy(par_list[i][-lim2])
#            F_temp = pyasl.unred(lbd * 1e4, 10**logF_list[i],
#                                 ebv=-1 * ebv_temp, R_V=rv)
#            plt.plot(lbd, (flux - F_temp) / dflux, 'bs', alpha=0.2)
#    else:
#        plt.plot(lbd, (flux - 10.**logF_list[i])/dflux, 'bs', alpha=0.2)
#    # plt.plot(lbd, (flux - flux_mod) / dflux, 'bs', markersize=5, alpha=0.1)
#    # plt.ylabel('$(\mathrm{F}-F_\mathrm{model})/\sigma$', fontsize=20)
#    plt.ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=20)
#    plt.xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=20)
#    # plt.ylim(-100, 100)
#    plt.hlines(y=0, xmin=min(lbd), xmax=max(lbd),
#               linestyles='--', color='black')
#
#    plt.xlim(min(lbd), max(lbd))
#    if flag.model == 'aara' or flag.model == 'beatlas' or\
#       flag.model == 'acol' or flag.model == 'bcmi':
#        plt.xscale('log')
#    plt.tick_params(direction='in', length=6, width=2, colors='gray',
#                    which='both')
#    plt.legend(loc='upper right')
#
#    return

def plot_line(line, lista_obs, minfo, logF_grid, par, listpar, dims, logF, dlogF, lbd, par_list, current_folder, fig_name):
    # Finding position
    u = np.where(lista_obs == line)
    index = u[0][0]
    # Finding the corresponding flag.model (interpolation)
    if check_list(lista_obs, 'UV'):
        lim = 2
        if flag.include_rv:
            lim = 3
    else: lim=0
    
    if check_list(lista_obs, 'UV'):
        logF_mod_line = griddataBA(minfo, logF_grid[index], par[:-lim] ,listpar, dims)
    else:
        logF_mod_line = griddataBA(minfo, logF_grid[index], par ,listpar, dims)
        
    #logF_mod_line = griddataBA(minfo, logF_grid[index], par[:-lim],listpar, dims)
    flux_mod_line = 10.**logF_mod_line
    # Observations
    logF_line = logF[index]
    flux_line = 10.**logF_line
    
    logF_list = np.zeros([len(par_list), len(logF_mod_line)])
    chi2 = np.zeros(len(logF_list))
    for i in range(len(par_list)):
        if check_list(lista_obs, 'UV') is False:
            logF_list[i] = griddataBA(minfo, logF_grid[index], par_list[i],
                                  listpar, dims)
        else:
            logF_list[i] = griddataBA(minfo, logF_grid[index], par_list[i, :-lim],
                                  listpar, dims)
                                  
    
    # Data
    keep_fx = np.where(flux_line > 0) # avoid plot zero flux
    lbd_line = lbd[index]
    #print(len(lbd), lbd)
    dflux_line = dlogF[index] * flux_line
    # Plot
    fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})                              
    # Plot models
    for i in range(len(par_list)):
        ax1.plot(lbd_line, 10**logF_list[i], color='gray', alpha=0.1)
    ax1.errorbar(lbd_line[keep_fx], flux_line[keep_fx], yerr= dflux_line[keep_fx], ls='', marker='o', alpha=0.5, ms=5, color='blue', linewidth=1) 

    # Best fit
    #ax1.plot(lbd_line, flux_mod_line, color='red', ls='-', lw=3.5, alpha=0.4, label='$\mathrm{Best\, fit}$')
    
    ax1.set_ylabel('Normalized Flux',fontsize=18)
    ax1.set_xlim(min(lbd_line), max(lbd_line))
    #ax1.legend(loc='lower right')
    ax1.set_title(line)
    # Residuals
    ax2.plot(lbd_line[keep_fx], (flux_line[keep_fx] - flux_mod_line[keep_fx])/dflux_line[keep_fx], marker='o', alpha=0.5)
    
    ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=16)
    ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=16)
    ax2.set_xlim(min(lbd_line), max(lbd_line))
    plt.tight_layout()
    plt.savefig(current_folder + fig_name + '_new_residuals_' + line +'.png', dpi=100)
    plt.close()
    
