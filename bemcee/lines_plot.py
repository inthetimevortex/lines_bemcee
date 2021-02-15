from PyAstronomy import pyasl
import numpy as np
import matplotlib.pylab as plt
from .be_theory import hfrac2tms, oblat2w
from .utils import beta, geneva_interp_fast, griddataBAtlas, griddataBA, linfit, jy2cgs
import corner
from .constants import G, Msun, Rsun, sigma, Lsun
import seaborn as sns
from scipy.special import erf
import importlib
from __init__ import mod_name
flag = importlib.import_module(mod_name)
import organizer as info

sns.set_style("white", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})
#plt.rc('xtick',labelsize=8)
#plt.rc('ytick',labelsize=8)

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
    '''
    Prints output of the MCMC simulation to terminal
    CURRENTLY ONLY WORKS FOR MODEL = 'AERI' OPTION
    
    Usage:
    print_output(params_fit, errors_fit)
    '''
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
def print_to_latex(params_fit, errors_fit, current_folder, fig_name, hpds):
    '''
    Prints results in latex table format
    
    Usage:
    params_to_print = print_to_latex(params_fit, errors_fit, current_folder, fig_name, labels, hpds)
    '''
    params_fit=params_fit[0]
    errors_fit=errors_fit[0]
    fname = current_folder+fig_name+ '.txt'
    
    if flag.model == 'aeri':
        names = ['Mstar', 'W', 't/tms', 'i', 'Dist', 'E(B-V)']
        if flag.include_rv:
            names = names + ['RV']
        if flag.binary_star:
            names = names + ['M2']
    if flag.model == 'acol':
        names = ['Mstar', 'W', 't/tms', 'logn0', 'Rd', 'n', 'i', 'Dist', 'E(B-V)']
        if flag.include_rv:
            names = names + ['RV']
        if flag.binary_star:
            names = names + ['M2']
    if flag.model == 'beatlas':
        names = ['Mstar', 'W', 'Sig0', 'n', 'i', 'Dist', 'E(B-V)']
        if flag.include_rv:
            names = names + ['RV']
        if flag.binary_star:
            names = names + ['M2']
    
    file1 = open(fname, 'w')
    L = [r"\begin{table}"+' \n',
        '\centering \n',
        r"\begin{tabular}{lll}"+' \n',
        '\hline \n',
        'Parameter  & Value & Type \\\ \n', 
        '\hline \n']
    file1.writelines(L)
    
    params_to_print = []
    #print(errors_fit[0][1])
    for i in range(len(params_fit)):
        params_to_print.append(names[i] + '= {0:.2f} +{1:.2f} -{2:.2f}'.format(params_fit[i], errors_fit[i][0], errors_fit[i][1]))
        file1.writelines(info.labels[i] + '& ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Free \\\ \n'.format(params_fit[i], errors_fit[i][0], errors_fit[i][1]))
    
    #if len(hpds[0]) > 1:
        
    
    Mstar = params_fit[0]
    Mstar_range = [Mstar + errors_fit[0][0], Mstar - errors_fit[0][1]]
    
    W = params_fit[1]
    W_range = [W + errors_fit[1][0], W - errors_fit[1][1]]
    
    tms = params_fit[2]
    tms_range = [tms + errors_fit[2][0], tms - errors_fit[2][1]]
    
    cosi = params_fit[3]
    cosi_range = [cosi + errors_fit[3][0], cosi - errors_fit[3][1]]
    
    oblat = 1 + 0.5*(W**2) # Rimulo 2017
    ob_max, ob_min = 1 + 0.5*(W_range[0]**2), 1 + 0.5*(W_range[1]**2)
    oblat_range = [ob_max, ob_min]
    
    Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014')
    
    Rpole_range = [0., 100.]
    logL_range = [0., 100000.]
    
    for mm in Mstar_range:
        for oo in oblat_range:
            for tt in tms_range:
                Rpolet, logLt, _ = geneva_interp_fast(mm, oo, tt, Zstr='014')
                if Rpolet > Rpole_range[0]:
                    Rpole_range[0] = Rpolet
                    #print('Rpole max is now = {}'.format(Rpole_range[0]))
                if Rpolet < Rpole_range[1]:
                    Rpole_range[1] = Rpolet
                    #print('Rpole min is now = {}'.format(Rpole_range[1]))
                if logLt > logL_range[0]:
                    logL_range[0] = logLt
                    #print('logL max is now = {}'.format(logL_range[0]))
                if logLt < logL_range[1]:
                    logL_range[1] = logLt
                    #print('logL min is now = {}'.format(logL_range[1]))
        
    beta_range = [beta(oblat_range[0], is_ob=True), beta(oblat_range[1], is_ob=True)]
        
    beta_par = beta(oblat, is_ob=True)
    
    Req = oblat*Rpole
    Req_max, Req_min = oblat_range[0]*Rpole_range[0], oblat_range[1]*Rpole_range[1]
    
    wcrit = np.sqrt(8. / 27. * G * Mstar * Msun / (Rpole * Rsun)**3)
    vsini = W * wcrit * (Req * Rsun) * np.sin(cosi * np.pi/180.) * 1e-5
    
    w_ = oblat2w(oblat)
    A_roche = 4.*np.pi* (Rpole*Rsun)**2 * (1. + 0.19444*w_**2 \
                      + 0.28053*w_**4 - 1.9014*w_**6 + 6.8298*w_**8 \
                     - 9.5002*w_**10 + 4.6631*w_**12)
                     
    Teff = ((10.**logL)*Lsun / sigma / A_roche)**0.25
    
    
    Teff_range = [0., 50000.]
    for oo in oblat_range:
        for rr in Rpole_range:
            for ll in logL_range:
                
                w_ = oblat2w(oo)
                A_roche = 4.*np.pi* (rr*Rsun)**2 * (1. + 0.19444*w_**2 \
                      + 0.28053*w_**4 - 1.9014*w_**6 + 6.8298*w_**8 \
                     - 9.5002*w_**10 + 4.6631*w_**12)
                     
                Teff_ = ((10.**ll)*Lsun / sigma / A_roche)**0.25
                if Teff_ > Teff_range[0]:
                    Teff_range[0] = Teff_
                    #print('Teff max is now = {}'.format(Teff_range[0]))
                if Teff_ < Teff_range[1]:
                    Teff_range[1] = Teff_
                    #print('Teff min is now = {}'.format(Teff_range[1]))

    
    
    
    vsini_range = [0., 10000.]
    for mm in Mstar_range:
        for rr in Rpole_range:
            for oo in oblat_range:
                for ww in W_range:
                    for ii in cosi_range: 
                        wcrit = np.sqrt(8. / 27. * G * mm * Msun / (rr * Rsun)**3)
                        #print(wcrit)
                        vsinit = ww * wcrit * (oo * rr * Rsun) * np.sin(ii * np.pi/180.) * 1e-5
                        if vsinit > vsini_range[0]:
                            vsini_range[0] = vsinit
                            #print('vsini max is now = {}'.format(vsini_range[0]))

                        if vsinit < vsini_range[1]:
                            vsini_range[1] = vsinit
                            #print('vsini min is now = {}'.format(vsini_range[1]))


    
    file1.writelines(r"$R_{\rm eq}/R_{\rm p}$"+' & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n'.format(oblat, oblat_range[0] - oblat, oblat- oblat_range[1]))
    params_to_print.append('Oblateness = {0:.2f} +{1:.2f} -{2:.2f}'.format(oblat, oblat_range[0] - oblat, oblat- oblat_range[1]))
    file1.writelines(r"$R_{\rm eq}\,[R_\odot]$"+' & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n'.format(Req, Req_max - Req, Req - Req_min))
    params_to_print.append('Equatorial radius = {0:.2f} +{1:.2f} -{2:.2f}'.format(Req, Req_max - Req, Req - Req_min))
    file1.writelines(r"$\log(L)\,[L_\odot]$"+' & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n'.format(logL, logL_range[0] - logL, logL - logL_range[1]))
    params_to_print.append('Log Luminosity  = {0:.2f} +{1:.2f} -{2:.2f}'.format(logL, logL_range[0] - logL, logL - logL_range[1]))
    file1.writelines(r"$\beta$"+' & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived \\\ \n'.format(beta_par, beta_range[1] - beta_par, beta_par - beta_range[0]))
    params_to_print.append('Beta  = {0:.2f} +{1:.2f} -{2:.2f}'.format(beta_par, beta_range[1] - beta_par, beta_par - beta_range[0]))
    file1.writelines(r"$v \sin i\,\rm[km/s]$"+' & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n'.format(vsini, vsini_range[0] - vsini, vsini - vsini_range[1]))
    params_to_print.append('vsini = {0:.2f} +{1:.2f} -{2:.2f}'.format(vsini, vsini_range[0] - vsini, vsini - vsini_range[1]))
    file1.writelines(r"$T_{\rm eff}$"+' & ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Derived  \\\ \n'.format(Teff, Teff_range[0]-Teff, Teff-Teff_range[1]))
    params_to_print.append('Teff = {0:.2f} +{1:.2f} -{2:.2f}'.format(Teff, Teff_range[0]-Teff, Teff-Teff_range[1]))
    
    L = ['\hline \n',
        '\end{tabular} \n'
        '\end{table} \n']
    
    file1.writelines(L)

    file1.close()

    params_print = ' \n'.join(map(str, params_to_print))
    
    return params_to_print
    

# ==============================================================================
def print_output_means(samples):
    """ Print the mean values obtained with the samples 
    CURRENTLY ONLY WORKS FOR MODEL = 'AERI' OPTION
    
    Usage: 
    print_output_means(samples)
    """
    
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
def plot_residuals(par, npy, current_folder, fig_name):
    '''
    Create residuals plot separated from the corner 
    For the SED and the lines
    
    Usage:
    plot_residuals(par, npy, current_folder, fig_name)

    '''
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')

    chain = np.load(npy)
    par_list=[]
    flat_samples = chain.reshape((-1, info.Ndim))
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        params = flat_samples[ind]
        par_list.append(params)
    

    
    if flag.SED:
        if flag.model == 'aeri':
            dist = par[4][0]
            ebv = par[5][0]
            if flag.include_rv:
                rv = par[6][0]
            else:
                rv=3.1
        elif flag.model == 'acol':
            dist = par[7][0]
            ebv = par[8][0]
            if flag.include_rv:
                rv = par[9][0]
            else:
                rv=3.1
        elif flag.model == 'beatlas':
            dist = par[5][0]
            ebv = par[6][0]
            if flag.include_rv:
                rv = par[7][0]
            else:
                rv=3.1
        #print(dist)
        u = np.where(info.lista_obs == 'UV')
        index = u[0][0]
          
        # Observations
        logF_UV = info.logF[index]
        flux_UV = 10.**logF_UV
        dlogF_UV = info.dlogF[index]
        lbd_UV = info.wave[index]
        
        dist = 1e3/dist
        norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
        uplim = info.dlogF[index] == 0.0
        keep = np.logical_not(uplim)
        dflux = dlogF_UV * flux_UV
        logF_list = np.zeros([len(par_list), len(logF_UV)])
        
        #par_list = chain[:, -1, :]
        #inds = np.random.randint(len(flat_samples), size=100)
        for i, params in enumerate(par_list):
            if flag.binary_star:
                logF_mod_UV_1_list = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
                logF_mod_UV_2_list = griddataBA(info.minfo, info.logF_grid[index], np.array([params[-1], 0.1, params[2], params[3]]), info.listpar, info.dims)
                logF_list[i] = np.log10(10.**np.array(logF_mod_UV_1_list) + 10.**np.array(logF_mod_UV_2_list))
            else:
                if flag.model != 'beatlas':
                    logF_list[i] = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim],
                                      info.listpar, info.dims)
                else:
                    logF_list[i] = griddataBAtlas(info.minfo, info.logF_grid[index], params[:-info.lim],
                                      info.listpar, info.dims, isig = info.dims["sig0"])
            #par_list[i] = params
        logF_list += np.log10(norma)
        
        
        fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Plot Models
        for i in range(len(par_list)):
            if flag.model == 'aeri':
                ebv_temp = par_list[i][5]
            elif flag.model == 'beatlas':
                ebv_temp = par_list[i][6]
            else:
                ebv_temp = par_list[i][8]
            F_temp = pyasl.unred(lbd_UV * 1e4, 10**logF_list[i],
                                 ebv=-1 * ebv_temp, R_V=rv)
            ax1.plot(lbd_UV, F_temp, color='gray', alpha=0.1, lw=0.6)

        
        ax2.plot(lbd_UV, (flux_UV - F_temp) / dflux, 'ks', ms = 5, alpha=0.2)
        #ax2.set_ylim(-10,10)
        if flag.votable or flag.data_table:
            #ax2.set_xscale('log')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            
        # Plot Data
        keep = np.where(flux_UV > 0) # avoid plot zero flux
        ax1.errorbar(lbd_UV[keep], flux_UV[keep], yerr=dflux[keep], ls='', marker='o',
                     alpha=0.5, ms=5, color='k', linewidth=1)

                     
        ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$')#, fontsize=12)
        ax1.set_ylabel(r'$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}\, \mu m^{-1}]}$')
                   #fontsize=12)	
        ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$')#, fontsize=12)

        plt.tight_layout()
        plt.savefig(current_folder + fig_name + '_new_residuals-UV' + '.png', dpi=100)
        plt.close()
    

    if flag.Ha:
        line = 'Ha'
        plot_line(line, par, par_list, current_folder, fig_name)
    if flag.Hb:
        line = 'Hb'
        plot_line(line, par, par_list, current_folder, fig_name)   
    if flag.Hd:
        line = 'Hd'
        plot_line(line, par, par_list, current_folder, fig_name)            
    if flag.Hg:
        line = 'Hg'
        plot_line(line, par, par_list, current_folder, fig_name)              
                           
    return    


# ==============================================================================
def plot_line(line, par, par_list, current_folder, fig_name):
    '''
    Plots residuals for the lines
    
    Usage:
    plot_line(line, par, par_list, current_folder, fig_name)
    '''
    # Finding position
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    u = np.where(info.lista_obs == line)
    index = u[0][0]

    # Observations
    #logF_line = logF[index]
    flux_line = info.logF[index]
    dflux_line = info.dlogF[index]
    #dflux_line = dlogF[index] * flux_line

    keep = np.where(flux_line > 0) # avoid plot zero flux
    lbd_line = info.wave[index]
    
    F_list = np.zeros([len(par_list), len(flux_line)])
    F_list_unnorm = np.zeros([len(par_list), len(flux_line)])
    #chi2 = np.zeros(len(F_list))
    for i, params in enumerate(par_list):
        if flag.binary_star:
            F_mod_line_1_list = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim],info.listpar, info.dims)
            F_mod_line_2_list = griddataBA(info.minfo, info.logF_grid[index], np.array([params[-1], 0.1,params[2], params[3]]), info.listpar, info.dims)
            F_list[i]  = linfit(lbd[index], F_mod_line_1_list + F_mod_line_2_list)
            #logF_list[i] = np.log(norm_spectra(lbd[index], F_list))
        else:
            F_list_unnorm[i] = griddataBA(info.minfo, info.logF_grid[index], params[:-info.lim], info.listpar, info.dims)
            F_list[i]  = linfit(info.wave[index], F_list_unnorm[i])
    
    
    #np.savetxt(current_folder + fig_name + '_new_residuals_' + line +'.dat', np.array([lbd_line, flux_mod_line]).T)
    
    # Plot
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [3, 1]})                              
    # Plot models
    for i in range(len(par_list)):
        ax1.plot(lbd_line, F_list[i], color='gray', alpha=0.1)
    ax1.errorbar(lbd_line, flux_line, yerr= dflux_line, ls='', marker='o', alpha=0.5, ms=5, color='blue', linewidth=1) 

    # Best fit
    #ax1.plot(lbd_line, flux_mod_line, color='red', ls='-', lw=3.5, alpha=0.4, label='Best fit \n chi2 = {0:.2f}'.format(chi2_line))
    
    ax1.set_ylabel('Normalized Flux')#,fontsize=14)
    ax1.set_xlim(min(lbd_line), max(lbd_line))
    #ax1.legend(loc='lower right')
    ax1.set_title(line)
    # Residuals
    ax2.plot(lbd_line, (flux_line - F_list[-1])/dflux_line, marker='o', alpha=0.5)
    
    ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$')#, fontsize=14)
    ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$')#, fontsize=14)
    ax2.set_xlim(min(lbd_line), max(lbd_line))
    plt.tight_layout()
    plt.savefig(current_folder + fig_name + '_new_residuals_' + line +'.png', dpi=100)
    plt.close()
    
    return
