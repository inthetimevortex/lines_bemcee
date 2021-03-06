from PyAstronomy import pyasl
import numpy as np
import matplotlib.pylab as plt
from be_theory import hfrac2tms, oblat2w
from utils import beta, geneva_interp_fast, griddataBAtlas, griddataBA, linfit, jy2cgs
from lines_reading import check_list, find_lim
import corner
from constants import G, Msun, Rsun, sigma, Lsun
import seaborn as sns
#import user_settings as flag
from scipy.special import erf
import sys
import importlib
mod_name = sys.argv[1]+'_'+'user_settings'
#print(sys.argv[1])
flag = importlib.import_module(mod_name)

sns.set_style("white", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)

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


def print_to_latex(params_fit, errors_fit, current_folder, fig_name, labels, hpds):
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
        file1.writelines(labels[i] + '& ${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$ & Free \\\ \n'.format(params_fit[i], errors_fit[i][0], errors_fit[i][1]))
    
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

def plot_residuals_new(par, Nwalk, Nmcmc, npy,
                       current_folder, fig_name):
    '''
    Create residuals plot separated from the corner 

    '''
        
    lim = find_lim()
    chain = np.load(npy)
    par_list=[]
    flat_samples = chain.reshape((-1, flag.Ndim))
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        params = flat_samples[ind]
        #if flag.model == 'aeri':
        #    params[3] = np.cos(params[3] * np.pi/180.)
        #elif flag.model == 'acol':
        #    params[1] = 1 + 0.5*(params[1]**2)
        #    params[2] = hfrac2tms(params[2], inverse=True)
        #    params[6] = np.cos(params[6] * np.pi/180.)
        par_list.append(params)
    

    
    if check_list(flag.lista_obs, 'UV'):
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
        #print(dist)
        u = np.where(flag.lista_obs == 'UV')
        index = u[0][0]
          
        # Observations
        logF_UV = flag.logF[index]
        flux_UV = 10.**logF_UV
        dlogF_UV = flag.dlogF[index]
        lbd_UV = flag.wave[index]
        
        dist = 1e3/dist
        norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
        uplim = flag.dlogF[index] == 0.0
        keep = np.logical_not(uplim)
        dflux = dlogF_UV * flux_UV
        logF_list = np.zeros([len(par_list), len(logF_UV)])
        
        #par_list = chain[:, -1, :]
        #inds = np.random.randint(len(flat_samples), size=100)
        for i, params in enumerate(par_list):
            if flag.binary_star:
                logF_mod_UV_1_list = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
                logF_mod_UV_2_list = griddataBA(flag.minfo, flag.logF_grid[index], np.array([params[-1], 0.1, params[2], params[3]]), flag.listpar, flag.dims)
                logF_list[i] = np.log10(10.**np.array(logF_mod_UV_1_list) + 10.**np.array(logF_mod_UV_2_list))
            else:
                logF_list[i] = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim],
                                      flag.listpar, flag.dims)
            #par_list[i] = params
        logF_list += np.log10(norma)
        
        
    
    
    
    
    
    
    #if flag.model == 'aeri':
    #    if check_list(flag.lista_obs, 'UV'):
    #
    #        if flag.include_rv and flag.binary_star:
    #            Mstar, W, tms, cosi, dist, ebv, rv, M2 = par
    #        elif flag.include_rv and not flag.binary_star:
    #            Mstar, W, tms, cosi, dist, ebv, rv = par
    #        elif flag.binary_star and not flag.include_rv:
    #            Mstar, W, tms, cosi, dist, ebv, M2 = par
    #            rv = 3.1
    #        else:
    #            Mstar, W, tms, cosi, dist, ebv = par
    #            rv = 3.1
    #    else:
    #        if flag.binary_star:
    #            Mstar, W, tms, cosi, M2 = par
    #        else:
    #            Mstar, W, tms, cosi = par
    #        rv=3.1
    #    
    #    if isinstance(W, list):
    #        oblat = 1 + 0.5*(np.array(W)**2) # Rimulo 2017
    #    else:
    #        oblat = 1 + 0.5*(W**2)
    #    
    #    par[3] = np.cos(par[3] * np.pi/180.)
    #        
    #
    #if flag.model == 'acol':
    #    if check_list(flag.lista_obs, 'UV'):
    #        if flag.include_rv and flag.binary_star:
    #            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, dist, ebv, rv, M2 = par
    #        elif flag.include_rv and not flag.binary_star:
    #            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, dist, ebv, rv = par
    #        elif flag.binary_star and not flag.include_rv:
    #            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, dist, ebv, M2 = par
    #            rv = 3.1
    #        else:
    #            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, dist, ebv = par
    #            rv=3.1
    #    else:
    #        if flag.binary_star:
    #            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi, M2 = par
    #        else:
    #            Mstar, oblat, Hfrac, Sig0, Rd, n, cosi = par
    #    oblat = 1 + 0.5*(oblat**2) # Rimulo 2017
    #    par[1] = 1 + 0.5*(par[1]**2)
    #    tms = hfrac2tms(Hfrac, inverse=True)
    #    par[2] = hfrac2tms(par[2], inverse=True)
    #    rv = 3.1
    #    par[6] = np.cos(par[6] * np.pi/180.)
    #
    #
    #cosi = np.cos(cosi*np.pi/180.)
    ##print(par)
    ##print(flag.listpar)
    #Rpole, logL, _ = geneva_interp_fast(Mstar, oblat, tms, Zstr='014')
    #
    ## ***
    #
    #
    #if check_list(flag.lista_obs, 'UV'):
	#	# Finding position
    #    u = np.where(flag.lista_obs == 'UV')
    #    index = u[0][0]
    #      
    #    # Observations
    #    logF_UV = flag.logF[index]
    #    flux_UV = 10.**logF_UV
    #    dlogF_UV = flag.dlogF[index]
    #    lbd_UV = flag.wave[index]
    #    
    #    dist = 1e3/dist
    #    norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
    #    uplim = flag.dlogF[index] == 0.0
    #    keep = np.logical_not(uplim) 
    #
    #        
    #    if flag.binary_star:
    #            logF_mod_UV_1 = griddataBA(flag.minfo, flag.logF_grid[index], par[:-lim], flag.listpar, flag.dims)
    #            logF_mod_UV_2 = griddataBA(flag.minfo, flag.logF_grid[index], np.array([M2, 0.1, par[2], par[3]]), flag.listpar, flag.dims)
    #            logF_mod_UV = np.log10(10.**logF_mod_UV_1 + 10**logF_mod_UV_2)
    #        
    #    else:
    #        logF_mod_UV = griddataBA(flag.minfo, flag.logF_grid[index], par[:-lim], flag.listpar, flag.dims)
    #    
    #
    #    # convert to physical units
    #    logF_mod_UV += np.log10(norma)
    #    
    #    flux_mod_UV = 10.**logF_mod_UV
    #    dflux = dlogF_UV * flux_UV
    #    
    #    flux_mod_UV = pyasl.unred(lbd_UV * 1e4, flux_mod_UV, ebv=-1 * ebv, R_V=rv)
    #    
    #    rms = np.array([1e-3*0.1, 1e-3*0.1, 1e-3*0.1])
    #
    #    upper_lim = jy2cgs(10**logF_UV[uplim], lbd_UV[uplim], inverse=True)
    #    mod_upper = jy2cgs(10**logF_mod_UV[uplim], lbd_UV[uplim], inverse=True)
    #    
    #    chi2_UV = np.sum(((logF_UV[keep] - logF_mod_UV[keep])**2. / (dlogF_UV[keep])**2.)) - 2. * np.sum( np.log( (np.pi/2.)**0.5 * rms * (1. + erf(((upper_lim- mod_upper)/((2**0.5)*rms))))))
    #    N_UV = len(logF_UV)
    #    chi2_UV = chi2_UV/N_UV
    #    
    #    logF_list = np.zeros([len(par_list), len(logF_mod_UV)])
    #    chi2 = np.zeros(len(logF_list))
    #    for i in range(len(par_list)):
    #        if flag.binary_star:
    #            logF_mod_UV_1_list = griddataBA(flag.minfo, flag.logF_grid[index], par_list[i, :-lim], flag.listpar, flag.dims)
    #            logF_mod_UV_2_list = griddataBA(flag.minfo, flag.logF_grid[index], np.array([par_list[i, -1], 0.1, par_list[i, 2], par_list[i, 3]]), flag.listpar, flag.dims)
    #            logF_list[i] = np.log10(10.**np.array(logF_mod_UV_1_list) + 10.**np.array(logF_mod_UV_2_list))
    #        else:
    #            logF_list[i] = griddataBA(flag.minfo, flag.logF_grid[index], par_list[i, :-lim],
    #                                  flag.listpar, flag.dims)
    #    
    #    
    #    logF_list += np.log10(norma)

        
        
        #for j in range(len(logF_list)):
        #    chi2[j] = np.sum((logF_UV[keep] - logF_list[j][keep])**2 / (dlogF_UV[keep])**2)
        
        fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})

        # Plot Models
        for i in range(len(par_list)):
            if flag.model == 'aeri':
                ebv_temp = par_list[i][5]
            else:
                ebv_temp = par_list[i][8]
            F_temp = pyasl.unred(lbd_UV * 1e4, 10**logF_list[i],
                                 ebv=-1 * ebv_temp, R_V=rv)
            ax1.plot(lbd_UV, F_temp, color='gray', alpha=0.1)

        
        
        # Applying reddening to the best model
 
        # Best fit
        #ax1.plot(lbd_UV, flux_mod_UV, color='red', ls='-', lw=3.5, alpha=0.4,
        #         label='Best fit \n chi2 = {0:.2f}'.format(chi2_UV))

        #ax2.plot(lbd_UV, (flux_UV - flux_mod_UV) / dflux, 'bs', alpha=0.2)
        ax2.plot(lbd_UV, (flux_UV - F_temp) / dflux, 'bs', alpha=0.2)
        ax2.set_ylim(-10,10)
        if flag.votable:
            ax2.set_xscale('log')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            
        # Plot Data
        keep = np.where(flux_UV > 0) # avoid plot zero flux
        ax1.errorbar(lbd_UV[keep], flux_UV[keep], yerr=dflux[keep], ls='', marker='o',
                     alpha=0.5, ms=5, color='blue', linewidth=1)

                     
        ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=14)
        ax1.set_ylabel(r'$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2}\, \mu m^{-1}]}$',
                   fontsize=14)	
        ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=14)
        #plt.tick_params(labelbottom='off')
        #ax1.set_xlim(min(lbd_UV), max(lbd_UV))
        #ax2.set_xlim(min(lbd_UV), max(lbd_UV))
        #plt.tick_params(direction='in', length=6, width=2, colors='gray',
        #    which='both')
        #ax1.legend(loc='upper right')
        #ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(current_folder + fig_name + '_new_residuals-UV' + '.png', dpi=100)
        plt.close()
    

    if check_list(flag.lista_obs, 'Ha'):
        # Finding position
        line = 'Ha'
        plot_line(line, par, par_list, current_folder, fig_name)
    if check_list(flag.lista_obs, 'Hb'):
        line = 'Hb'
        plot_line(line, par, par_list, current_folder, fig_name)   
                       
    if check_list(flag.lista_obs, 'Hd'):
        line = 'Hd'
        plot_line(line, par, par_list, current_folder, fig_name)            
    if check_list(flag.lista_obs, 'Hg'):
        line = 'Hg'
        plot_line(line, par, par_list, current_folder, fig_name)              
                           
        


def plot_line(line, par, par_list, current_folder, fig_name):
    # Finding position
    u = np.where(flag.lista_obs == line)
    index = u[0][0]
    # Finding the corresponding flag.model (interpolation)
    lim = find_lim()
    
    #if check_list(lista_obs, 'UV'):
    #    logF_mod_line = griddataBA(minfo, logF_grid[index], par[:-lim] ,listpar, dims)
    #else:
    #    logF_mod_line = griddataBA(minfo, logF_grid[index], par ,listpar, dims)
        
    #if flag.binary_star:
    #    F_mod_line_1 = griddataBA(flag.minfo, flag.logF_grid[index], par[:-lim], flag.listpar, flag.dims)
    #    F_mod_line_2 = griddataBA(flag.minfo, flag.logF_grid[index], np.array([par[-1], 0.1, par[2], par[3]]), flag.listpar, flag.dims)
    #    flux_mod_line = linfit(lbd[index], F_mod_line_1 + F_mod_line_2)
    #    #logF_mod_line = np.log10(F_mod_line)
    #    #logF_mod_Ha = np.log(norm_spectra(lbd[index], F_mod_Ha_unnormed))
    #else:
    #    F_mod_line_unnorm = griddataBA(flag.minfo, flag.logF_grid[index], par[:-lim], flag.listpar, flag.dims)
    #    flux_mod_line = linfit(flag.wave[index], F_mod_line_unnorm)
    #    #logF_mod_Ha = np.log10(F_mod_Ha)
    #    #logF_mod_line = np.log(norm_spectra(lbd[index], F_mod_line_unnormed))
    #    #logF_mod_line = np.log10(10.**logF_mod_line_1 + 10.**logF_mod_line_2)

        
    #logF_mod_line = griddataBA(minfo, logF_grid[index], par[:-lim],listpar, dims)
    #flux_mod_line = 10.**logF_mod_line
    # Observations
    #logF_line = logF[index]
    flux_line = flag.logF[index]
    dflux_line = flag.dlogF[index]
    #dflux_line = dlogF[index] * flux_line

    keep = np.where(flux_line > 0) # avoid plot zero flux
    lbd_line = flag.wave[index]
    
    F_list = np.zeros([len(par_list), len(flux_line)])
    F_list_unnorm = np.zeros([len(par_list), len(flux_line)])
    #chi2 = np.zeros(len(F_list))
    for i, params in enumerate(par_list):
        if flag.binary_star:
            F_mod_line_1_list = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim],flag.listpar, flag.dims)
            F_mod_line_2_list = griddataBA(flag.minfo, flag.logF_grid[index], np.array([params[-1], 0.1,params[2], params[3]]), flag.listpar, flag.dims)
            F_list[i]  = linfit(lbd[index], F_mod_line_1_list + F_mod_line_2_list)
            #logF_list[i] = np.log(norm_spectra(lbd[index], F_list))
        else:
            F_list_unnorm[i] = griddataBA(flag.minfo, flag.logF_grid[index], params[:-lim], flag.listpar, flag.dims)
            F_list[i]  = linfit(flag.wave[index], F_list_unnorm[i])
            
    #logF_list[i]= np.log10(flux_mod_line_list)
    #chi2_line = np.sum((flux_line[keep] - flux_mod_line[keep])**2 / (dflux_line[keep])**2.)
    #N_line = len(flux_line[keep])
    #chi2_line = chi2_line/N_line
    # Data

    #print(len(lbd), lbd)
    
    
    #np.savetxt(current_folder + fig_name + '_new_residuals_' + line +'.dat', np.array([lbd_line, flux_mod_line]).T)
    
    # Plot
    fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})                              
    # Plot models
    for i in range(len(par_list)):
        ax1.plot(lbd_line, F_list[i], color='gray', alpha=0.1)
    ax1.errorbar(lbd_line, flux_line, yerr= dflux_line, ls='', marker='o', alpha=0.5, ms=5, color='blue', linewidth=1) 

    # Best fit
    #ax1.plot(lbd_line, flux_mod_line, color='red', ls='-', lw=3.5, alpha=0.4, label='Best fit \n chi2 = {0:.2f}'.format(chi2_line))
    
    ax1.set_ylabel('Normalized Flux',fontsize=14)
    ax1.set_xlim(min(lbd_line), max(lbd_line))
    #ax1.legend(loc='lower right')
    ax1.set_title(line)
    # Residuals
    ax2.plot(lbd_line, (flux_line - F_list[-1])/dflux_line, marker='o', alpha=0.5)
    
    ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=14)
    ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=14)
    ax2.set_xlim(min(lbd_line), max(lbd_line))
    plt.tight_layout()
    plt.savefig(current_folder + fig_name + '_new_residuals_' + line +'.png', dpi=100)
    plt.close()
    
