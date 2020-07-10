#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  redo_plots.py
#  
#  Copyright 2020 Amanda Rubio <amanda@Pakkun>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


from PyAstronomy import pyasl
import numpy as np
import matplotlib.pylab as plt
from be_theory import hfrac2tms
from utils import beta, geneva_interp_fast, griddataBAtlas, griddataBA, lineProf
from lines_reading import check_list, create_list, read_star_info, read_BAphot2_xdr, read_observables, create_tag
import corner
import corner_HDR
from constants import G, Msun, Rsun
import seaborn as sns
from pymc3.stats import hpd
import user_settings as flag

lines_dict = {
'Ha':6562.801,
'Hb':4861.363,
'Hd':4101.74,
'Hg':4340.462 }

sns.set_style("white", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})


lista_obs = create_list()
tag = create_tag()
star= np.copy(flag.stars)
ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_BAphot2_xdr(lista_obs)

ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs, Ndim = read_star_info(star, lista_obs, listpar)


logF, dlogF, logF_grid, wave, box_lim = read_observables(models, lbdarr, lista_obs)



Nwalk = 300
nint_mcmc = 1000 
af = 0.33

current_folder = str(flag.folder_fig) + str(flag.stars) + '/'
fig_name = 'Walkers_' + np.str(Nwalk) + '_Nmcmc_' +\
            np.str(nint_mcmc) + '_af_' + str(af) + '_a_' +\
            str(flag.a_parameter) + tag
file_npy = flag.folder_fig + str(flag.stars) + '/' + 'Walkers_' +\
            str(Nwalk) + '_Nmcmc_' + str(nint_mcmc) +\
            '_af_' + str(af) + '_a_' + str(flag.a_parameter) +\
            tag + ".npy"

chain = np.load(file_npy)

flatchain_1 = chain.reshape((-1, Ndim))
samples = np.copy(flatchain_1)


for i in range(len(samples)):
    if flag.model == 'aeri':
        # Converting angles to degrees
        samples[i][3] = (np.arccos(samples[i][3])) * (180. / np.pi)

if flag.model == 'aeri':
    ranges[3] = np.array([0., 90.])
#    #ranges[3] = np.array([ranges[3][1], ranges[3][0]])
    
best_pars = []
best_errs = []

for i in range(Ndim):
    #mode_val = mode1(np.round(samples[:,i], decimals=2))
    qvalues = hpd(samples[:,i], alpha=0.32)
    cut = samples[samples[:,i] > qvalues[0], i]
    cut = samples[samples[:,i] < qvalues[1], i]
    median_val = np.median(cut)

    best_errs.append([qvalues[1] - median_val, median_val - qvalues[0]])
    best_pars.append(median_val)
    

if flag.model == 'aeri':
    if check_list(lista_obs, 'UV'):
        labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
                r'$i[\mathrm{^o}]$', r'$d\,[mas]$', r'E(B-V)']
        if flag.include_rv is True:
            labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
                r'$i[\mathrm{^o}]$', r'$d\,[mas]$', r'E(B-V)',
                r'$R_\mathrm{V}$']
    else:
        labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
                r'$i[\mathrm{^o}]$']




def plot_residuals_new(par, lbd, logF, dlogF, minfo, listpar, lbdarr, logF_grid,
                       isig, dims, Nwalk, Nmcmc, npy, box_lim, lista_obs,
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
        

        #print(cosi)
        cosi = np.cos(cosi * np.pi/180.)
        par[3] = np.cos(par[3] * np.pi/180.)
        #print(cosi)

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
            
            dist = 1e3/dist
            norma = (10. / dist)**2  # (Lstar*Lsun) / (4. * pi * (dist*pc)**2)
            uplim = dlogF[index] == 0
            keep = np.logical_not(uplim)  
            
            
            # convert to physical units
            logF_mod_UV += np.log10(norma)
            
            flux_mod_UV = 10.**logF_mod_UV
            dflux = dlogF_UV * flux_UV
            
            flux_mod_UV = pyasl.unred(lbd_UV * 1e4, flux_mod_UV, ebv=-1 * ebv, R_V=rv)

            
            chi2_UV = np.sum((flux_UV - flux_mod_UV)**2. / dflux**2.)
            N_UV = len(logF_UV)
            chi2_UV = chi2_UV/N_UV
            logF_list = np.zeros([len(par_list), len(logF_mod_UV)])
            chi2 = np.zeros(len(logF_list))
            for i in range(len(par_list)):
                logF_list[i] = griddataBA(minfo, logF_grid[index], par_list[i, :-lim],
                                          listpar, dims)
            # Plot
            #fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
            
            logF_list += np.log10(norma)

            bottom, left = 0.84, 0.51 #0.80, 0.48  # 0.75, 0.48
            width, height = 0.96 - left, 0.97 - bottom
            ax1 = plt.axes([left, bottom, width, height])
            
            for j in range(len(logF_list)):
                chi2[j] = np.sum((logF_UV[keep] - logF_list[j][keep])**2 / (dlogF_UV[keep])**2)
            
            
        
            # Plot Models
            for i in range(len(par_list)):
                
                ebv_temp = np.copy(ebv)
                F_temp = pyasl.unred(lbd_UV * 1e4, 10**logF_list[i],
                                     ebv=-1 * ebv_temp, R_V=rv)
                plt.plot(lbd_UV, F_temp, color='gray', alpha=0.1)
                # Residuals  --- desta forma plota os residuos de todos os modelos, mas acho que nao eh o que quero  
                #ax2.plot(lbd_UV, (flux_UV - F_temp) / dflux, 'bs', alpha=0.2)
            

            # Applying reddening to the best model
 
            # Best fit
            ax1.plot(lbd_UV, flux_mod_UV, color='red', ls='-', lw=3.5, alpha=0.4,
                     label=r'Best Fit' '\n' '$\chi^2$ = {0:.2f}'.format(chi2_UV))
            #ax2.plot(lbd_UV, (flux_UV - flux_mod_UV) / dflux, 'bs', alpha=0.2)
            #ax2.set_ylim(-10,10)
            # Plot Data
            keep = np.where(flux_UV > 0) # avoid plot zero flux
            ax1.errorbar(lbd_UV[keep], flux_UV[keep], yerr=dflux[keep], ls='', marker='o',
                         alpha=0.5, ms=5, color='blue', linewidth=1)

                         
            ax1.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=14)
            ax1.set_ylabel(r'$F_{\lambda}\,\mathrm{[erg\, s^{-1}\, cm^{-2} \mu m^{-1}}$',
                       fontsize=14)	
            #ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=14)
            #plt.tick_params(labelbottom='off')
            ax1.set_xlim(min(lbd_UV), max(lbd_UV))
            #ax2.set_xlim(min(lbd_UV), max(lbd_UV))
            #plt.tick_params(direction='in', length=6, width=2, colors='gray',
            #    which='both')
            ax1.legend(loc='upper right', fontsize=13)
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            #plt.tight_layout()
            #plt.savefig(current_folder + fig_name + '_new_residuals-UV-REDONE' + '.png', dpi=100)
            #plt.close()
        

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
    dflux_line = dlogF[index] * flux_line

    keep = np.where(flux_line > 0) # avoid plot zero flux
    lbd_line = lbd[index]
    
    logF_list = np.zeros([len(par_list), len(logF_mod_line)])
    chi2 = np.zeros(len(logF_list))
    for i in range(len(par_list)):
        if check_list(lista_obs, 'UV') is False:
            logF_list[i] = griddataBA(minfo, logF_grid[index], par_list[i],
                                  listpar, dims)
        else:
            logF_list[i] = griddataBA(minfo, logF_grid[index], par_list[i, :-lim],
                                  listpar, dims)
                                  
    chi2_line = np.sum((flux_line[keep] - flux_mod_line[keep])**2 / (dflux_line[keep])**2.)
    N_line = len(logF_line[keep])
    chi2_line = chi2_line/N_line
    #np.savetxt(current_folder + fig_name + '_new_residuals_' + line +'.dat', np.array([lbd_line, flux_mod_line]).T)
    
    bottom, left = 0.65, 0.51 #0.80, 0.48  # 0.75, 0.48
    width, height = 0.96 - left, 0.78 - bottom
    ax2 = plt.axes([left, bottom, width, height])
    lbc = lines_dict[line]
    print(lbc)
    # Plot
    #fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})                              
    # Plot models
    #ax2 = plt.subplot(211)
    vel, fx = lineProf(lbd_line, flux_line, hwidth=2500., lbc=lbc*1e-4)
    for i in range(len(par_list)):
        ax2.plot(vel, 10**logF_list[i], color='gray', alpha=0.1)
    ax2.errorbar(vel[keep], flux_line[keep], yerr= dflux_line[keep], ls='', marker='o', alpha=0.5, ms=5, color='blue', linewidth=1) 

    # Best fit
    ax2.plot(vel, flux_mod_line, color='red', ls='-', lw=3.5, alpha=0.4, label=r'Best Fit' '\n' '$\chi^2$ = {0:.2f}'.format(chi2_line))
    
    ax2.set_ylabel('Normalized Flux',fontsize=14)
    ax2.set_xlim(min(vel), max(vel))
    ax2.legend(loc='lower right', fontsize=13)
    ax2.set_title(line)
    ax2.set_xlabel('Vel [km/s]', fontsize=14)
    # Residuals
    #ax2.plot(lbd_line[keep], (flux_line[keep] - flux_mod_line[keep])/dflux_line[keep], marker='o', alpha=0.5)
    
    #ax2.set_ylabel('$(F-F_\mathrm{m})/\sigma$', fontsize=14)
    #ax2.set_xlabel('$\lambda\,\mathrm{[\mu m]}$', fontsize=14)
    #ax2.set_xlim(min(lbd_line), max(lbd_line))
    #plt.tight_layout()
    #plt.savefig(current_folder + fig_name + '_new_residuals_REDONE' + line +'.png', dpi=100)
    #plt.close()


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

truth_color = 'k'

corner_HDR.corner(samples, labels=labels, labels2=labels, range=ranges, quantiles=None, plot_contours=True, show_titles=True, 
                title_kwargs={'fontsize': 15}, label_kwargs={'fontsize': 19}, truths = best_pars, hdr=True,
                truth_color=truth_color, color=color, color_hist=color_hist, color_dens=color_dens, 
                smooth=1, plot_datapoints=False, fill_contours=True, combined=True)

plot_residuals_new(best_pars, wave, logF, dlogF, minfo,
                           listpar, lbdarr, logF_grid, isig, dims,
                           Nwalk, nint_mcmc, file_npy, box_lim,
                           lista_obs, current_folder, fig_name)
                           
                           
plt.savefig(current_folder + fig_name + '_REDONE.png', dpi=100)
