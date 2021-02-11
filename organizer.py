#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  user_settings.py
#  
#  Copyright 2020 Amanda Rubio <amanda.rubio@usp.br>
#  


import numpy as np
import random
import importlib
from operator import is_not
from functools import partial
from __init__ import mod_name
flag = importlib.import_module(mod_name)


from bemcee import read_models, create_tag, create_list, read_stellar_prior, read_observables

# ==============================================================================
def read_stars(stars_table):
    ''' Reads info in the star.txt file in data/star folder
        
        Usage
        stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs, 
        list_pre_ebmv, incl0, sig_incl0 = read_stars(star)
    '''
    typ = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    file_data = flag.folder_data + stars_table

    a = np.genfromtxt(file_data, usecols=typ, unpack=True,
                      comments='#',
                      dtype={'names': ('star', 'plx', 'sig_plx', 'vsini',
                                       'sig_vsini', 'pre_ebmv', 'inc', 'sinc', 'lbd_range'),
                             'formats': ('S9', 'f2', 'f2', 'f4',
                                         'f4', 'f4', 'f4', 'f4', 
                                         'U40')})

    stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs,\
        list_pre_ebmv, incl0, sig_incl0,  lbd_range =\
        a['star'], a['plx'], a['sig_plx'], a['vsini'], a['sig_vsini'],\
        a['pre_ebmv'], a['inc'], a['sinc'], a['lbd_range']

    if np.size(stars) == 1:
        stars = stars.astype('str')
    else:
        for i in range(len(stars)):
            stars[i] = stars[i].astype('str')

    return list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs,\
        list_pre_ebmv, incl0, sig_incl0

# ==============================================================================
def find_lim():
    ''' Defines the value of "lim", to only use the model params in the 
    interpolation
    
    Usage:
    lim = find_lim()
        
    '''
    if flag.SED:
        if flag.include_rv and not flag.binary_star:
            lim = 3
        elif flag.include_rv and flag.binary_star:
            lim = 4
        elif flag.binary_star and not flag.include_rv:
            lim = 3
        else:
            lim = 2
    else:
        if flag.binary_star:
            lim = 1
        else:
            if flag.model == 'aeri':
                lim = -4
            else:
                lim= -7
    
    return lim

# ==============================================================================
def set_ranges(star, lista_obs, listpar):
    ''' Defines the ranges and Ndim
    
    Usage:
    ranges, Ndim = set_ranges(star, lista_obs, listpar)
        
    '''
    print(75 * '=')


    print('\nRunning star: %s\n' % star)
    print(75 * '=')
    

    if flag.SED or flag.normal_spectra is False:
        if flag.include_rv:
            ebmv, rv = [[0.0, 0.8], [1., 5.8]]
        else:
            rv = 3.1
            ebmv, rv = [[0.0, 1.8], None]
        
        dist_min = file_plx - flag.Nsigma_dis * file_dplx
        dist_max = file_plx + flag.Nsigma_dis * file_dplx
     
        
        addlistpar = [ebmv, [dist_min, dist_max], rv]
        addlistpar = list(filter(partial(is_not, None), addlistpar))
        

    
        if flag.model == 'aeri' or flag.model == 'befavor':
            
            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                                [listpar[1][0], listpar[1][-1]],
                                [listpar[2][0], listpar[2][-1]],
                                [listpar[3][0], listpar[3][-1]],
                                [dist_min, dist_max],
                                [ebmv[0], ebmv[-1]]])

                                            
        elif flag.model == 'aara' or flag.model == 'acol':

            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                            [listpar[1][0], listpar[1][-1]],
                            [listpar[2][0], listpar[2][-1]],
                            [listpar[3][0], listpar[3][-1]],
                            [listpar[4][0], listpar[4][-1]],
                            [listpar[5][0], listpar[5][-1]],
                            [listpar[6][0], listpar[6][-1]],
                            [dist_min, dist_max],
                            [ebmv[0], ebmv[-1]]])

            
        elif flag.model == 'beatlas':

            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                            [listpar[1][0], listpar[1][-1]],
                            [listpar[2][0], listpar[2][-1]],
                            [listpar[3][0], listpar[3][-1]],
                            [listpar[4][0], listpar[4][-1]],
                            [dist_min, dist_max],
                            [ebmv[0], ebmv[-1]]])
        
        if flag.include_rv:
            ranges = np.concatenate([ranges, [rv]])
            
        if flag.binary_star:
            M2 = [listpar[0][0], listpar[0][-1]]
            ranges = np.concatenate([ranges, [M2]])
            


    else:
        if flag.model == 'aeri' or flag.model == 'befavor':
            
            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                               [listpar[1][0], listpar[1][-1]],
                               [listpar[2][0], listpar[2][-1]],
                               [listpar[3][0], listpar[3][-1]]])

                           
        elif flag.model == 'acol' or flag.model == 'aara':

            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                               [listpar[1][0], listpar[1][-1]],
                               [listpar[2][0], listpar[2][-1]],
                               [listpar[3][0], listpar[3][-1]],
                               [listpar[4][0], listpar[4][-1]],
                               [listpar[5][0], listpar[5][-1]],
                               [listpar[6][0], listpar[6][-1]]])

                
        elif flag.model == 'beatlas':
            
            ranges = np.array([[listpar[0][0], listpar[0][-1]],
                            [listpar[1][0], listpar[1][-1]],
                            [listpar[2][0], listpar[2][-1]],
                            [listpar[3][0], listpar[3][-1]],
                            [listpar[4][0], listpar[4][-1]]])
        
        if flag.binary_star:
            M2 = [listpar[0][0], listpar[0][-1]]
            ranges = np.concatenate([ranges, [M2]])
    
    if flag.box_W:
        if flag.box_W_max == 'max':
            ranges[1][0] = flag.box_W_min
        elif flag.box_W_min == 'min':
            ranges[1][1] = flag.box_W_max
        else:
            ranges[1][0], ranges[1][1] = flag.box_W_min, flag.box_W_max
    
    
        if flag.box_i:
            if flag.model == 'aeri':
                indx = 3
            elif flag.model == 'acol':
                indx = 6
            if flag.box_i_max == 'max':
                ranges[indx][0] = flag.box_i_min
            elif flag.box_i_min == 'min':
                ranges[indx][1] = flag.box_i_max
            else:
                ranges[indx][0], ranges[indx][1] = flag.box_i_min, flag.box_i_max
    
    Ndim = len(ranges)

    return ranges, Ndim






###########################################################################
#
###########################################################################

list_of_stars = flag.stars + '/' + flag.stars + '.txt'

file_plx, file_dplx, file_vsini, file_dvsini,\
file_ebmv, file_incl, file_dincl = read_stars(list_of_stars)

lista_obs = create_list() 

lim = find_lim()

tags = create_tag()

print(tags)
print(flag.lbd_range)

ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_models(lista_obs)

grid_priors, pdf_priors = read_stellar_prior()

ranges, Ndim = set_ranges(flag.stars, lista_obs, listpar)


logF, dlogF, logF_grid, wave, box_lim =\
        read_observables(models, lbdarr, lista_obs)



if flag.model == 'aeri':
    if UV:
        labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
              r'$i[\mathrm{^o}]$', r'$\pi\,[mas]$', r'E(B-V)']
        if flag.include_rv is True:
            labels = labels + [r'$R_\mathrm{V}$']

    else:
        labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
              r'$i[\mathrm{^o}]$']
    labels2 = labels
    
if flag.model == 'acol':
    if flag.SED:
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
if flag.model == 'beatlas':
    labels = [r'$M\,[\mathrm{M_\odot}]$', r'$W$',
                r'$\Sigma_0 \, [\mathrm{g/cm^{-2}}]$',
                r'$n$', r'$i[\mathrm{^o}]$', r'$\pi\,[\mathrm{pc}]$',
                r'E(B-V)']
    labels2 = [r'$M$', r'$W$',
                r'$\\Sigma_0 $',
                r'$R_\mathrm{D}$',
                r'$n$', r'$i$', r'$\pi$',
                r'E(B-V)']                
    if flag.include_rv is True:
            labels = labels + [r'$R_\mathrm{V}$']
            labels2 = labels2 + [r'$R_\mathrm{V}$']
        
if flag.binary_star:
        labels = labels + [r'$M2\,[M_\odot]$']
        labels2 = labels2 + [r'$M2\,[M_\odot]$']


#############################################################################################################################
if flag.corner_color == 'random' or flag.corner_color == '':
    corner_color = random.choice(['blue', 'dark blue', 'teal', 'green', 'yellow', 'orange', 'red', 'purple', 'violet', 'pink'])
else:
    corner_color = flag.corner_color
#############################################################################################################################

if corner_color == 'blue':
    color='xkcd:cornflower'
    color_hist='xkcd:powder blue'
    color_dens='xkcd:clear blue'
    
elif corner_color == 'dark blue':
    color='xkcd:dark teal'
    color_hist='xkcd:pale sky blue'
    color_dens='xkcd:ocean'
    
elif corner_color == 'teal':
    color='xkcd:dark sea green'
    color_hist='xkcd:pale aqua'
    color_dens='xkcd:seafoam blue'
    
elif corner_color == 'green':
    color='xkcd:forest green'
    color_hist='xkcd:light grey green'
    color_dens='xkcd:grass green'
    
elif corner_color == 'yellow':
    color='xkcd:sandstone'
    color_hist='xkcd:pale gold'
    color_dens='xkcd:sunflower'
    
elif corner_color == 'orange':
    color='xkcd:cinnamon'
    color_hist='xkcd:light peach'
    color_dens='xkcd:bright orange'
    
elif corner_color == 'red':
    color='xkcd:deep red'
    color_hist='xkcd:salmon'
    color_dens='xkcd:reddish'
    
elif corner_color == 'purple':
    color='xkcd:medium purple'
    color_hist='xkcd:soft purple'
    color_dens='xkcd:plum purple'
    
elif corner_color == 'violet':
    color='xkcd:purpley'
    color_hist='xkcd:pale violet'
    color_dens='xkcd:blue violet'
    
elif corner_color == 'pink':
    color='xkcd:pinky'
    color_hist='xkcd:light pink'
    color_dens='xkcd:pink red'


truth_color='k'


