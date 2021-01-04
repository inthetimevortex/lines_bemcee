#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  user_settings.py
#  
#  Copyright 2020 Amanda Rubio <amanda.rubio@usp.br>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,

#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import numpy as np
import random
from lines_reading import read_models, create_tag, create_list, read_stellar_prior, read_star_info, read_observables

# ==============================================================================
# General Options

a_parameter =       2.0   # Set internal steps of each walker
extension =         '.png'  # Figure flag.extension to be saved
include_rv =        False  # If False: fix Rv = 3.1, else Rv will be inferead
af_filter =         False  # Remove walkers outside the range 0.2 < af < 0.5
long_process =      True  # Run with few walkers or many?
list_of_stars =     'HD37795'  # Star name
Nsigma_dis =        2.  # Set the range of values for the distance
model =             'acol'  # 'beatlas', 'befavor', 'aara', or 'acol'


binary_star = False

folder_data = '../data/'
folder_fig = '../figures/'
folder_defs = '../defs/'
folder_tables = '../tables/'
folder_models = '../models/'


vsini_prior =   True # Uses a gaussian vsini prior
dist_prior =    True # Uses a gaussian distance prior

box_W =         False # Constrain the W lower limit, not actual a prior, but restrain the grid
box_W_min, box_W_max = [0.6, 'max']

box_i =         False # Constrain the i limits, not actual a prior, but restrain the grid
box_i_min, box_i_max = [np.cos(50.*np.pi/180.), 'max']

incl_prior =    False # Uses a gaussian inclination prior 

normal_spectra =    True # True if the spectra is normalized (for lines), distance and e(b-v) are not computed
only_wings =        False # Run emcee with only the wings
only_centerline =   False # Run emcee with only the center of the line
Sigma_Clip =        True # If you want telluric lines/outilier points removed
remove_partHa =     False # Remove a lbd interval in flag.Halpha that has too much absorption (in the wings)

# Line and continuum combination

UV =            True
iue =           True
votable =       True
data_table=     False
Ha =            True
Hb =            False
Hd =            False
Hg =            False

   
corner_color = '' # OPTIONS ARE: blue, dark blue, teal, green, yellow, orange, red, purple, violet, pink.
                  # IF YOU DO NOT CHOOSE A COLOR, A RANDOM ONE WILL BE SELECTED

# Plot options
compare_results = False # Plot the reference Achernar values in the corner (only for model aeri)


# ------------------------------------------------------------------------------
# if True: M, Age, Oblat are set as priors for the choosen input, npy_star
stellar_prior = False
npy_star = 'Walkers_500_Nmcmc_1000_af_0.28_a_1.4_rv_false+hip.npy'

# ------------------------------------------------------------------------------
# Alphacrucis' options
acrux = True # If True, it will run in Nproc processors in the cluster
Nproc = 24  # Number of processors to be used in the cluster

# ==============================================================================

# ==============================================================================
# Read the list of flag.stars and takes the prior information about them from the stars_table txt file
def read_stars(stars_table):

    typ = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    file_data = folder_data + stars_table

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

    return stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs,\
        list_pre_ebmv, incl0, sig_incl0, lbd_range

# ==============================================================================
# Reading the list of stars

list_of_stars = list_of_stars + '/' + list_of_stars + '.txt'
stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs,\
    list_pre_ebmv, incl0, sig_incl0, lbd_range =\
    read_stars(list_of_stars)

lbd_range = 'UV+VIS+NIR+MIR+FIR+MICROW+RADIO'

#############################################################################################################################
if corner_color == 'random' or corner_color == '':
    corner_color = random.choice(['blue', 'dark blue', 'teal', 'green', 'yellow', 'orange', 'red', 'purple', 'violet', 'pink'])
#############################################################################################################################


lista_obs = create_list() 

tags = create_tag()

print(tags)
print(lbd_range)
ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_models(lista_obs)

grid_priors, pdf_priors = read_stellar_prior()


ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
    Ndim = read_star_info(stars, lista_obs, listpar)


logF, dlogF, logF_grid, wave, box_lim =\
        read_observables(models, lbdarr, lista_obs)


