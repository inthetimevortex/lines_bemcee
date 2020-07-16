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
import numpy as np
import random
# ==============================================================================
# General Options
a_parameter =       4.   # Set internal steps of each walker
extension =         '.png'  # Figure flag.extension to be saved
include_rv =        False # If False: fix Rv = 3.1, else Rv will be inferead
af_filter =         False  # Remove walkers outside the range 0.2 < af < 0.5
long_process =      False  # Run with few walkers or many?
list_of_stars =     'hd19818.txt'  # The list of flag.stars with prior data in /tables/ folder
plot_fits =         True  # NOT USING //// Include fits in the corner plot NOT USING 
plot_in_log_scale=  False  # NOT USING ///
Nsigma_dis =        3.  # Set the range of values for the distance
model =             'aeri'  # 'beatlas', 'befavor', 'aara', or 'acol'


binary_star = False

folder_data = '../data/'
folder_fig = '../figures/'
folder_defs = '../defs/'
folder_tables = '../tables/'
folder_models = '../models/'


vsini_prior =   False # Uses a gaussian vsini prior
dist_prior =    True # Uses a gaussian distance prior

box_W =         True # Constrain the W lower limit, not actual a prior, but restrain the grid
box_W_min, box_W_max = [0.6, 'max']

incl_prior =    False # Uses a gaussian inclination prior 

normal_spectra =    True # True if the spectra is normalized (for lines), distance and e(b-v) are not computed
only_wings =        False # Run emcee with only the wings
only_centerline =   False # Run emcee with only the center of the line
Sigma_Clip =        True # If you want telluric lines/outilier points removed
remove_partHa =     False # Remove a lbd interval in flag.Halpha that has too much absorption (in the wings)

# Line and continuum combination
combination =   True
UV =            True
iue =           False
votable =       False
data_table=     True
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
acrux = False  # If True, it will run in Nproc processors in the cluster
Nproc = 24  # Number of processors to be used in the cluster

# ==============================================================================

# ==============================================================================
# Read the list of flag.stars and takes the prior information about them from the stars_table txt file
def read_stars(stars_table):

    typ = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    file_data = folder_tables + stars_table

    a = np.genfromtxt(file_data, usecols=typ, unpack=True,
                      comments='#',
                      dtype={'names': ('star', 'plx', 'sig_plx', 'vsini',
                                       'sig_vsini', 'pre_ebmv', 'inc', 'sinc',
                                       'bump', 'lbd_range'),
                             'formats': ('S9', 'f2', 'f2', 'f4',
                                         'f4', 'f4', 'f4', 'f4', 'S5',
                                         'S24')})

    stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs,\
        list_pre_ebmv, incl0, sig_incl0, bump0, lbd_range =\
        a['star'], a['plx'], a['sig_plx'], a['vsini'], a['sig_vsini'],\
        a['pre_ebmv'], a['inc'], a['sinc'], a['bump'], a['lbd_range']

    if np.size(stars) == 1:
        stars = stars.astype('str')
    else:
        for i in range(len(stars)):
            stars[i] = stars[i].astype('str')

    return stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs,\
        list_pre_ebmv, incl0, sig_incl0, bump0, lbd_range

# ==============================================================================
# Reading the list of flag.stars
stars, list_plx, list_sig_plx, list_vsini_obs, list_sig_vsini_obs,\
    list_pre_ebmv, incl0, sig_incl0, bump0, lbd_range=\
    read_stars(list_of_stars)

#############################################################################################################################
if corner_color == 'random' or corner_color == '':
    corner_color = random.choice(['blue', 'dark blue', 'teal', 'green', 'yellow', 'orange', 'red', 'purple', 'violet', 'pink'])
#############################################################################################################################

def init():
    global flags
    flags = [a_parameter, extension, include_rv, af_filter,
            long_process, list_of_flag.stars, plot_fits, plot_in_log_scale, 
            Nsigma_dis, model, vsini_prior, dist_prior, box_W ,incl_prior, 
            Halpha, normal_spectra, only_wings, only_centerline, Sigma_Clip, 
            remove_partHa, combination, UV, Ha, Hb, Hg, Hd, compare_results,
            stellar_prior, npy_star, acrux, Nproc, stars, list_plx, list_sig_plx, 
            list_vsini_obs, list_sig_vsini_obs, list_pre_ebmv, incl0, sig_incl0, bump0, 
            lbd_range, corner_color, data_table, binary_star, iue]

