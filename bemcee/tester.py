import numpy as np
import random
import emcee
from lines_emcee import run_emcee, lnlike, lnprob, lnprior
import sys
from schwimmbad import MPIPool

import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob
import utils as ut
from operator import is_not
from functools import partial
import os
from utils import bin_data, find_nearest, jy2cgs
from scipy.interpolate import griddata
import atpy
from astropy.io import fits
from pyhdust import spectools as spec
from matplotlib import pyplot as plt
from astropy.stats import SigmaClip
from astropy.stats import median_absolute_deviation as MAD
from scipy.signal import detrend
from lines_radialv import delta_v, Ha_delta_v

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
from lines_convergence import plot_convergence
from astropy.stats import SigmaClip
import seaborn as sns

sns.set_style("white", {"xtick.major.direction": 'in',
              "ytick.major.direction": 'in'})

lines_dict = {
'Ha':6562.801,
'Hb':4861.363,
'Hd':4101.74,
'Hg':4340.462 }


#==============================================================================
def find_nearest2(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx




#=======================================================================
# Read the xdr combining continuum and lines
def read_BAphot2_xdr(lista_obs):

    dims = ['M', 'W', 'tms', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = None # photospheric  model is None

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN] 

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    xdrPL =  folder_models + 'BAphot__UV2M_Ha_Hb_Hg_Hd.xdr'
    ninfo, ranges, lbdarr, minfo, models = ut.readXDRsed(xdrPL, quiet=False)
    models = 1e-8 * models # erg/s/cm2/micron
    
    # Correction for negative parameter values (in cosi for instance)
    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0 and (j != 0 or j != len(models[i][j]) - 1):
                models[i][j] = (models[i][j - 1] + models[i][j + 1]) / 2.
    
    # Combining models and lbdarr
    models_combined = []
    lbd_combined = []
    
    if check_list(lista_obs, 'UV'):
        if  votable or  data_table:
            lbd_UV, models_UV = xdr_remove_lines(lbdarr, models)
	    # UV is from lbdarr[0] to lbdarr[224]
        else:
            j = 0
            shape = (5500,225)
            models_UV = np.zeros(shape)
            while j < 5500:
                models_UV[j] = models[j][:225]
                j += 1
   
            lbd_UV = lbdarr[:225]
        
        #plt.plot(lbd_UV, models_UV[30])
        #plt.xscale('log')
        #plt.yscale('log')
        models_combined.append(models_UV)
        lbd_combined.append(lbd_UV)
   
    if check_list(lista_obs, 'Ha'): 
        lbdc = 0.656
        models_combined, lbd_combined = select_xdr_part(lbdarr, models, models_combined, lbd_combined, lbdc)
        
    if check_list(lista_obs, 'Hb'):
        lbdc = 0.486
        models_combined, lbd_combined = select_xdr_part(lbdarr, models, models_combined, lbd_combined, lbdc)
    
    if check_list(lista_obs, 'Hd'):
        lbdc = 0.410
        models_combined, lbd_combined = select_xdr_part(lbdarr, models, models_combined, lbd_combined, lbdc)
    
    if check_list(lista_obs, 'Hg'):
        lbdc = 0.434
        models_combined, lbd_combined = select_xdr_part(lbdarr, models, models_combined, lbd_combined, lbdc)
  
          
        
    listpar = [np.unique(minfo[:,0]), np.unique(minfo[:,1]),np.unique(minfo[:,2]), np.unique(minfo[:,3])]

    return ctrlarr, minfo, models_combined, lbd_combined, listpar, dims, isig  


# =============================================================================
def select_xdr_part(lbdarr, models, models_combined, lbd_combined, lbdc):
    line_peak = find_nearest2(lbdarr, lbdc)
    keep_a = find_nearest2(lbdarr, lbdc - 0.004)
    keep_b = find_nearest2(lbdarr, lbdc + 0.004)
    lbd_line = lbdarr[keep_a:keep_b]
    models_line = models[:, keep_a:keep_b]
    lbdarr_line = lbd_line*1e4
    lbdarr_line = lbdarr_line/(1.0 + 2.735182E-4 + 131.4182/lbdarr_line**2 + 2.76249E8/lbdarr_line**4)
    models_combined.append(models_line)
    lbd_combined.append(lbdarr_line*1e-4)

    return models_combined, lbd_combined

# =============================================================================
def xdr_remove_lines(lbdarr, models):
    for line in lines_dict:
        keep_a = find_nearest2(lbdarr, lines_dict[line] - 0.007)
        keep_b = find_nearest2(lbdarr, lines_dict[line] + 0.007)
        lbdarr1 = lbdarr[:keep_a]
        lbdarr2 = lbdarr[keep_b:]
        lbdarr = np.concatenate([lbdarr1,lbdarr2])
        novo_models1 = models[:, :keep_a]
        novo_models2 = models[:, keep_b:]
        novo_models = np.hstack((novo_models1, novo_models2))
        
    return lbdarr, novo_models
        
# ==============================================================================
def read_aara_xdr():


    dims = ['M', 'ob', 'Hfrac', 'sig0', 'Rd', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    xdrPL =  folder_models + 'aara_sed.xdr'  # 


    listpar, lbdarr, minfo, models = ut.readBAsed(xdrPL, quiet=False)

    # F(lbd)] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0. or models[i][j] == 0.:
                models[i][j] = (models[i][j + 1] + models[i][j - 1]) / 2.

    # n0 to logn0
    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    minfo[:, 4] = np.log10(minfo[:, 4])

    if True:
        mask = []
        tmp, idx = find_nearest(lbdarr, 1000)
        for i in range(len(models)):
            if models[i][idx] > 2.21834e-10:
                mask.append(i)
                # print(i)
                # plt.plot(lbdarr, models[i], alpha=0.1)
        tmp, idx = find_nearest(lbdarr, 80)
        for i in range(len(models)):
            if models[i][idx] > 2e-8:
                mask.append(i)
                # print(i)
                # # plt.plot(lbdarr, models[i], alpha=0.1)
        tmp, idx = find_nearest(lbdarr, 850)
        for i in range(len(models)):
            if models[i][idx] > 7e-11:
                mask.append(i)
        #         print(i)
        #         plt.plot(lbdarr, models[i], alpha=0.1)
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()

        new_models = np.delete(models, mask, axis=0)
        new_minfo = np.delete(minfo, mask, axis=0)

        models = np.copy(new_models)
        minfo = np.copy(new_minfo)

    # delete columns of fixed par
    cols2keep = [0, 1, 3, 4, 5, 7, 8]
    cols2delete = [2, 6]
    listpar = [listpar[i] for i in cols2keep]
    minfo = np.delete(minfo, cols2delete, axis=1)
    listpar[3].sort()

    # for i in range(len(models)):
    #     plt.plot(lbdarr, models[i], alpha=0.1)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# ==============================================================================
def read_befavor_xdr():



    dims = ['M', 'ob', 'Hfrac', 'sig0', 'Rd', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, 0.014, np.NaN, 0.0, 50.0, 60.0, 3.5, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.
    xdrPL =  folder_models + 'BeFaVOr.xdr'
    listpar, lbdarr, minfo, models = ut.readBAsed(xdrPL, quiet=False)
    # [models] = [F(lbd)]] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0 and (j != 0 or j != len(models[i][j]) - 1):
                models[i][j] = (models[i][j - 1] + models[i][j + 1]) / 2.

    # delete columns of fixed par
    cols2keep = [0, 1, 3, 8]
    cols2delete = [2, 4, 5, 6, 7]
    listpar = [listpar[i] for i in cols2keep]
    minfo = np.delete(minfo, cols2delete, axis=1)
    listpar[3].sort()
    listpar[3][0] = 0.

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig

## ==============================================================================
## Function to read the xdr in the same format as Rodrigo (made with bat.createXDRsed)
#def read_BAphot2_xdr():
#
#     folder_models = 'models/'
#
#    #dims = ['M', 'W', 'tms', 'sig0', 'cosi']
#    #dims = dict(zip(dims, range(len(dims))))
#    #isig = dims["sig0"]
#    dims = ['M', 'W', 'tms', 'cosi']
#    dims = dict(zip(dims, range(len(dims))))
#    isig = None # photospheric  model is None
#
#    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN] # tambem nao tenho certeza o que eh
#
#    tmp = 0
#    cont = 0
#    while tmp < len(ctrlarr):
#        if math.isnan(ctrlarr[tmp]) is True:
#            cont = cont + 1
#            tmp = tmp + 1
#        else:
#            tmp = tmp + 1
#
#    # Read the grid models, with the interval of parameters.
#    xdrPL =  folder_models + 'BAphot_v2_UV.xdr'
#    # codigo segundo o Rodrigo
#    ninfo, ranges, lbdarr, minfo, models = bat.readXDRsed(xdrPL, quiet=False)
#    models = 1e-8 * models # erg/s/cm2/micron
#    
#    #logmodels = np.log(1e-8 * models) # erg/s/cm2/micron, at 10 pc, warning de 'division by zero'
#    #listpar, lbdarr, minfo, models = bat.readBAsed(xdrPL, quiet=False)
#    # [models] = [F(lbd)]] = 10^-4 erg/s/cm2/Ang
#
#    # Correction for negative parameter values (in cosi for instance)
#    for i in range(np.shape(minfo)[0]):
#        for j in range(np.shape(minfo)[1]):
#            if minfo[i][j] < 0:
#                minfo[i][j] = 0.
#
#    for i in range(np.shape(models)[0]):
#        for j in range(np.shape(models)[1]):
#            if models[i][j] < 0 and (j != 0 or j != len(models[i][j]) - 1):
#                models[i][j] = (models[i][j - 1] + models[i][j + 1]) / 2.
#
#    # delete columns of fixed par -- talvez nem precise dessa parte
#    #cols2keep = [0, 1, 3, 8]
#    #cols2delete = [2, 4, 5, 6, 7]
#    #listpar = [listpar[i] for i in cols2keep]
#    #minfo = np.delete(minfo, cols2delete, axis=1)
#    #listpar[3].sort()
#    #listpar[3][0] = 0.
#    
#    # versao anterior tinha o cosi negativo no listpar, eu troquei agora para zero dado que o minfo de fato considera igual a 0
#    listpar = [np.array([1.7000000476837158,2.0,2.5,3.0,4.0,5.0,7.0,9.0,12.0,15.0]), np.array([0.0,0.33000001311302185,0.4699999988079071,0.5699999928474426,0.6600000262260437,0.7400000095367432,0.8100000023841858,0.8700000047683716,0.9300000071525574]), np.array([0.0,0.25,0.5,0.75,1.0]), np.array([0,0.11146999895572662,0.22155000269412994,0.3338100016117096,0.44464001059532166,0.5548400282859802,0.6665300130844116,0.7782400250434875,0.8886200189590454,1.0])]
#
#    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig

# ==============================================================================
# Function to read the UV part of the complete xdr (made with bat.createXDRsed)
#def read_BAphot2_UV_xdr():
#
#     folder_models = 'models/'
#
#    dims = ['M', 'W', 'tms', 'cosi']
#    dims = dict(zip(dims, range(len(dims))))
#    isig = None # photospheric  model is None
#
#    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN] # tambem nao tenho certeza o que eh
#
#    tmp = 0
#    cont = 0
#    while tmp < len(ctrlarr):
#        if math.isnan(ctrlarr[tmp]) is True:
#            cont = cont + 1
#            tmp = tmp + 1
#        else:
#            tmp = tmp + 1
#
#    # Read the grid models, with the interval of parameters.
#    xdrPL =  folder_models + 'BAphot_UV2M_ Halpha.xdr'
#    # codigo segundo o Rodrigo
#    ninfo, ranges, lbdarr, minfo, models = bat.readXDRsed(xdrPL, quiet=False)
#    models = 1e-8 * models # erg/s/cm2/micron
#    
#    # Correction for negative parameter values (in cosi for instance)
#    for i in range(np.shape(minfo)[0]):
#        for j in range(np.shape(minfo)[1]):
#            if minfo[i][j] < 0:
#                minfo[i][j] = 0.
#
#    for i in range(np.shape(models)[0]):
#        for j in range(np.shape(models)[1]):
#            if models[i][j] < 0 and (j != 0 or j != len(models[i][j]) - 1):
#                models[i][j] = (models[i][j - 1] + models[i][j + 1]) / 2.
#    
#    # UV is from lbdarr[0] to lbdarr[224]
#    j = 0
#    shape = (5500,225)
#    models_UV = np.zeros(shape)
#    while j < 5500:
#        models_UV[j] = models[j][:225]
#        j += 1
#    
#    listpar = [np.array([1.7000000476837158,2.0,2.5,3.0,4.0,5.0,7.0,9.0,12.0,15.0]), np.array([0.0,0.33000001311302185,0.4699999988079071,0.5699999928474426,0.6600000262260437,0.7400000095367432,0.8100000023841858,0.8700000047683716,0.9300000071525574]), np.array([0.0,0.25,0.5,0.75,1.0]), np.array([0,0.11146999895572662,0.22155000269412994,0.3338100016117096,0.44464001059532166,0.5548400282859802,0.6665300130844116,0.7782400250434875,0.8886200189590454,1.0])]
#
#    return ctrlarr, minfo, models_UV, lbdarr[:225], listpar, dims, isig

#def read_acol_Ha_xdr():
#
#    dims = ['M', 'ob', 'Hfrac', 'sig0', 'Rd', 'mr', 'cosi']
#    dims = dict(zip(dims, range(len(dims))))
#    isig = dims["sig0"]
#
#    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
#
#    tmp = 0
#    cont = 0
#    while tmp < len(ctrlarr):
#        if math.isnan(ctrlarr[tmp]) is True:
#            cont = cont + 1
#            tmp = tmp + 1
#        else:
#            tmp = tmp + 1
#
#    # Read the grid models, with the interval of parameters.
#     folder_models = 'models/'
#    xdrPL =  folder_models + 'acol_Ha_only.xdr'
#
#
#    listpar, lbdarr, minfo, models = bat.readBAsed(xdrPL, quiet=False)
#
#
#    #
#    mask = np.ones(len(minfo[0]), dtype=bool)
#    mask[[2, 6]] = False
#    result = []
#    for i in range(len(minfo)):
#        result.append(minfo[i][mask])
#    minfo = np.copy(result)
#
#    for i in range(np.shape(minfo)[0]):
#        minfo[i][3] = np.log10(minfo[i][3])
#
#    listpar[4] = np.log10(listpar[4])
#    listpar[4].sort()
#    listpar = list([listpar[0], listpar[1], listpar[3], listpar[4],
#                    listpar[5], listpar[7], listpar[8]])
#    
#
#        # Change vacuum wavelength to air wavelength
#    lbdarr = lbdarr*1e4 # to Angstrom # estava lbdarr[255:495], que eh o  Halpha completo de fato
#    lbdarr = lbdarr/(1.0 + 2.735182E-4 + 131.4182/lbdarr**2 + 2.76249E8/lbdarr**4) # valid if lbd in Angstrom
#    lbdarr = lbdarr*1e-4 # to mum again
#
#
#    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig

# ==============================================================================
# Function to read the  Halpha part of the complete xdr (made with bat.createXDRsed)
#def read_BAphot2_ Halpha_xdr():
#
#     folder_models = 'models/'
#
#    dims = ['M', 'W', 'tms', 'cosi']
#    dims = dict(zip(dims, range(len(dims))))
#    isig = None # photospheric  model is None
#
#    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN] # tambem nao tenho certeza o que eh
#
#    tmp = 0
#    cont = 0
#    while tmp < len(ctrlarr):
#        if math.isnan(ctrlarr[tmp]) is True:
#            cont = cont + 1
#            tmp = tmp + 1
#        else:
#            tmp = tmp + 1
#
#    # Read the grid models, with the interval of parameters.
#    xdrPL =  folder_models + 'BAphot_UV2M_ Halpha.xdr'
#    # codigo segundo o Rodrigo
#    ninfo, ranges, lbdarr, minfo, models = bat.readXDRsed(xdrPL, quiet=False)
#    models = 1e-8 * models # erg/s/cm2/micron
#    
#    # Correction for negative parameter values (in cosi for instance)
#    for i in range(np.shape(minfo)[0]):
#        for j in range(np.shape(minfo)[1]):
#            if minfo[i][j] < 0:
#                minfo[i][j] = 0.
#
#    for i in range(np.shape(models)[0]):
#        for j in range(np.shape(models)[1]):
#            if models[i][j] < 0 and (j != 0 or j != len(models[i][j]) - 1):
#                models[i][j] = (models[i][j - 1] + models[i][j + 1]) / 2.
#    
#    #  Halpha is from lbdarr[255] to lbdarr[495]
#    j = 0
#    shape = (5500,238) # Estava (5500,240)
#    models_ Halpha = np.zeros(shape)
#    while j < 5500:
#        models_ Halpha[j] = models[j][256:494] # estava models[j][255:495]
#        j += 1
#    
#    listpar = [np.array([1.7000000476837158,2.0,2.5,3.0,4.0,5.0,7.0,9.0,12.0,15.0]), np.array([0.0,0.33000001311302185,0.4699999988079071,0.5699999928474426,0.6600000262260437,0.7400000095367432,0.8100000023841858,0.8700000047683716,0.9300000071525574]), np.array([0.0,0.25,0.5,0.75,1.0]), np.array([0,0.11146999895572662,0.22155000269412994,0.3338100016117096,0.44464001059532166,0.5548400282859802,0.6665300130844116,0.7782400250434875,0.8886200189590454,1.0])]
#
#    # Change vacuum wavelength to air wavelength
#    lbdarr = lbdarr[256:494]*1e4 # to Angstrom # estava lbdarr[255:495], que eh o  Halpha completo de fato
#    lbdarr = lbdarr/(1.0 + 2.735182E-4 + 131.4182/lbdarr**2 + 2.76249E8/lbdarr**4) # valid if lbd in Angstrom
#    lbdarr = lbdarr*1e-4 # to mum again
#    
#
#    return ctrlarr, minfo, models_ Halpha, lbdarr, listpar, dims, isig
    
# ==============================================================================
def read_beatlas_xdr():

    dims = ['M', 'ob', 'sig0', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]
    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1


    xdrPL =  folder_models + 'disk_flx.xdr'  # 'PL.xdr'

    listpar, lbdarr, minfo, models = ut.readBAsed(xdrPL, quiet=False)

    # F(lbd)] = 10^-4 erg/s/cm2/Ang

    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(models)[0]):
        for j in range(np.shape(models)[1]):
            if models[i][j] < 0. or models[i][j] == 0.:
                models[i][j] = (models[i][j + 1] + models[i][j - 1]) / 2.

    listpar[-1][0] = 0.

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# ==============================================================================
def read_acol_xdr():

    # print(params_tmp)
    dims = ['M', 'ob', 'Hfrac', 'sig0', 'Rd', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.

    #xdrPL =  folder_models + 'acol_06.xdr'

    listpar, lbdarr, minfo, models = ut.readBAsed(xdrPL, quiet=False)

    # Filter (removing bad models)
    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(listpar)[0]):
        for j in range(len(listpar[i])):
            if listpar[i][j] < 0:
                listpar[i][j] = 0.

    mask = np.ones(len(minfo[0]), dtype=bool) # isso provavelmente remove os parâmetros com só 1 valor
    mask[[2, 6]] = False
    result = []
    for i in range(len(minfo)):
        result.append(minfo[i][mask])
    minfo = np.copy(result)

    for i in range(np.shape(minfo)[0]):
        minfo[i][3] = np.log10(minfo[i][3])

    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    listpar = list([listpar[0], listpar[1], listpar[3], listpar[4], listpar[5],
                    listpar[7], listpar[8]])
        
    return ctrlarr, minfo, [models], [lbdarr], listpar, dims, isig


#===============================================================================

def read_acol_Ha_xdr(lista_obs):

    # print(params_tmp)
    dims = ['M', 'ob', 'Hfrac', 'sig0', 'Rd', 'mr', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = dims["sig0"]

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1

    # Read the grid models, with the interval of parameters.

    xdrPL =  folder_models + 'acol_Ha.xdr'

    listpar, lbdarr, minfo, models = ut.readBAsed(xdrPL, quiet=False)

    # Filter (removing bad models)
    for i in range(np.shape(minfo)[0]):
        for j in range(np.shape(minfo)[1]):
            if minfo[i][j] < 0:
                minfo[i][j] = 0.

    for i in range(np.shape(listpar)[0]):
        for j in range(len(listpar[i])):
            if listpar[i][j] < 0:
                listpar[i][j] = 0.

    mask = np.ones(len(minfo[0]), dtype=bool) # isso provavelmente remove os parâmetros com só 1 valor
    mask[[2, 6]] = False
    result = []
    for i in range(len(minfo)):
        result.append(minfo[i][mask])
    minfo = np.copy(result)

    for i in range(np.shape(minfo)[0]):
        minfo[i][3] = np.log10(minfo[i][3])

    listpar[4] = np.log10(listpar[4])
    listpar[4].sort()
    listpar = list([listpar[0], listpar[1], listpar[3], listpar[4], listpar[5],
                    listpar[7], listpar[8]])
    
    # Combining models and lbdarr
    models_combined = []
    lbd_combined = []
    
    
    if check_list(lista_obs, 'UV'):
        lbd_UV, models_UV = xdr_remove_lines(lbdarr, models)
    
        models_combined.append(models_UV)
        lbd_combined.append(lbd_UV)
   
    if check_list(lista_obs, 'Ha'): 
        lbdc = 0.6563
        models_combined, lbd_combined = select_xdr_part(lbdarr, models, models_combined, lbd_combined, lbdc)
    
    
    
        
    return ctrlarr, minfo, models_combined, lbd_combined, listpar, dims, isig


# ==============================================================================

# ==============================================================================
def read_star_info(star, lista_obs, listpar):

        print(75 * '=')

        star_r = star

        print('\nRunning star: %s\n' % star_r)
        print(75 * '=')

        plx = np.copy( list_plx)
        dplx = np.copy( list_sig_plx)
        vsin_obs = np.copy( list_vsini_obs)
        sig_vsin_obs = np.copy( list_sig_vsini_obs)
        
        
        # New parameters limits
        #box_W_inf = 0.8
        #vrad_inf = -100  # km/s
        #vrad_sup = 100 # km/s

# ------------------------------------------------------------------------------
        # Reading known stellar parameters
        dist_pc = plx
        sig_dist_pc = dplx
        #if plx != 0:
        #    dist_pc = 1e3 / plx  # pc
        #    sig_dist_pc = (1e3 * dplx) / plx**2
        #    
        #else:
        #    dist_pc = 50.0
        #    sig_dist_pc = 100.0

        
        if check_list(lista_obs, 'UV') or  normal_spectra is False:
            if  include_rv:
                ebmv, rv = [[0.0, 0.8], [1., 5.8]]
            else:
                rv = 3.1
                ebmv, rv = [[0.0, 0.8], None]
            
            dist_min = dist_pc -  Nsigma_dis * sig_dist_pc
            dist_max = dist_pc +  Nsigma_dis * sig_dist_pc
    
            #if dist_min < 0:
            #    dist_min = 1    
            
            addlistpar = [ebmv, [dist_min, dist_max], rv]
            addlistpar = list(filter(partial(is_not, None), addlistpar))
            

    
            if  model == 'aeri' or  model == 'befavor':
                
                ranges = np.array([[listpar[0][0], listpar[0][-1]],
                                    [listpar[1][0], listpar[1][-1]],
                                    [listpar[2][0], listpar[2][-1]],
                                    [listpar[3][0], listpar[3][-1]],
                                    [dist_min, dist_max],
                                    [ebmv[0], ebmv[-1]]])

                                                
            elif  model == 'aara' or  model == 'acol':

                ranges = np.array([[listpar[0][0], listpar[0][-1]],
                                [listpar[1][0], listpar[1][-1]],
                                [listpar[2][0], listpar[2][-1]],
                                [listpar[3][0], listpar[3][-1]],
                                [listpar[4][0], listpar[4][-1]],
                                [listpar[5][0], listpar[5][-1]],
                                [listpar[6][0], listpar[6][-1]],
                                [dist_min, dist_max],
                                [ebmv[0], ebmv[-1]]])

                
            elif  model == 'beatlas':

                ranges = np.array([[listpar[0][0], listpar[0][-1]],
                                [listpar[1][0], listpar[1][-1]],
                                [listpar[2][0], listpar[2][-1]],
                                [listpar[3][0], listpar[3][-1]],
                                [listpar[4][0], listpar[4][-1]],
                                [dist_min, dist_max],
                                [ebmv[0], ebmv[-1]]])
            
            if  include_rv:
                ranges = np.concatenate([ranges, [rv]])
                
            if  binary_star:
                M2 = [listpar[0][0], listpar[0][-1]]
                ranges = np.concatenate([ranges, [M2]])
                


        else:
            if  model == 'aeri' or  model == 'befavor':
                
                ranges = np.array([[listpar[0][0], listpar[0][-1]],
                                   [listpar[1][0], listpar[1][-1]],
                                   [listpar[2][0], listpar[2][-1]],
                                   [listpar[3][0], listpar[3][-1]]])

                               
            elif  model == 'acol' or  model == 'aara':

                ranges = np.array([[listpar[0][0], listpar[0][-1]],
                                   [listpar[1][0], listpar[1][-1]],
                                   [listpar[2][0], listpar[2][-1]],
                                   [listpar[3][0], listpar[3][-1]],
                                   [listpar[4][0], listpar[4][-1]],
                                   [listpar[5][0], listpar[5][-1]],
                                   [listpar[6][0], listpar[6][-1]]])

                    
            elif  model == 'beatlas':
                
                ranges = np.array([[listpar[0][0], listpar[0][-1]],
                                [listpar[1][0], listpar[1][-1]],
                                [listpar[2][0], listpar[2][-1]],
                                [listpar[3][0], listpar[3][-1]],
                                [listpar[4][0], listpar[4][-1]]])
            
            if  binary_star:
                M2 = [listpar[0][0], listpar[0][-1]]
                ranges = np.concatenate([ranges, [M2]])
        
        if  box_W:
            if  box_W_max == 'max':
                ranges[1][0] =  box_W_min
            elif  box_W_min == 'min':
                ranges[1][1] =  box_W_max
            else:
                ranges[1][0], ranges[1][1] =  box_W_min,  box_W_max
        
        Ndim = len(ranges)
        
        
        

        return ranges, dist_pc, sig_dist_pc, vsin_obs,\
            sig_vsin_obs, Ndim


def read_stellar_prior():
    if  stellar_prior is True:
        chain = np.load('npys/' +  npy_star)
        Ndim = np.shape(chain)[-1]
        flatchain = chain.reshape((-1, Ndim))

        mas = flatchain[:, 0]
        obl = flatchain[:, 1]
        age = flatchain[:, 2]
        dis = flatchain[:, -2]
        ebv = flatchain[:, -1]

        # grid_mas = np.linspace(np.min(mas), np.max(mas), 100)
        # grid_obl = np.linspace(np.min(obl), np.max(obl), 100)
        # grid_age = np.linspace(np.min(age), np.max(age), 100)
        # grid_ebv = np.linspace(np.min(ebv), np.max(ebv), 100)

        grid_mas = np.linspace(3.4, 14.6, 100)
        grid_obl = np.linspace(1.00, 1.45, 100)
        grid_age = np.linspace(0.08, 0.78, 100)
        grid_dis = np.linspace(0.00, 140, 100)
        grid_ebv = np.linspace(0.00, 0.10, 100)

        pdf_mas = kde_scipy(x=mas, x_grid=grid_mas, bandwidth=0.005)
        pdf_obl = kde_scipy(x=obl, x_grid=grid_obl, bandwidth=0.005)
        pdf_age = kde_scipy(x=age, x_grid=grid_age, bandwidth=0.01)
        pdf_dis = kde_scipy(x=dis, x_grid=grid_dis, bandwidth=0.01)
        pdf_ebv = kde_scipy(x=ebv, x_grid=grid_ebv, bandwidth=0.0005)

    else:
        grid_mas = 0
        grid_obl = 0
        grid_age = 0
        grid_dis = 0
        grid_ebv = 0

        pdf_mas = 0
        pdf_obl = 0
        pdf_age = 0
        pdf_dis = 0
        pdf_ebv = 0
        
    grid_priors = [grid_mas, grid_obl, grid_age, grid_dis, grid_ebv]
    pdf_priors = [pdf_mas, pdf_obl, pdf_age, pdf_dis, pdf_ebv]
    
        
    return grid_priors, pdf_priors



def read_espadons(fname):

    # read fits
    if str( stars) == 'HD37795':
        hdr_list = fits.open(fname)
        fits_data = hdr_list[0].data
        fits_header = hdr_list[0].header
    #read MJD
        MJD = fits_header['MJDATE']   
        lat = fits_header['LATITUDE']
        lon = fits_header['LONGITUD']
        lbd = fits_data[0, :]
        ordem = lbd.argsort()
        lbd = lbd[ordem] * 10
        flux_norm = fits_data[1, ordem]
    #vel, flux = spt.lineProf(lbd, flux_norm, lbc=lbd0)
    else :
        lbd, flux_norm, MJD, dateobs, datereduc, fitsfile = spec.loadfits(fname)
#    plt.plot (lbd, flux_norm)
#    plt.show()
    return lbd, flux_norm, MJD
    



# ==================================================================================
# READS LINES FROM DATA
# ==================================================================================
def read_line_spectra(models, lbdarr, linename):
        
    
    table =  folder_data + str( stars) + '/' + 'spectra/' + 'list_spectra.txt'
    
    if os.path.isfile(table) is False or os.path.isfile(table) is True:
        os.system('ls ' +  folder_data + str( stars) + '/spectra' + '/* | xargs -n1 basename >' +  folder_data + '/' + 'list_spectra.txt')
        spectra_list = np.genfromtxt(table, comments='#', dtype='str')
        file_name = np.copy(spectra_list)
    
    fluxes, waves, errors = [], [], []

    
    if file_name.tolist()[-3:] == 'csv':
        file_ha = str( folder_data) + str( stars) + '/spectra/' + str(file_name)
        wl, normal_flux = np.loadtxt(file_ha, delimiter=' ').T
    else:
        file_ha = str( folder_data) + str( stars) + '/spectra/' + str(file_name)
        #print(file_ha)  
        wl, normal_flux, MJD = read_espadons(file_ha)
    
    #--------- Selecting an approximate wavelength interval 
    c = 299792.458 #km/s
    lbd_central = lines_dict[linename]
    
        
    if  Sigma_Clip:
        #gives the line profiles in velocity space
        vl, fx = ut.lineProf(wl, normal_flux, hwidth=2500., lbc=lbd_central)
        vel, fluxes = Sliding_Outlier_Removal(vl, fx, 50, 8, 15)
        wl = c*lbd_central/(c - vel)
    else:
        vel = (wl - lbd_central)*c/lbd_central
        largura = 6000 #km/s    # 6000 is compatible with sigmaclip
        lim_esq1 = np.where(vel < -largura/2.)
        lim_dir1 = np.where(vel > largura/2.)

        inf = lim_esq1[0][-1]
        sup = lim_dir1[0][0]

        wl = wl[inf:sup]
        fluxes = normal_flux[inf:sup]

    
    radv = delta_v(vel, fluxes, 'Ha')
#AMANDA_VOLTAR: o que fazer quando o 
    #radv = 2.8
    print('RADIAL VELOCITY = {0}'.format(radv))
    vel = vel - radv
    wl = c*lbd_central/(c - vel)

    # Changing the units
    waves = wl * 1e-4 # mum
    errors = np.zeros(len(waves)) # nao tenho informacao sobre os erros ainda

    # Aqui faz o corte pra pegar so o que ta no intervalo de lbd da grade (lbdarr)
    idx = np.where((waves >= np.min(lbdarr)-0.001) & (waves <= np.max(lbdarr)+0.001))
    #wave = waves[idx]
    #flux = fluxes[idx]
    #sigma = errors[idx]
    #obs_new = np.zeros(len(lbdarr))


    #new_wave = waves
    #new_flux = fluxes


    # Bin the spectra data to the lbdarr (model)
    # Creating box limits array
    box_lim = np.zeros(len(lbdarr) - 1)
    for i in range(len(lbdarr) - 1):
        box_lim[i] = (lbdarr[i] + lbdarr[i+1])/2.

    # Binned flux  
    bin_flux = np.zeros(len(box_lim) - 1)

    lbdarr = lbdarr[1:-1]
    errors = np.zeros(len(lbdarr))
    #sigma_new.fill(0.017656218)
#AMANDA_VOLTAR: erros devem ser estimados numa rotina de precondicionamento
#HD6226: 0.04
#MT91-213: 0.01 
    for i in range(len(box_lim) - 1):
        # lbd observations inside the box
        index = np.argwhere((waves > box_lim[i]) & (waves < box_lim[i+1]))
        if len(index) == 0:
            bin_flux[i] = -np.inf
        elif len(index) == 1:
            bin_flux[i] = fluxes[index[0][0]]
            errors[i] = 0.04
        else: 
            # Calculating the mean flux in the box
            bin_flux[i] = np.sum(fluxes[index[0][0]:index[-1][0]])/len(fluxes[index[0][0]:index[-1][0]])
            errors[i] = 0.04/np.sqrt(len(index))

    flux = bin_flux
    
    #que estou interpolando sao as observacoes nos modelos, entao nao eh models_new que tenho que fazer. o models ta pronto
    # log space
    #logF = np.log10(obs_new)
    #mask = np.where(obs_new == -np.inf)
    #logF[mask] = -np.inf # so that dlogF == 0 in these points and the chi2 will not be computed
    #dlogF = sigma_new/obs_new
    
    
    #rmalize the flux of models to the continuum
    #norm_model = np.zeros((len(models),len(lbdarr)))
    #for i in range(len(models)):
    #    norm_model[i] = ut.linfit(lbdarr, models[i][1:-1])
    #models = norm_model
    novo_models = np.zeros((len(models),len(lbdarr)))
    for i in range(len(models)):
        novo_models[i] = models[i][1:-1]
    #logF_grid = np.log10(models)  
    
    if linename == 'Ha':
        if  remove_partHa:
            lbdarr1 = lbdarr[:149]
            lbdarr2 = lbdarr[194:]
            lbdarr = np.append(lbdarr1,lbdarr2)
            novo_models = np.zeros((5500,191))
            obs_new1 = flux[:149]
            obs_new2 = flux[194:]
            flux = np.append(obs_new1,obs_new2)
            #logF = np.log10(obs_new)
            errors = np.append(errors[:149], errors[194:])
            #dlogF = sigma_new2 / obs_new
            i = 0
            while i < len(models):
                novo_models[i] = np.append(novo_models[i][:149],novo_models[i][194:])
                i+=1
            #logF_grid = np.log10(novo_models)       
    
    
    if  only_wings:
        plt.plot(waves, fluxes)
        point_a, point_b = plt.ginput(2)
        plt.close()
        #H_peak = find_nearest2(lbdarr, lines_dict[linename])
        keep_a = find_nearest2(lbdarr, point_a[0])
        keep_b = find_nearest2(lbdarr, point_b[0])
        #keep_b = find_nearest2(lbdarr, lines_dict[linename] + 0.0009)
        #keep_b = find_nearest2(lbdarr, lines_dict[linename] + 0.0009)
        lbdarr1 = lbdarr[:keep_a]
        lbdarr2 = lbdarr[keep_b:]
        lbdarr = np.concatenate([lbdarr1,lbdarr2])
        obs_new1 = flux[:keep_a]
        obs_new2 = flux[keep_b:]
        flux = np.concatenate([obs_new1,obs_new2])
        #logF = np.log10(obs_neww)
        sigma_new1 = errors[:keep_a]
        sigma_new2 = errors[keep_b:]
        errors = np.concatenate([sigma_new1, sigma_new2])
        #dlogF = sigma_new/obs_neww
        novo_models1 = novo_models[:, :keep_a]
        novo_models2 = novo_models[:, keep_b:]
        novo_models = np.hstack((novo_models1, novo_models2))
        #logF_grid = np.log10(novo_models)
        
    if  only_centerline:
        plt.plot(waves, fluxes)
        point_a, point_b = plt.ginput(2)
        plt.close()
        #H_peak = find_nearest2(lbdarr, lines_dict[linename])
        keep_a = find_nearest2(lbdarr, point_a[0])
        keep_a = find_nearest2(lbdarr, point_b[0])
        lbdarr1 = lbdarr[keep_a:]
        lbdarr2 = lbdarr[:keep_b]
        lbdarr = np.concatenate([lbdarr1,lbdarr2])
        obs_new1 = flux[keep_a:]
        obs_new2 = flux[:keep_b]
        flux = np.concatenate([obs_new1,obs_new2])
        #logF = np.log10(obs_neww)
        sigma_new1 = errors[keep_a:]
        sigma_new2 = errors[:keep_b]
        errors = np.concatenate([sigma_new1, sigma_new2])
        #dlogF = sigma_new/obs_neww
        novo_models1 = novo_models[:, keep_a:]
        novo_models2 = novo_models[:, :keep_b]
        novo_models = np.hstack((novo_models1, novo_models2))
        #logF_grid = np.log10(novo_models)
  
    #print(len(flux), len(errors), len(novo_models[0]))
    return flux, errors, novo_models, lbdarr, box_lim
    
    
    
# ===================================================================================
#def read_Ha(models, lbdarr,  folder_data, folder_fig, star,
#                      model,  only_wings,  only_centerline,  Sigma_Clip,
#                       remove_partHa):
#
#    table =  folder_data + str( stars) + '/' + 'spectra/' + 'list_spectra.txt'
#    
#    if os.path.isfile(table) is False or os.path.isfile(table) is True:
#        os.system('ls ' +  folder_data + str( stars) + '/spectra' + '/* | xargs -n1 basename >' +  folder_data + '/' + 'list_feros.txt')
#        spectra_list = np.genfromtxt(table, comments='#', dtype='str')
#        file_name = np.copy(spectra_list)
#    
#    fluxes, waves, errors = [], [], []
#
#    
#    if file_name.tolist()[-3:] == 'csv':
#        file_ha = str( folder_data) + str( stars) + '/spectra/' + str(file_name)
#        wl, normal_flux = np.loadtxt(file_ha, delimiter=' ').T
#    else:    
#        wl, normal_flux, MJD = read_espadons(file_name)
#    
#    #--------- Selecting an approximate wavelength interval for  Halpha
#    c = 299792.458 #km/s
#    lbd_central = 6562.801    
#        
#    if  Sigma_Clip:
#        #gives the line profiles of H-alpha in velocity space
#        vl, fx = spec.lineProf(wl, normal_flux, hwidth=2500., lbc=lbd_central)
#        vel, fluxes = Sliding_Outlier_Removal(vl, fx, 50, 8, 15)
#        wl = c*lbd_central/(c - vel)
#        #plt.plot(wl,fluxes)
#        #plt.show()
#    else:
#        vel = (wl - lbd_central)*c/lbd_central
#        largura = 6000 #km/s    # 6000 is compatible with sigmaclip
#        lim_esq1 = np.where(vel < -largura/2.)
#        lim_dir1 = np.where(vel > largura/2.)
#
#        inf = lim_esq1[0][-1]
#        sup = lim_dir1[0][0]
#
#        wl = wl[inf:sup]
#        fluxes = normal_flux[inf:sup]
#
#    
#    radv = Ha_delta_v(vel, fluxes, 'Ha')
#    print('RADIAL VELOCITY = {0}'.format(radv))
#    vel = vel - radv
#    wl = c*lbd_central/(c - vel)
#
#    # Changing the units
#    waves = wl * 1e-4 # mum
#    errors = np.zeros(len(waves)) # nao tenho informacao sobre os erros ainda
#
#    if  model == 'beatlas' or  model == 'aara' or  model == 'aeri': # Aqui faz o corte pra pegar so o que ta no intervalo de lbd da grade (lbdarr)
#        idx = np.where((waves >= np.min(lbdarr)-0.001) & (waves <= np.max(lbdarr)+0.001))
#        wave = waves[idx]
#        flux = fluxes[idx]
#        sigma = errors[idx]
#        obs_new = np.zeros(len(lbdarr))
#
#
#    new_wave = waves
#    new_flux = fluxes
#    #sigma = np.zeros(len(new_wave))
#    #sigma.fill(0.017656218) # new test 
#    #sigma.fill(0.0017656218)  #uncertainty obtained from read_feros.py 
#
#    # Bin the spectra data to the lbdarr (model)
#    # Creating box limits array
#    box_lim = np.zeros(len(lbdarr) - 1)
#    for i in range(len(lbdarr) - 1):
#        box_lim[i] = (lbdarr[i] + lbdarr[i+1])/2.
#
#    # Binned flux  
#    bin_flux = np.zeros(len(box_lim) - 1)
#
#    lbdarr = lbdarr[1:-1]
#    sigma_new = np.zeros(len(lbdarr))
#    #sigma_new.fill(0.017656218)
#
#    for i in range(len(box_lim) - 1):
#        # lbd observations inside the box
#        index = np.argwhere((new_wave > box_lim[i]) & (new_wave < box_lim[i+1]))
#        if len(index) == 0:
#            bin_flux[i] = -np.inf
#        elif len(index) == 1:
#            bin_flux[i] = new_flux[index[0][0]]
#        else: 
#            # Calculating the mean flux in the box
#            bin_flux[i] = np.sum(new_flux[index[0][0]:index[-1][0]])/len(new_flux[index[0][0]:index[-1][0]])
#            sigma_new[i] = 0.017656218/np.sqrt(len(index))
#
#    obs_new = bin_flux
#    
#    #que estou interpolando sao as observacoes nos modelos, entao nao eh models_new que tenho que fazer. o models ta pronto
#    # log space
#    logF = np.log10(obs_new)
#    mask = np.where(obs_new == -np.inf)
#    logF[mask] = -np.inf # so that dlogF == 0 in these points and the chi2 will not be computed
#    dlogF = sigma_new/obs_new
#    
#    
#    #rmalize the flux of models to the continuum
#    norm_ model = np.zeros((len(models),len(lbdarr)))
#    for i in range(len(models)):
#        norm_model[i] = spec.linfit(lbdarr, models[i][1:-1])
#    models = norm_model
#
#
#    logF_grid = np.log10(models)  
#    
#    if  remove_partHa:
#        lbdarr1 = lbdarr[:149]
#        lbdarr2 = lbdarr[194:]
#        lbdarr = np.append(lbdarr1,lbdarr2)
#        novo_models = np.zeros((5500,191))
#        obs_new1 = obs_new[:149]
#        obs_new2 = obs_new[194:]
#        obs_new = np.append(obs_new1,obs_new2)
#        logF = np.log10(obs_new)
#        sigma_new2 = np.append(sigma_new[:149], sigma_new[194:])
#        dlogF = sigma_new2 / obs_new
#        i = 0
#        while i < len(models):
#            novo_models[i] = np.append(models[i][:149],models[i][194:])
#            i+=1
#        logF_grid = np.log10(novo_models)       
#    
#    if  only_centerline:
#        lbdarr = lbdarr[100:139]
#        logF = np.log10(obs_new[100:139])
#        novo_models = np.zeros((5500,39))
#        #sigma_new = np.zeros(len(lbdarr))
#        #sigma_new.fill(0.0017656218)
#        dlogF = sigma_new[100:139] / obs_new[100:139]
#        i = 0
#        while i < len(models):
#            novo_models[i] = models[i][100:139]
#            i+=1
#        logF_grid = np.log10(novo_models)
#    
#    
#    
#    
#    
#    if  only_wings:
#        Ha_peak = find_nearest2(lbdarr, 0.65628)
#        keep_a = find_nearest2(lbdarr, 0.65628 - 0.0009)
#        keep_b = find_nearest2(lbdarr, 0.65628 + 0.0009)
#        lbdarr1 = lbdarr[:keep_a]
#        lbdarr2 = lbdarr[keep_b:]
#        lbdarr = np.concatenate([lbdarr1,lbdarr2])
#        obs_new1 = obs_new[:keep_a]
#        obs_new2 = obs_new[keep_b:]
#        obs_neww = np.concatenate([obs_new1,obs_new2])
#        logF = np.log10(obs_neww)
#        sigma_new1 = sigma_new[:keep_a]
#        sigma_new2 = sigma_new[keep_b:]
#        sigma_new = np.concatenate([sigma_new1, sigma_new2])
#        dlogF = sigma_new/obs_neww
#        novo_models1 = models[:, :keep_a]
#        novo_models2 = models[:, keep_b:]
#        novo_models = np.hstack((novo_models1, novo_models2))
#        logF_grid = np.log10(novo_models)
#
#    return logF, dlogF, logF_grid, lbdarr, box_lim

# ==============================================================================
def read_iue(models, lbdarr):

    table =  folder_data + str( stars) + '/' + 'list_iue.txt'

    # os.chdir( folder_data + str( stars) + '/')
    if os.path.isfile(table) is False:
        os.system('ls ' +  folder_data + str( stars) +
                  '/*.FITS | xargs -n1 basename >' +
                   folder_data + str( stars) + '/' + 'list_iue.txt')
    
    iue_list = np.genfromtxt(table, comments='#', dtype='str')
    file_name = np.copy(iue_list)

    fluxes, waves, errors = [], [], []
    
    if file_name.tolist()[-3:] == 'csv':
        file_iue = str( folder_data) + str( stars) + '/' + str(file_name)
        wave, flux, sigma = np.loadtxt(str(file_iue), delimiter=',').T
        fluxes = np.concatenate((fluxes, flux*1e4), axis=0)
        waves = np.concatenate((waves, wave*1e-4), axis=0)
        errors = np.concatenate((errors, sigma*1e4), axis=0)

    else:
    # Combines the observations from all files in the folder, taking the good quality ones
        for k in range(len(file_name)):
            file_iue = str( folder_data) + str( stars) + '/' + str(file_name[k])
            hdulist = fits.open(file_iue)
            tbdata = hdulist[1].data
            wave = tbdata.field('WAVELENGTH') * 1e-4  # mum
            flux = tbdata.field('FLUX') * 1e4  # erg/cm2/s/A -> erg/cm2/s/mum
            sigma = tbdata.field('SIGMA') * 1e4  # erg/cm2/s/A -> erg/cm2/s/mum
    
            # Filter of bad data: '0' is good data
            qualy = tbdata.field('QUALITY')
            idx = np.where((qualy == 0))
            wave = wave[idx]
            sigma = sigma[idx]
            flux = flux[idx]
            
            idx = np.where((flux>0.))
            wave = wave[idx]
            sigma = sigma[idx]
            flux = flux[idx]
    
            fluxes = np.concatenate((fluxes, flux), axis=0)
            waves = np.concatenate((waves, wave), axis=0)
            errors = np.concatenate((errors, sigma), axis=0)

    if os.path.isdir( folder_fig + str( stars)) is False:
        os.mkdir( folder_fig + str( stars))

# ------------------------------------------------------------------------------
    # Would you like to cut the spectrum?
    #if  cut_iue_regions is True:
    #    wave_lim_min_iue = 0.13
    #    wave_lim_max_iue = 0.30
    #
    #    # Do you want to select a range to middle UV? (2200 bump region)
    #    wave_lim_min_bump_iue = 0.20  # 0.200 #0.195  #0.210 / 0.185
    #    wave_lim_max_bump_iue = 0.30  # 0.300 #0.230  #0.300 / 0.335
    #
    #    indx = np.where(((waves >= wave_lim_min_iue) &
    #                     (waves <= wave_lim_max_iue)))
    #    indx2 = np.where(((waves >= wave_lim_min_bump_iue) &
    #                      (waves <= wave_lim_max_bump_iue)))
    #    indx3 = np.concatenate((indx, indx2), axis=1)[0]
    #    waves, fluxes, errors = waves[indx3], fluxes[indx3], errors[indx3]

    #else: # remove observations outside the range
    #    wave_lim_min_iue = min(waves)
    #    wave_lim_max_iue = 0.300
    #    indx = np.where(((waves >= wave_lim_min_iue) &
    #                     (waves <= wave_lim_max_iue)))
    #    waves, fluxes, errors = waves[indx], fluxes[indx], errors[indx]

    # sort the combined observations in all files
    new_wave, new_flux, new_sigma = \
        zip(*sorted(zip(waves, fluxes, errors)))


    nbins = 200
    xbin, ybin, dybin = bin_data(new_wave, new_flux, nbins,
                                exclude_empty=True)

    # just to make sure that everything is in order
    ordem = xbin.argsort()
    wave = xbin[ordem]
    flux = ybin[ordem]
    sigma = dybin[ordem]


    return wave, flux, sigma


def combine_sed(wave, flux, sigma, models, lbdarr):
    if  lbd_range == 'UV':
        wave_lim_min = 0.13  # mum
        wave_lim_max = 0.3  # mum
    if  lbd_range == 'UV+VIS':
        wave_lim_min = 0.1  # mum
        wave_lim_max = 0.7  # mum
    if  lbd_range == 'UV+VIS+NIR':
        wave_lim_min = 0.1  # mum
        wave_lim_max = 5.0  # mum
    if  lbd_range == 'UV+VIS+NIR+MIR':
        wave_lim_min = 0.1  # mum
        wave_lim_max = 40.0  # mum
    if  lbd_range == 'UV+VIS+NIR+MIR+FIR':
        wave_lim_min = 0.1  # mum
        wave_lim_max = 350.  # mum
    if  lbd_range == 'UV+VIS+NIR+MIR+FIR+MICROW+RADIO':
        wave_lim_min = 0.1  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if  lbd_range == 'VIS+NIR+MIR+FIR+MICROW+RADIO':
        wave_lim_min = 0.39  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if  lbd_range == 'NIR+MIR+FIR+MICROW+RADIO':
        wave_lim_min = 0.7  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if  lbd_range == 'MIR+FIR+MICROW+RADIO':
        wave_lim_min = 5.0  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if  lbd_range == 'FIR+MICROW+RADIO':
        wave_lim_min = 40.0  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if  lbd_range == 'MICROW+RADIO':
        wave_lim_min = 1e3  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if  lbd_range == 'RADIO':
        wave_lim_min = 1e6  # mum
        wave_lim_max = np.max(lbdarr)  # mum


    band = np.copy( lbd_range)

    ordem = wave.argsort()
    wave = wave[ordem]
    flux = flux[ordem]
    sigma = sigma[ordem]

# ------------------------------------------------------------------------------
    # select lbdarr to coincide with lbd
    #models_new = np.zeros([len(models), len(wave)])
    
    idx = np.where((wave >= wave_lim_min) & (wave <= wave_lim_max))
    wave = wave[idx]
    flux = flux[idx]
    sigma = sigma[idx]
    models_new = np.zeros([len(models), len(wave)])
    
    for i in range(len(models)): # A interpolacao 
        models_new[i, :] = 10.**griddata(np.log10(lbdarr),
                                         np.log10(models[i]),
                                         np.log10(wave), method='linear')
    # to log space
    logF = np.log10(flux)
    dlogF = sigma / flux
    logF_grid = np.log10(models_new)
    
# ==============================================================================

    return logF, dlogF, logF_grid, wave

def read_votable():
    
    table =  folder_data + str( stars) + '/' + 'list.txt'

    #os.chdir(folder_data + str(star) + '/')
    #if os.path.isfile(table) is False or os.path.isfile(table) is True:
    os.system('ls ' +  folder_data + str( stars) +
                '/*.xml | xargs -n1 basename >' +
                 folder_data + str( stars) + '/' + 'list.txt')
    vo_list = np.genfromtxt(table, comments='#', dtype='str')
    table_name = np.copy(vo_list)
    vo_file =  folder_data + str( stars) + '/' + str(table_name)
    #thefile = 'data/HD37795/alfCol.sed.dat' #folder_data + str(star) + '/' + str(table_name)
    #table = np.genfromtxt(thefile, usecols=(1, 2, 3))
    #wave, flux, sigma = table[:,0], table[:,1], table[:,2]
    
    try:
        t1 = atpy.Table(vo_file)
        wave = t1['Wavelength'][:]  # Angstrom
        flux = t1['Flux'][:]  # erg/cm2/s/A
        sigma = t1['Error'][:]  # erg/cm2/s/A
    except:
        t1 = atpy.Table(vo_file, tid=1)
        wave = t1['SpectralAxis0'][:]  # Angstrom
        flux = t1['Flux0'][:]  # erg/cm2/s/A
        sigma = [0.] * len(flux)  # erg/cm2/s/A

    new_wave, new_flux, new_sigma = zip(*sorted(zip(wave, flux, sigma)))

    new_wave = list(new_wave)
    new_flux = list(new_flux)
    new_sigma = list(new_sigma)

    # Filtering null sigmas
    #for h in range(len(new_sigma)):
    #    if new_sigma[h] == 0.:
    #        new_sigma[h] = 0.002 * new_flux[h]
    

    wave = np.copy(new_wave) * 1e-4
    flux = np.copy(new_flux) * 1e4
    sigma = np.copy(new_sigma) * 1e4
    keep = wave > 0.34
    wave, flux, sigma = wave[keep], flux[keep], sigma[keep]
    
    if  stars == 'HD37795':
        fname =  folder_data + '/HD37795/alfCol.txt'
        data = np.loadtxt(fname, dtype={'names':('lbd', 'flux', 'dflux', 'source'), 'formats':(np.float, np.float, np.float, '|S20')})
        wave = np.hstack([wave, data['lbd']])
        flux = np.hstack([flux, jy2cgs(1e-3*data['flux'], data['lbd'])])
        sigma = np.hstack([sigma, jy2cgs(1e-3*data['dflux'], data['lbd'])])

    return wave, flux, sigma

def find_lim():
    if  model == 'aeri':
        if  UV:
            if  include_rv and not  binary_star:
                lim = 3
            elif  include_rv and  binary_star:
                lim = 4
            elif  binary_star and not  include_rv:
                lim = 3
            else:
                lim = 2
        else:
            if  binary_star:
                lim = 1
            else:
                lim = -4
    if  model == 'acol':
        if  UV:
            if  include_rv and not  binary_star:
                lim = 3
            elif  include_rv and  binary_star:
                lim = 4
            elif  binary_star and not  include_rv:
                lim = 3
            else:
                lim = 2
        else:
            if  binary_star:
                lim = 1
            else:
                lim = -7
    
    return lim


# ==============================================================================
# Calls the xdr reading function
def read_models(lista_obs):

    if  model == 'aeri':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_BAphot2_xdr(lista_obs)
    if  model == 'befavor':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_befavor_xdr()
    if  model == 'aara':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_aara_complete()
    if  model == 'beatlas':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_beatlas_xdr()
    if  model == 'acol':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_acol_Ha_xdr(lista_obs)

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig

#=======================================================================
# Sigmaclip routine by Jonathan Labadie-Bartz

def Sliding_Outlier_Removal(x, y, window_size, sigma=3.0, iterate=1):
  # remove NANs from the data
  x = x[~np.isnan(y)]
  y = y[~np.isnan(y)]

  #make sure that the arrays are in order according to the x-axis 
  y = y[np.argsort(x)]
  x = x[np.argsort(x)]

  # tells you the difference between the last and first x-value
  x_span = x.max() - x.min()  
  i = 0
  x_final = x
  y_final = y
  while i < iterate:
    i+=1
    x = x_final
    y = y_final
    
    # empty arrays that I will append not-clipped data points to
    x_good_ = np.array([])
    y_good_ = np.array([])
    
    # Creates an array with all_entries = True. index where you want to remove outliers are set to False
    tf_ar = np.full((len(x),), True, dtype=bool)
    ar_of_index_of_bad_pts = np.array([]) #not used anymore
    
    #this is how many days (or rather, whatever units x is in) to slide the window center when finding the outliers
    slide_by = window_size / 5.0 
    
    #calculates the total number of windows that will be evaluated
    Nbins = int((int(x.max()+1) - int(x.min()))/slide_by)
    
    for j in range(Nbins+1):
        #find the minimum time in this bin, and the maximum time in this bin
        x_bin_min = x.min()+j*(slide_by)-0.5*window_size
        x_bin_max = x.min()+j*(slide_by)+0.5*window_size
        
        # gives you just the data points in the window
        x_in_window = x[(x>x_bin_min) & (x<x_bin_max)]
        y_in_window = y[(x>x_bin_min) & (x<x_bin_max)]
        
        # if there are less than 5 points in the window, do not try to remove outliers.
        if len(y_in_window) > 5:
            
            # Removes a linear trend from the y-data that is in the window.
            y_detrended = detrend(y_in_window, type='linear')
            y_in_window = y_detrended
            #print(np.median(m_in_window_))
            y_med = np.median(y_in_window)
            
            # finds the Median Absolute Deviation of the y-pts in the window
            y_MAD = MAD(y_in_window)
            
            #This mask returns the not-clipped data points. 
            # Maybe it is better to only keep track of the data points that should be clipped...
            mask_a = (y_in_window < y_med+y_MAD*sigma) & (y_in_window > y_med-y_MAD*sigma)
            #print(str(np.sum(mask_a)) + '   :   ' + str(len(m_in_window)))
            y_good = y_in_window[mask_a]
            x_good = x_in_window[mask_a]
            
            y_bad = y_in_window[~mask_a]
            x_bad = x_in_window[~mask_a]
            
            #keep track of the index --IN THE ORIGINAL FULL DATA ARRAY-- of pts to be clipped out
            try:
                clipped_index = np.where([x == z for z in x_bad])[1]
                tf_ar[clipped_index] = False
                ar_of_index_of_bad_pts = np.concatenate([ar_of_index_of_bad_pts, clipped_index])
            except IndexError:
                #print('no data between {0} - {1}'.format(x_in_window.min(), x_in_window.max()))
                pass
        # puts the 'good' not-clipped data points into an array to be saved
        
        #x_good_= np.concatenate([x_good_, x_good])
        #y_good_= np.concatenate([y_good_, y_good])
        
        #print(len(mask_a))
        #print(len(m
        #print(m_MAD)
    
    ##multiple data points will be repeated! We don't want this, so only keep unique values. 
    #x_uniq, x_u_indexs = np.unique(x_good_, return_index=True)
    #y_uniq = y_good_[x_u_indexs]
    
    ar_of_index_of_bad_pts = np.unique(ar_of_index_of_bad_pts)
    #print('step {0}: remove {1} points'.format(i, len(ar_of_index_of_bad_pts)))
    #print(ar_of_index_of_bad_pts)
    
    #x_bad = x[ar_of_index_of_bad_pts]
    #y_bad = y[ar_of_index_of_bad_pts]
    #x_final = x[
    
    x_final = x[tf_ar]
    y_final = y[tf_ar]
  return(x_final, y_final)

#=======================================================================
# Creates file tag

def create_tag():

#def create_tag(model,Ha, Hb, Hd, Hg):    
    tag = '+' +  model
    

    if  only_wings:
        tag = tag + '_onlyWings'
    if  only_centerline:
        tag = tag + '_onlyCenterLine'
    if  remove_partHa:
        tag = tag + '_removePartHa'
    if  Sigma_Clip is True:
        tag = tag + '_SigmaClipData'
    if  vsini_prior:
        tag = tag + '_vsiniPrior'
    if  dist_prior:
        tag = tag + '_distPrior'
    if  box_W:
        tag = tag + '_boxW'
    if  incl_prior:
        tag = tag + '_inclPrior'
    if  UV:
        tag = tag + '+UV'
    if  Ha:
        tag = tag + '+Ha'
    if  Hb:
        tag = tag + '+Hb'
    if  Hd:
        tag = tag + '+Hd'
    if  Hg:
        tag = tag + '+Hg'
                    
    return tag

#=======================================================================
# Creates list of observables

def create_list():
    
    lista = np.array([])
    
    if  UV:
        lista = np.append(lista,'UV')
    if  Ha:
        lista = np.append(lista,'Ha')
    if  Hb:
        lista = np.append(lista,'Hb')
    if  Hd:
        lista = np.append(lista,'Hd')
    if  Hg:
        lista = np.append(lista,'Hg')
        
    return lista
    
#=======================================================================
# Check if an observable is in the list

def check_list(lista_obs, x):
    a = np.where(lista_obs == x)
    if(len(a[0]) == 0):
        return False
    else:
        return True

#========================================================================

def read_table():
    table =  folder_data + str( stars) + '/sed_data/' + 'list_sed.txt'

    #os.chdir(folder_data + str(star) + '/')
    #if os.path.isfile(table) is False or os.path.isfile(table) is True:
    os.system('ls ' +  folder_data + str( stars) + '/sed_data/*.dat /*.csv /*.txt | xargs -n1 basename >' +
                 folder_data + str( stars) + '/sed_data/' + 'list_sed.txt')
    vo_list = np.genfromtxt(table, comments='#', dtype='str')
    table_name = np.copy(vo_list)
    table =  folder_data + str( stars) + '/sed_data/' + str(table_name)
    

    typ = (0, 1, 2, 3)

    a = np.genfromtxt(table, usecols=typ, unpack=True,
                      delimiter='\t', comments='#',
                      dtype={'names': ('B', 'V', 'R', 'I'),
                             'formats': ('f4', 'f4', 'f4', 'f4')})
    
    B_mag, V_mag, R_mag, I_mag = a['B'], a['V'], a['R'], a['I']

    # lbd array (center of bands)
    wave = np.array([0.4361, 0.5448, 0.6407, 0.7980])

    c = 299792458e6 #um/s

	# Change observed magnitude to flux
	# Zero magnitude flux 10^-20.erg.s^-1.cm^2.Hz^-1  ---> convert to erg.s^-1.cm^2.um^-1
    B_flux0 = 4.26*10**(-20)
    V_flux0 = 3.64*10**(-20)
    R_flux0 = 3.08*10**(-20)
    I_flux0 = 2.55*10**(-20)
	
    B_flux = B_flux0*10.**(- B_mag/2.5)
    V_flux = V_flux0*10.**(- V_mag/2.5) 
    R_flux = R_flux0*10.**(- R_mag/2.5) 
    I_flux = I_flux0*10.**(- I_mag/2.5) 
    
	#---> convert to erg.s^-1.cm^2.um^-1
    B_flux = B_flux*c/(wave[0]**2)
    V_flux = V_flux*c/(wave[1]**2) 
    R_flux = R_flux*c/(wave[2]**2) 
    I_flux = I_flux*c/(wave[3]**2)
	
    flux = np.array([B_flux, V_flux, R_flux, I_flux])
    logF = np.log10(np.array([B_flux, V_flux, R_flux, I_flux]))
	# Uncertainty (?)	
    #dlogF = 0.01*logF  
    sigma = 0.01 * flux
    
    
    return wave, flux, sigma  

#=======================================================================
# Read the data files

def read_observables(models, lbdarr, lista_obs):
   
    logF_combined = []
    dlogF_combined = []
    logF_grid_combined = []
    wave_combined = []
    box_lim_combined = []
    

    if check_list(lista_obs, 'UV'):     
        
        u = np.where(lista_obs == 'UV')
        index = u[0][0]
        
        if  iue:
            wave, flux, sigma = read_iue(models[index], lbdarr[index])
        else:
            wave, flux, sigma = [], [], []

        if  votable:
            wave0, flux0, sigma0 = read_votable()
        elif  data_table:
            wave0, flux0, sigma0 = read_table()
        else:    
            wave0, flux0, sigma0 = [], [], []
            
        wave = np.hstack([wave0, wave])
        flux = np.hstack([flux0, flux])
        sigma = np.hstack([sigma0, sigma])
        
        logF_UV, dlogF_UV, logF_grid_UV, wave_UV =\
            combine_sed(wave, flux, sigma, models[index], lbdarr[index])

        logF_combined.append(logF_UV)
        dlogF_combined.append(dlogF_UV)
        logF_grid_combined.append(logF_grid_UV) 
        wave_combined.append(wave_UV)


    if check_list(lista_obs, 'Ha'):
        
        u = np.where(lista_obs == 'Ha')
        index = u[0][0]
        logF_Ha, dlogF_Ha, logF_grid_Ha, wave_Ha, box_lim_Ha =\
            read_line_spectra(models[index], lbdarr[index], 'Ha')

        logF_combined.append(logF_Ha)
        dlogF_combined.append(dlogF_Ha)
        logF_grid_combined.append(logF_grid_Ha) 
        wave_combined.append(wave_Ha)
        box_lim_combined.append(box_lim_Ha)
        
    if check_list(lista_obs, 'Hb'):

        u = np.where(lista_obs == 'Hb')
        index = u[0][0]
        logF_Hb, dlogF_Hb, logF_grid_Hb, wave_Hb, box_lim_Hb =\
            read_line_spectra(models[index], lbdarr[index], 'Hb')

        logF_combined.append(logF_Hb)
        dlogF_combined.append(dlogF_Hb)
        logF_grid_combined.append(logF_grid_Hb) 
        wave_combined.append(wave_Hb)
        box_lim_combined.append(box_lim_Hb)
    
    if check_list(lista_obs, 'Hd'):

        u = np.where(lista_obs == 'Hd')
        index = u[0][0]
        logF_Hd, dlogF_Hd, logF_grid_Hd, wave_Hd, box_lim_Hd =\
            read_line_spectra(models[index], lbdarr[index], 'Hd')

        logF_combined.append(logF_Hd)
        dlogF_combined.append(dlogF_Hd)
        logF_grid_combined.append(logF_grid_Hd) 
        wave_combined.append(wave_Hd)
        box_lim_combined.append(box_lim_Hd)
    
    if check_list(lista_obs, 'Hg'):

        u = np.where(lista_obs == 'Hg')
        index = u[0][0]
        logF_Hg, dlogF_Hg, logF_grid_Hg, wave_Hg, box_lim_Hg =\
            read_line_spectra(models[index], lbdarr[index], 'Hg')

        logF_combined.append(logF_Hg)
        dlogF_combined.append(dlogF_Hg)
        logF_grid_combined.append(logF_grid_Hg) 
        wave_combined.append(wave_Hg)
        box_lim_combined.append(box_lim_Hg)

    return logF_combined, dlogF_combined, logF_grid_combined, wave_combined, box_lim_combined


# ==============================================================================
# General Options
a_parameter =       1.4   # Set internal steps of each walker
extension =         '.png'  # Figure  extension to be saved
include_rv =        False  # If False: fix Rv = 3.1, else Rv will be inferead
af_filter =         False  # Remove walkers outside the range 0.2 < af < 0.5
long_process =      False  # Run with few walkers or many?
list_of_stars =     'HD37795'  # Star name
Nsigma_dis =        3.  # Set the range of values for the distance
model =             'acol'  # 'beatlas', 'befavor', 'aara', or 'acol'


binary_star = False

folder_data = '../data/'
folder_fig = '../figures/'
folder_defs = '../defs/'
folder_tables = '../tables/'
folder_models = '../models/'


vsini_prior =   False # Uses a gaussian vsini prior
dist_prior =    True # Uses a gaussian distance prior

box_W =         False # Constrain the W lower limit, not actual a prior, but restrain the grid
box_W_min, box_W_max = [0.6, 'max']

incl_prior =    False # Uses a gaussian inclination prior 

normal_spectra =    True # True if the spectra is normalized (for lines), distance and e(b-v) are not computed
only_wings =        False # Run emcee with only the wings
only_centerline =   False # Run emcee with only the center of the line
Sigma_Clip =        True # If you want telluric lines/outilier points removed
remove_partHa =     False # Remove a lbd interval in  Halpha that has too much absorption (in the wings)

# Line and continuum combination

UV =            False
iue =           False
votable =       False
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
acrux = False  # If True, it will run in Nproc processors in the cluster
Nproc = 24  # Number of processors to be used in the cluster

# ==============================================================================

# ==============================================================================
# Read the list of  stars and takes the prior information about them from the stars_table txt file
def read_stars(stars_table):

    typ = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    file_data = folder_data + stars_table

    a = np.genfromtxt(file_data, usecols=typ, unpack=True,
                      comments='#',
                      dtype={'names': ('star', 'plx', 'sig_plx', 'vsini',
                                       'sig_vsini', 'pre_ebmv', 'inc', 'sinc', 'lbd_range'),
                             'formats': ('S9', 'f2', 'f2', 'f4',
                                         'f4', 'f4', 'f4', 'f4', 
                                         'U24')})

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

#############################################################################################################################
if corner_color == 'random' or corner_color == '':
    corner_color = random.choice(['blue', 'dark blue', 'teal', 'green', 'yellow', 'orange', 'red', 'purple', 'violet', 'pink'])
#############################################################################################################################


lista_obs = create_list() 

tags = create_tag()

print(tags)

ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_models(lista_obs)

grid_priors, pdf_priors = read_stellar_prior()


ranges, dist_pc, sig_dist_pc, vsin_obs, sig_vsin_obs,\
    Ndim = read_star_info(stars, lista_obs, listpar)


logF, dlogF, logF_grid, wave, box_lim =\
        read_observables(models, lbdarr, lista_obs)


if  long_process is True:
    Nwalk = 100  # 200  # 500
    nint_burnin = 300  # 50
    nint_mcmc = 1000  # 500  # 1000
else:
    Nwalk = 20
    nint_burnin = 30
    nint_mcmc = 80
    

    
p0 = [np.random.rand( Ndim) * ( ranges[:, 1] -  ranges[:, 0]) +
       ranges[:, 0] for i in range(Nwalk)]

start_time = time.time()




if  acrux is True:
    sampler = emcee.EnsembleSampler(Nwalk,  Ndim, lnprob,  moves=[(emcee.moves.StretchMove( a_parameter))],
                                    pool=pool)
else:
    sampler = emcee.EnsembleSampler(Nwalk,  Ndim, lnprob, moves=[(emcee.moves.StretchMove( a_parameter))])

sampler_tmp = run_emcee(p0, sampler, nint_burnin, nint_mcmc,
                         Ndim, Nwalk, file_name= stars)
                         
print("--- %s minutes ---" % ((time.time() - start_time) / 60))

sampler, params_fit, errors_fit, maxprob_index,\
    minprob_index, af, file_npy_burnin = sampler_tmp

chain = sampler.chain

if  af_filter is True:
    acceptance_fractions = sampler.acceptance_fraction
    chain = chain[(acceptance_fractions >= 0.20) &
                  (acceptance_fractions <= 0.50)]
    af = acceptance_fractions[(acceptance_fractions >= 0.20) &
                              (acceptance_fractions <= 0.50)]
    af = np.mean(af)

af = str('{0:.2f}'.format(af))

# Saving first sample
file_npy =  folder_fig + str( stars) + '/' + 'Walkers_' +\
    str(Nwalk) + '_Nmcmc_' + str(nint_mcmc) +\
    '_af_' + str(af) + '_a_' + str( a_parameter) +\
     tags + ".npy"
np.save(file_npy, chain)



flatchain_1 = chain.reshape((-1,  Ndim)) # cada walker em cada step em uma lista só


samples = np.copy(flatchain_1)



for i in range(len(samples)):
   

    if  model == 'acol':
        samples[i][1] = obl2W(samples[i][1])
        samples[i][2] = hfrac2tms(samples[i][2])
        samples[i][6] = (np.arccos(samples[i][6])) * (180. / np.pi)

    if  model == 'aeri':
        # Converting angles to degrees
        samples[i][3] = (np.arccos(samples[i][3])) * (180. / np.pi)


# plot corner
quantiles = [0.16, 0.5, 0.84]

if  model == 'aeri':
    if check_list( lista_obs, 'UV'):
        labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
              r'$i[\mathrm{^o}]$', r'$\pi\,[mas]$', r'E(B-V)']
        if  include_rv is True:
            labels = labels + [r'$R_\mathrm{V}$']

    else:
        labels = [r'$M\,[M_\odot]$', r'$W$', r"$t/t_\mathrm{ms}$",
              r'$i[\mathrm{^o}]$']
    labels2 = labels
    
if  model == 'acol':
    if check_list( lista_obs, 'UV'):
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
        if  include_rv is True:
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
        
        
if  binary_star:
        labels = labels + [r'$M2\,[M_\odot]$']
        labels2 = labels2 + [r'$M2\,[M_\odot]$']

if  corner_color == 'blue':
    truth_color='xkcd:cobalt'
    color='xkcd:cornflower'
    color_hist='xkcd:powder blue'
    color_dens='xkcd:clear blue'
    
elif  corner_color == 'dark blue':
    truth_color='xkcd:deep teal'
    color='xkcd:dark teal'
    color_hist='xkcd:pale sky blue'
    color_dens='xkcd:ocean'
    
elif  corner_color == 'teal':
    truth_color='xkcd:charcoal'
    color='xkcd:dark sea green'
    color_hist='xkcd:pale aqua'
    color_dens='xkcd:seafoam blue'
    
elif  corner_color == 'green':
    truth_color='xkcd:forest'
    color='xkcd:forest green'
    color_hist='xkcd:light grey green'
    color_dens='xkcd:grass green'
    
elif  corner_color == 'yellow':
    truth_color='xkcd:mud brown'
    color='xkcd:sandstone'
    color_hist='xkcd:pale gold'
    color_dens='xkcd:sunflower'
    
elif  corner_color == 'orange':
    truth_color='xkcd:chocolate'
    color='xkcd:cinnamon'
    color_hist='xkcd:light peach'
    color_dens='xkcd:bright orange'
    
elif  corner_color == 'red':
    truth_color='xkcd:mahogany'
    color='xkcd:deep red'
    color_hist='xkcd:salmon'
    color_dens='xkcd:reddish'
    
elif  corner_color == 'purple':
    truth_color='xkcd:deep purple'
    color='xkcd:medium purple'
    color_hist='xkcd:soft purple'
    color_dens='xkcd:plum purple'
    
elif  corner_color == 'violet':
    truth_color='xkcd:royal purple'
    color='xkcd:purpley'
    color_hist='xkcd:pale violet'
    color_dens='xkcd:blue violet'
    
elif  corner_color == 'pink':
    truth_color='xkcd:wine'
    color='xkcd:pinky'
    color_hist='xkcd:light pink'
    color_dens='xkcd:pink red'


new_ranges = np.copy( ranges)

if  model == 'aeri':

    new_ranges[3] = (np.arccos( ranges[3])) * (180. / np.pi)
    new_ranges[3] = np.array([ ranges[3][1],  ranges[3][0]])
if  model == 'acol':
    new_ranges[1] = obl2W( ranges[1])            
    new_ranges[2][0] = hfrac2tms( ranges[2][1])
    new_ranges[2][1] = hfrac2tms( ranges[2][0])
    new_ranges[6] = (np.arccos( ranges[6])) * (180. / np.pi)
    new_ranges[6] = np.array([new_ranges[6][1], new_ranges[6][0]])



best_pars = []
best_errs = []
for i in range( Ndim):
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
if  compare_results:
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





current_folder = str( folder_fig) + str( stars) + '/'
fig_name = 'Walkers_' + np.str(Nwalk) + '_Nmcmc_' +\
    np.str(nint_mcmc) + '_af_' + str(af) + '_a_' +\
    str( a_parameter) +  tags

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
