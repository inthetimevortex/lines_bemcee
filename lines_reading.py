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
import user_settings as flag



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
    isig = None # photospheric flag.model is None

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
    xdrPL = flag.folder_models + 'BAphot__UV2M_Ha_Hb_Hg_Hd.xdr'
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
        if flag.votable or flag.data_table:
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
    keep_a = find_nearest2(lbdarr, lbdc - 0.007)
    keep_b = find_nearest2(lbdarr, lbdc + 0.007)
    lbd_line = lbdarr[keep_a:keep_b]
    models_line = models[:, keep_a:keep_b]
    lbdarr_line = lbd_line*1e4
    lbdarr_line = lbdarr_line/(1.0 + 2.735182E-4 + 131.4182/lbdarr_line**2 + 2.76249E8/lbdarr_line**4)
    models_combined.append(models_line)
    lbd_combined.append(lbdarr_line*1e-4)

    return models_combined, lbd_combined


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
    xdrPL = flag.folder_models + 'aara_sed.xdr'  # 


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
    xdrPL = flag.folder_models + 'BeFaVOr.xdr'
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
#    flag.folder_models = 'models/'
#
#    #dims = ['M', 'W', 'tms', 'sig0', 'cosi']
#    #dims = dict(zip(dims, range(len(dims))))
#    #isig = dims["sig0"]
#    dims = ['M', 'W', 'tms', 'cosi']
#    dims = dict(zip(dims, range(len(dims))))
#    isig = None # photospheric flag.model is None
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
#    xdrPL = flag.folder_models + 'BAphot_v2_UV.xdr'
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
#    flag.folder_models = 'models/'
#
#    dims = ['M', 'W', 'tms', 'cosi']
#    dims = dict(zip(dims, range(len(dims))))
#    isig = None # photospheric flag.model is None
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
#    xdrPL = flag.folder_models + 'BAphot_UV2M_flag.Halpha.xdr'
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
#    flag.folder_models = 'models/'
#    xdrPL = flag.folder_models + 'acol_Ha_only.xdr'
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
#    lbdarr = lbdarr*1e4 # to Angstrom # estava lbdarr[255:495], que eh o flag.Halpha completo de fato
#    lbdarr = lbdarr/(1.0 + 2.735182E-4 + 131.4182/lbdarr**2 + 2.76249E8/lbdarr**4) # valid if lbd in Angstrom
#    lbdarr = lbdarr*1e-4 # to mum again
#
#
#    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig

# ==============================================================================
# Function to read the flag.Halpha part of the complete xdr (made with bat.createXDRsed)
#def read_BAphot2_flag.Halpha_xdr():
#
#    flag.folder_models = 'models/'
#
#    dims = ['M', 'W', 'tms', 'cosi']
#    dims = dict(zip(dims, range(len(dims))))
#    isig = None # photospheric flag.model is None
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
#    xdrPL = flag.folder_models + 'BAphot_UV2M_flag.Halpha.xdr'
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
#    # flag.Halpha is from lbdarr[255] to lbdarr[495]
#    j = 0
#    shape = (5500,238) # Estava (5500,240)
#    models_flag.Halpha = np.zeros(shape)
#    while j < 5500:
#        models_flag.Halpha[j] = models[j][256:494] # estava models[j][255:495]
#        j += 1
#    
#    listpar = [np.array([1.7000000476837158,2.0,2.5,3.0,4.0,5.0,7.0,9.0,12.0,15.0]), np.array([0.0,0.33000001311302185,0.4699999988079071,0.5699999928474426,0.6600000262260437,0.7400000095367432,0.8100000023841858,0.8700000047683716,0.9300000071525574]), np.array([0.0,0.25,0.5,0.75,1.0]), np.array([0,0.11146999895572662,0.22155000269412994,0.3338100016117096,0.44464001059532166,0.5548400282859802,0.6665300130844116,0.7782400250434875,0.8886200189590454,1.0])]
#
#    # Change vacuum wavelength to air wavelength
#    lbdarr = lbdarr[256:494]*1e4 # to Angstrom # estava lbdarr[255:495], que eh o flag.Halpha completo de fato
#    lbdarr = lbdarr/(1.0 + 2.735182E-4 + 131.4182/lbdarr**2 + 2.76249E8/lbdarr**4) # valid if lbd in Angstrom
#    lbdarr = lbdarr*1e-4 # to mum again
#    
#
#    return ctrlarr, minfo, models_flag.Halpha, lbdarr, listpar, dims, isig
    
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


    xdrPL = flag.folder_models + 'disk_flx.xdr'  # 'PL.xdr'

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

    xdrPL = flag.folder_models + 'acol_06.xdr'

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

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig


# ==============================================================================
def read_star_info(star, lista_obs, listpar):

        print(75 * '=')

        star_r = star

        print('\nRunning star: %s\n' % star_r)
        print(75 * '=')

        plx = np.copy(flag.list_plx)
        dplx = np.copy(flag.list_sig_plx)
        vsin_obs = np.copy(flag.list_vsini_obs)
        sig_vsin_obs = np.copy(flag.list_sig_vsini_obs)
        
        
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

        
        if check_list(lista_obs, 'UV') or flag.normal_spectra is False:
            if flag.include_rv:
                ebmv, rv = [[0.0, 0.8], [1., 5.8]]
            else:
                rv = 3.1
                ebmv, rv = [[0.0, 0.8], None]
            
            dist_min = dist_pc - flag.Nsigma_dis * sig_dist_pc
            dist_max = dist_pc + flag.Nsigma_dis * sig_dist_pc
    
            #if dist_min < 0:
            #    dist_min = 1    
            
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

        if flag.box_W:
            if flag.box_W_max == 'max':
                ranges[1][0] = flag.box_W_min
            elif flag.box_W_min == 'min':
                ranges[1][1] = flag.box_W_max
            else:
                ranges[1][0], ranges[1][1] = flag.box_W_min, flag.box_W_max
        
        Ndim = len(ranges)

        return ranges, dist_pc, sig_dist_pc, vsin_obs,\
            sig_vsin_obs, Ndim

# ==============================================================================
# function that returns the interpolated spectra from files
#def read_Hg(models, lbdarr, flag.folder_data, folder_fig, star): # flag.Halpha is boolean
#
#    table = flag.folder_data + str(flag.stars) + '/' + 'spectra/' + 'list_spectra.txt'
#    
#    if os.path.isfile(table) is False or os.path.isfile(table) is True:
#        os.system('ls ' + flag.folder_data + str(flag.stars) + '/spectra' + '/* | xargs -n1 basename >' + flag.folder_data + '/' + 'list_feros.txt')
#        spectra_list = np.genfromtxt(table, comments='#', dtype='str')
#        file_name = np.copy(spectra_list)
#    
#    fluxes, waves, errors = [], [], []
#    #
#    ## Collecting just the first file
#    #for k in range(1):
#    #    file_feros = str(flag.folder_data) + str(flag.stars) + '/' + 'feros/' + str(file_name)
#    wl, normal_flux, MJD = read_espadons(lines[0])
#
#    
#    #--------- Selecting an approximate wavelength interval for flag.Halpha
#    c = 299792.458 #km/s
#    lbd_central = 4340.462  
#        
#    if flag.Sigma_Clip:
#        #gives the line profiles in velocity space
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
#        # flag.Halpha line 
#        wl = wl[inf:sup]
#        fluxes = normal_flux[inf:sup]
#
#    radv = delta_v(vel, fluxes, 'Hg')
#    print('RADIAL VELOCITY = {0}'.format(radv))
#    vel = vel - radv
#    wl = c*lbd_central/(c - vel)
#    # Changing the units
#    waves = wl * 1e-4 # mum
#    errors = np.zeros(len(waves)) # nao tenho informacao sobre os erros ainda
#
#     # Aqui faz o corte pra pegar so o que ta no intervalo de lbd da grade (lbdarr)
#    idx = np.where((waves >= np.min(lbdarr)-0.001) & (waves <= np.max(lbdarr)+0.001))
#    wave = waves[idx]
#    flux = fluxes[idx]
#    sigma = errors[idx]
#    obs_new = np.zeros(len(lbdarr))
#    
#    new_wave = waves
#    new_flux = fluxes
#
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
#    #sigma_new.fill(10*0.006613449)
#
#    for i in range(len(box_lim) - 1):
#        # lbd observations inside the box
#        index = np.argwhere((new_wave > box_lim[i]) & (new_wave < box_lim[i+1]))
#        if len(index) == 0:
#           bin_flux[i] = -np.inf
#        elif len(index) == 1:
#            bin_flux[i] = new_flux[index[0][0]]
#        else: 
#            # Calculating the mean flux in the box
#            bin_flux[i] = np.sum(new_flux[index[0][0]:index[-1][0]])/len(new_flux[index[0][0]:index[-1][0]])
#            sigma_new[i] = 10*0.006613449/np.sqrt(len(index))
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
#    norm_flag.model = np.zeros((len(models),len(lbdarr)))
#    for i in range(len(models)):
#        norm_model[i] = spec.linfit(lbdarr, models[i][1:-1])
#    models = norm_model
#
#
#    logF_grid = np.log10(models)  
#        
#    
#    #THIS IS NOT GENERAL!!!
#    if flag.only_centerline:
#        lbdarr = lbdarr[100:139]
#        logF = np.log10(obs_new[100:139])
#        novo_models = np.zeros((5500,39))
#        dlogF = sigma_new[100:139] / obs_new[100:139]
#        i = 0
#        while i < len(models):
#            novo_models[i] = models[i][100:139]
#            i+=1
#        logF_grid = np.log10(novo_models)
#    
#    if flag.only_wings:
#        Hd_peak = find_nearest2(lbdarr, 0.434046)
#        keep_a = find_nearest2(lbdarr, 0.434046 - 0.0004)
#        keep_b = find_nearest2(lbdarr, 0.434046 + 0.0004)
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
#    
#        logF_grid = np.log10(novo_models)
#
#    return logF, dlogF, logF_grid, lbdarr, box_lim
#
#
#def read_Hd(models, lbdarr, flag.folder_data, folder_fig, star,
#                      model, flag.only_wings, flag.only_centerline, flag.Sigma_Clip): # flag.Halpha is boolean
#
#    table = flag.folder_data + str(flag.stars) + '/' + 'spectra/' + 'list_spectra.txt'
#    
#    if os.path.isfile(table) is False or os.path.isfile(table) is True:
#        os.system('ls ' + flag.folder_data + str(flag.stars) + '/spectra' + '/* | xargs -n1 basename >' + flag.folder_data + '/' + 'list_feros.txt')
#        spectra_list = np.genfromtxt(table, comments='#', dtype='str')
#        file_name = np.copy(spectra_list)
#    
#    fluxes, waves, errors = [], [], []
#    #
#    ## Collecting just the first file
#    #for k in range(1):
#    #    file_feros = str(flag.folder_data) + str(flag.stars) + '/' + 'feros/' + str(file_name)
#    wl, normal_flux, MJD = read_espadons(lines[0])
#    
#    #--------- Selecting an approximate wavelength interval for flag.Halpha
#    c = 299792.458 #km/s
#    lbd_central = 4101.74   
#        
#    if flag.Sigma_Clip:
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
#        # flag.Halpha line 
#        wl = wl[inf:sup]
#        fluxes = normal_flux[inf:sup]
#
#    radv = delta_v(vel, fluxes, 'Hd')
#    print('RADIAL VELOCITY = {0}'.format(radv))
#    vel = vel - radv
#    wl = c*lbd_central/(c - vel)
#    # Changing the units
#    waves = wl * 1e-4 # mum
#    errors = np.zeros(len(waves)) # nao tenho informacao sobre os erros ainda
#
#    if flag.model == 'beatlas' or flag.model == 'aara' or flag.model == 'aeri': # Aqui faz o corte pra pegar so o que ta no intervalo de lbd da grade (lbdarr)
#        idx = np.where((waves >= np.min(lbdarr)-0.001) & (waves <= np.max(lbdarr)+0.001))
#        wave = waves[idx]
#        flux = fluxes[idx]
#        sigma = errors[idx]
#        obs_new = np.zeros(len(lbdarr))
#    
#
#
#    new_wave = waves
#    new_flux = fluxes
#
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
#           bin_flux[i] = -np.inf
#        elif len(index) == 1:
#            bin_flux[i] = new_flux[index[0][0]]
#        else: 
#            # Calculating the mean flux in the box
#            bin_flux[i] = np.sum(new_flux[index[0][0]:index[-1][0]])/len(new_flux[index[0][0]:index[-1][0]])
#            sigma_new[i] = 10*0.00805128748/np.sqrt(len(index))
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
#    norm_flag.model = np.zeros((len(models),len(lbdarr)))
#    for i in range(len(models)):
#        norm_model[i] = spec.linfit(lbdarr, models[i][1:-1])
#    models = norm_model
#
#
#    logF_grid = np.log10(models)  
#        
#    
#    if flag.only_centerline:
#        lbdarr = lbdarr[100:139]
#        logF = np.log10(obs_new[100:139])
#        novo_models = np.zeros((5500,39))
#        dlogF = sigma_new[100:139] / obs_new[100:139]
#        i = 0
#        while i < len(models):
#            novo_models[i] = models[i][100:139]
#            i+=1
#        logF_grid = np.log10(novo_models)
#    
#    if flag.only_wings:
#        Hd_peak = find_nearest2(lbdarr, 0.4102)
#        keep_a = find_nearest2(lbdarr, 0.41017 - 0.0003)
#        keep_b = find_nearest2(lbdarr, 0.41017 + 0.0003)
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
#
#
#
#def read_Hb(models, lbdarr, flag.folder_data, folder_fig, star,
#                      model, flag.only_wings, flag.only_centerline, flag.Sigma_Clip): # flag.Halpha is boolean
#
#    table = flag.folder_data + str(flag.stars) + '/' + 'spectra/' + 'list_spectra.txt'
#    
#    if os.path.isfile(table) is False or os.path.isfile(table) is True:
#        os.system('ls ' + flag.folder_data + str(flag.stars) + '/spectra' + '/* | xargs -n1 basename >' + flag.folder_data + '/' + 'list_feros.txt')
#        spectra_list = np.genfromtxt(table, comments='#', dtype='str')
#        file_name = np.copy(spectra_list)
#    
#    fluxes, waves, errors = [], [], []
#    #
#    ## Collecting just the first file
#    #for k in range(1):
#    #    file_feros = str(flag.folder_data) + str(flag.stars) + '/' + 'feros/' + str(file_name)
#    wl, normal_flux, MJD = read_espadons(lines[0])
#
#    
#    #--------- Selecting an approximate wavelength interval for flag.Halpha
#    c = 299792.458 #km/s
#    lbd_central = 4861.363    
#        
#    if flag.Sigma_Clip:
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
#        # flag.Halpha line 
#        wl = wl[inf:sup]
#        fluxes = normal_flux[inf:sup]
#
#    radv = delta_v(vel, fluxes, 'Hb')
#    print('RADIAL VELOCITY = {0}'.format(radv))
#    vel = vel - radv
#    wl = c*lbd_central/(c - vel)
#    # Changing the units
#    waves = wl * 1e-4 # mum
#    errors = np.zeros(len(waves)) # nao tenho informacao sobre os erros ainda
#
#    if flag.model == 'beatlas' or flag.model == 'aara' or flag.model == 'aeri': # Aqui faz o corte pra pegar so o que ta no intervalo de lbd da grade (lbdarr)
#        idx = np.where((waves >= np.min(lbdarr)-0.001) & (waves <= np.max(lbdarr)+0.001))
#        wave = waves[idx]
#        flux = fluxes[idx]
#        sigma = errors[idx]
#        obs_new = np.zeros(len(lbdarr))
#    
#
#
#    new_wave = waves
#    new_flux = fluxes
#    #sigma = np.zeros(len(new_wave))
#    #sigma.fill(10*0.00344927) # new test 
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
#           bin_flux[i] = -np.inf
#        elif len(index) == 1:
#            bin_flux[i] = new_flux[index[0][0]]
#        else: 
#            # Calculating the mean flux in the box
#            bin_flux[i] = np.sum(new_flux[index[0][0]:index[-1][0]])/len(new_flux[index[0][0]:index[-1][0]])
#            sigma_new[i] = 10*0.00344927/np.sqrt(len(index))
#
#    obs_new = bin_flux
#
#    #plt.plot(lbdarr, obs_new, linestyle = 'none', marker='o')
#    #plt.show()
#
#
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
#    norm_flag.model = np.zeros((len(models),len(lbdarr)))
#    for i in range(len(models)):
#        norm_model[i] = spec.linfit(lbdarr, models[i][1:-1])
#    models = norm_model
#
#
#    logF_grid = np.log10(models)  
#        
#    
#    if flag.only_centerline:
#        lbdarr = lbdarr[100:139]
#        logF = np.log10(obs_new[100:139])
#        novo_models = np.zeros((5500,39))
#        dlogF = sigma_new[100:139] / obs_new[100:139]
#        i = 0
#        while i < len(models):
#            novo_models[i] = models[i][100:139]
#            i+=1
#        logF_grid = np.log10(novo_models)
#    
#    if flag.only_wings:
#        Hb_peak = find_nearest2(lbdarr, 0.486136)
#        keep_a = find_nearest2(lbdarr, 0.486136 - 0.0004)
#        keep_b = find_nearest2(lbdarr, 0.486136 + 0.0004)
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


def read_espadons(fname):

    # read fits
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
    
    return lbd, flux_norm, MJD


# ==================================================================================
# READS LINES FROM DATA
# ==================================================================================
def read_line_spectra(models, lbdarr, linename):
        
    
    table = flag.folder_data + str(flag.stars) + '/' + 'spectra/' + 'list_spectra.txt'
    
    if os.path.isfile(table) is False or os.path.isfile(table) is True:
        os.system('ls ' + flag.folder_data + str(flag.stars) + '/spectra' + '/* | xargs -n1 basename >' + flag.folder_data + '/' + 'list_spectra.txt')
        spectra_list = np.genfromtxt(table, comments='#', dtype='str')
        file_name = np.copy(spectra_list)
    
    fluxes, waves, errors = [], [], []

    
    if file_name.tolist()[-3:] == 'csv':
        file_ha = str(flag.folder_data) + str(flag.stars) + '/spectra/' + str(file_name)
        wl, normal_flux = np.loadtxt(file_ha, delimiter=' ').T
    else:
        file_ha = str(flag.folder_data) + str(flag.stars) + '/spectra/' + str(file_name)    
        wl, normal_flux, MJD = read_espadons(file_ha)
    
    #--------- Selecting an approximate wavelength interval 
    c = 299792.458 #km/s
    lbd_central = lines_dict[linename]
    
        
    if flag.Sigma_Clip:
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
    print('RADIAL VELOCITY = {0}'.format(radv))
    vel = vel - radv
    wl = c*lbd_central/(c - vel)

    # Changing the units
    waves = wl * 1e-4 # mum
    errors = np.zeros(len(waves)) # nao tenho informacao sobre os erros ainda

    # Aqui faz o corte pra pegar so o que ta no intervalo de lbd da grade (lbdarr)
    idx = np.where((waves >= np.min(lbdarr)-0.001) & (waves <= np.max(lbdarr)+0.001))
    wave = waves[idx]
    flux = fluxes[idx]
    sigma = errors[idx]
    obs_new = np.zeros(len(lbdarr))


    new_wave = waves
    new_flux = fluxes


    # Bin the spectra data to the lbdarr (model)
    # Creating box limits array
    box_lim = np.zeros(len(lbdarr) - 1)
    for i in range(len(lbdarr) - 1):
        box_lim[i] = (lbdarr[i] + lbdarr[i+1])/2.

    # Binned flux  
    bin_flux = np.zeros(len(box_lim) - 1)

    lbdarr = lbdarr[1:-1]
    sigma_new = np.zeros(len(lbdarr))
    #sigma_new.fill(0.017656218)

    for i in range(len(box_lim) - 1):
        # lbd observations inside the box
        index = np.argwhere((new_wave > box_lim[i]) & (new_wave < box_lim[i+1]))
        if len(index) == 0:
            bin_flux[i] = -np.inf
        elif len(index) == 1:
            bin_flux[i] = new_flux[index[0][0]]
            sigma_new[i] = 0.017656218
        else: 
            # Calculating the mean flux in the box
            bin_flux[i] = np.sum(new_flux[index[0][0]:index[-1][0]])/len(new_flux[index[0][0]:index[-1][0]])
            sigma_new[i] = 0.017656218/np.sqrt(len(index))

    obs_new = bin_flux
    
    #que estou interpolando sao as observacoes nos modelos, entao nao eh models_new que tenho que fazer. o models ta pronto
    # log space
    logF = np.log10(obs_new)
    mask = np.where(obs_new == -np.inf)
    logF[mask] = -np.inf # so that dlogF == 0 in these points and the chi2 will not be computed
    dlogF = sigma_new/obs_new
    
    
    #rmalize the flux of models to the continuum
    norm_model = np.zeros((len(models),len(lbdarr)))
    for i in range(len(models)):
        norm_model[i] = ut.linfit(lbdarr, models[i][1:-1])
    models = norm_model


    logF_grid = np.log10(models)  
    
    if linename == 'Ha':
        if flag.remove_partHa:
            lbdarr1 = lbdarr[:149]
            lbdarr2 = lbdarr[194:]
            lbdarr = np.append(lbdarr1,lbdarr2)
            novo_models = np.zeros((5500,191))
            obs_new1 = obs_new[:149]
            obs_new2 = obs_new[194:]
            obs_new = np.append(obs_new1,obs_new2)
            logF = np.log10(obs_new)
            sigma_new2 = np.append(sigma_new[:149], sigma_new[194:])
            dlogF = sigma_new2 / obs_new
            i = 0
            while i < len(models):
                novo_models[i] = np.append(models[i][:149],models[i][194:])
                i+=1
            logF_grid = np.log10(novo_models)       
    
    
    if flag.only_wings:
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
        obs_new1 = obs_new[:keep_a]
        obs_new2 = obs_new[keep_b:]
        obs_neww = np.concatenate([obs_new1,obs_new2])
        logF = np.log10(obs_neww)
        sigma_new1 = sigma_new[:keep_a]
        sigma_new2 = sigma_new[keep_b:]
        sigma_new = np.concatenate([sigma_new1, sigma_new2])
        dlogF = sigma_new/obs_neww
        novo_models1 = models[:, :keep_a]
        novo_models2 = models[:, keep_b:]
        novo_models = np.hstack((novo_models1, novo_models2))
        logF_grid = np.log10(novo_models)
        
    if flag.only_centerline:
        plt.plot(waves, fluxes)
        point_a, point_b = plt.ginput(2)
        plt.close()
        #H_peak = find_nearest2(lbdarr, lines_dict[linename])
        keep_a = find_nearest2(lbdarr, point_a[0])
        keep_a = find_nearest2(lbdarr, point_b[0])
        lbdarr1 = lbdarr[keep_a:]
        lbdarr2 = lbdarr[:keep_b]
        lbdarr = np.concatenate([lbdarr1,lbdarr2])
        obs_new1 = obs_new[keep_a:]
        obs_new2 = obs_new[:keep_b]
        obs_neww = np.concatenate([obs_new1,obs_new2])
        logF = np.log10(obs_neww)
        sigma_new1 = sigma_new[keep_a:]
        sigma_new2 = sigma_new[:keep_b]
        sigma_new = np.concatenate([sigma_new1, sigma_new2])
        dlogF = sigma_new/obs_neww
        novo_models1 = models[:, keep_a:]
        novo_models2 = models[:, :keep_b]
        novo_models = np.hstack((novo_models1, novo_models2))
        logF_grid = np.log10(novo_models)

    return logF, dlogF, logF_grid, lbdarr, box_lim
    
    
    
# ===================================================================================
#def read_Ha(models, lbdarr, flag.folder_data, folder_fig, star,
#                      model, flag.only_wings, flag.only_centerline, flag.Sigma_Clip,
#                      flag.remove_partHa):
#
#    table = flag.folder_data + str(flag.stars) + '/' + 'spectra/' + 'list_spectra.txt'
#    
#    if os.path.isfile(table) is False or os.path.isfile(table) is True:
#        os.system('ls ' + flag.folder_data + str(flag.stars) + '/spectra' + '/* | xargs -n1 basename >' + flag.folder_data + '/' + 'list_feros.txt')
#        spectra_list = np.genfromtxt(table, comments='#', dtype='str')
#        file_name = np.copy(spectra_list)
#    
#    fluxes, waves, errors = [], [], []
#
#    
#    if file_name.tolist()[-3:] == 'csv':
#        file_ha = str(flag.folder_data) + str(flag.stars) + '/spectra/' + str(file_name)
#        wl, normal_flux = np.loadtxt(file_ha, delimiter=' ').T
#    else:    
#        wl, normal_flux, MJD = read_espadons(file_name)
#    
#    #--------- Selecting an approximate wavelength interval for flag.Halpha
#    c = 299792.458 #km/s
#    lbd_central = 6562.801    
#        
#    if flag.Sigma_Clip:
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
#    if flag.model == 'beatlas' or flag.model == 'aara' or flag.model == 'aeri': # Aqui faz o corte pra pegar so o que ta no intervalo de lbd da grade (lbdarr)
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
#    norm_flag.model = np.zeros((len(models),len(lbdarr)))
#    for i in range(len(models)):
#        norm_model[i] = spec.linfit(lbdarr, models[i][1:-1])
#    models = norm_model
#
#
#    logF_grid = np.log10(models)  
#    
#    if flag.remove_partHa:
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
#    if flag.only_centerline:
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
#    if flag.only_wings:
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

    table = flag.folder_data + str(flag.stars) + '/' + 'list_iue.txt'

    # os.chdir(flag.folder_data + str(flag.stars) + '/')
    if os.path.isfile(table) is False:
        os.system('ls ' + flag.folder_data + str(flag.stars) +
                  '/*.FITS | xargs -n1 basename >' +
                  flag.folder_data + str(flag.stars) + '/' + 'list_iue.txt')
    
    iue_list = np.genfromtxt(table, comments='#', dtype='str')
    file_name = np.copy(iue_list)

    fluxes, waves, errors = [], [], []
    
    if file_name.tolist()[-3:] == 'csv':
        file_iue = str(flag.folder_data) + str(flag.stars) + '/' + str(file_name)
        wave, flux, sigma = np.loadtxt(str(file_iue), delimiter=',').T
        fluxes = np.concatenate((fluxes, flux*1e4), axis=0)
        waves = np.concatenate((waves, wave*1e-4), axis=0)
        errors = np.concatenate((errors, sigma*1e4), axis=0)

    else:
    # Combines the observations from all files in the folder, taking the good quality ones
        for k in range(len(file_name)):
            file_iue = str(flag.folder_data) + str(flag.stars) + '/' + str(file_name[k])
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
    
            fluxes = np.concatenate((fluxes, flux), axis=0)
            waves = np.concatenate((waves, wave), axis=0)
            errors = np.concatenate((errors, sigma), axis=0)

    if os.path.isdir(flag.folder_fig + str(flag.stars)) is False:
        os.mkdir(flag.folder_fig + str(flag.stars))

# ------------------------------------------------------------------------------
    # Would you like to cut the spectrum?
    #if flag.cut_iue_regions is True:
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
    if flag.lbd_range == 'UV':
        wave_lim_min = 0.13  # mum
        wave_lim_max = 0.3  # mum
    if flag.lbd_range == 'UV+VIS':
        wave_lim_min = 0.1  # mum
        wave_lim_max = 0.7  # mum
    if flag.lbd_range == 'UV+VIS+NIR':
        wave_lim_min = 0.1  # mum
        wave_lim_max = 5.0  # mum
    if flag.lbd_range == 'UV+VIS+NIR+MIR':
        wave_lim_min = 0.1  # mum
        wave_lim_max = 40.0  # mum
    if flag.lbd_range == 'UV+VIS+NIR+MIR+FIR':
        wave_lim_min = 0.1  # mum
        wave_lim_max = 350.  # mum
    if flag.lbd_range == 'UV+VIS+NIR+MIR+FIR+MICROW+RADIO':
        wave_lim_min = 0.1  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if flag.lbd_range == 'VIS+NIR+MIR+FIR+MICROW+RADIO':
        wave_lim_min = 0.39  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if flag.lbd_range == 'NIR+MIR+FIR+MICROW+RADIO':
        wave_lim_min = 0.7  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if flag.lbd_range == 'MIR+FIR+MICROW+RADIO':
        wave_lim_min = 5.0  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if flag.lbd_range == 'FIR+MICROW+RADIO':
        wave_lim_min = 40.0  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if flag.lbd_range == 'MICROW+RADIO':
        wave_lim_min = 1e3  # mum
        wave_lim_max = np.max(lbdarr)  # mum
    if flag.lbd_range == 'RADIO':
        wave_lim_min = 1e6  # mum
        wave_lim_max = np.max(lbdarr)  # mum


    band = np.copy(flag.lbd_range)

    ordem = wave.argsort()
    wave = wave[ordem]
    flux = flux[ordem]
    sigma = sigma[ordem]

# ------------------------------------------------------------------------------
    # select lbdarr to coincide with lbd
    models_new = np.zeros([len(models), len(wave)])
    if flag.model == 'beatlas' or flag.model == 'aara' or flag.model == 'aeri': # Aqui faz o corte pra pegar so o que ta no intervalo de lbd
        idx = np.where((wave >= np.min(lbdarr)) & (wave <= np.max(lbdarr)))
        wave = wave[idx]
        flux = flux[idx]
        sigma = sigma[idx]
        models_new = np.zeros([len(models), len(wave)])

    for i in range(len(models)): # A interpolacao 
        models_new[i, :] = 10.**griddata(np.log(lbdarr),
                                         np.log10(models[i]),
                                         np.log(wave), method='linear')
    # to log space
    logF = np.log10(flux)
    dlogF = sigma / flux
    logF_grid = np.log10(models_new)

# ==============================================================================

    return logF, dlogF, logF_grid, wave

def read_votable():
    
    table = flag.folder_data + str(flag.stars) + '/' + 'list.txt'

    #os.chdir(folder_data + str(star) + '/')
    #if os.path.isfile(table) is False or os.path.isfile(table) is True:
    os.system('ls ' + flag.folder_data + str(flag.stars) +
                '/*.xml | xargs -n1 basename >' +
                flag.folder_data + str(flag.stars) + '/' + 'list.txt')
    vo_list = np.genfromtxt(table, comments='#', dtype='str')
    table_name = np.copy(vo_list)
    vo_file = flag.folder_data + str(flag.stars) + '/' + str(table_name)
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
    
    if flag.stars == 'HD37795':
        fname = flag.folder_data + '/HD37795/alfCol.txt'
        data = np.loadtxt(fname, dtype={'names':('lbd', 'flux', 'dflux', 'source'), 'formats':(np.float, np.float, np.float, '|S20')})
        wave = np.hstack([wave, data['lbd']])
        flux = np.hstack([flux, jy2cgs(1e-3*data['flux'], data['lbd'])])
        sigma = np.hstack([sigma, jy2cgs(1e-3*data['dflux'], data['lbd'])])

    return wave, flux, sigma


# ==============================================================================
# Calls the xdr reading function
def read_models(lista_obs):

    if flag.model == 'aeri':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_BAphot2_xdr(lista_obs)
    if flag.model == 'befavor':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_befavor_xdr()
    if flag.model == 'aara':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_aara_complete()
    if flag.model == 'beatlas':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_beatlas_xdr()
    if flag.model == 'acol':
        ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_acol_xdr()

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
    tag = '+' + flag.model
    

    if flag.only_wings:
        tag = tag + '_onlyWings'
    if flag.only_centerline:
        tag = tag + '_onlyCenterLine'
    if flag.remove_partHa:
        tag = tag + '_removePartHa'
    if flag.Sigma_Clip is True:
        tag = tag + '_SigmaClipData'
    if flag.vsini_prior:
        tag = tag + '_vsiniPrior'
    if flag.dist_prior:
        tag = tag + '_distPrior'
    if flag.box_W:
        tag = tag + '_boxW'
    if flag.incl_prior:
        tag = tag + '_inclPrior'
    if flag.combination:
        tag = tag + '_combination'
    if flag.UV:
        tag = tag + '+UV'
    if flag.Ha:
        tag = tag + '+Ha'
    if flag.Hb:
        tag = tag + '+Hb'
    if flag.Hd:
        tag = tag + '+Hd'
    if flag.Hg:
        tag = tag + '+Hg'
                    
    return tag

#=======================================================================
# Creates list of observables

def create_list():
    
    lista = np.array([])
    
    if flag.UV:
        lista = np.append(lista,'UV')
    if flag.Ha:
        lista = np.append(lista,'Ha')
    if flag.Hb:
        lista = np.append(lista,'Hb')
    if flag.Hd:
        lista = np.append(lista,'Hd')
    if flag.Hg:
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
    table = flag.folder_data + str(flag.stars) + '/sed_data/' + 'list_sed.txt'

    #os.chdir(folder_data + str(star) + '/')
    #if os.path.isfile(table) is False or os.path.isfile(table) is True:
    os.system('ls ' + flag.folder_data + str(flag.stars) + '/sed_data/*.dat /*.csv /*.txt | xargs -n1 basename >' +
                flag.folder_data + str(flag.stars) + '/sed_data/' + 'list_sed.txt')
    vo_list = np.genfromtxt(table, comments='#', dtype='str')
    table_name = np.copy(vo_list)
    table = flag.folder_data + str(flag.stars) + '/sed_data/' + str(table_name)
    

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
        
        if flag.iue:
            wave, flux, sigma = read_iue(models[index], lbdarr[index])
        else:
            wave, flux, sigma = [], [], []
        
        if flag.votable:
            wave0, flux0, sigma0 = read_votable()
        elif flag.data_table:
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

