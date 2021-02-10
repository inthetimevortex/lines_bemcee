
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
import pyhdust.rotstars as rot
import math

def oblat2w(oblat):
    '''
    Author: Rodrigo Vieira
    Converts oblateness into wc=Omega/Omega_crit
    Ekstrom et al. 2008, Eq. 9

    Usage:
    w = oblat2w(oblat)
    '''
    if (np.min(oblat) < 1.) or (np.max(oblat) > 1.5):
        print('Warning: values out of allowed range')

    oblat = np.array([oblat]).reshape((-1))
    nw = len(oblat)
    w = np.zeros(nw)

    for iw in range(nw):
        if oblat[iw] <= 1.:
            w[iw] = 0.
        elif oblat[iw] >= 1.5:
            w[iw] = 1.
        else:
            w[iw] = (1.5**1.5) * np.sqrt(2. * (oblat[iw] - 1.) / oblat[iw]**3.)

    if nw == 1:
        w = w[0]

    return w

def read_BAphot2_xdr():

    dims = ['M', 'W', 'tms', 'cosi']
    dims = dict(zip(dims, range(len(dims))))
    isig = None # photospheric model is None

    ctrlarr = [np.NaN, np.NaN, np.NaN, np.NaN] 

    tmp = 0
    cont = 0
    while tmp < len(ctrlarr):
        if math.isnan(ctrlarr[tmp]) is True:
            cont = cont + 1
            tmp = tmp + 1
        else:
            tmp = tmp + 1
    
    folder_models = '../models/'
    
    # Read the grid models, with the interval of parameters.
    
    xdrPL = folder_models + 'BAphot__UV2M_Ha_Hb_Hg_Hd.xdr'
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
    #models_combined = []
    #lbd_combined = []
    
    listpar = [np.unique(minfo[:,0]), np.unique(minfo[:,1]),np.unique(minfo[:,2]), np.unique(minfo[:,3])]

    return ctrlarr, minfo, models, lbdarr, listpar, dims, isig
    
    
ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_BAphot2_xdr()

#Massa da estrela, Raio do polo, Luminosidade da estrela, Omega estrela, Beta, inclinação

Mass = 18.5
ttms = 0.5
oblat = 1.1
W = np.sqrt((oblat - 1)*2)

omega = oblat2w(oblat)
Rpole, logL, age = ut.geneva_interp_fast(Mass, oblat, ttms, Zstr='014')
Beta = rot.beta(oblat, is_ob = True)

#in deg
incl = 60


params = [Mass, W, ttms, np.cos(np.deg2rad(incl))]

mod = ut.griddataBA(minfo, np.log10(models), params, listpar, dims)

np.savetxt('M{:.2f}_Ob{:.2f}_omega{:.2f}_Rp{:.2f}_Beta{:.2f}_incl{:.2f}_logL{:.2f}.txt'.format(Mass, oblat, omega, Rpole, Beta, incl, logL), np.c_[lbdarr,mod])

