#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  lines_gauss.py
#
#  Copyright 2021 Amanda Rubio <amanda.rubio@usp.br>
#


import numpy as np
import scipy.stats as sp
import scipy.constants as const
import scipy.interpolate as interp
import scipy.integrate as integrate
import pyhdust.spectools as spc
import importlib
import bemcee.constants as const
from pyhdust import spectools as spt
import operator
import matplotlib.pyplot as plt
from icecream import ic
from __init__ import mod_name
flag = importlib.import_module(mod_name)

#Makes a 1D Gaussian defined by its FWHM
def gaussian1d(npix, fwhm, normalize=True):
    # Initialize Gaussian params
    cntrd = (npix + 1.0) * 0.5
    st_dev = 0.5 * fwhm / np.sqrt( 2.0 * np.log(2) )
    x = np.linspace(1,npix,npix)

    # Make Gaussian
    ampl = (1/(np.sqrt(2*np.pi*(st_dev**2))))
    expo = np.exp(-((x - cntrd)**2)/(2*(st_dev**2)))
    gaussian = ampl * expo

    # Normalize
    if normalize:
        gaussian /= gaussian.sum()

    return gaussian


def gaussfold(lam, flux, fwhm):

    ic(len(lam), len(flux), fwhm)
    lammin = min(lam)
    lammax = max(lam)

    dlambda = fwhm / float(17)

    interlam = lammin + dlambda * np.arange(float((lammax-lammin)/dlambda+1))
    x = interp.interp1d(lam, flux, kind='linear', fill_value='extrapolate')
    interflux = x(interlam)


    fwhm_pix = fwhm / dlambda
    #window = int(17 * fwhm_pix)
    window = len(interlam)
    ic(lammin, lammax, dlambda, window)
    #print(len(interlam), len(interflux))
    #window = 1000

    # Get a 1D Gaussian Profile
    gauss = gaussian1d(window, fwhm_pix)
    # Convolve input spectrum with the Gaussian profile
    fold = np.convolve(interflux, gauss, mode='same')

    ic(len(interlam), len(fold))

    y = interp.interp1d(interlam, fold, kind='linear', fill_value='extrapolate')
    fluxfold = y(lam)

    return fluxfold


def gaussconv(fac_e, v_e, F_mod_Ha, wave):


    vel, flx = spt.lineProf(wave, F_mod_Ha, lbc=0.65628, hwidth=1380)

    ##Adding junk ones to either side of the halpha continuum
    ##other gaussfold complains about x and y not being the same
    ##length, as lammin and lammax
    ##Start by finding the second largest and smallest velocity to extend from
    #velo2ndlarg = max(n for n in vel if n!=max(vel))
    #velo2ndsmal = min(n for n in vel if n!=min(vel))
    ##velocity
    ##vel = extend(np.linspace(velo2ndlarg+1,24000,10000))
    vel = np.concatenate((vel, np.linspace(max(vel),6000,2000)), axis=0)
    ##vel = extend(np.linspace(-24000, velo2ndsmal-1, 10000))
    vel = np.concatenate((np.linspace(-6000, min(vel), 2000), vel), axis=0)
    ##flux
    flx = np.concatenate((flx, np.ones(2000)))
    flx = np.concatenate((np.ones(2000), flx))

    # vel = vel[1000:-1000]
    # flx = flx[1000:-1000]
    ic(len(wave), len(vel), len(flx))
    #flx.extend(np.ones(10000))
    #Sort the junk values so that x is in numerically increasing order
    #L = sorted(zip(vel,flx), key=operator.itemgetter(0))
    #vel, flx = zip(*L)
    #ss = np.argsort(vel)
    #vel = vel[ss]
    #flx=flx[ss]

    ###fac_e = [0.2, 0.5] #Fraction of light scattered by electrons
    ###v_e = [500.0, 650.0] #Speed of electron motion
    ###v_h = [10.0, 20.0] #Sound speed of the disk

    notscat_flux = [i*(1-fac_e) for i in flx]
    scat_flux = [i*fac_e for i in flx]
    #print(len(notscat_flux), len(scat_flux))
    #notscat_conv = gaussfold(vel, notscat_flux, v_h)
    scat_conv = gaussfold(vel, scat_flux, v_e)
    flux_conv = scat_conv + notscat_flux #flux after convolution
    #plt.plot(vel, flux_conv, 'r')
    #plt.plot(vel, flx, 'b')
    #plt.show()

    wave2 = const.c * 0.65628/(const.c - vel)
    plt.plot(wave2, flux_conv, 'r')
    plt.show()
    return wave2[1000:-1000], flux_conv[1000:-1000]
