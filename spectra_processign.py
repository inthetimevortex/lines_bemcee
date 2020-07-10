#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  
#  Copyright 2020 Amanda Rubio <amanda@Pakkun>
#  


import cmd
import numpy as np
import utils as ut
from lines_radialv import delta_v, Ha_delta_v
from lines_reading import read_BAphot2_xdr
from scipy.signal import detrend
from astropy.stats import SigmaClip
from astropy.stats import median_absolute_deviation as MAD
import matplotlib.pyplot as plt

def find_nearest2(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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
   
    ar_of_index_of_bad_pts = np.unique(ar_of_index_of_bad_pts)

    
    x_final = x[tf_ar]
    y_final = y[tf_ar]
  return(x_final, y_final)


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
    
    return lbd, flux_norm


lines_dict = {
'Ha':6562.801,
'Hb':4861.363,
'Hd':4101.74,
'Hg':4340.462 }

print('WELCOME TO THE PRE-PROCESSING CODE FOR BEMCEE')
print('Place your spectra file in the "data/your star name" directory before proceeding')
print('This code accepts spectra in FITS, DAT, TXT and CSV formats')

lista_obs = np.array(['Ha', 'Hb', 'Hd', 'Hg'])

ctrlarr, minfo, models_combined, lbd_combined, listpar, dims, isig = read_BAphot2_xdr(lista_obs)



star = input('What is the name of the star?')
folder_data = '../data/' + star+ '/'

master = np.genfromtxt(folder_data+'master.txt', delimiter='    ', skip_header=1, dtype=str)
fnames = master[:,0]
data_type = master[:,1]
domain = master[:,2]

for i, fname in enumerate(fnames):
    fname = folder_data + fname
    file_type = fname.split('.')[-1]

    if file_type == 'dat' or file_type == 'csv' or file_type == 'txt':
        delimiter = input('What delimeter? (. , \s or \t)')
        wave, flux = np.loadtxt(fname, delimiter=delimiter).T
        
    else:
        wave, flux = read_espadons(fname)
    
    plt.plot(wave, flux)
    plt.show()
    line = domain[i]
    c = 299792.458 #km/s
    lbd_central = lines_dict[line]
    
    u = np.where(lista_obs == line)
    index = u[0][0]
    lbdarr = lbd_combined[index]
        
    vl, fx = ut.lineProf(wave, flux, hwidth=2500., lbc=lbd_central)
    vel, fluxes = Sliding_Outlier_Removal(vl, fx, 50, 8, 15)
    radv = delta_v(vel, fluxes, line)
    print('RADIAL VELOCITY = {0}'.format(radv))
    vel = vel - radv
    wave = c*lbd_central/(c - vel)
    
    # Changing the units
    wave = wave * 1e-4 # mum
    errors = np.zeros(len(wave))


    idx = np.where((wave >= np.min(lbdarr)-0.001) & (wave <= np.max(lbdarr)+0.001))
    wave = wave[idx]
    flux = flux[idx]
    
    
    plt.plot(wave, flux)
    plt.title('Select part to estimate errors')
    point_a, point_b = plt.ginput(2)
    plt.close()
    
    keep_a = find_nearest2(wave, point_a[0])
    keep_b = find_nearest2(wave, point_b[0])
    use_lbd = wave[keep_a:keep_b]
    use_flux = flux[keep_a:keep_b]
    
    m, b = np.polyfit(use_lbd, use_flux, 1)
    fit = b + m * use_lbd
    
    residue = use_flux - fit
    sigma = np.std(residue)
    
    # Bin the spectra data to the lbdarr (model)
    # Creating box limits array
    box_lim = np.zeros(len(lbdarr) - 1)
    for i in range(len(lbdarr) - 1):
        box_lim[i] = (lbdarr[i] + lbdarr[i+1])/2.
    
    # Binned flux  
    bin_flux = np.zeros(len(box_lim) - 1)
    
    lbdarr = lbdarr[1:-1]
    #sigma = np.zeros(len(lbdarr))
    
    for i in range(len(box_lim) - 1):
        # lbd observations inside the box
        index = np.argwhere((wave > box_lim[i]) & (wave < box_lim[i+1]))
        if len(index) == 0:
            bin_flux[i] = -np.inf
        elif len(index) == 1:
            bin_flux[i] = flux[index[0][0]]
            sigma_new[i] = 0.017656218
        else: 
            # Calculating the mean flux in the box
            bin_flux[i] = np.sum(flux[index[0][0]:index[-1][0]])/len(flux[index[0][0]:index[-1][0]])
            sigma_new[i] = 0.017656218/np.sqrt(len(index))
    
    obs_new = bin_flux







