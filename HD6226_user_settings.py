# ==============================================================================
# General Options
# ==============================================================================


stars =             'HD6226'  # Star name
a_parameter =       2.0   # Set internal steps of each walker
extension =         '.png'  # Figure flag.extension to be saved
include_rv =        False  # If False: fix Rv = 3.1, else Rv will be inferead
af_filter =         False  # Remove walkers outside the range 0.2 < af < 0.5
long_process =      True  # Run with few walkers or many?
Nsigma_dis =        2.  # Set the range of values for the distance
model =             'aeri'  # 'beatlas', 'aeri', or 'acol'


if long_process is True:
    Nwalk = 400
    Sburn = 500
    Smcmc = 5000
else:
    Nwalk = 1500
    Sburn = 200
    Smcmc = 1000

binary_star = False

folder_data = '../data/'
folder_fig = '../figures/'
folder_defs = '../defs/'
folder_tables = '../tables/'
folder_models = '../models/'

lbd_range = 'UV'

vsini_prior =   False # Uses a gaussian vsini prior
dist_prior =    True # Uses a gaussian distance prior

box_W =        True # Constrain the W lower limit, not actual a prior, but restrain the grid
box_W_min, box_W_max = [0.6, 'max']

box_i =         False # Constrain the i limits, not actual a prior, but restrain the grid
box_i_min, box_i_max = [0.86, 'max']

incl_prior =    False # Uses a gaussian inclination prior

normal_spectra =    True # True if the spectra is normalized (for lines), distance and e(b-v) are not computed
only_wings =        False # Run emcee with only the wings
only_centerline =   False # Run emcee with only the center of the line
Sigma_Clip =        True # If you want telluric lines/outilier points removed
remove_partHa =     True # Remove a lbd interval in flag.Halpha that has too much absorption (in the wings)

# Line and continuum combination

SED =           True #True if you have any kind of SED data (IUE, visible, etc)
iue =           True
votable =       False
data_table=     False

Ha =            False
Hb =            False
Hd =            False
Hg =            False

pol = False

corner_color = 'violet' # OPTIONS ARE: blue, dark blue, teal, green, yellow, orange, red, purple, violet, pink.
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
