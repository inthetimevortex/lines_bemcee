# ==============================================================================
# -*- coding:utf-8 -*-
# ==============================================================================
# importing packages
import emcee
from lines_emcee import new_emcee_inference
import sys
import importlib
#import user_settings as flag
from schwimmbad import MPIPool
#from emcee.utils import MPIPool
#import os

#os.environ["OMP_NUM_THREADS"] = "1"

# ==============================================================================

mod_name = sys.argv[1]+'_'+'user_settings'
#print(sys.argv[1])
flag = importlib.import_module(mod_name)


# Acrux
if flag.acrux is True:
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        new_emcee_inference(pool)
else:
    pool = False
    new_emcee_inference(pool)




# ==============================================================================
# The End
print(75 * '=')
print('\nSimulation Finished\n')
print(75 * '=')

if flag.acrux is True:
    pool.close()

# ==============================================================================

