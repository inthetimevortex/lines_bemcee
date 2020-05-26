# ==============================================================================
# -*- coding:utf-8 -*-
# ==============================================================================
# importing packages
from lines_reading import read_models, create_tag, create_list
import emcee
from lines_emcee import run
import sys
import user_settings as flag
from schwimmbad import MPIPool


# ==============================================================================
lista_obs = create_list() 

tags = create_tag()

                 
print(tags)

# Acrux
if flag.acrux is True:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
else:
    pool = False



# Reading Models
ctrlarr, minfo, models, lbdarr, listpar, dims, isig = read_models(lista_obs)

# ==============================================================================
# Run code
input_params = ctrlarr, minfo, models, lbdarr, listpar, dims, isig, tags, pool, lista_obs

run(input_params)

# ==============================================================================
# The End
print(75 * '=')
print('\nSimulation Finished\n')
print(75 * '=')

if flag.acrux is True:
    pool.close()

# ==============================================================================

