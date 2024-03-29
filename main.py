# ==============================================================================
# -*- coding:utf-8 -*-
# ==============================================================================
# importing packages

import emcee
from bemcee import emcee_inference
import sys
from schwimmbad import MPIPool
from organizer import flag

# ==============================================================================

# # Acrux
# if flag.acrux is True:
#     with MPIPool() as pool:
#         if not pool.is_master():
#             pool.wait()
#             sys.exit(0)
#         emcee_inference(pool)
# else:
#     pool = False
#     emcee_inference(pool)

emcee_inference()


# ==============================================================================
# The End
print(75 * "=")
print("\nSimulation Finished\n")
print(75 * "=")

# if flag.acrux is True:
#     pool.close()

# ==============================================================================
