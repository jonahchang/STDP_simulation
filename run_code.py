# ============================================================================================================
#
#
# -----------------------------------------------------------------------
#
# Author: Victor Pedrosa <v.pedrosa15@imperial.ac.uk>
# Imperial College London, London, UK - Feb 2017
# ====================================================================================================


# Import modules -------------------------------------------------------------------------------------
import subprocess
import numpy as np
import os
from time import time as time_now
import multiprocessing as mp

# Create new directories to store data ---------------------------------------------------------------

newpath = r'Data/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
newpath = r'Figures/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

# start to count the time spent with simulations -----------------------------------------------------
time_in = time_now() 


# ====================================================================================================
# Run the simulations
# ====================================================================================================

# Run the main code for homogeneous stimulation ------------------------------------------------------
def run_trial(tr):
	subprocess.call('python UP-state-mediated_plast_fig4E.py {0}'.format(tr),shell=True)

NTrials = 200
quant_proc = np.max((mp.cpu_count()-1,1))
pool = mp.Pool(processes=quant_proc)

res = pool.map(run_trial,[i for i in range(NTrials)])

# stop counting the time and show the total time spent -----------------------------------------------
time_end = time_now()
total_time = (time_end-time_in)/60. # [min]

print('\n')
print('Simulation finally finished!')
print('Total time = {0:.2f} minutos'.format(total_time))


# Run the code to generate the figures ---------------------------------------------------------------

subprocess.call('python Make_fig4E.py',shell=True)
