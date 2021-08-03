##################################################################################
# UP-state-mediated_plast_fig4E.py -- Uses the simulator from network_simulator.py
# and runs a simulation of a plastic feedforward network following the findings
# of González-Rueda et al. (Neuron, 2018).
#
# On these simulations, synaptic weights are updated following the Up-state-mediated 
# plasticity described in the paper.
#
# Author: Victor Pedrosa
# Imperial College London, London, UK - Dec 2017
##################################################################################


# Clear everything!
def clearall():
	all = [var for var in globals() if var[0] != "_"]
	for var in all:
		del globals()[var]

clearall()

# -------------------------- Import libraries ------------------------------------

import numpy as np
from time import time as time_now
import sys
from neuron import h

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

args = sys.argv
try:   
    trial = int(args[1])
except IndexError:
    trial = 0

# ------------------------ Import parameters -------------------------------------

from imp import reload
import params; reload(params); import params as p
import SimStep; reload(SimStep); import SimStep as SS

time_in = time_now()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initialization of time-dependent variables -------------------------------------

# Synaptic weights ---------------------------------------------------------------

WEE = np.zeros((p.LE,p.NE))	# E-E connections

WEE[:] = np.linspace(0.1,1.0,p.NE)    # TODO update weights in the matrix with p.LE rows instead of 1 row

WEE += np.random.normal(0,0.000001,(1,p.NE))
WEE = SS._rect(WEE) - SS._rect(WEE-p.w_max)


# Other variables ----------------------------------------------------------------

xbar_pre = np.zeros(p.NE) 		# Synaptic traces for presynaptic events
xbar_post = np.zeros(1) 		# Synaptic traces for postsynaptic events
Vmemb = np.zeros(p.NE+1)		# [mV] Membrane potential
ref = np.zeros(p.NE+1)			# Variable to identify the neurons within the refractory time
gSynE = np.zeros(p.NE)			# Synaptic conductance for excitatory connections
Iext = np.zeros(p.NE+1)		# [pA] External current for each neuron
vlayer2 = np.zeros(p.LE + 1)   # membrane potential for each layer

stim = h.IClamp(cell.section(1))    # TODO find individual cells to target
stim.delay = 0
stim.dur = 1e9
vec = h.Vector(np.array())
stim.amp = vec.play(stim._ref_amp, time_in, True)
dend_v = h.Vector().record(my_cell.dend(0.5)._ref_v)    # TODO update arguments


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create variables to "save" the results -----------------------------------------

subsampling = 100
WEE_all = np.zeros((int(p.nSteps/subsampling)+1,p.LE,p.NE))
WEE_all[0] = WEE    # TODO might need to change to reflect p.LE instead of 1
WEE_var = 1.*WEE

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Run the code -------------------------------------------------------------------

for step in range(p.nSteps):
	WEE_in = WEE
	
	Vmemb,ref,xbar_pre,xbar_post,gSynE,WEE,Iext \
	= SS.SimStep (Vmemb,ref,xbar_pre,xbar_post,gSynE,WEE,Iext,"up")
	
	WEE_diff = WEE - WEE_in
	WEE_var = WEE_var + WEE_diff
	WEE = WEE_in
		
	if (((step+1) % subsampling) == 0):
		WEE_all[int(step/subsampling)+1,:,:] = WEE_var # save the synaptic weights, added third dimension for LE postsynaptic neurons

	
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
# establishing connections between layers    
import matplotlib.pyplot as plt



def create_WS_network(N, k, P):
    """
    Parameters
    ----------
    N: int
       Number of neurons
    k: int
       Number of outgoing connections per neuron.
    P: float
       Randomness parameter.
    """
    if N < 1:
        raise ValueError("N must be > 0")
    if k % 2 != 0:
        raise ValueError("k must be even")
    if P < 0 or P > 1:
        raise ValueError("P must be >= 0 and <= 1")
        
    mat = np.zeros([N, N], dtype=int)
    for i in range(N):
        for j in range(i-k//2, i+k//2 + 1):
            if j == i:
                continue
            ind = j % N
            mat[i, ind] = 1
            
    if P > 0:
        for i in range(N):
            for j in range(i-k//2, i+k//2 + 1):
                if j == i:
                    continue
                ind = j % N
                if np.random.random() <= P:
                    mat[i, ind] = 0
                    while True:
                        loc = np.random.randint(0, N)
                        if loc == i or mat[i, loc] == 1 or loc == int:
                            continue
                        mat[i, loc] = 1
                        break
    return mat
                    

WEE[50:, 50:] = create_WS_network(p.NE // 2, 4, 0.2)    # layer 2/3  (bottom right)
WEE[0:50, 0:50] = create_WS_network(p.LE // 2, 2, 0.1)    # layer 4  (top left)
WEE[50:, 0:50] = create_WS_network(p.NE // 2, 8, 0.3)    # layer 2/3 to layer 4 (bottom left)
WEE[0:50, 50:] = create_WS_network(p.NE // 2, 4, 0.1)    # layer 4 to layer 2/3 (top right)
plt.imshow(WEE, cmap="gray")
        
        



# Compute the total time spent with the simulation -------------------------------

time_end = time_now()
time_total = time_end - time_in

np.save('Data/Wall_{0:03d}'.format(trial),WEE_all[[0,-1]])

print('')
print('Trial = {1:3d} \n Total time = {0:.3f} seconds'.format(time_total,trial))
print("finished")