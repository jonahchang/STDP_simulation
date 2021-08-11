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
import math
#from neuron import h

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

WEE = np.zeros((p.LE,p.NE))    # E-E connections

WEE[:] = np.linspace(0.1,1.0,p.LE*p.NE).reshape(p.LE,p.NE) 

WEE += np.random.normal(0,0.000001,(p.LE,p.NE))
WEE = SS._rect(WEE) - SS._rect(WEE-p.w_max)

import alt_approach_network as alt

WSNET = alt.create_WS_network(100, 10, 0.1)

def zero_out(w):
    return w * WSNET
                
WEE = zero_out(WEE)
# Other variables ----------------------------------------------------------------

xbar_pre = np.zeros(p.NE)         # Synaptic traces for presynaptic events
xbar_post = np.zeros(p.LE)         # Synaptic traces for postsynaptic events
Vmemb = np.zeros(p.NE)        # [mV] Membrane potential for layer 4 neurons
vlayer2 = np.zeros(p.LE)   # membrane potential for each neuron in layer 2/3

ref = np.zeros(p.NE)            # Variable to identify the L4 neurons within the refractory time
ref2 = np.zeros(p.LE)            # Variable to identify the L2/3 neurons within the refractory time

gSynE = np.zeros(p.NE)            # Synaptic conductance for excitatory connections in L4
gSynE2 = np.zeros(p.LE)            # Synaptic conductance for excitatory connections in L2/3

Iext = 0        # [pA] External current for each neuron in layer 4
Iext2 = 0        # [pA] External current for each layer 2/3 neuron

# Not using neuron module so removed h.IClamp, h.Vector, etc:
# Instead, you could edit the Iext2 external current provided
# to the layer 2/3 cells, and you could access the voltage of
# the layer 2/3 cells from vlayer2 variable
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create variables to "save" the results -----------------------------------------

subsampling = 100

# WEE_all stores the WEE weight matrix every subsampledth time step
WEE_all = np.zeros((int(p.nSteps/subsampling)+1,p.LE,p.NE))
WEE_all[0] = WEE
WEE_var = 1.*WEE

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Run the code -------------------------------------------------------------------

spks_1 = [[] for  i in range(100)]
spks_2 = [[] for i in range(100)]

for step in range(p.nSteps):
    WEE_in = WEE
    Iext2 = 0 * np.sin(2 * math.pi * (4 * step * p.dt / 1000 + 0)) * 1
    Iext = 12 * np.sin(2 * math.pi * (4 * step * p.dt / 1000 + 0)) * 1
    Vmemb,ref,ref2,xbar_pre,xbar_post,gSynE,gSynE2,WEE,Iext, vlayer2, Iext2, spikes1, spikes2 \
    = SS.SimStep (Vmemb,ref,ref2,xbar_pre,xbar_post,gSynE,gSynE2,WEE,Iext,vlayer2, Iext2,"up")
    
    WEE_diff = WEE - WEE_in
    WEE_var = WEE_var + WEE_diff
    WEE = WEE_in
        
    if (((step+1) % subsampling) == 0):
        WEE_all[int(step/subsampling)+1,:,:] = WEE_var # save the synaptic weights, added third dimension for LE postsynaptic neurons
        
    
    # save spike times
    for num in range(100):
        if spikes1[num]:
            spks_1[num].append(step)
        if spikes2[num]:
            spks_2[num].append(step)

    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
# establishing connections between layers    
import matplotlib.pyplot as plt

f = plt.figure()
plt.imshow(WEE_all[0], cmap="gray")
plt.title("Initial Weight Matrix")
plt.xlabel("Postsynaptic cell (Layer 2/3)")
plt.ylabel("Presynaptic cell (Layer 4)")

f2 = plt.figure()
plt.imshow(WEE_all[-1], cmap="gray")
plt.title("Last Weight Matrix")
plt.xlabel("Postsynaptic cell (Layer 2/3)")
plt.ylabel("Presynaptic cell (Layer 4)")
        

f3 = plt.figure(figsize = (10, 8))
ax = f3.add_subplot(1, 1, 1)
for i in range(100):
    lst = spks_1[i]
    if lst:
        Y = i * np.ones(len(lst))
        ax.scatter(lst, Y, color = 'red', s = 1)
    lst = spks_2[i]
    if lst:
        Y = (i + 100) * np.ones(len(lst))
        ax.scatter(lst, Y, color = 'blue', s = 1)
ax.set_xlim((0, p.nSteps))

# Compute the total time spent with the simulation -------------------------------

time_end = time_now()
time_total = time_end - time_in

np.save('Data/Wall_{0:03d}'.format(trial),WEE_all[[0,-1]])

print('')
print('Trial = {1:3d} \n Total time = {0:.3f} seconds'.format(time_total,trial))
print("finished")