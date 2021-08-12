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

# remove previous results!
import os, os.path
mypath = "Data"
for root, dirs, files in os.walk(mypath):
    for file in files:
        os.remove(os.path.join(root, file))

# -------------------------- Import libraries ------------------------------------

import numpy as np
from time import time as time_now
import sys

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

plotflag = 1


args = sys.argv
try:   
    trials2run = [int(args[1])]
    plotflag = 0
except IndexError:
    trials2run = list(range(100))
    if len(trials2run)>2:
        plotflag = 2
# ------------------------ Import parameters -------------------------------------

from imp import reload
import params; reload(params); import params as p
import SimStep; reload(SimStep); import SimStep as SS


alt_approach = False
# if alt_approach:
#    import alt_approach_network; reload(alt_approach_network); import alt_approach_network as aan


    
for trial in trials2run:
    
    time_in = time_now()
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Initialization of time-dependent variables -------------------------------------
    
    # Synaptic weights ---------------------------------------------------------------
    
    WEE = np.zeros((p.LE,p.NE))    # E-E connections
    
    for c2 in range(p.LE):
        WEE[c2,:] = np.roll(np.concatenate((np.linspace(0.1,1.0,int(p.NE//2)),np.linspace(1.0,0.1,p.NE-int(p.NE//2)))), c2*int(p.NE//p.LE)) 
    
    WEE += np.random.normal(0,0.000001,(p.LE,p.NE))
    WEE = SS._rect(WEE) - SS._rect(WEE-p.w_max)
    
    if alt_approach:
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
    
    Iext = np.zeros(p.NE)        # [pA] External current for each neuron in layer 4
    Iext2 = np.zeros(p.LE)        # [pA] External current for each layer 2/3 neuron
    
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
    
    layer2_example = []
    layer4_example = []
    
    layer2ct = [0]*100
    layer4ct = [0]*100
    
    for step in range(p.nSteps):
        WEE_in = WEE
        Vmemb,ref,ref2,xbar_pre,xbar_post,gSynE,gSynE2,WEE,Iext, vlayer2, Iext2, spikes1, spikes2 \
        = SS.SimStep (Vmemb,ref,ref2,xbar_pre,xbar_post,gSynE,gSynE2,WEE,Iext,vlayer2, Iext2,step,"up")
        vlayer2_out = vlayer2
        WEE_diff = WEE - WEE_in
        WEE_var = WEE_var + WEE_diff
        WEE = WEE_in
            
        if (((step+1) % subsampling) == 0):
            WEE_all[int(step/subsampling)+1,:,:] = WEE_var # save the synaptic weights, added third dimension for LE postsynaptic neurons
            
        
        # save spike times
        for num in range(p.NE):
            if spikes1[num]:
                spks_1[num].append(step)
                layer4ct[num] += 1
        for num in range(p.LE):
            if spikes2[num]:
                spks_2[num].append(step)
                layer2ct[num] += 1
    
        layer2_example.append(vlayer2_out[0])
        layer4_example.append(Vmemb[0])
        
    if plotflag==1 or (plotflag==2 and trial==1):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(layer4_example, label='Layer 4')
        plt.plot(layer2_example, label='Layer 2')
        plt.legend()
        plt.xlim(200, 700)
        plt.xlabel('Time (ms)')
        plt.ylabel('Modified Membrane Potential')
        plt.title('Representative cells in each layer')
        plt.show()	    
        
        L4Hz = np.asarray(layer4ct)*1000/p.t_max
        plt.figure()
        plt.hist(L4Hz,list(range(15)))
        plt.title('Firing Rates across Layer 4')
        plt.xlabel('Average Firing Rate (Hz)')
        plt.ylabel('Number of cells with this firing rate')
        plt.show()	
          
        L2Hz = np.asarray(layer2ct)*1000/p.t_max
        plt.figure()
        plt.hist(L2Hz,list(range(15)))
        plt.title('Firing Rates across Layer 2')
        plt.xlabel('Average Firing Rate (Hz)')
        plt.ylabel('Number of cells with this firing rate')
        plt.show()	    
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            
    # establishing connections between layers  
    if plotflag==1 or (plotflag==2 and trial==1):  
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
        plt.xlabel('Time (ms)')
        plt.ylabel('Specific cell')
        plt. title('Spike Raster')
    
    # Compute the total time spent with the simulation -------------------------------
    
    time_end = time_now()
    time_total = time_end - time_in
    
    np.save('Data/Wall_{0:03d}'.format(trial),WEE_all[[0,-1]])
    
    print('')
    print('Trial = {1:3d} \n Total time = {0:.3f} seconds'.format(time_total,trial))
    print("finished")
