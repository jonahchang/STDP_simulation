##################################################################################
# params.py -- Set of parameters for UP-state-mediated_plast_fig4E.py
#
#
# Author: Victor Pedrosa
# Imperial College London, London, UK - Dec 2017
##################################################################################

import numpy as np

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define the parameters -----------------------------------------------------------------------------

sinwave=0

# Neuron model parameters  (integrate and fire) -----------------------------------------------------
tau_m = 10. 		      		# [ms] Time constant for the leaky integrator
R = 1. 						# [Ohm] Resistance of the leaky I-F model for excitatory neurons
El = 0.						# [mV] Resting potential (leaky)
Vth = 10. 					# [mV] Threshold potential
Vres = 0. 					# [mV] Reset potential
Vspike = 10.		      		# [mV] Extra potential to indicate a spike - Just to make it prettier
Tref = 3. 					# [ms] Refractory time

# Network parameters --------------------------------------------------------------------------------
NE = 100	                   	# Number of excitatory neurons in the input layer
LE = 100                    # Number of excitatory neurons in the output layer
# Simulation parameters -----------------------------------------------------------------------------
t_max = 10000	      	    # [ms] Total time of simulation
dt = 1. 		                  # [ms] Simulation time step
nSteps = np.int(t_max/dt)       # Number of steps in simulation

# Excitatory plasticity -----------------------------------------------------------------------------
tp_plast = 20.                	# [ms] Decay time for pre->post activity
tm_plast = 20. 	           	# [ms] Decay time for post->pre activity

a_pre = {}		      		# Pre-synaptic activity term (non-Hebbian)
a_post = {}	     			# Post-synaptic activity term
a_plus = {}					# Coefficient related to pre->post activity
a_minus = {}		       	# Coefficient related to post->pre activity

# Down states -> standard STDP.
# Up states -> Pre alone leads to depression + standard STDP

a_pre["up"] = -0.0002  
a_post["up"] = -0.0 
a_plus["up"] = 0.005 
a_minus["up"] = -0.005 

a_pre["down"] = 0.0 
a_post["down"] = -0.0 
a_plus["down"] = 0.005 
a_minus["down"] = -0.005  


# Connections (synapses) -----------------------------------------------------------------------------
EsynE = 30.					# [mV] Reversal potential for exc. syn.
w_min = 0. 					# lower bound on all weights
w_max = 5. 					# upper bound on exc weights

# Synaptic conductance
tauSynEx = 10.				# [ms] Time constant for excitatory postsynaptic potential
tauSynIn = 10.				# [ms] Time constant for inhibitory postsynaptic potential
gBarEx = .4 				# Peak synaptic conductance for excitatory synapses

# Synaptic weights (constants) ------------------------------------------------------------------------


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Parameters for speeding up the simulation +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

step_tauSynEx = dt/tauSynEx
step_tau_m = dt/tau_m
step_tp_plast = dt/tp_plast
step_tm_plast = dt/tm_plast

