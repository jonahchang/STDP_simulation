##################################################################################
# SimStep.py -- Network simulator 
#
#
# Author: Victor Pedrosa
# Imperial College London, London, UK - Dec 2016
##################################################################################


import numpy as np

# ------------------------ Import parameters -------------------------------------

from imp import reload
import params; reload(params); import params as p


# ================================================================================
# Main code ----------------------------------------------------------------------
def _rect (x): return x*(x>0.)

def _Iext (Ipre,numcells):
	taufilt = 20 # [ms] filtering time constant
	Iext = Ipre - (Ipre-_rect((np.random.normal(0,1.8,numcells))**3.))*p.dt/taufilt
	Iext[-1] = Ipre[-1] - (Ipre[-1]-(9+_rect((np.random.normal(0,0.0001,1)))))*p.dt/taufilt
	return Iext

def SimStep (u,ref,ref2,xbar_pre,xbar_post,gSynE,gSynE2,WEE,Iext,vlayer2, Iext2,s):
	# ----------------------------------------------------------------------------
	# - Takes as the input:
	# u: [NE] Membrane potential at time t
	# xbar_pre: [NE] synaptic traces for presynaptic events
	# xbar_post: [NE] synaptic traces for postsynaptic events
	# gSyn_E: [NE] synaptic conductances for excitatory connections
	# WEE: [LExNE] Excitatory synaptic weights
	# Iext: [NE] External current for each layer 2 neuron
    # vlayer2: [LE] Membrane potential at time t
    # Iext2: [LE] External current for each layer 4 neuron
	# s: [str] "up" or "down" - state of the network
	# ----------------------------------------------------------------------------
	# - Returns as output: (all variables for time t+dt)
	# u_out: 
	# xbar_pre_out: 
	# xbar_post_out 
	# gSynE_out 
	# WEE_out
    # vlayer2_out
	# ----------------------------------------------------------------------------
	
    ## SPIKE CALCS
    
    # Layer 4 spikes
	spikes = (u>p.Vth) # Verify all the L4 neurons that fired an action potential
	ref += spikes*p.Tref  # update the refractory variable

    # Layer 2/3 spikes
	spikes2 = (vlayer2>p.Vth) # Verify all the neurons that fired an action potential
	ref2 += spikes2*p.Tref  # update the refractory variable

    ## SYNAPSE CALCS
	
	# Layer 4 synaptic conductances
	gSynE_out = gSynE + p.gBarEx * spikes
	gSynE_out = gSynE_out - gSynE_out*p.step_tauSynEx
	
	# Layer 2 synaptic conductances
	gSynE2_out = gSynE2 + p.gBarEx * spikes2
	gSynE2_out = gSynE2_out - gSynE2_out*p.step_tauSynEx

    ## CURRENTS
	
	Isyn = -(vlayer2[:] - p.EsynE)*np.dot(WEE,gSynE) # Postsynaptic currents in Layer 2
	Iext = _Iext(Iext,p.NE) # Layer 4 external currents
	Iext2 = _Iext(Iext2,p.LE) # Layer 2 external currents

    ## MEMBRANE POTENTIAL	



    # Layer 4 membrane potential calculations
	u = u + (p.Vres-u)*spikes # reset the voltage for those who spiked
	u_out = u + (-u + p.R*Iext)*p.step_tau_m
	u_out[(ref>0.001)] = p.Vres
	u_out = u_out + (p.Vspike-u_out+p.Vth)*(u_out>p.Vth) # add a constant to "see" the spikes



    # Layer 2/3 membrane potential calculations
	vlayer2 = vlayer2 + (p.Vres-vlayer2)*spikes # reset the voltage for those who spiked
	vlayer2_out = vlayer2 + (-vlayer2 + p.R*Iext2)*p.step_tau_m
	vlayer2_out = vlayer2_out + Isyn*p.step_tau_m
	vlayer2_out[(ref>0.001)] = p.Vres
	vlayer2_out = vlayer2_out + (p.Vspike-vlayer2_out+p.Vth)*(vlayer2_out>p.Vth) # add a constant to "see" the spikes


	## REFRACTORY PERIODS
    
    # Layer 4
	ref = _rect(ref - p.dt)
	ref_out = ref
	
    # Layer 2
	ref2 = _rect(ref2 - p.dt)
	ref2_out = ref2
	
	# Update the synaptic traces
    # TODO figure out what makes sense for the Layer 2 cells (suggestion is added in comment after xbar_post)
	xbar_pre_out = xbar_pre + (10.-xbar_pre)*spikes
	xbar_pre_out = _rect(xbar_pre_out - p.dt)
	xbar_post_out = xbar_post + (10.-xbar_post)*spikes2
	
	# Update the synaptic weights from Layer 4 to Layer 2/3
	auxMat = np.ones((p.LE,p.NE))    # changed 1 to p.LE
	WEE_out = WEE + p.a_pre[s]*(auxMat*spikes) \
		+ (-1.*p.a_pre[s])*(auxMat*spikes2) * (xbar_pre>0.)  
	
    # TODO check if it makes sense to look at the activity of all
    # Layer 2/3 cells in sum now that we have expanded this model
    # to a large number of layer 2/3 cells (previously a single cell
    # that got checked)
	if sum(spikes2) > 0:
		xbar_pre_out[:] = 0.
	
	WEE_out = _rect(WEE_out) - _rect(WEE_out-p.w_max) # apply bounds
		
	return u_out,ref_out,ref2_out,xbar_pre_out,xbar_post_out,gSynE_out,gSynE2_out,WEE_out,Iext,vlayer2_out,Iext2