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

def _Iext (Ipre):
	taufilt = 20 # [ms] filtering time constant
	Iext = Ipre - (Ipre-_rect((np.random.normal(0,1.8,p.NE+1))**3.))*p.dt/taufilt
	Iext[-1] = Ipre[-1] - (Ipre[-1]-(9+_rect((np.random.normal(0,0.0001,1)))))*p.dt/taufilt    # changed 1 to p.LE
	return Iext

def SimStep (u,ref,xbar_pre,xbar_post,gSynE,WEE,Iext,s):
	# ----------------------------------------------------------------------------
	# - Takes as the input:
	# u: [NE] Membrane potential at time t
	# xbar_pre: [NE] synaptic traces for presynaptic events
	# xbar_post: [NE] synaptic traces for postsynaptic events
	# gSyn_E: [NE] synaptic conductances for excitatory connections
	# WEE: [NExNE] Excitatory synaptic weights
	# Iext: [NE] External current for each neuron
	# s: [str] "up" or "down" - state of the network
	# ----------------------------------------------------------------------------
	# - Returns as output: (all variables for time t+dt)
	# u_out: 
	# xbar_pre_out: 
	# xbar_post_out 
	# gSynE_out 
	# WEE_out
	# ----------------------------------------------------------------------------
	
	spikes = (u>p.Vth) # Verify all the neurons that fired an action potential
	spikesE = spikes[:p.NE] # Excitatory neurons
	ref += spikes*p.Tref  # update the refractory variable
	
	# Update the synaptic conductances
	gSynE_out = gSynE + p.gBarEx * spikesE
	gSynE_out = gSynE_out - gSynE_out*p.step_tauSynEx
	
	# Update the membrane potential
	IsynE = -(u[-1] - p.EsynE)*np.dot(WEE,gSynE)
	Isyn = IsynE 
	Iext = _Iext(Iext)
	
	u = u + (p.Vres-u)*spikes # reset the voltage for those who spiked
	u_out = u + (-u + p.R*Iext)*p.step_tau_m
	u_out[-1] = u_out[-1] + Isyn[-1]*p.step_tau_m
	u_out[(ref>0.001)] = p.Vres
	u_out = u_out + (p.Vspike-u_out+p.Vth)*(u_out>p.Vth) # add a constant to "see" the spikes
	
	ref = _rect(ref - p.dt)
	ref_out = ref
	
	# Update the synaptic traces
	xbar_pre_out = xbar_pre + (10.-xbar_pre)*spikesE
	xbar_pre_out = _rect(xbar_pre_out - p.dt)
	xbar_post_out = xbar_post 
	
	# Update the synaptic weights
	auxMat = np.ones((p.LE,p.NE))    # changed 1 to p.LE
	WEE_out = WEE + p.a_pre[s]*(auxMat*spikesE) \
		+ (-1.*p.a_pre[s])*(auxMat*spikes[-1]) * (xbar_pre>0.)  
	
	if spikes[-1] > 0:
		xbar_pre_out[:] = 0.
	
	WEE_out = _rect(WEE_out) - _rect(WEE_out-p.w_max) # apply bounds
		
	return u_out,ref_out,xbar_pre_out,xbar_post_out,gSynE_out,WEE_out,Iext