##################################################################################
# Make_fig4E.py - Load the data and generate figure 4E
#
# Author: Victor Pedrosa
# Imperial College London, London, UK - Jan 2017
##################################################################################

# ------------------------------------- Import modules -------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import os
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm
import matplotlib as mpl
import UP_state_mediated_plast_fig4E as UP
from imp import reload
import params; reload(params); import params as p

mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['font.weight'] = 900
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['xtick.major.pad']= 10
mpl.rcParams['ytick.major.pad']= 10
mpl.rcParams['xtick.direction']='out'
mpl.rcParams['ytick.direction']='out'

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=19)
plt.rc('font', weight=400)
plt.ion()

cmap = cm.inferno
color0 = cmap(70)

fig = plt.figure(num=1,figsize=(7*0.7, 6*0.7), dpi=100, facecolor='w')
gs1 = GridSpec(1, 1)

def hide_frame(ax):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')

# ================================================================================================================
# Use the data generated with 1-Neuromodulation_and_plasticity.py 
# ================================================================================================================

# ----------------------------------------------------------------------------------------------------------------
# Plot the weights after a fixed time

# Choose the directory and rearrange the files
Dir = 'Data/'
fnames = os.listdir(Dir)
fnames.sort()


# Create the first subplot to show the weights
ax1 = plt.subplot(gs1[0, 0])
hide_frame(ax1)

plt.xlabel(r'Initial weight, $w_0$',fontsize='20',fontweight=900)
plt.ylabel(r'Rel. weight change, $\Delta w$', fontsize='19')
#plt.ylabel(r'Rel. weight change, $\Delta w/w_0$', fontsize='19')
plt.xlim((0,1.05))

W0 = np.array([])
Wend = np.array([])

for i in range(len(fnames)):
	data = np.load(Dir+fnames[i])
	W0 = np.append(W0,data[0]) # added this middle index to capture all Layer 2 activity, not sure if that's the right strategy here tho
	Wend = np.append(Wend,data[1]) # added this middle index to capture all Layer 2 activity, not sure if that's the right strategy here tho

n_trials = len(fnames)
#%%

if not UP.alt_approach:

    # If NOT using the alt approach, run this code
    # reshape num trials, p.LE (Layer 2), p.NE (Layer 4) to compute stats
    # and then reshape final output so that it is a vector
    W0_mean = np.mean(np.reshape(W0,(n_trials,p.LE,p.NE)),axis=0)#.reshape(100*100,)
    Wend_mean = np.mean(np.reshape((Wend-W0)/W0,(n_trials,p.LE,p.NE)),axis=0)#.reshape(100*100,)
    #Wend_mean = np.mean(np.reshape((Wend-W0)/W0,(n_trials,100,100)),axis=0).reshape(100*100,)
    
    Wend_std = np.std(np.reshape((Wend-W0)/W0,(n_trials,p.LE,p.NE)),axis=0)#.reshape(100*100,)
    #Wend_std = np.std(np.reshape((Wend-W0)/W0,(n_trials,100,100)),axis=0).reshape(100*100,)
    
    for r in range(0,W0_mean.shape[0],max(1,int(W0_mean.shape[0]//10))):
        ind = np.argsort(W0_mean[r,:], axis=0)
        wo = np.take_along_axis(W0_mean[r,:], ind, axis=0)
        mn = np.take_along_axis(Wend_mean[r,:], ind, axis=0)
        std = np.take_along_axis(Wend_std[r,:], ind, axis=0)

        plt.plot(wo,mn,color=color0,lw=1.5)
        plt.fill_between(wo,mn-std,mn+std,alpha=.3,color=color0)
        plt.title("Layer 2 Cell #{}".format(r))
        plt.xlabel(r'Initial weight, $w_0$',fontsize='20',fontweight=900)
        plt.ylabel(r'Rel. weight change, $\Delta w/W_0$', fontsize='19')
        plt.xlim((0,1.05))
        plt.show()
    

else: # If using alt approach, run this code
    # reshape num trials, p.LE (Layer 2), p.NE (Layer 4) to compute stats
    # and then reshape final output so that it is a vector
    W0_mean = np.mean(np.reshape(W0,(n_trials,p.LE,p.NE)),axis=0)
    Wend_mean = np.mean(np.reshape((Wend-W0),(n_trials,p.LE,p.NE)),axis=0)
    
    Wend_std = np.std(np.reshape((Wend-W0),(n_trials,p.LE,p.NE)),axis=0)
    
    title_list = ['L4->L4','L2->L4','L4->L2','L2->L2']
    # we need four different graphs of this, one for each quadrant of the W matrices
    for row_start in [0, p.NE//2]:
        for col_start in [0, p.LE//2]:
            plt.plot(W0_mean[row_start:(row_start+p.NE//2),col_start:(col_start+p.LE//2)].reshape((p.NE//2)*(p.LE//2),),Wend_mean[row_start:(row_start+p.NE//2),col_start:(col_start+p.LE//2)].reshape((p.NE//2)*(p.LE//2),),color=color0,lw=1.5)
            plt.fill_between(W0_mean[row_start:(row_start+p.NE//2),col_start:(col_start+p.LE//2)].reshape((p.NE//2)*(p.LE//2),),Wend_mean[row_start:(row_start+p.NE//2),col_start:(col_start+p.LE//2)].reshape((p.NE//2)*(p.LE//2),)-Wend_std[row_start:(row_start+p.NE//2),col_start:(col_start+p.LE//2)].reshape((p.NE//2)*(p.LE//2),),Wend_mean[row_start:(row_start+p.NE//2),col_start:(col_start+p.LE//2)].reshape((p.NE//2)*(p.LE//2),)+Wend_std[row_start:(row_start+p.NE//2),col_start:(col_start+p.LE//2)].reshape((p.NE//2)*(p.LE//2),),alpha=.3,color=color0)
            plt.title(title_list[row_start//(p.NE//2) + (col_start//(p.LE//2))*2])
            plt.xlabel(r'Initial weight, $w_0$',fontsize='20',fontweight=900)
            plt.ylabel(r'Rel. weight change, $\Delta w$', fontsize='19')
            plt.xlim((0,1.05))
            plt.show()


# ----------------------------------------------------------------------------------------------------------------
# Choose the directory and save the figures
plt.savefig('Figures/fig4E.png',dpi=400)

#%%
# ----------------------------------------------------------------------------------------------------------------
# making the histogram

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(UP.WEE.reshape(p.LE*p.NE, 1), bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], rwidth = 0.8)
#plt.ylim(0, 150)
plt.xlim(0.1, 1)
plt.xlabel('initial weight', fontsize = '20')
plt.ylabel('synaptic weight', fontsize = '20')
# Show plot
plt.show()
# 90% of weight changes are 0, other 10% are not


