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
plt.ylabel(r'Rel. weight change, $\Delta w/w_0$', fontsize='19')
plt.xlim((0,1.05))

W0 = np.array([])
Wend = np.array([])

for i in range(len(fnames)):
	data = np.load(Dir+fnames[i])
	W0 = np.append(W0,data[0,:,0])
	Wend = np.append(Wend,data[1,:,0])

n_trials = len(fnames)

W0_mean = np.mean(np.reshape(W0,(n_trials,100)),axis=0)
Wend_mean = np.mean(np.reshape((Wend-W0)/W0,(n_trials,100)),axis=0)

Wend_std = np.std(np.reshape((Wend-W0)/W0,(n_trials,100)),axis=0)

plt.plot(W0_mean,Wend_mean,color=color0,lw=1.5)
plt.fill_between(W0_mean,Wend_mean-Wend_std,Wend_mean+Wend_std,alpha=.3,color=color0)


# ----------------------------------------------------------------------------------------------------------------
# Choose the directory and save the figures
plt.savefig('Figures/fig4E.png',dpi=400)