# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:18:35 2021

@author: maria
"""

import numpy as np
from imp import reload
import params; reload(params); import params as p


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
                    
WEE = np.zeros((p.LE,p.NE))	# E-E connections

WEE[50:, 50:] = create_WS_network(p.NE // 2, 4, 0.2)    # layer 2/3  (bottom right)
WEE[0:50, 0:50] = create_WS_network(p.LE // 2, 2, 0.1)    # layer 4  (top left)
WEE[50:, 0:50] = create_WS_network(p.NE // 2, 8, 0.3)    # layer 2/3 to layer 4 (bottom left)
WEE[0:50, 50:] = create_WS_network(p.LE // 2, 4, 0.1)    # layer 4 to layer 2/3 (top right)
import matplotlib.pyplot as plt

f = plt.figure()
plt.imshow(WEE, cmap="gray")
plt.plot([(p.LE + p.NE)//4, (p.LE + p.NE)//4],[0, (p.LE + p.NE)//2 -1],'y')
plt.plot([0, (p.LE + p.NE)//2 -1],[(p.LE + p.NE)//4, (p.LE + p.NE)//4],'y')
plt.title("Weight Matrix")
plt.xlabel("Layer 4 <-- Postsynaptic cells --> Layer 2/3")
plt.ylabel("Layer 2/3 <-- Presynaptic cells --> Layer 4")
