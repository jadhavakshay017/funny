# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 00:43:12 2016

@author: ROHAN
"""

 
#"""Solve the 1D diffusion equation using CN and finite differences."""
import numpy as np
from time import sleep
import matplotlib.pyplot as plt 
import networkx as nx

#	The total number of nodes nnodes = 10 

#	The total number of times ntimes = 500 

#	The time step 
ntimes = 500 
nnodes = 10 
dt = 0.5
D = 0.1 
h = 1.0 
#	The diffusion constant D = 0.1 

#	The spatial mesh size h = 1.0 

G = nx.grid_graph(dim=[nnodes])

L = np.matrix(nx.laplacian(G))

# The rhs of the diffusion equation rhs = -D*L/h**2
# Setting initial temperature

T = 60*np.matrix(np.ones((nnodes, ntimes))) 
for i in range(nnodes/2):
    T[i,0] = 0;

# Setup the time propagator. In t his case the rhs is time-independent so we

# can do this once.

id = np.matrix(np.eye(nnodes,nnodes)) 
pmat = id+(dt/2.0)*rhs
mmat = id-(dt/2.0)*rhs

propagator = np.linalg.inv(mmat)*pmat

# Propagate

for i in range(ntimes-1):
    T[nnodes/2,i] = T[nnodes/2, i] + T[:,i+1] = propagator*T[:,i]

plt.plot(T[:,300])
plt.show()  
#	To plot 1 time plt.plot(T[:,300]) plt.show() 

#	To plot all times #plt.plot(T) 
