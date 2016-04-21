# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 00:49:37 2016

@author: ROHAN
"""

"""Solve the 1D diffusion equation using CN and finite differences.""" 
import numpy as np
from time import sleep
import matplotlib.pyplot as plt 
import networkx as nx
from pylab import *

# The total number of nodes 
nodx = 3
nody = 3
nnodes = nodx*nody

#this is for the plotting xmin = 0.0
xmin = 0.0
xmax = 1.0 
ymin = 0.0 
ymax = 1.0
dt = 0.5
ntimes = 100
Ny = 4
Nx = 4
tmax = 1000.0
Nt = 3000
# Ny = 4
# Nx = 4
# tmin = 0.0
# tmax = 1000.0
# Nt = 3000

# The total number of times ntimes = 100

# The time step dt = 0.5
# The diffusion constant

D = np.matrix(np.eye(nnodes,nnodes)) 
D = .1*D
Drod = 2
# this loop trys to set a few of the central D values to be different #so we can simulate something being stuck in the center of the device 

for i in range(4):
    D[i + nnodes/2, i + nnodes/2] =  1
    D[i + nnodes/2, i + nnodes/2] =  Drod*D[i + nnodes/2, i + nnodes/2]
# The spatial mesh size

h = 1.0
tmin = 0.0
tmax = dt*ntimes

x, dx = np.linspace(xmin, xmax, nodx, retstep=True) 
y, dy = np.linspace(ymin, ymax, nody, retstep=True) 
t, dt = np.linspace(tmin, tmax, ntimes, retstep=True) 
G = nx.grid_graph(dim=[nodx,nody])
L = np.matrix(nx.laplacian(G))

#making an expression for the heat source to go into the rhs 
C = np.matrix(np.zeros((nnodes,nnodes)))
C[nnodes/2,nnodes/2] = 0
#	The rhs of the diffusion equation rhs = -D*L/h**2 + C 

#	Setting initial temperature 

T = 60*np.matrix(np.ones((nnodes,ntimes))) 
for i in range(nnodes/2):
    T[i,0] = 0;

#	Setup the time propagator. In this case the rhs is time-independent so we 
#	can do this once. 

ident = np.matrix(np.eye(nnodes,nnodes)) 
pmat = ident+(dt/2.0)*rhs
mmat = ident-(dt/2.0)*rhs
propagator = np.linalg.inv(mmat)*pmat

#	Propagate E is for energy conservation E = np.zeros(ntimes) 

for i in range(ntimes-1): E[i] = sum(T[:,i]) 
T[:,i+1] = propagator*T[:,i] 

#	To plot 1 time 
print E[2]

#need to convert the big string T into a matrix for plotting and visualization w = 0

#	R = np.matrix(np.zeros((nodx,nody,ntimes))) 
#	t[:,:,:] = R[:,:,:] 

#	a 3d array (two stacked 2d arrays) 
t = np.zeros((nodx, nody, ntimes))

w = 0
for p in range(ntimes):
    for i in range(nodx):

        for j in range(nody): 
            t[i,j,p] = T[w, p] 
            w = w + 1
w = 0
# print w

print t[:,:,1]
#cannot plot numpy arrays! so we have to use a normal u array
# u[:,:] = T[:,:]

V, dV = np.linspace(0, 70, 4, retstep=True)

print V
CS = plt.contourf(x,y,t[:,:,0], V)

plt.ylabel('distance (m)') 
plt.xlabel('distance (m)')

# Make a colorbar for the ContourSet returned by the contourf call. cbar = colorbar(CS)
cbar.ax.set_ylabel('verbosity coefficient')

plt.show()

# To plot all times #plt.plot(T)
