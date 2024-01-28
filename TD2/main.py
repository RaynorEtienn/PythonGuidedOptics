# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:39:46 2024

@author: 33670
"""

# %% On importe les librairies :
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import legume

# %% On définit les paramètres du guide et on initialize les variables :
eps_I = 1
eps_II = 2.0**2 
d = 0.7
w = 2 # length in microns -> length unit is l=1 micron

nbands_slab = 16 
nk = 200
b = np.arange(0,1,0.005)
f_ridge = np.empty((len(b),1))
neff_ridge = np.empty((len(b),1))
beta_ridge = np.empty((len(b),1))

# %% On calcule en premier terme le mode fondamental TE du guide transverse, i.e. confiné dans z :
V0_TE = 2 / np.sqrt(1-b) * np.arctan(np.sqrt(b/(1-b)))
neff0 = np.sqrt((1-b)*eps_I+b*eps_II)
k0 = V0_TE/(d*np.sqrt(eps_II-eps_I))
f0 = k0 / (2*np.pi)
beta0 = k0 * neff0

# %% On calcule V_TM pour chaque b[i], et on trouve l’indice des arrays k0 et neff0 qui satisfont les
# Eqs. 1 et 2, en utilisant la fonction nanargmin de Numpy :
for i in np.arange(len(b)):
    V0_TM = 2 / np.sqrt(1-b[i]) * (np.arctan(neff0**2/eps_I * np.sqrt(b[i]/(1-b[i]))))
    index = np.nanargmin(np.fabs(V0_TM**2-k0**2*w**2*(neff0**2-eps_I))) #Solves the

# implicit equation V0_TM=k0 w sqrt(neff(k0)^2-n2^2) for k0
    f_ridge[i] = k0[index]/(2*np.pi)
    neff_ridge[i] = np.sqrt((1-b[i]) * eps_I + b[i] * neff0[index]**2)
    beta_ridge[i] = k0[index] * neff_ridge[i]

# %% On trace le mode :
k_ll = np.linspace(0,np.pi*10,nk+1).reshape(nk+1,1)
fig1, ax1 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))
ax1.plot(beta_ridge, f_ridge, '-r', k_ll, k_ll / (2*np.pi*np.sqrt(eps_I)) ,\
          '--k', k_ll, k_ll / (2*np.pi*np.sqrt(eps_II)), '--k')
ax1.set_xlabel("$ \\beta $")
ax1.set_ylabel("$ f = k_0 / 2 \pi $")
fig1.suptitle("Slab's fundamental mode TE")

# Not super clean but useful for the next part
nbands_slab = 2                    # total number of modes (TE+TM)
V = np.empty((len(b), int(nbands_slab/2)))
k0 = np.empty((len(b), int(nbands_slab/2)))
f = np.empty((len(b), int(nbands_slab/2))) 
neff = np.empty((len(b), int(nbands_slab/2)));
beta = np.empty((len(b), int(nbands_slab/2))) # tuple (len(b), int(nbands_slab/2))
    
f0 = np.empty((len(b),int(nbands_slab/2)))

for m in np.arange(int(nbands_slab/2)):
    V[:,m] = 2/np.sqrt(1-b) * (np.arctan(np.sqrt(b/(1-b))) + m * np.pi/2)
    neff[:,m] = np.sqrt((1-b) * eps_I + b * eps_II)
    k0[:,m] = V[:,m] / (d * np.sqrt(eps_II-eps_I))
    f[:,m] = k0[:,m] / (2 * np.pi)
    beta[:,m] = k0[:,m] * neff[:,m]
    f0[:,m] = beta[:,m] / (2 * np.pi * neff[:,m])

fig2, ax2 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))
ax2.plot(f_ridge, neff_ridge, '-r')
ax2.plot(f0, neff, '-b')

ax2.set_xlabel("$ f = k_0 / 2 \pi $")
ax2.set_ylabel("$ n_{eff} $")

fig2.suptitle("Slab's indices'. Blue: 1D Red: 3D")
