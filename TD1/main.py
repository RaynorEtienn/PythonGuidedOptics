# %% Commençons par définir les constantes du problème et un vecteur b entre 0 et 1 avec Numpy 2:

import numpy as np
eps_I = 1; eps_II = 2.0**2; d = 0.7 # unit length l=1 micron  # d = 2 when using part 2
b = np.arange(0, 1, 0.005)

# %% On initialize V, k0, f, neff et beta comme des arrays de dimension 2 vides :
nbands_slab = 16                    # total number of modes (TE+TM)
V = np.empty((len(b), int(nbands_slab/2)))
k0=np.empty((len(b), int(nbands_slab/2)))
f = np.empty((len(b), int(nbands_slab/2))) 
neff=np.empty((len(b), int(nbands_slab/2)));
beta = np.empty((len(b),int(nbands_slab/2))) # tuple (len(b), int(nbands_slab/2))
# is matrix dimension: length of b times nbands_slab/2.

# %% On évalue Eq. 1 pour chaque mode m dans une boucle for3; k0 et neff obtenus avec Eqs. 2-3 :
 
f0 = np.empty((len(b),int(nbands_slab/2)))

for m in np.arange(int(nbands_slab/2)):
    V[:,m] = 2/np.sqrt(1-b) * (np.arctan(np.sqrt(b/(1-b))) + m * np.pi/2)
    neff[:,m] = np.sqrt((1-b) * eps_I + b * eps_II)
    k0[:,m] = V[:,m] / (d * np.sqrt(eps_II-eps_I))
    f[:,m] = k0[:,m] / (2 * np.pi)
    beta[:,m] = k0[:,m] * neff[:,m]
    f0[:,m] = beta[:,m] / (2 * np.pi * neff[:,m])

# %% Nous pouvons maintenant tracer les modes TE en utilisant Matplotlib comme suit 4:

import numpy.matlib
import matplotlib.pyplot as plt

nk = 80 # number of k-intervals
k_ll = np.linspace(0,np.pi*10,nk+1).reshape(nk+1,1)
fig1, ax1 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))

plt.plot(beta,f,'-b',k_ll,k_ll/(2*np.pi*np.sqrt(eps_I)),'--k', \
k_ll, k_ll / (2*np.pi*np.sqrt(eps_II)), '--k')

ax1.set_ylim([0, 4])
ax1.set_xlim([0, 10*np.pi])
ax1.set_xlabel("$ \\beta \ell $")
ax1.set_ylabel("$ \ell/\lambda $")
fig1.suptitle('Bands 1D slab. Blue: TE.') # labels and title


# %% Question 2

nbands_slab = 16                    # total number of modes (TE+TM)
V_TM = np.empty((len(b), int(nbands_slab/2))) 
k0_TM=np.empty((len(b), int(nbands_slab/2)))
f_TM = np.empty((len(b), int(nbands_slab/2))) 
neff_TM=np.empty((len(b), int(nbands_slab/2)))
beta_TM = np.empty((len(b),int(nbands_slab/2))) # tuple (len(b), int(nbands_slab/2))
# is matrix dimension: length of b times nbands_slab/2.

f0_TM = np.empty((len(b),int(nbands_slab/2)))

for m in np.arange(int(nbands_slab/2)):
    V_TM[:,m] = 2/np.sqrt(1-b) * (np.arctan((eps_II/eps_I) * np.sqrt(b/(1-b))) + m * np.pi/2)
    neff_TM[:,m] = np.sqrt((1-b) * eps_I + b * eps_II)
    k0_TM[:,m] = V_TM[:,m] / (d * np.sqrt(eps_II-eps_I))
    f_TM[:,m] = k0_TM[:,m] / (2 * np.pi)
    beta_TM[:,m] = k0_TM[:,m] * neff_TM[:,m]
    f0_TM[:,m] = beta_TM[:,m] / (2 * np.pi * neff_TM[:,m])

# plotting of TM modes
fig2, ax2 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))
plt.plot(beta_TM, f0_TM, '-r', k_ll, k_ll / (2*np.pi*np.sqrt(eps_I)), '--k', \
k_ll, k_ll / (2*np.pi*np.sqrt(eps_II)), '--k')

ax2.set_ylim([0, 4])
ax2.set_xlim([0, 10*np.pi])
ax2.set_xlabel("$ \\beta \ell $") 
ax2.set_ylabel("$ \ell/\lambda $")
fig2.suptitle('Bands 1D slab. Red: TM') # labels and title

fig3, ax3 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))
plt.plot(beta_TM, f0_TM, '-r', k_ll, k_ll / (2*np.pi*np.sqrt(eps_I)), '--k', \
k_ll, k_ll / (2*np.pi*np.sqrt(eps_II)), '--k', \
beta,f,'-b')

ax3.set_ylim([0, 4])
ax3.set_xlim([0, 10*np.pi])
ax3.set_xlabel("$ \\beta \ell $") 
ax3.set_ylabel("$ \ell/\lambda $")
fig3.suptitle('Bands 1D slab. Blue: TE. Red: TM') # labels and title

fig4, ax4 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))
plt.plot(neff, f0)

