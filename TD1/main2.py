# %% Commençons par définir les constantes du problème et un vecteur b entre 0 et 1 avec Numpy 2:

import numpy as np
eps_I = 1; eps_II = 2.0**2; d = 2 # unit length l=1 micron  # d = 2 when using part 2
b = np.arange(0, 1, 0.005)

# %% On initialize V, k0, f, neff et beta comme des arrays de dimension 2 vides :
nbands_slab = 16                    # total number of modes (TE+TM)
V = np.empty((len(b), int(nbands_slab/2)))
k0 = np.empty((len(b), int(nbands_slab/2)))
f = np.empty((len(b), int(nbands_slab/2))) 
neff = np.empty((len(b), int(nbands_slab/2)));
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

plt.plot(beta, f, '-b', k_ll, k_ll / (2*np.pi*np.sqrt(eps_I)), '--k', \
         k_ll, k_ll / (2*np.pi*np.sqrt(eps_II)), '--k')

ax1.set_ylim([0, 1.5])
ax1.set_xlim([0, 10])
ax1.set_xlabel("$ \\beta \ell $")
ax1.set_ylabel("$ \ell/\lambda $")
fig1.suptitle('Bands 1D slab. Blue: TE.') # labels and title


# %% Question 2

nbands_slab = 16                    # total number of modes (TE+TM)
V_TM = np.empty((len(b), int(nbands_slab/2))) 
k0_TM = np.empty((len(b), int(nbands_slab/2)))
f_TM = np.empty((len(b), int(nbands_slab/2))) 
neff_TM = np.empty((len(b), int(nbands_slab/2)))
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

ax2.set_ylim([0, 1.5])
ax2.set_xlim([0, 10])
ax2.set_xlabel("$ \\beta \ell $") 
ax2.set_ylabel("$ \ell/\lambda $")
fig2.suptitle('Bands 1D slab. Red: TM') # labels and title

fig3, ax3 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))
plt.plot(beta_TM, f0_TM, '-r', k_ll, k_ll / (2*np.pi*np.sqrt(eps_I)), '--k', \
         k_ll, k_ll / (2*np.pi*np.sqrt(eps_II)), '--k', \
         beta,f,'-b')

ax3.set_ylim([0, 1.5])
ax3.set_xlim([0, 10])
ax3.set_xlabel("$ \\beta \ell $") 
ax3.set_ylabel("$ \ell/\lambda $")
fig3.suptitle('Bands 1D slab. Blue: TE. Red: TM') # labels and title



# %% On importe Legume :
import legume

# %% On définit les paramètres de la structure :
a_x = 0.1; a_y = 3 # x and y-period of the Bravais lattice in microns
w = 2 # WG width
gmax = 9 # size of the region in reciprocal space containing the G base vectors

# %% On définit la structure :
lattice = legume.Lattice([a_x,0], [0,a_y]) # rectangular lattice
rectangle = legume.Poly(eps=eps_II, x_edges = [a_x/2, a_x/2, -a_x/2, -a_x/2], \
y_edges = [-w/2, w/2, w/2, -w/2]) # WG core; corners must be in counter-clockwise order
layer = legume.ShapesLayer(lattice, eps_b = eps_I) # layer= lattice points & bg index
layer.add_shape(rectangle) # add the core structure to the layer

# %% On définit le chemin de vecteurs d’onde de Bloch k :
path = lattice.bz_path(['G', [np.pi/a_x,0]], [nk])

# %% Expansion en ondes planes of the permittivity (layer PWE) :
pwe = legume.PlaneWaveExp(layer, gmax = gmax) # performs PWE of layer. G vectors inside

# a region of size 2*pi*g_max are utilized
legume.viz.eps_ft(pwe, figsize=(5, 4)) # visualize the reconstructed permittivity
fig4 = plt.gcf() 
fig4.suptitle('1D WG in xz (PWE). Perm') # fig title
fig4.get_axes()[0].set_xlabel("x") 
fig4.get_axes()[0].set_ylabel("y") # axes labels

# %% Run PWE pour les modes TE du slab 1D 5:    
pwe.run(kpoints = path['kpoints'], pol = 'tm',numeig = int(nbands_slab/2))
freqs_te = pwe.freqs

# %% Run PWE pour les modes TM du slab 1D :
pwe.run(kpoints = path['kpoints'], pol = 'te', numeig = int(nbands_slab/2))
freqs_tm = pwe.freqs

# %% Plot :
kx = path['kpoints'][0]

fig5, ax5 = plt.subplots(1, constrained_layout = True, figsize=(5, 4))
ax5.plot(kx, freqs_te, '-g', kx, freqs_tm, '-k', beta_TM, f0_TM, '-r', \
         beta, f0, '-b', k_ll, k_ll/(2*np.pi*np.sqrt(eps_I)), \
         '--k', k_ll, k_ll/(2*np.pi*np.sqrt(eps_II)), '--k')

fig5.suptitle('Bands 1D slab y-normal. Blue & green: TE. Red & Black: TM')
ax5.set_ylim([0, 1])
ax5.set_xlim([0, 10])
ax5.set_ylabel("$ \omega \ell/2 \pi c$")
ax5.set_xlabel("$ \\beta \ell $")
ax5.xaxis.grid('True') # labels and title
