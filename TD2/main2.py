# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:01:21 2024

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
eps_III = 1.5**2
d = 0.7
w = 0.7 # length in microns -> length unit is l=1 micron

nbands_slab = 2
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
# fig1, ax1 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))
# ax1.plot(beta_ridge, f_ridge, '-c', k_ll, k_ll / (2*np.pi*np.sqrt(eps_I)) ,\
#           '--k', k_ll, k_ll / (2*np.pi*np.sqrt(eps_II)), '--k')

# A refaire au propre
nbands_slab = 16                    # total number of modes (TE+TM)
V = np.empty((len(b), int(nbands_slab/2)))
k0=np.empty((len(b), int(nbands_slab/2)))
f = np.empty((len(b), int(nbands_slab/2))) 
neff=np.empty((len(b), int(nbands_slab/2)));
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
ax2.plot(f0, neff)

ax2.set_xlabel("$ f = k_0 / 2 \pi $")
ax2.set_ylabel("$ n_{eff} $")
fig2.suptitle("Slab's indices'. Blue: 1D. Red: 3D. Colors: 3D-GME")



# %% On définit les paramètres de la structure :
a_x = 0.1
a_y = 5 # x and y-period of the Bravais lattice in microns
gmax = 9 # size of the region in reciprocal space containing the G base vectors
lambda_target = 1.5 # target wavelength of fund. mode to plot fields

# %% On définit la structure :
lattice = legume.Lattice([a_x,0], [0,a_y]) # rectangular lattice
rectangle_u = legume.Poly(eps=eps_I, x_edges=[a_x/2,a_x/2,-a_x/2,-a_x/2], \
                          y_edges = [w/2, a_y/2, a_y/2, w/2]) # air slots in material background
rectangle_d = legume.Poly(eps=eps_I, x_edges=[a_x/2,a_x/2,-a_x/2,-a_x/2], \
                          y_edges=[-a_y/2,-w/2,-w/2,-a_y/2]) # corners must be in counter-clockwise order


ridge = legume.PhotCryst(lattice, eps_l=eps_I, eps_u=eps_I)#Changing here for substrat
# ridge.add_layer(d=w, eps_b=eps_III)
ridge.add_layer(d=d, eps_b=eps_II)
ridge.add_shape([rectangle_u, rectangle_d])

# %% On définit le chemin de vecteurs d’onde de Bloch k :
path = lattice.bz_path(['G', [np.pi/a_x,0]], [nk])

# %% Expansion en modes guidés de la structure (ridge GME) :
gme = legume.GuidedModeExp(ridge, gmax=9,truncate_g='tbt') # truncate_g=’abs’
# results in G within a circle of r=gmax; ’tbt’ is square of side gmax in recip. space.
legume.viz.structure(ridge, yz=True, xy=False, figsize=(5, 4))
fig2 = plt.gcf(); fig2.suptitle('Ridge WG in yz (GME). Perm')
fig2.get_axes()[0].set_xlabel("y")
fig2.get_axes()[0].set_ylabel("z")

# %% Run GME pour guide ridge 7:
gme.run(kpoints=path['kpoints'], gmode_inds=[0,1,2,3], \
        numeig=nbands_slab, verbose=False, eps_eff='background') # effective slab = bg index
freqsridge_te = gme.freqs[:, [0, 2, 4, 6, 7]]
freqsridge_tm = gme.freqs[:, [1, 3, 5]]
index_target = np.argmin(np.fabs(freqsridge_te[:,0]-1/lambda_target)) # for plot field

# %% Plot bands 8:
kx = np.reshape(path['kpoints'][0],(nk+1,1))

kmatrix_ridge_te = np.tile(kx,(1,np.shape(freqsridge_te)[-1]))
kmatrix_ridge_tm = np.tile(kx,(1,np.shape(freqsridge_tm)[-1]))

freqsridge_te[freqsridge_te>kmatrix_ridge_te/(2*np.pi*np.sqrt(eps_I))] = np.nan #eps_III pour cacher sous eps_III radiatif
freqsridge_tm[freqsridge_tm>kmatrix_ridge_tm/(2*np.pi*np.sqrt(eps_I))] = np.nan

fig3, ax3 = plt.subplots(1, constrained_layout=True, figsize=(5, 4))
ax3.plot(kx, freqsridge_te, 'b', kx, freqsridge_tm, 'r',\
          beta_ridge, f_ridge, '-c', kx, kx/(2*np.pi*np.sqrt(eps_I)), '--k',\
              kx, kx/(2*np.pi*np.sqrt(eps_II)), '--k',)
    
plt.plot([kx[index_target], kx[index_target]], [0,5],'k')

fig3.suptitle('Bands ridge WG.')
ax3.set_ylim([0., 1.])
ax3.set_xlim([0, 10])
ax3.set_ylabel("$ \omega \ell/2 \pi c$")
ax3.set_xlabel("$ \\beta \ell $")
ax3.xaxis.grid('True') # labels and title


# # %% Plot fields (optionnel) :
# # TE00; sigma_xy=1 (TE-like); sigma_kx=-1; Dz!=0, Dx!=0
# fig4=legume.viz.field(gme, field='d', kind=index_target, mind=0, x=0, \
# component='y', val='im', N1=200, N2=100, cbar=True, eps=True)
# fig4.get_axes()[0].set_xlabel("y"); fig4.get_axes()[0].set_ylabel("z")

# # TM00; sigma_xy=-1 (TM-like); sigma_kx=1; Dy!=0, Dx!=0
# fig5=legume.viz.field(gme, field='d', kind=index_target, mind=1, x=0, \
# component='z', val='re', N1=200, N2=100, cbar=True, eps=True)
# fig5.get_axes()[0].set_xlabel("y"); fig5.get_axes()[0].set_ylabel("z")

# # TE01; sigma_xy=1 (TE-like); sigma_kx=1; Dz!=0; Dx!=0
# fig6=legume.viz.field(gme, field='d', kind=index_target, mind=2, x=0, \
# component='y', val='re', N1=200, N2=100, cbar=True, eps=True)
# fig6.get_axes()[0].set_xlabel("y"); fig6.get_axes()[0].set_ylabel("z")

# # TM01; sigma_xy=-1 (TM-like); sigma_kx=-1; Dy!=0, Dx!=0
# fig7=legume.viz.field(gme, field='d', kind=index_target, mind=3, x=0, \
# component='z', val='im', N1=200, N2=100, cbar=True, eps=True)
# fig7.get_axes()[0].set_xlabel("y"); fig7.get_axes()[0].set_ylabel("z")

# # TE02; sigma_xy=1 (TM-like); sigma_kx=-1; Dy!=0, Dx !=0
# fig8=legume.viz.field(gme, field='d', kind=index_target, mind=4, x=0, \
# component='y', val='im', N1=200, N2=100, cbar=True, eps=True)
# fig8.get_axes()[0].set_xlabel("y"); fig8.get_axes()[0].set_ylabel("z")

# # TM02; sigma_xy=-1 (TE-like); sigma_kx=1; Dz!=0, Dx!=0
# fig9=legume.viz.field(gme, field='d', kind=index_target, mind=5, x=0, \
# component='z', val='re', N1=200, N2=100, cbar=True, eps=True)
# fig9.get_axes()[0].set_xlabel("y"); fig9.get_axes()[0].set_ylabel("z")

# # TE10; sigma_xy=-1 (TM-like); sigma_kx=-1; Dy!=0, Dx !=0
# fig10=legume.viz.field(gme, field='d', kind=index_target, mind=6, x=0, \
# component='y', val='im', N1=200, N2=100, cbar=True, eps=True)
# fig10.get_axes()[0].set_xlabel("y"); fig10.get_axes()[0].set_ylabel("z")

# # TE03; sigma_xy=1 (TE-like); sigma_kx=1; Dz!=0, Dx!=0
# fig11=legume.viz.field(gme, field='d', kind=index_target, mind=7, x=0, \
# component='y', val='re', N1=200, N2=100, cbar=True, eps=True)
# fig11.get_axes()[0].set_xlabel("y"); fig11.get_axes()[0].set_ylabel("z")

# # Create a list to store images
# fig_list = [fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11]; imglist = []
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# for fig in fig_list:
# # Create a canvas
#     canvas = FigureCanvas(fig)
#     # Render the figure (draw the figure)
#     canvas.draw()
#     # Get the image as a numpy array and append to imglist
#     img = np.array(canvas.renderer.buffer_rgba())
#     imglist.append(img)

# # Create a 4x2 grid of subplots
# fig12, axs = plt.subplots(4, 2, figsize=(10, 8))
# # Flatten the axs array for easier indexing
# axs = axs.flatten()
# # Iterate over images and subplots
# for i in range(len(imglist)):
#     # Get the image at the corresponding index
#     image = imglist[i]
#     # Display the image in the corresponding subplot
#     axs[i].imshow(image)
#     axs[i].axis('off')
# # Adjust layout to prevent overlap
# plt.tight_layout()

