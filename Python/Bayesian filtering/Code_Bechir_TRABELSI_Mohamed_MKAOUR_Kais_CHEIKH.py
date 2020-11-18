# -*- coding: utf-8 -*-
"""
SOD 333 : filtrage bayésian
 
@author: Kais CHEIKH
         Bechir TRABELSI
         Mohamed MKAOUAR
"""
# Import des bibliothéques
import numpy as np # calcul numerique
import numpy.random as rnd # fonctions pseudo-aleatoires
import matplotlib.pyplot as plt # fonctions graphiques a la MATLAB
import matplotlib.animation as anim # fonctions d'animation
import scipy.io as io # fonctions pour l'ouverture des fichiers .mat de MATLAB
from math import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
 
# Initialisation des varibles du problemes: 
# Ils sont considéres comme des varibales globales
X1MIN,X1MAX = -1e4, 1e4 
X2MIN,X2MAX = -1e4, 1e4
r0 = (-6000,2000)
v0 = (120,0)
sigma_r0 = 100
sigma_v0 = 10
sigma_INS = 7 
sigma_ALT = 10 
sigma_BAR = 20 
Delta = 1 
T =100
 

# definition d'une fonction donnant les indices des ancetres dans la redistribution multinomiale
def resampling_multi(w,N):
    u_tild = np.zeros((N))
    expo = np.zeros((N))
    alpha = np.zeros((N))
    u_ord = np.zeros((N))
    uu = np.zeros((N+1))
    s = np.zeros((N))
#
    w = w/w.sum()
    s = np.cumsum(w)
    u_tild = rnd.uniform(0,1,N)
#
    for i in range(N):
        alpha[i] = u_tild[i]**(1/float(i+1))
    alpha = np.cumprod(alpha)
    u_ord = alpha[N-1]/alpha
    u = np.append(u_ord,float("inf"))
#
    ancestor = np.zeros(N,dtype=int)
    offsprings = np.zeros(N,dtype=int)
    i = 0
    for j in range(N):
        o = 0
        while u[i]<=s[j]:
            ancestor[i] = j
            i = i+1
            o = o+1
        offsprings[j] = o
    return ancestor

# fonction de simulation des algortithmes 
# c = coefficient de sensibilité de l'algorithme
# T = la duree total des sequences 
# N = Nombre des particules a simuler
# sim = varible binaire, s'il est True elle affiche l'évolution des particules sur un graphe.
# evol = varible binaire, s'il est True elle affiche le déroulemnet de l'algo. 
# Elle retourne 3 variables : 
# RV : matrice contenant les valeurs de (r,v) estimée à chaque particule à chaque instant.
# weights : les poids respectifs associé à chaque particule à chaque instant.
# X_i :matrice contenat les particules (dr,dv)
# h : Les altitudes associées à chaque compesant de RV.

def Adaptive(c,T,N,sim=True,evol=True) : 
    #### Création des matrices contenant tous les états pour toutes les particules
    X_i = np.zeros(shape=(T+1,N,4)) # Matrice des particules(les écarts : dr,dv)
    RV = np.zeros(shape=(T+1,N,4))  # Matrice des positions et des vitesses estimés (r_INS+dr,v_INS+dv)
    h = np.zeros(shape=(T+1,N)) # Matrice des altitudes des particules
   
    #### Tirage des premiers élements.
    for i in range(N):
        X_i[0,i,0:2] = rnd.normal(size=2,loc=0,scale=sigma_r0)
        X_i[0,i,2:4] = rnd.normal(size=2,loc=0,scale=sigma_v0)
    RV[0,:,0:2] = r0 + X_i[0,:,0:2]
    RV[0,:,2:4] = v0 + X_i[0,:,2:4]
 
    weights = np.zeros(shape=(T+1,N)) # Création de la matrice des poids 
 
    #### Les poids initiaux
    
    # Determination des indices 
    i_ind= (X2MAX - RV[0,:,1]) * N2 /(X2MAX-X2MIN) 
    i_ind=np.ceil(i_ind) 
    j_ind= (RV[0,:,0] - X1MIN) * N1 /(X1MAX-X1MIN) 
    j_ind=np.ceil(j_ind) 
 
    #On calcule le h0 
    h[0,:] = map[i_ind.astype(int),j_ind.astype(int)] 
    weights[0,:]= vraisemblance(h=h_ALT[0], mu= h[0,:])
    weights[0,:] = weights[0,:] / np.sum(weights[0,:])
 
    #### La boucle du filtre     
    for k in range(1,T+1): 
        #On calcule N effective :
        Neff= 1 / (np.sum(weights[k-1,:]**2))
 
        ### algorithme SIR 
        if Neff <= c*N :
            if(evol==True):
                print(" instant ",k,"  : SIR avec Neff = ",Neff)
            ### Phase de tirage
            #indices = rnd.choice(range(0,N),size=
                                 #N,p=weights[k-1,:])
            indices = resampling_multi(w=weights[k-1,:],
                                       N=N)
            Xi_hat = X_i[k-1,indices,:] 
 
            ### Phase de prédiction 
            w_INS = rnd.multivariate_normal(mean=[0,0],cov=[[sigma_INS,0],[0,sigma_INS]],size=N)
            X_i[k,:,0:2] = Xi_hat[:,0:2]+Delta*Xi_hat[:,2:4]
            X_i[k,:,2:4] = Xi_hat[:,2:4]-Delta*w_INS
 
            ### Phase de correction 
            #On calcule r et v 
            RV[k,:,0:2] = r_INS[:,k]+X_i[k,:,0:2]
            RV[k,:,2:4] = v_INS[:,k]+X_i[k,:,2:4]
 
            ### Calcul des poids     
            # Determination des indices 
            i_ind= (X2MAX - RV[k,:,1]) * N2 /(X2MAX-X2MIN) 
            i_ind=np.clip(np.ceil(i_ind), 0,N2-1) 
            j_ind= (RV[k,:,0] - X1MIN) * N1 /(X1MAX-X1MIN) 
            j_ind=np.clip(np.ceil(j_ind),0,N1-1) 
            
            ### On calcule le hk 
            h[k,:] = map[i_ind.astype(int),j_ind.astype(int)]  
            weights[k,:] = vraisemblance(h=h_ALT[k], mu= h[k,:])
            weights[k,:] = weights[k,:] / np.sum(weights[k,:])
 
        ### algorithme SIS : 
        elif Neff > c*N : 
            if(evol==True) :
                print(" instant ",k,"  : SIS avec Neff = ",Neff)
            
            ### Phase de prédiction 
            w_INS = rnd.multivariate_normal(mean=[0,0],cov=[[sigma_INS,0],[0,sigma_INS]],size=N)
            X_i[k,:,0:2] = X_i[k-1,:,0:2]+Delta*X_i[k-1,:,2:4]
            X_i[k,:,2:4] = X_i[k-1,:,2:4]-Delta*w_INS
 
            ### Phase de correction 
            #On calcule r et v 
            RV[k,:,0:2] = r_INS[:,k]+X_i[k,:,0:2]
            RV[k,:,2:4] = v_INS[:,k]+X_i[k,:,2:4]
 
            ### Calcul des poids     
            # Determination des indices 
            i_ind= (X2MAX - RV[k,:,1]) * N2 /(X2MAX-X2MIN) 
            i_ind=np.clip(np.ceil(i_ind),0,N2-1)  
            j_ind= (RV[k,:,0] - X1MIN) * N1 /(X1MAX-X1MIN) 
            j_ind=np.clip(np.ceil(j_ind),0,N1-1)
            
            ### On calcule le hk 
            h[k,:] = map[i_ind.astype(int),j_ind.astype(int)] 
            q_k = vraisemblance(h=h_ALT[k], mu= h[k,:])
            weights[k,:] = (weights[k-1,:]*q_k) / np.sum(weights[k-1,:]*q_k)
 
        # Affichage de l'évolution 
        if(sim==True): 
            plt.clf()
            plt.imshow(map,cmap='jet',extent=[X1MIN,X1MAX,X2MIN,X2MAX])
            plt.plot(rtrue[0,:],rtrue[1,:],'r-',label='Real')
            plt.plot(r_INS[0,:],r_INS[1,:],'y-',label='INS')
            plt.scatter(RV[k,:,0],RV[k,:,1],s=1,color = 'black',label="Particules")
            plt.plot(np.sum(weights[k,:]*RV[k,:,0]),np.sum(weights[k,:]*RV[k,:,1]),'r*'
                 ,label="Moyenne des particules")
            plt.legend()
            plt.pause(0.1)
            plt.show()
 
    return (RV, weights,X_i,h)
# On dessine la grille 3D : 
val2=[]
for i in range(map.shape[1]):
               for j in range(map.shape[0]):
                  
                   val2.append((i,j,map[j][i]))
x, y, z = zip(*val2)
#z = list(map(float, z))
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
grid_z = griddata((y, x), z, (grid_y, grid_x), method='cubic')
fig = plt.figure()
ax = fig.gca(projection='3d')
im=ax.plot_surface(grid_y, grid_x, grid_z, cmap='jet',alpha=1)
ax.contour3D(grid_y, grid_x, grid_z, 200, cmap='jet')
fig.colorbar(im)
plt.show()

# On dessine la grille :
map = io.loadmat('mnt')['map']
N1 = map.shape[1]
N2 = map.shape[0]
 
plt.imshow(map,cmap='jet',extent=[X1MIN,X1MAX,X2MIN,X2MAX])
 

# le plot de trajectoire reelle : 
traj = io.loadmat('traj.mat')
rtrue = traj['rtrue']
vtrue = traj['vtrue']   
 
plt.plot(rtrue[0,:],rtrue[1,:],'r-',label='Real')
 
 
# Lecture des valeurs INS :
a_INS = io.loadmat('ins.mat')['a_INS']
 
nmax = np.shape(a_INS)[1]
r_INS = np.zeros(rtrue.shape)
v_INS = np.zeros(vtrue.shape)
r_INS[:,0] = r0
v_INS[:,0] = v0
for k in range(1,nmax):
    r_INS[:,k] = r_INS[:,k-1]+Delta*v_INS[:,k-1]
    v_INS[:,k] = v_INS[:,k-1]+Delta*a_INS[:,k-1]
 
plt.plot(r_INS[0,:],r_INS[1,:],'m-', label='INS')
 
'''
    Il y a une grande déviation relativement 
    à l'échelle du plot(environs 2500).
'''

# La trajectoire estimé par le calcul d'erreur
delta_rk = np.zeros(rtrue.shape)
delta_vk = np.zeros(vtrue.shape)
 
for k in range(1,nmax):
    w_INS = rnd.normal(loc=0,scale=sigma_INS,size=2)
    delta_rk[:,k] = delta_rk[:,k-1]-Delta*delta_vk[:,k-1]
    delta_vk[:,k] = delta_vk[:,k-1]-Delta*w_INS
 
plt.plot(r_INS[0,:]+delta_rk[0,:],r_INS[1,:]+delta_rk[1,:],'g-',label='INS_c')
 
plt.legend()
 
 
# Question 4 : 
rtrue_map = np.zeros(rtrue.shape[1])
N = len(rtrue_map)
N2, N1 = np.shape(map)
for k in range(N): 
    i= int( (X2MAX - rtrue[1,k]) * N2 /(X2MAX-X2MIN) ) 
    j= int( (rtrue[0,k] - X1MAX) * N1 /(X1MAX-X1MIN) )
    rtrue_map[k] = map[i,j]
plt.show()
plt.plot(range(N),rtrue_map,'b-',label='profil réel')
#plt.show()
 
 
#Question 5 : 
h_ALT = io.loadmat('alt.mat')['h_ALT'][0]
plt.plot(range(N),h_ALT,'r+',label='mesures altimétriques')
plt.legend()
plt.show()
 
#Question 6 : vraisemblance 
def vraisemblance(h,mu): 
    sigma = sqrt( sigma_ALT**2 + sigma_BAR**2) # the std of the Normal law 
    return np.exp(-0.5*(h-mu)**2 / sigma**2)
 
# Calcul des erreurs pour differents N.
L2NormSIR=[]
ListN=np.arange(50,10000,500)
for l in ListN:
    RV, weights,X_i,h = Adaptive(c=1, T=T, N=l)
    y_hat=[]
    for k in range(T):
        y_hat.append([np.sum(weights[k,:]*RV[k,:,0]),np.sum(weights[k,:]*RV[k,:,1])])
    y=[]
    for k in range(T):
        y.append([rtrue[0,k],rtrue[1,k]])
    L2NormSIR.append(np.linalg.norm(np.subtract(y,y_hat),2))
L2NormSIS=[]
ListN=np.arange(50,10000,500)
for l in ListN:
    RV, weights,X_i,h = Adaptive(c=0, T=T, N=l)
    y_hat=[]
    for k in range(T):
        y_hat.append([np.sum(weights[k,:]*RV[k,:,0]),np.sum(weights[k,:]*RV[k,:,1])])
    y=[]
    for k in range(T):
        y.append([rtrue[0,k],rtrue[1,k]])
    L2NormSIS.append(np.linalg.norm(np.subtract(y,y_hat),2))

# MSE for diffrent c 
c=np.linspace(0,1,num=10)
MSE = []
for i in range(10):
    RV, weights,X_i,h = Adaptive(c=c[i], T=T, N=1000)
    MSE_inter= np.zeros(shape=(T+1,))
    for i in range (T) : 
        MSE_inter[k] = (rtrue[0,k]-np.sum(weights[k,:]*RV[k,:,0]))**2 + (rtrue[1,k]-np.sum(weights[k,:]*RV[k,:,1]))**2
    MSE.append(np.mean(MSE_inter))
plt.plot(c,MSE)
plt.show()