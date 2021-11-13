# -*- coding: utf-8 -*-
"""
Data generation for figure 02
"""
import sys
sys.path.append("../main/")
import numpy as np
from ODEdrop import *
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from matplotlib.gridspec import GridSpec

# Linear Mass loss
def V(t):
    return (  np.pi*(2-t*1e-3), -1e-3*np.pi)

# HETEROGENEITY FUNCTION
het_data = loadmat("Figure02_het.mat")
het_X = het_data["X"].flatten()
het_Y = het_data["Y"].flatten()
het_g = het_data["F"].flatten()
ix = (np.abs(het_X)<=2) & (np.abs(het_Y)<=2)
het_X = het_X[ix]
het_Y = het_Y[ix]
het_g = het_g[ix]

# Compute the triangulation
tri = Delaunay(list(zip(het_X, het_Y)))  

# Perform the interpolation with the given values
ginterp = LinearNDInterpolator(tri, het_g)
g = lambda x,y: ginterp(x,y)

# SHADE PROFILE
Npt = 150
xf,yf = np.meshgrid(np.linspace(-2,2,Npt),np.linspace(-2,2,Npt))
Gf = g(xf,yf)

# LOAD PDE DATA
PDEdata = loadmat("Figure02_PDE.mat")

# Azimuthal points for PDE
n = 100
Tplotab = np.array([10,200,400,600,800,1000,1200,1400,1600,1800,1990]) 
u = 2*np.pi*np.arange(1,n+1)/n
Xc = PDEdata["X"]
Yc = PDEdata["Y"]
CL = PDEdata["Bound"]
tPDE=PDEdata["tPDE"]

fig = plt.figure(figsize=(9,6),tight_layout=True)
gs = GridSpec(3, 2, figure=fig)
axtop = (fig.add_subplot(gs[0:2, 0],adjustable='box',aspect=1), \
         fig.add_subplot(gs[0:2, 1],adjustable='box',aspect=1))
axbot = (fig.add_subplot(gs[2, 0]),fig.add_subplot(gs[2, 1]))

for ax in [0,1]:
    # DRAW SURFACE
    axtop[ax].pcolor(xf[0],yf[:,0],Gf,shading='auto',cmap='Greys')
    
    # MAKE LEGENDS
    ls = ['r--','b:']
    lab = ['Hybrid','Reduced']
    axtop[ax].plot([],'k',lw=0.5,label='PDE')
    axtop[ax].plot([],ls[ax],label=lab[ax])
    axtop[ax].legend()
    
    # SOLVE ODE SYSTEM
    drop = ODEdrop(n=100,het=g,t_end=1999,V=V,method='LSODA',bim=(ax==0))
    drop.solve()
    
    # CONTACT LINE PROFILES - SUBPLOTS (a) and (b)
    for i in Tplotab:
        # PLOT PDE SOLUTION
        X = Xc[10*i] + CL[10*i,:]*np.cos(u)
        Y = Yc[10*i] + CL[10*i,:]*np.sin(u)
        axtop[ax].fill(X,Y,edgecolor='k',fill=False,lw=0.5)
        
        # PLOT ODE SOLUTION
        X,Y = drop.getcl(i)
        axtop[ax].fill(X,Y,ls=ls[ax][1:],edgecolor=ls[ax][0],fill=False)
    
    # SUBPLOT (c)
    scale1 = 1.7
    scale2 = 1.5
    Tplot = tPDE[:-10:10].flatten()
    Vol = V(Tplot)[0].flatten()
    Xode,Yode,Rode = drop.getcl(Tplot,coord='polar')
    if ax==0:
        axbot[0].plot([0,2000],[1,0],'k',label='$v$',lw=0.5)
        Rmean = np.mean(CL[:-10:10,:],axis=1)
        θmean = 4*Vol/(np.pi*Rmean**3)
        axbot[0].plot(Tplot,0.5*Rmean*θmean/scale2,label='$h_{max}$',lw=0.5)
        axbot[0].plot(Tplot,Rmean/scale1,label='$a_0$',lw=0.5)
        axbot[0].plot(Tplot,θmean/scale1,label='$\\bar{\\vartheta}$',lw=0.5)
        
        axbot[0].set_xlim([0,2000])
        axbot[0].set_ylim([0,1.2])
        axbot[0].set_xlabel('$t$')
        axbot[0].set_ylabel('Scaled units')
        axbot[0].legend()

    axbot[0].set_prop_cycle(None)
    Rmean = np.mean(Rode,axis=1)
    θmean = 4*Vol/(np.pi*Rmean**3)
    axbot[0].plot(Tplot,0.5*Rmean*θmean/scale2,ls[ax][1:],lw=1.5)
    axbot[0].plot(Tplot,Rmean/scale1,ls[ax][1:],lw=1.5)
    axbot[0].plot(Tplot,θmean/scale1,ls[ax][1:],lw=1.5)
    
    
    # SUBPLOT (d)
    if ax==0:
        axbot[1].plot(Tplot,Xc[:-10:10],label='$x_c$',lw=0.5)
        axbot[1].plot(Tplot,Yc[:-10:10],label='$y_c$',lw=0.5)
        axbot[1].set_xlim([0,2000])
        axbot[1].set_xlabel('$t$')
        axbot[1].set_ylabel('$x_c,y_c$')
        axbot[1].legend()
    
    axbot[1].set_prop_cycle(None)
    axbot[1].plot(Tplot,Xode,ls[ax][1:],lw=1.5)
    axbot[1].plot(Tplot,Yode,ls[ax][1:],lw=1.5)
        
plt.savefig('Figure02.png',bbox_inches='tight', dpi=200)