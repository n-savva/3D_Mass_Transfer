# -*- coding: utf-8 -*-
"""
Data generation for figure 03
"""
import sys
sys.path.append("../main/")
import numpy as np
from ODEdrop import *
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Colormap
cmap = LinearSegmentedColormap.from_list('custom greys', [(1,1,1),(0.7,0.7,0.7)], N=256)

# Heterogeneity function
het_data = loadmat("Figure03_het.mat")
het_X = het_data["X"].flatten()
het_Y = het_data["Y"].flatten()
het_g = het_data["F"].flatten()
ix = (np.abs(het_X)<=5) & (np.abs(het_Y)<=5)
het_X = het_X[ix]
het_Y = het_Y[ix]
het_g = het_g[ix]

# Compute the triangulation of the heterogeneity
tri = Delaunay(list(zip(het_X, het_Y)))  

# Perform the interpolation with the given values
ginterp = LinearNDInterpolator(tri, het_g)
g = lambda x,y: ginterp(x,y)

# Volume variations
def V(t):
    return Vperiodic(t,p=Period[ax],Vamp=1.5*np.pi)
    
# SHADE PROFILE
Npt = 150
xf,yf = np.meshgrid(np.linspace(-4.5,4.5,Npt),np.linspace(-4.5,4.5,Npt))
Gf = g(xf,yf)


fig = plt.figure(figsize=(9,6),tight_layout=True)
gs = GridSpec(2, 3, figure=fig)


axL = (fig.add_subplot(gs[0, 0],adjustable='box',aspect=1), \
         fig.add_subplot(gs[1, 0],adjustable='box',aspect=1))
axR = (fig.add_subplot(gs[0,1:]),fig.add_subplot(gs[1,1:]))

PDEfiles = ["Figure03_PDE_a","Figure03_PDE_b"]
Period = [200,600]
for ax in [0,1]:
    axL[ax].pcolor(xf[0],yf[:,0],Gf,shading='auto',cmap=cmap)

    # Load PDE data
    n = 100
    PDEdata = loadmat(PDEfiles[ax])
    u = 2*np.pi*np.arange(1,n+1)/n
    Xc = PDEdata["X"]
    Yc = PDEdata["Y"]
    CL = PDEdata["Bound"]
    tPDE=PDEdata["tPDE"]
    
    X = Xc[-1] + CL[-1,:]*np.cos(u)
    Y = Yc[-1] + CL[-1,:]*np.sin(u)
    axL[ax].fill(X,Y,edgecolor='k',fill=False,lw=0.5)
    axL[ax].plot(Xc,Yc,lw=0.7)
    axR[ax].plot(tPDE[::10],Xc[::10],label='$x_c$',lw=0.5)
    axR[ax].plot(tPDE[::10],Yc[::10],label='$y_c$',lw=0.5)
    axR[ax].legend()
    
    # Three cases investigated [bim,order,line style]
    #   Case 1: Bim, all terms, dashed
    #   Case 2: Reduced, all terms, dotted
    #   Case 3: Bim, Low-order, dot-dashed
    t_end = 3200
    tp = np.arange(t_end+1)
    cases = ([True,2,'--'],[False,2,':'],[True,0,'-.'])
    i = 0
    cas = ['black,','black!30,']
    style = ['dashed,line width=1pt','dotted,line width=1pt','dash pattern={on 4pt off 2pt on 1pt off 2pt},line width=1pt']
    for case in cases:
        
        drop = ODEdrop(het=g,t_end=t_end,V=V,bim=case[0],order=case[1])
        drop.solve()
    
        # Draw contact line
        X,Y = drop.getcl(t_end)
        axL[ax].fill(X,Y,ls=case[2],fill=False)
        Xc,Yc,__ = drop.getcl(tp,coord='polar')
    
        # Draw centroid
        axR[ax].set_prop_cycle(None)
        axR[ax].plot(tp,Xc,case[2])
        axR[ax].plot(tp,Yc,case[2])
        
    
        tikzplot(str(ax)+'_'+str(i)+'x.dat',tp[::10],Xc[::10],style=cas[0]+style[i])
        tikzplot(str(ax)+'_'+str(i)+'y.dat',tp[::10],Yc[::10],style=cas[1]+style[i])
        tikzplot(str(ax)+'_'+str(i)+'cl.dat',X,Y,style=style[i])
                 
        i+=1