# -*- coding: utf-8 -*-
"""
Data generation for figure 06
"""
import sys
sys.path.append("../main/")
import numpy as np
from ODEdrop import *
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


# Volume variations
def V(t):
    return Vperiodic(t,p=200,Vamp=np.pi)

# Heterogeneity profile
cmap = LinearSegmentedColormap.from_list('custom greys', [(1,1,1),(0.5,0.5,0.5)], N=256)
g = lambda x,y: 1-0.15*(np.cos(2*np.pi*(x+y)) + np.cos(2*np.pi*(x-y)))
    
# Shaded profile
Npt = 150
xf,yf = np.meshgrid(np.linspace(-3,3,Npt),np.linspace(-3,3,Npt))
Gf = g(xf,yf)


fig = plt.figure(figsize=(9,6),tight_layout=True)
gs = GridSpec(2, 3, figure=fig)


axL = (fig.add_subplot(gs[0, 0],adjustable='box',aspect=1), \
         fig.add_subplot(gs[1, 0],adjustable='box',aspect=1))
axR = (fig.add_subplot(gs[0,1:]),fig.add_subplot(gs[1,1:]))

for ax in axR:
    ax.set_xlim([0,800])
    ax.set_ylim([-0.1,0.4])
    ax.set_xlabel('$t$')
axR[0].set_ylabel('$x_c$')
axR[1].set_ylabel('$y_c$')
    
PDEfiles = ["Figure06_PDE_a","Figure06_PDE_b"]
Period = [200,600]
Flux = [(0.75,0.25,1),(0,0,1)]
Colors = ['C0','C1']
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
    
    for Tp in [650,750]:
        X = Xc[Tp*10] + CL[Tp*10,:]*np.cos(u)
        Y = Yc[Tp*10] + CL[Tp*10,:]*np.sin(u)
        axL[ax].fill(X,Y,edgecolor='k',fill=False,lw=0.5)
    
    axL[ax].plot(Flux[ax][0],Flux[ax][1],'x',markersize=6)
    
    # Show centroid evolution
    axR[0].plot(tPDE[::10],Xc[::10],lw=0.5,color=Colors[ax],label='(%1.2f, %1.2f)' % (Flux[ax][:-1]))
    axR[1].plot(tPDE[::10],Yc[::10],lw=0.5,color=Colors[ax],label='(%1.2f, %1.2f)' % (Flux[ax][:-1]))
    
    
    tp = np.linspace(0,800,801)
    style = ['--',':']
    for case in [0,1]:
        drop = ODEdrop(n=150,het=g,V=V,t_end=800,bim=(case==0),flux=Flux[ax])
        drop.solve()
        
        # Draw contact line shapes
        plt.sca(axL[ax])
        drop.drawcl([650,750],style=style[case])
        
        # Get Centroid evolution
        Xc,Yc,__ = drop.getcl(tp,coord='polar')
        
        axR[0].plot(tp,Xc,style[case],color=Colors[ax])
        axR[1].plot(tp,Yc,style[case],color=Colors[ax])

    axR[0].legend()
    axR[1].legend()
plt.savefig('Figure06.png',bbox_inches='tight', dpi=200)