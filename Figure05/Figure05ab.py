# -*- coding: utf-8 -*-
"""
Data generation for figure 05
"""
import sys
sys.path.append("../main/")
from ODEdrop import *
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('custom greys', [(1,1,1),(0.5,0.5,0.5)], N=256)

# Heterogeneity
def g(x,y):
    return 1.2+0.25*(-np.tanh(50*(x+1.5))+np.tanh(50*(x-1.5))-np.tanh(50*(x-1.75)))

# Volume variations
def V(t):
    return ( np.pi*(1+2*np.tanh(2*np.pi*t/100)), np.pi**2/(25*(np.cosh(np.pi*t/50))**2) )

# Shade surface profile
Npt = 150
xf,yf = np.meshgrid(np.linspace(-4,4,Npt),np.linspace(-4,4,Npt))
Gf = g(xf,yf)
    

fig = plt.figure(figsize=(8,4),tight_layout=True)
gs = GridSpec(1, 2, figure=fig)

# Locations of delta functions
Xd = [0,0.75]

# Data files
PDEfiles = ["Figure05_PDE_a","Figure05_PDE_b"]

# Times at which profiles are plotted
Tp = [0,5,10,20,30,300]

for ax in [0,1]:
    fig.add_subplot(gs[ax],adjustable='box',aspect=1)
    plt.pcolor(xf[0],yf[:,0],Gf,shading='auto',cmap=cmap)
    
    # Legend
    plt.plot([],'k',lw=0.5,label='PDE')
    plt.plot([],'r--',label='Hybrid')
    plt.legend()
    
    # Plot contact line shapes from PDE
    n = 100
    PDEdata = loadmat(PDEfiles[ax])
    u = 2*np.pi*np.arange(1,n+1)/n
    Xc = PDEdata["X"]
    Yc = PDEdata["Y"]
    CL = PDEdata["Bound"]
    tPDE=PDEdata["tPDE"]
    
    for i in Tp:
        X = Xc[i*10] + CL[i*10,:]*np.cos(u)
        Y = Yc[i*10] + CL[i*10,:]*np.sin(u)
        plt.fill(X,Y,edgecolor='k',fill=False,lw=0.5)
    
    # Plot contact line shapes from BIM
    drop = ODEdrop(ic=1, t_end=300, V=V, flux=(Xd[ax],0,1),het=g,method='RK45')
    drop.solve()
    drop.drawcl(Tp,color='r',style='--')

    # Show location of delta function
    plt.plot(Xd[ax],0,'x',markersize=6)

# Export
plt.savefig('Figure05ab.png', bbox_inches='tight', dpi=200)
