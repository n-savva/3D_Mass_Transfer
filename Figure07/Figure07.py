# -*- coding: utf-8 -*-
"""
Data generation for figure 07
"""
import sys
sys.path.append("../main/")
import os
from ODEdrop import *
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Colormap
cmap = LinearSegmentedColormap.from_list('custom greys', [(1,1,1),(0.7,0.7,0.7)], N=256)

# Volume variations
def V(t):
    return Vperiodic(t,p=200,Vamp=1.25*np.pi)

# Heterogeneity
def g(x,y):
    Rxy = np.sqrt((x-het_X)**2+(y-het_Y)**2)
    return 1.0 + np.sum(0.5*(np.tanh(200*Rxy + 10) - np.tanh(200*Rxy - 10)),axis=0)

het_data = loadmat("Figure07_het.mat")
het_X = het_data["Xd"]
het_Y = het_data["Yd"]

# Check if output data is there; otherwise compute solution
if not os.path.isfile('Figure07.npy'):
    # Solve the problem (takes some time)
    drop = ODEdrop(ic=1.8, n=512,t_end=800, V=V, flux=(0,0,1),het=g,method='RK45',bim=False)
    drop.solve()
    
    # Export data
    Tp = np.arange(drop.t_end+1)
    X,Y = drop.getcl(Tp)
    Xc, Yc, R = drop.getcl(Tp,coord='polar')
    
    with open('Figure07.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, Y)
        np.save(f, Xc)
        np.save(f, Yc)
        np.save(f, R)   
else:
    with open('Figure07.npy', 'rb') as f:
        X = np.load(f)
        Y = np.load(f)
        __ = np.load(f)
        __ = np.load(f)
        Ro = np.load(f)


Tp=np.arange(350,455,5)
t = np.arange(801)
R = np.mean(Ro,axis=1)

# Shade surface profile
Npt = 400
xf,yf = np.meshgrid(np.linspace(-4,4,Npt),np.linspace(-4,4,Npt))
Gf = g(xf.flatten(),yf.flatten()).reshape(Npt,Npt)

# Plot layout
fig = plt.figure(figsize=(8,10),tight_layout=True)
gs = GridSpec(6,4, figure=fig)


# Plot profiles during Injection
ax = fig.add_subplot(gs[:2,:2],adjustable='box',aspect=1)
ax.pcolor(xf[0],yf[:,0],Gf,shading='auto',cmap=cmap)
ax.plot(X[Tp].T,Y[Tp].T,'k',lw=0.7)
ax.set_ylim([-2.7,2.3])
ax.set_xlim([-2.5,2.5])
ax.set_title('Injection')

# Plot profiles during Withdrawal
ax = fig.add_subplot(gs[:2,2:4],adjustable='box',aspect=1)
ax.pcolor(xf[0],yf[:,0],Gf,shading='auto',cmap=cmap)
ax.plot(X[Tp+100].T,Y[Tp+100].T,'k',lw=0.7)
ax.set_ylim([-2.7,2.3])
ax.set_xlim([-2.5,2.5])
ax.set_title('Withdrawal')

# Plot of mean radius
ax = fig.add_subplot(gs[2,:])
ax.plot(t,R)
ax.set_xlim((0,800))
ax.set_ylabel("$a_0$")

# Plot of mean apparent angle
ax = fig.add_subplot(gs[3,:])
Vo = V(t)[0]
theta = 4*Vo/(np.pi*R**3)
ax.plot(t,theta)
ax.set_ylabel(r"$\bar\vartheta$")
ax.set_xlim((0,800))

# PLot of V(t)
ax = fig.add_subplot(gs[4,:])
ax.plot(t,Vo/np.pi)
ax.set_xlim((0,800))
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$V/\pi$")

# Export
plt.savefig('Figure07.png', bbox_inches='tight', dpi=200)

