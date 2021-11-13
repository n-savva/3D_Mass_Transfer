# -*- coding: utf-8 -*-
"""
Data generation for figure 04
"""
import sys
sys.path.append("../main/")
from ODEdrop import *

# Volume variations
def V(t):
    T = 30
    return ( np.pi*(2+np.tanh(t/T)) , np.pi/(T*(np.cosh(t/T))**2) )
    
# load PDE data
n = 100
PDEdata = loadmat('Figure04_PDE.mat')
u = 2*np.pi*np.arange(1,n+1)/n
Xc = PDEdata["X"]
Yc = PDEdata["Y"]
CL = PDEdata["Bound"]
tPDE=PDEdata["tPDE"]

# Times to plot
Tp = [0, 2, 8, 16, 32, 60]

# Show sources and sinks
plt.plot([1.8,0,-1],[0,1.8,-1],'x',markersize=6)

# Legend
plt.plot([],'k',lw=0.5,label='PDE')
plt.plot([],'r--',label='Hybrid')
plt.legend()

for i in Tp:
    X = Xc[i*10] + CL[i*10,:]*np.cos(u)
    Y = Yc[i*10] + CL[i*10,:]*np.sin(u)
    plt.fill(X,Y,'k',lw=0.5,fill=None)
    
# Solve the reduced system
flux = [(1.8,0,1),(0,1.8,1),(-1,-1,-1)]
drop = ODEdrop(ic=2, t_end=60, V=V, flux=flux)
drop.solve()
drop.drawcl(Tp,color='r',style='--')
plt.axis('equal')

# Export
plt.savefig('Figure04.png', bbox_inches='tight', dpi=200)