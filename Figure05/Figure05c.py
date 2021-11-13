# -*- coding: utf-8 -*-
"""
Data generation for figure 05c
"""
import sys
sys.path.append("../main/")
from ODEdrop import *

# Heterogeneity
def g(x,y):
    return 1.2+a*(-np.tanh(50*(x+1.5))+np.tanh(50*(x-1.5))-np.tanh(50*(x-1.75)))

# Volume variations
def V(t):
    return ( np.pi*(1+2*np.tanh(2*np.pi*t/100)), np.pi**2/(25*(np.cosh(np.pi*t/50))**2) )

def events(t,U):
    m = 49
    a_hat = U[:m+1].astype(np.complex128)
    a_hat[1:m+1] -= 1j*U[m+1:-2]
    return 1.75 - max(U[-2] +  ifs(a_hat)*cos_u)


Ad = [0.25,0.265]
Xds = [np.hstack((0.3288,0.33,0.335,0.34,np.linspace(0.35,0.75,num=21))),\
      np.hstack((0.6299034,0.63,0.6315,0.632,0.6325,0.6330,0.6335,0.64,np.linspace(0.65,0.75,num=6)))]

Tds = [[],[]]                
for i in [0,1]:
    a = Ad[i]
    print('\nWorking with a = %1.2f' % a)
    
    for Xd in Xds[i]:
        print('\nXd = %1.4f' % Xd)               
        drop = ODEdrop(ic=1, t_end=40, V=V, flux=(Xd,0,1),het=g,method='LSODA',events=events)
        cos_u = np.cos(drop.u)
        drop.solve()
        
        if drop.solution.t_events[0].size>0:
            Tds[i].append(drop.solution.t_events[0][0])
        else:
            Tds[i].append(np.nan)
                          
# Plot    
plt.plot(Xds[0],Tds[0],label='0.25')
plt.plot(Xds[1],Tds[1],label='0.265')
plt.ylim([0,30])
plt.xlim([0,0.75])
plt.ylabel('$t_b$')
plt.xlabel('$x_0$')
plt.legend()

# Export
plt.savefig('Figure05c.png', bbox_inches='tight', dpi=200)
