"""
Figure 9 of JFM 2019 vs PF2021
"""
import sys
sys.path.append("../main/")
from ODEdrop import *

# DEFINE HETEROGENEITY
def g(x,y):
    m1 = 20
    m2 = 3
    F = 1+0.3*np.tanh(m2*np.cos(np.pi*x))*np.tanh(m2*np.sin(np.pi*y))

    return 2.2+(1-0.5*(1-np.tanh(m1*(y+1))*np.tanh(m1*(y-1)))*(1-np.tanh(m1*(x-1)))/2)*np.arctan2(y,x)/np.pi*F+(1-np.tanh(m1*(y+1))*np.tanh(m1*(y-1)))*(1-np.tanh(m1*(x-1)))/3

# DRAW SURFACE
xf,yf = np.meshgrid(np.linspace(-6,6,150),np.linspace(-6,6,150))
Gf = g(xf,yf)
plt.pcolor(xf[0],yf[:,0],Gf,shading='auto',cmap='Greys_r')
plt.axis('equal')

# LOAD & PLOT PDE DATA
PDEdata = loadmat("PDE.mat")
Tplot = np.array([1,38,75,118,175,400]) 
u = np.ravel(PDEdata["u"])
for i in Tplot*10:
    X = PDEdata["U_PDE"][i,-2] + PDEdata["U_PDE"][i,:-2]*np.cos(u)
    Y = PDEdata["U_PDE"][i,-1] + PDEdata["U_PDE"][i,:-2]*np.sin(u)
    plt.fill(X,Y,edgecolor='y',fill=False,lw=2)

# PREPARE LEGEND
plt.plot([],label='PDE',color='y')
col = ['b','g','r']

for i in range(3):
    plt.plot([],label='order = %2d' % i,color=col[i])
plt.legend()
    
# LOOP THROUGH DIFFERENT ORDERS OF APPROXIMATION
for order in range(3):
    drop = ODEdrop(het=g,t_end=400,order=order,Xc=-3,Yc=2.5,method='LSODA')
    drop.solve()
    drop.drawcl(Tplot,color=col[order])
    
# EXPORT FIGURE    
plt.axis('equal')
plt.savefig('Figure9_JFM2019.png',bbox_inches='tight', dpi=200)