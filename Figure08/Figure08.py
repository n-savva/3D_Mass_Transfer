# -*- coding: utf-8 -*-
"""
Data generation for Figure 08
"""
import sys
sys.path.append("../main/")
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

pars = loadmat("../main/parameters.mat")
m = np.arange(1,513)
pars['beta_0'][0,0] = np.nan
pars['beta_m'][0,:2] = np.nan


for idx in [('beta','$\\beta$'),\
            ('gamma','$\\gamma$'),\
            ('beta_m','$\\beta_m^-$'),\
            ('beta_p','$\\beta_m^+$'),\
            ('beta_0','$\\beta_m^0$')]:
    plt.semilogx(m,pars[idx[0]][0],label=idx[1],base=2)

mm = np.array([1,512])
eulergamma = 0.57721566

# Asymptotics
plt.semilogx(mm,np.log(mm)-1+0.5*np.pi+eulergamma,'k--',label='asymptotics')
plt.legend()
plt.semilogx(mm,np.log(mm)-1+eulergamma+3*np.log(2),'k--')
plt.semilogx(mm,-(3*np.log(mm)-4.882058651),'k--')
plt.semilogx(mm,-2*(1.244330053+1.5*np.log(mm)),'k--')

# labels    
plt.xlabel('$m$')
plt.ylabel('$\\beta, \\gamma, \\beta_m^\\pm, \\beta_m^0$')
plt.savefig('Figure08.png', bbox_inches='tight',dpi=200)