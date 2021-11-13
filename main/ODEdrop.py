# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.integrate import solve_ivp
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.animation as animation
import os


# Complex Fourier Series
def fs(F):
    n = int(len(F)/2)
    F = np.fft.rfft(F)
    F /= n
    F[0]*= 0.5
    return F[:-1]  # Cutoff frequency removed

# Inverse Complex Fourier Series
def ifs(F):
    m = len(F)
    G = m*np.concatenate((F,[0.0]))
    G[0] *= 2
    return np.fft.irfft(G)

# Fourier Interpolation
def fourmat(n,y):
    x = 2*np.pi*np.arange(n)/n
    w = (-1.)**np.arange(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = 1/np.tan(0.5*(x-y))
        P = w*P
        P = P/np.sum(P)
        P[np.isnan(P)] = 1
    return P

# Matrix for barycentric interpolation
def barymat(w,x,X):
    P = X - x
    with np.errstate(divide='ignore',invalid='ignore'):
        P = w/P
        P = P/np.sum(P)
        P[np.isnan(P)] = 1
    return P


# Periodic volume variations
def Vperiodic(t,m=20,p=100,Vavg=2*np.pi,Vamp=np.pi):
    a = 2*np.pi/p
    A = np.sqrt(1+(m*np.cos(a*t))**2)
    V = Vavg + Vamp*np.arctan(m*np.sin(a*t)/A)/np.arctan(m)
    Vdot = Vamp*a*m*np.cos(a*t)/A/np.arctan(m)
    return V,Vdot


class ODEdrop(object):
    """
    A class for a drop object.
    
    Attributes
    ----------
    slip: float  [1e-3]
        Slip length.
    V: float [2*np.pi]
        Droplet volume. It can be a constant or a function of time.
    n: int [100]
        Number of discrete points for solving. It must be even.
    het: float or callable [1.0]
        The heterogeneity profile.
    ic: float or callable [1.0]
        The initial condition for the radius.
    t_end: float [None]
        The final time.
    Xc,Yc: float [0.0]
        The initial coordinates of the centroid.
    order:  integer [2]
        An integer specifying the order of approximation
        0: Leading-order expression for the normal speed
        1: Correction terms included in JFM2019
        2: Correction terms included in PRF2021
    flux: list of tuples [None]
        Specifies whether a delta-function flux is applied. Each tuple consists of three elements (x,y,s), where (x,y) are the x and y coordinates of the flux position and s is the strength. Avoid placing the flux too near to the contact line.
    bim: bool [True]
        Do not change at present. For future development.
    method: str ['RK45']
        The method to be used with solve_ivp. If it is slow use 'LSODA'.
    φ,u: array_like
        Polar angle.
    soltion: OdeSolution
        OdeSolution instance containing the solution to the equations; See 
        documentation for solve_ivp.
    events: callable function [None]
        Events functionality passed to solve_ivp
    
    
    Methods
    -------
    ic(ic), het(het), V(vol) 
        Sets the attributes for ic het and V
    
    solve()
        Computes the solution if t_end is specified. Otherwise it returns the
        angle.
        
    drawcl(T,color='b',style='-')    
        Draws contact line shapes for given values of time in T.

        Parameters
        ----------
        T : array_like
            The times at which the shapes are to be plotted.
        color: string, optional
            The color of the plot. The default is 'b'.
        style: string, optional
            The line style for the plot. The default is '-'
        
        Raises
        ------
        Exception
            - If t_end is undefined.
            - If solution has not yet been computed.
            - If some element of T lies outside the time range.

        Returns
        -------
        None

    
    angle(t)    
        Returns an array with the angle at a specified time, t.

        Parameters
        ----------
        t : float 
           The time for which the angle is to be returned. The default is None.

        Raises
        ------
        Exception
            - If t lies outside the solution range
            - If solution has yet to be computed
            - If t is not a scalar

        Returns
        -------
        angle: array_like
            The apparent contact angle. 

        
    getcl(t)
        Returns the X and Y coordinates of the contact line for prescribed 
        times.

        Parameters
        ----------
        t : float or array_like
            The times at which the coordinates are to be returned. An exception
            is thrown if some element of t lies outside the solution range.
        coord: string
            The coordinate system to output the contact line. It can be either 'cartesian' or 'polar'
            
        Returns
        -------
        X,Y : array_like
            Coordinates of the contact line shape if coord = 'cartesian'
            
        X,Y,R: array_like
            Centroid coordinates and radius of contact line if coord='polar


    resume(t)   
        Resumes a completed simulation.

        Parameters
        ----------
        t : 2-tuple of floats
            Interval of integration. The solver should start with the previous
            t_end. Otherwise an exception is thrown.

        Returns
        -------
        None.

    makegif(file=None,fps=5,duration=10)
        Creates a gif saved with the name file, for given fps and duration in
        seconds. An exception is thrown if solution has not been computed.

        Parameters
        ----------
        file : str
            The name of the file. The default is None.
        fps : float
            The number of frames per second. The default is 5.
        duration : flat
            The duration of the animation. The default is 10.

        Returns
        -------
        None.
    """
    def __init__(self, slip = 1e-3, V=2*np.pi, n = 100, het = 1., ic = 1.0,
                 order = 2, flux=None, t_end=None, bim=True, Xc=0., Yc=0.,
                 method='RK45',move_origin=True,events=None):
        if (n%2 != 0):
            raise Exception("An even number of points are required")

        # Discretization
        self.slip = slip
        self.n = n
        self.m = int(n/2-1)
        self.φ = 2*np.pi*np.arange(n)/n
        self.u = self.φ
        self.bim = bim
        self.order = order
        self.__logλ = np.log(slip)
        
        # Parse simulation parameters
        self.V = V
        self.flux = flux
        if isinstance(self.flux,tuple):
                self.flux = [self.flux]
        
        # Initial condition
        self.Xc = Xc
        self.Yc = Yc
        self.ic = ic
        
        # ODE integrator
        self.events = events
        self.t_end = t_end
        self.method = method
        self.solution = None
        self.move_origin = move_origin
    
        # Chemical Heterogeneity
        self.het = het
        
        # Define a set of private variables for BIM
        self.__mm = np.arange(1,self.m+1)
        self.__m0 = np.arange(self.m)
        if self.bim:
            self.__j = np.arange(n)
            self.__W = toeplitz(1.0/self.n*np.sum(np.cos((self.__j[:,None]*self.__mm)*np.pi/(self.m+1))/self.__mm,axis=1) \
                               + 0.25*(-1.)**self.__j/(self.m+1)**2)
            self.__δφ = self.φ - self.φ[:,None]
            self.__sin_δφ = np.sin(self.__δφ)
            self.__cos_δφ = np.cos(self.__δφ)
            self.__sin2_δφ = np.sin(.5*self.__δφ)**2
            self.__k = np.fft.rfftfreq(self.n,d=1/self.n)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                self.__log_4_sin2_δφ = np.log(4*self.__sin2_δφ)
        
        self.__cosφ = np.cos(self.φ)
        self.__sinφ = np.sin(self.φ)
        
        # Simulation parameters using the Low-order model
        self.__β0 = np.full(self.m,-self.__logλ)
        self.__βm = np.full(self.m,-self.__logλ)
        self.__βp = np.full(self.m,-self.__logλ)
        self.__βt = np.zeros(self.m)
        self.__β00 = - self.__logλ 
   
        # Augment parameters following JFM2019 
        if self.order > 0:
            pars = loadmat(os.path.join(os.path.dirname(__file__),"parameters.mat"))
            self.__β0 += 1 - pars["beta"][0,:self.m]
            self.__βm += 1 - pars["gamma"][0,:self.m]
            self.__βp += 1 - 2*pars["beta"][0,:self.m] + pars["gamma"][0,:self.m]
            self.__β00 -= 1 + np.log(2)
        
        # Augment parameters following PoF2021
        if self.order > 1:
            self.__βm += pars["beta_m"][0,:self.m]
            self.__βp -= pars["beta_p"][0,:self.m]
            self.__βt = pars["beta_0"][0,:self.m]

        # Flux functions
        if self.flux is not None:
             pars = loadmat(os.path.join(os.path.dirname(__file__),"hypergeom.mat"))
             self.__Xq = 2*pars["x"][0]-1
             self.__Wq = pars["w"][0]
             self.__Iq = pars["I"][0,:self.m]
             self.__Fq = pars["F"][:self.m,:]
             self.__sqxq = pars["sqx"][0]
             self.__Wbaryq = pars["wbary"][0]
                         
    # Volume property
    @property
    def V(self):
        return self._V
    
    @V.setter
    def V(self,value):
        if not callable(value):
            self._V = lambda t: (value,0)
        else:
            self._V = value 
                        
            
    # Intial Condition
    @property
    def ic(self):
        return self._ic
    
    @ic.setter
    def ic(self,value):
        if not callable(value):
            self._ic = np.full(self.n,value,dtype='float64')
        else:
            self._ic = value(self.φ)
        self.solution = None
   
    # Heterogeneity profile
    @property
    def het(self):
        return self._g
    
    @het.setter
    def het(self,value):
        if not callable(value):
            self._g = lambda x,y: np.full(x.shape,value,dtype='float64')
        else:
            self._g = value
        self.solution = None
   
    # Evaluate Radius via BIM
    def __angle(self,Vo,Ro):
        # Scale data to avoid the Γ-contour
        self.__scale = 0.8/np.max(Ro)
        R = Ro*self.__scale
        V = Vo*self.__scale**3
        
        # Derivatives
        self.__Rhat = np.fft.rfft(R)
        self.__Rhat[-1] = 0
        self.__Ru = np.fft.irfft(1j*self.__k*self.__Rhat)
        self.__Ruu = np.fft.irfft(-self.__k**2*self.__Rhat)
        
        # Other variables
        self.__D = R**2 + self.__Ru**2
        self.__sqD = np.sqrt(self.__D)
        self.__RRo = R*R[:,None]
        self.__x_xo = (R-R[:,None])**2 + 4*self.__RRo*self.__sin2_δφ
        self.__x_dot_n = R**2/self.__sqD
        
        # Curvature (times sqD)
        self.__K = (R**2 + 2*self.__Ru**2 - R*self.__Ruu)/self.__D
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normal derivative of Green's function
            #   to obtain Gn multiply by  n/(0.5*π*sqD)
            self.__Gn = - (0.25/self.n)*(R**2 - self.__RRo*self.__cos_δφ \
                        - (R[:,None]*self.__Ru)*self.__sin_δφ)/self.__x_xo        
            np.fill_diagonal(self.__Gn,-(0.125/self.n)*self.__K)
            
            # Green's function
            #    2*π/m * (G - 0.5*log(4*sin(δφ)^2))
            self.__Gm = -0.5*(np.log(self.__x_xo)-self.__log_4_sin2_δφ)/self.n
            np.fill_diagonal(self.__Gm,-0.5*np.log(self.__D)/self.n)

        # Solve and determine the local angle
        self.__Wn = np.linalg.solve((self.__Gm + self.__W)*self.__sqD,0.125*R**2 + self.__Gn@(R**2))
        self.__kk = 4*V*self.n/(2*np.pi*np.sum((0.25*self.__x_dot_n-self.__Wn)*R**2*self.__sqD))
        
        return self.__kk*(0.5*self.__x_dot_n-self.__Wn)
    
    # Return Radius from fourier harmonics
    def __radius(self,Uo):
        a_hat = Uo[:self.m+1].astype(np.complex128)
        a_hat[1:self.m+1] -= 1j*Uo[self.m+1:]
        return ifs(a_hat)
        
    # ODE
    def __ode(self,t,U,pbar,state):
        dU = np.zeros(2*self.m+3)
        
        # Centroid and harmonics
        Xc,Yc = U[-2], U[-1]
        bm = (U[1:self.m+1] -1j*U[self.m+1:-2])/U[0]
        V,Vdot = self.V(t)
        
        # Contact line radius
        a = self.__radius(U[:-2])
        
        # Local contact angle
        θs = self._g(Xc + a*self.__cosφ, Yc + a*self.__sinφ)   
        θs3_hat = fs(θs**3)

        # Compute ψ
        if self.order == 1:
            self.__ψ[0]  = np.log(U[0]*np.mean(θs))
        elif self.order==2:
            self.__ψ  = fs(np.log(a*θs))
           
        # Apparent contact angle
        if self.bim:
            θ = self.__angle(V,a)
            θ3_hat = fs(θ**3)
        else:
            θo = 4*V/(np.pi*U[0]**3)
            θm = - bm*self.__m0
            θ = θo*ifs(np.concatenate(([1],θm)))
            θ3_hat = θo**3*np.concatenate(([1],3*θm))
        
        # Compute mass fluxes
        I = 0
        if self.flux is not None:
            I = np.zeros(self.m+1,dtype=np.cdouble)
            
            for delta in self.flux:
                Xo = delta[0]
                Yo = delta[1]
                So = delta[2]
                
                # Radial distance of delta function
                Rd = np.sqrt((Xo-Xc)**2 + (Yo-Yc)**2)
                φd = np.arctan2(Yo-Yc, Xo-Xc)
                fo = So*U[0]/V
                
                I[0] -= 0.25*fo
                I[1:] -= fo*bm*self.__Iq
            
                if Rd!=0:
                    # Interpolation at φd
                    M_δ = fourmat(self.n,φd)
                    φ_δ = np.dot(M_δ,θ)
                    a_δ = np.dot(M_δ,a)
                
                    # Additional contributions
                    ro = Rd/a_δ
                    f = a_δ**2*φ_δ*(1-ro**2)*np.pi
                
                    # Interpolate Hypergeometric
                    Io = (self.__Fq@barymat(self.__Wbaryq,self.__Xq,2*ro-1))/f
                    
                    I[0] += So*ro**2/f
                    I[1:]+= So*Io*np.exp(-1j*φd*self.__mm)
                 
                I[2:] -=1.5*fo*(2*self.__mm[1:]*I[0]*bm[1:]
                        + I[1]*(self.__mm[:-1]-1)*bm[:-1]
                        + np.append(np.conj(I[1])*(self.__mm[1:-1]+1)*bm[2:],[0]))       
               
        # Normal speed
        w = (θ3_hat - θs3_hat)/3. + Vdot*I
        
        # Axisymmetric component
        ψo = np.real(self.__ψ[0])
        Uo= np.real(w[0])/(ψo + self.__β00)
        
        # Miscellaneous variables
        ψ0plusβ0 = self.__β0 + ψo
        w[1:] -= Uo*(self.__mm*bm*self.__βt + self.__ψ[1:])
        w[1:] /= ψ0plusβ0
        
        # Centroid
        gg = (ψo + self.__βp[0])/ψ0plusβ0[0]
        Uc = np.linalg.solve(np.array([[1,-gg*bm[1]],[-gg*np.conj(bm[1]),1]],dtype=np.cdouble),
                             np.array([w[1],np.conj(w[1])],dtype=np.cdouble))
        
        # Correction terms for m>1
        dUh = w[2:] + 0.5*(Uc[0]*((1-self.__mm[1:])*(ψo + self.__βm[1:])*bm[:-1] - self.__ψ[1:-1])
                           + np.append(Uc[1]*((1+self.__mm[1:-1])*(ψo + self.__βp[1:-1])*bm[2:] - self.__ψ[3:]),[0]))/ψ0plusβ0[1:]
        
        # Assign outputs
        dU[0] = Uo
        dU[2:self.m+1] = np.real(dUh)
        dU[self.m+2:-2] =  - np.imag(dUh) 
        if self.move_origin:
            dU[-2] = np.real(Uc[0]) 
            dU[-1] = - np.imag(Uc[0])
        else:
            dU[1] = np.real(Uc[0]) 
            dU[self.m+1] = - np.imag(Uc[0])
        
            
       
        return dU
        
    def eventAttr():
        def decorator(func):
            func.direction = 0
            func.terminal = True
            return func
        return decorator

    # Progress Bar Display
    @eventAttr()    
    def __pbar(self,t,y,pbar,state):
        last_t, dt = state
        n = int((t-last_t)/dt)
        pbar.update(n)
        pbar.set_description("t = %1.2e" % t)
        state[0] = last_t + dt*n
        
        if self.events is not None:
            res = self.events(t,y)
            
            if res==0:
                pbar.set_description("Event triggered at t = %1.2e" % t)
            else:
                return res
        return 1   
        
    # Solve Method
    def solve(self):
        """
        Solves the system.

        Returns
        -------
        angle : array_like, shape (m,)
            If t_end is not prescribed, it returns the apparent angle.

        """
        if self.t_end is None:
            return self.__angle(self.V(0),self._ic)
        else:
            # Prepare the initial condition
            r_hat = fs(self._ic)
            
            self.__ψ = np.zeros(self.m+1)
            IC = np.zeros(2*self.m+3)
            IC[0] = np.real(r_hat[0])
            IC[1:self.m+1] = np.real(r_hat[1:]) 
            IC[self.m+1:-2] = - np.imag(r_hat[1:])
            IC[-2] = self.Xc
            IC[-1] = self.Yc
        
            print("\nSolving until t = %1.2f\n" % (self.t_end),end='',flush=True)
            with tqdm(total=100,unit="%") as pbar:
                self.solution = solve_ivp(self.__ode, (0,self.t_end), IC, 
                                          method=self.method,
                                          dense_output=True,
                                          events=self.__pbar,
                                          atol=1e-8,rtol=1e-8,
                                          args=[pbar,[0,self.t_end/100]])
                
    # Draw the contact line
    def drawcl(self,T,color='b',style='-'):
        """
        Draws contact line shapes for given values of time in T.

        Parameters
        ----------
        T : array_like
            The times at which the shapes are to be plotted.

        Raises
        ------
        Exception
            - If t_end is undefined.
            - If solution has not yet been computed.
            - If some element of T lies outside the time range.

        Returns
        -------
        None.

        """
        if self.t_end is None:
            raise Exception("Undefined t_end")
            
        if self.solution is None:
            raise Exception("No solution found. Run solve()")
        
        if not isinstance(T, (list, tuple, np.ndarray)):
                T = [T]    
                
        for t in T:
            if t<0 or t>self.t_end:
                raise Exception('Time out of range')
                
            U = self.solution.sol(t)
            Xc, Yc = U[-2], U[-1]
            a = self.__radius(U[:-2])
            plt.fill(Xc + a*self.__cosφ,Yc + a*self.__sinφ,ls=style,edgecolor=color,fill=False) 
                
    # Method for returning the angle for scalar t
    def angle(self,t=None):
        """
        Returns an array with the angle at a specified time, t.

        Parameters
        ----------
        t : float 
           The time for which the angle is to be returned. The default is None.

        Raises
        ------
        Exception
            - If t lies outside the solution range
            - If solution has yet to be computed
            - If t is not a scalar

        Returns
        -------
        angle: array_like
            The apparent contact angle. 

        """
        if t is None:
            return self.__angle(self.V(0),self._ic)
        elif self.solution is not None and not isinstance(t, (list, tuple, np.ndarray)):
            if t<0 or t>self.t_end:
                raise Exception('Time out of range')
            U = self.solution.sol(t)
            return self.__angle(self.V(t),self.__radius(U[:-2]))
        else:
            raise Exception("Either no solution found, or t is not given as a scalar")

    # Continue Solution
    def resume(self,t):
        """
        Continues a completed simulation.

        Parameters
        ----------
        t : float
            New ending time. The solver will start from the previous
            t_end. Otherwise an exception is thrown.

        Returns
        -------
        None.

        """
        if self.solution is None:
            raise Exception("No solution to continue")
            
        if t <= self.t_end:
            raise Exception("Cannot continue solution")
        
        print("\nSolving between t = %1.2f and %1.2f:" % (self.t_end,t),
              end='',flush=True)
        with tqdm(total=100,unit="%") as pbar:
            IC =  self.solution.sol(self.t_end)
            newsol = solve_ivp(self.__ode,(self.t_end,t), IC,
                                 method=self.method, 
                                 dense_output=True,events=self.__pbar,
                                 atol=1e-6,rtol=1e-6,
                                 args=[pbar,[self.t_end,(t-self.t_end)/100]])
        
        # Finalize Solution    
        self.solution.y = np.hstack((self.solution.y,newsol.y[:,1:]))
        self.solution.t = np.hstack((self.solution.t,newsol.t[1:]))
        self.solution.sol.ts = np.hstack((self.solution.sol.ts,newsol.sol.ts[1:]))
        self.solution.sol.interpolants = self.solution.sol.interpolants + newsol.sol.interpolants
        self.solution.sol.n_segments = len(self.solution.sol.interpolants)
        self.solution.sol.ts_sorted = self.solution.t
        self.solution.sol.t_max = newsol.sol.t_max
        self.t_end = newsol.sol.t_max   
        

    # Method for returning the X and Y coordinates of the contact line
    def getcl(self,t=None,coord='cartesian'):
        """
        Returns the X and Y coordinates of the contact line for prescribed 
        times.

        Parameters
        ----------
        t : float or array_like
            The times at which the coordinates are to be returned. An exception
            is thrown if some element of t lies outside the solution range.
        coord: string
            The coordinate system to output the contact line. It can be either 'cartesian' or 'polar'
            
        Returns
        -------
        X,Y : array_like
            Coordinates of the contact line shape if coord = 'cartesian'
            
        X,Y,R: array_like
            Centroid coordinates and radius of contact line if coord='polar'

        """
        if t is not None and self.solution is not None:
            if not isinstance(t, (list, tuple, np.ndarray)):
                t = [t]
            L = len(t)
            
            if coord=='cartesian':
                X = np.zeros((L,self.n))
                Y = np.zeros((L,self.n))
            
                for i in range(L):
                    if t[i]<0 or t[i]>self.t_end:
                        raise Exception('Time out of range')
                    U = self.solution.sol(t[i])
                    a = self.__radius(U[:-2])
                    X[i] = U[-2] + a*self.__cosφ
                    Y[i] = U[-1] + a*self.__sinφ
                     
                if L==1:
                    X, Y = X[0], Y[0]
    
                return X, Y
            elif coord=='polar':
                X = np.zeros((L,))
                Y = np.zeros((L,))
                R = np.zeros((L,self.n))
            
                for i in range(L):
                    if t[i]<0 or t[i]>self.t_end:
                        raise Exception('Time out of range')
                    U = self.solution.sol(t[i])
                    a = self.__radius(U[:-2])
                    X[i] = U[-2] 
                    Y[i] = U[-1]
                    R[i] = a
                
                if L==1:
                    X, Y, R = X[0], Y[0], R[0]
    
                return X, Y, R
            else:
                raise Exception('Incorrect Coordinate System')



    # Export to GIF            
    def makegif(self,file=None,fps=5,duration=10):
        """
        Creates a gif saved with the name file, for given fps and duration in
        seconds. An exception is thrown if solution has not been computed.

        Parameters
        ----------
        file : str
            The name of the file. The default is None.
        fps : float
            The number of frames per second. The default is 5.
        duration : flat
            The duration of the animation. The default is 10.

        Returns
        -------
        None.

        """
        def animate(i,pbar):
            plt.clf()
            plt.ylim(Ymin,Ymax)
            plt.xlim(Xmin,Xmax)
            plt.axis('equal')
            plt.pcolor(xf[0],yf[:,0],Gf,shading='auto',cmap='Greys')
            plt.fill(X[i],Y[i],label=i,facecolor=(0.45,0.7,1,0.5),edgecolor='b',lw=1.5)
            plt.title('t = %1.2f' % t[i])
            pbar.update(1)
            
        if self.solution is not None:
            if file is None:
                raise Exception("No filename specified")
            t = np.linspace(0,self.t_end,num=fps*duration+1)
            X,Y = self.getcl(t)
            Xmin = np.floor(X.min()-0.5)
            Xmax = np.ceil(X.max()+0.5)
            
            Ymin = np.floor(Y.min()-0.5)
            Ymax = np.ceil(Y.max()+0.5)
            
            xf,yf = np.meshgrid(np.linspace(Xmin,Xmax,150),np.linspace(Ymin,Ymax,150))
            Gf = self._g(xf,yf)
            fig = plt.figure(figsize=(5,5))
            Nframes = fps*duration+1
            
            print("\nGenerating frames for gif:", end='',flush=True)
            with tqdm(total=Nframes+1,unit="frames") as pbar:
                anim = animation.FuncAnimation(fig, animate, frames = Nframes,fargs=[pbar])
                writer = animation.PillowWriter(fps=fps)
                anim.save(file.replace('.gif','')+'.gif', writer=writer)   
        else:
            raise Exception("No solution computed yet.")