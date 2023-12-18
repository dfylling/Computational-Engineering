# %%

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
from envelope import *


#%%
class  TwoPhase:
    """
    Class  used  to  model  the  displacement of two fluids 
    described through a non-linear conservation law.
    """
    def  __init__(self,
                    
                    length = 2, # length of domain
                    nx = 500, # number of cells
                    t_final = 1.5, # Final simulation time
                    nt = 350,
                    a = 1):



        #################
        #  initial data #
        #################

        # grid cells
        self.x_min = -0.5
        self.x_max = 3.5
        self.length = self.x_max - self.x_min
        self.nx = 500
        self.dx = self.length/self.nx
        self.x = np.linspace(self.x_min + self.dx/2, self.x_max-self.dx/2, self.nx)

        # Time grid
        self.t_final = 1.5  # time of computation
        self.nt = 350
        self.dt= self.t_final/self.nt

        self.a = 1
        self.m=2

        #define vector for initial data
        self.u0 = np.zeros(self.nx)
        self.L1 = self.x <= 0
        self.L2 = np.logical_and(self.x > 0 , self.x <= 1)
        self.L3 = self.x > 1
        self.u0[self.L1] = 0
        self.u0[self.L2] = 0.75
        self.u0[self.L3] = 0

        # u0[L1] = 2*x[L1]
        # u0[L2] = 1
        # u0[L3] = 3-2*x[L3]

        plt.plot(self.x, self.u0,'--r')
        plt.xlabel('X')
        plt.ylabel('U(X)')
        plt.title('Initial data')
        plt.grid()

        #%%

        ########################
        #  analytical solution #
        ########################

        def f1(self, u):
            return u*(1-u)

        def f2(self, u):
            return u**2/(u**2 + (1-u)**2)

        
        def f_s(self, s, m):
            return s**self.nw/(s**self.nw+m*(1-s)**self.no)

        def Riemann(xR, v_l, v_r, nv, f):
            v = np.linspace(v_l, v_r, nv)

            points = np.c_[v, f(v)]

            # compute envelope depending on jump direction , +/-
            if v_l < v_r:
                envelope = lower_convex(points)
                Label = 'Lower Convex'
                [ev, ef] = np.array(envelope).T
            else:
                envelope = upper_concave(points)
                Label = 'Upper Concave'
                [ev, ef] = np.array(envelope).T
                ev = np.flip(ev)
                ef = np.flip(ef)

            # compute a velocity s_i for each v_i based on envelope
            speed_l = np.zeros(len(ev))
            speed_l[0] = (ef[1]-ef[0])/(ev[1]-ev[0])
            speed_l[1:-1] = (ef[2:]-ef[:-2])/(ev[2:]-ev[:-2]) 
            speed_l[-1] = (ef[-1]-ef[-2])/(ev[-1]-ev[-2])

            # compute the travelled distance from x_l
            xr = np.zeros(len(ev))
            xr = xR + speed_l*t_final
            return v, ev, ef, Label, xr

        # Give Riemann data for first jump

        xR  = 0     # position of jump
        v_l = 0     # u-value on left side of jump
        v_r = 0.75  # u-value on right side of jump
        nv = 50     # Number of values to divide jump, [vl,vr], into
        f = f_s      # Specify f(u)

        v, ev1, ef1, Label1, xr1 = Riemann(xR, v_l, v_r, nv, f)
        plt.figure(figsize=(15,8))

        plt.subplot(1,2,1)
        plt.plot(v, f(v), 'b.', label = 'Function')
        plt.plot(ev1, ef1, 'r--', label = Label1)
        plt.title(Label1 + ' Envelope')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(xr1,ev1)

        #%%
        # Give Riemann data for second jump
        xR  = 1
        v_l = 0.75
        v_r = 0

        v, ev2, ef2, Label2, xr2 = Riemann(xR, v_l, v_r, nv, f)

        plt.subplot(1,2,1)
        plt.plot(v, f(v), 'b.', label = 'Function')
        plt.plot(ev2, ef2, 'r--', label = Label2)
        plt.title(Label2 + ' Envelope')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(xr2,ev2)

        #%%

        xM = np.concatenate(([x_min], xr1, xr2, [x_max]))
        uM = np.concatenate(([0], ev1, ev2, [0]))
        u_sol = np.interp(x, xM, uM)

        plt.plot(xM, uM)
        plt.plot(x, u_sol, '--r')
        #%%
        if a*dt > dx:
            print('Decrease dt')

        u_old   = np.zeros(nx)

        # useful quantity for the discrete scheme
        h = (dt/dx)*a

        # calculate the numerical solution u by going through a loop of nt numberof time steps
        def Numerical(u0, h, nt, m):
            u_old = u0.copy()
            u_all = u_old.copy()

            for j in range(nt-1):
                # calculate solution at interior part of domain, that is, cells j=2,...,N-1  
                u_old[1:nx] = u_old[1:nx] - h*(f_s(u_old[1:nx], m) - f_s(u_old[0:nx-1], m))
                # solution at left boundary
                u_old[0] = u0[0]
                # solution at right boundary
                u_old[-1] = u0[-1]

                u_all = np.c_[u_all, u_old]
            return u_all

        plt.plot(x, Numerical(u0, h, nt, m)[:,-1], '-b')



        # %%

        ns = 50 # nunmber of saturation grid points
        s_all = np.linspace(0, 1, ns)

        M_all = np.linspace(0.5, 5, 5) #viscosity coefficient

        #saturation exponents
        nw = 3
        no = 2

        colors = ['r','orange','y','g','b']    
        for i, mi in enumerate(M_all):
            plt.plot(s_all, f_s(s=s_all,m=mi), color = colors[i])

        #%%
        colors = ['r','orange','y','g','b']    
        for i, mi in enumerate(M_all):
            plt.plot(x, Numerical(u0, h, nt, mi)[:,-1], color = colors[i])
        #%%

# %%

matplotlib.use('TkAgg')

fig, ax = plt.subplots()
line, = ax.plot(x,u0)

def init(): 
    line.set_data(x, u0)
    return line,

def animate(i):
    #line.set_ydata(np.sin(x + i / 50))  # update the data.
    line.set_data(x, u_all[:,i])  # update the data.
    return line,

plt.plot(x,u_all[:,0],'--r')
ani = animation.FuncAnimation(fig, animate, init_func = init, frames = NTime, interval = 20, blit=True)

plt.show()
# %%
