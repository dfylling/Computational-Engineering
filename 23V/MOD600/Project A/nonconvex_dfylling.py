#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
from envelope import *
%matplotlib inline
#%%

#################
#  initial data #
#################

# grid cells
x_min = -0.5
x_max = 3.5
length = x_max - x_min
nx = 500
dx = length/nx
x = np.linspace(x_min + dx/2, x_max-dx/2, nx)

# Time grid
t_final = 0.5  # time of computation
nt = 350
dt= t_final/nt

a = 1

#define vector for initial data
u0 = np.zeros(nx)
L1 = x <= 0
L2 = np.logical_and(x > 0 , x <= 1)
L3 = x > 1
u0[L1] = 0
u0[L2] = 0.75
u0[L3] = 0

if a*dt > dx:
    print('Decrease dt')

# useful quantity for the discrete scheme
h = (dt/dx)*a

plt.plot(x,u0,'--r')
plt.xlabel('X')
plt.ylabel('U(X)')
plt.title('Initial data')
plt.grid()

#%%

########################
#  analytical solution #
########################

def f1(u):
    return u*(1-u)

def f2(u):
    return u**2/(u**2 + (1-u)**2)

def Riemann(xR, v_l, v_r, nv, f, *args):
    v = np.linspace(v_l, v_r, nv)

    points = np.c_[v, f(v, *args)]

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
f = f2      # Specify f(u)

v, ev1, ef1, Label1, xr1 = Riemann(xR, v_l, v_r, nv, f)
plt.figure(figsize=(15,8))

# Give Riemann data for second jump
xR  = 1
v_l = 0.75
v_r = 0

v, ev2, ef2, Label2, xr2 = Riemann(xR, v_l, v_r, nv, f)

plt.subplot(2,2,1)
plt.plot(v, f(v), 'b.', label = 'Function')
plt.plot(ev1, ef1, 'r--', label = Label1)
plt.title(Label1 + ' Envelope')
plt.legend()

plt.subplot(2,2,2)
plt.plot(xr1,ev1)

plt.subplot(2,2,3)
plt.plot(v, f(v), 'b.', label = 'Function')
plt.plot(ev2, ef2, 'r--', label = Label2)
plt.title(Label2 + ' Envelope')
plt.legend()

plt.subplot(2,2,4)
plt.plot(xr2,ev2)

#%%

xM = np.concatenate(([x_min], xr1, xr2, [x_max]))
uM = np.concatenate(([0], ev1, ev2, [0]))
u_sol = np.interp(x, xM, uM)

plt.plot(xM, uM)
plt.plot(x, u_sol, '--r')
#%%

# calculate the numerical solution u by going through a loop of nt numberof time steps
def Numerical(u0, h, nt, *args):
    u_old = u0.copy()
    u_all = u_old.copy()

    for j in range(nt-1):
        # calculate solution at interior part of domain, that is, cells j=1,...,N-1  
        u_old[1:-1] = u_old[1:-1] - h*(f(u_old[1:-1], *args) - f(u_old[0:-2], *args))
        # solution at left boundary
        u_old[0] = u0[0]
        # solution at right boundary
        u_old[-1] = u0[-1]

        u_all = np.c_[u_all, u_old]
    return u_all

plt.plot(x, Numerical(u0, h, nt)[:,-1], '-b')
plt.plot(x, u_sol, '--r')


# %%
########################
#  Two Phaxe Flow data #
########################

# grid cells
x_min = 0
x_max = 1
length = x_max - x_min
nx = 300
dx = length/nx
x = np.linspace(x_min + dx/2, x_max-dx/2, nx)

# Time grid
t_final = 0.5  # time of computation
nt = 300
dt= t_final/nt

a = 1

#define vector for initial data
s0 = np.zeros(nx)
L1 = np.logical_and(x > 0 , x <= 0.1)
L2 = np.logical_and(x > 0.1 , x <= 1)
s0[L1] = 1
s0[L2] = 0

if a*dt > dx:
    print('Decrease dt')

# useful quantity for the discrete scheme
h = (dt/dx)*a

ns = 50 # nunmber of saturation grid points
s_all = np.linspace(0, 1, ns)

M_all = np.array([0.25, 0.5, 1, 2, 4]) #viscosity coefficient

#saturation exponents
nw = 3
no = 2

#%%
plt.plot(x,s0,'--r')
plt.xlabel('X')
plt.ylabel('S(X)')
plt.title('Initial data')
plt.grid()
#%%

def f_s(s, m=2, nw=3, no=2):
    return s**nw/(s**nw+m*(1-s)**no)

# Give Riemann data for jump

xR  = 0.1   # position of jump
v_l = 1     # u-value on left side of jump
v_r = 0     # u-value on right side of jump
nv = 50     # Number of values to divide jump, [vl,vr], into
f = f_s     # Specify f(u)
M = 0.5

v, ev1, ef1, Label1, xr1 = Riemann(xR, v_l, v_r, nv, f, M)

plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
plt.plot(v, f(v, M), 'b.', label = 'Function')
plt.plot(ev1, ef1, 'r--', label = Label1)
plt.title(Label1 + ' Envelope')
plt.legend()

plt.subplot(2,2,2)
plt.plot(xr1,ev1)

#%%

xM = np.concatenate(([x_min], xr1, [x_max]))
uM = np.concatenate(([1], ev1, [0]))
u_sol = np.interp(x, xM, uM)

plt.plot(xM, uM)
plt.plot(x, u_sol, '--r')

#%%

colors = ['r','orange','y','g','b']    
for i, mi in enumerate(M_all):
    plt.plot(s_all, f_s(s=s_all,m=mi), color = colors[i], label=f'M = {mi}')
    plt.legend()
#%%
u_num = Numerical(s0, h, nt, M)
plt.plot(x, u_num[:,-1])
plt.plot(x, u_sol, '--r')

print(f'Areal difference is {np.sum((u_num[:,-1]-u_sol)*dx):.3f}')
print(f'Sum of square difference is  {np.sum(((u_num[:,-1]-u_sol))**2):.3f}')

#%%
colors = ['r','orange','y','g','b']    
for i, mi in enumerate(M_all):
    plt.plot(x, Numerical(s0, h, nt, mi)[:,-1], color = colors[i], label=f'M = {mi}')
    plt.legend()

#%%
#since M = mu,water / mu,oil M > 1 means that the water has higher viscosity than the oil in front.
# Viscosity of reservoir oil relies on density of the oil as well as temperature and pressure.
# Viscosity of water also depends on temperature and pressure but is generally more stable.
# Usually the viscosity of reservoir oil is higher than water, so we should pay most attention to M < 1.
# The more mobile fluid, in our case water, will have a tendency to bypass the heavier fluid.
u0=s0
u_all = u_num

matplotlib.use('WebAgg')

fig, ax = plt.subplots()
line, = ax.plot(x,u0)

def init(): 
    line.set_data(x, u0)
    return line,

def animate(i):
    line.set_data(x, u_all[:,i])  # update the data.
    return line,

plt.plot(x,u_all[:,0],'--r')
ani = animation.FuncAnimation(fig, animate, init_func = init, frames = nt, interval = 200, blit=True)

plt.show()


#%%

#%%

#%%