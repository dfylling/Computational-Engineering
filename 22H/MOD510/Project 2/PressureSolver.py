#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sc
import warnings
warnings.filterwarnings('ignore') #stopping some irrelevant warnings from pandas

class PressureSolver:
    """
    A finite difference solver to solve pressure distribution in
    a reservoir, logarithmic grid has been used, y = ln(r/rw)
    The solver uses SI units internally, while "practical field units"
    are required as input.
    A and b is the system of style 'Ax=b'.
    Input arguments:
    name symbol unit
    --------------------------------------------------------------
    Number of grid points N dimensionless
    Constant time step dt days
    Well radius rw ft
    Outer reservoir boundary re ft
    Height of reservoir h ft
    Absolute permeability k mD
    Porosity phi dimensionless
    Fluid viscosity mu mPas (cP)
    Total (rock+fluid) compressibility ct 1 / psi
    Constant flow rate at well Q bbl / day
    Initial reservoir pressure pi psi
    14
    -------------------------------------------------------------
    """

    def __init__(self,
                 N=4,
                 dt=0.010,
                 t_final=1,
                 rw=0.318,
                 re=1000.0,
                 h=11.0,
                 phi=0.25,
                 mu=1.0,
                 ct=7.8e-6,
                 Q=1000.0,
                 k=500,
                 p_i=3900.0):

        # Unit conversion factors (input units --> SI)
        self.ft_to_m = 0.3048
        self.psi_to_pa = 6894.75729
        self.day_to_sec = 24*60*60
        self.hour_to_sec = 60*60
        self.bbl_to_m3 = 0.1589873

        # Grid
        self.N = N
        self.r_w = rw*self.ft_to_m
        self.r_e = re*self.ft_to_m
        self.h = h*self.ft_to_m

        self.y_e = np.log(self.r_e/self.r_w)  # dimensionless
        self.dy = self.y_e/N  # h = \Delta y
        # shifting nodes h/2 away from y_w and y_e
        self.y = np.linspace(self.dy/2, self.y_e-self.dy/2, self.N)
        # transforming y back into radius for plotting
        self.r = self.r_w*np.exp(self.y)

        # Rock and fluid properties
        self.k = k*1e-15 / 1.01325
        self.phi = phi
        self.mu = mu*1e-3
        self.ct = ct / self.psi_to_pa
        self.n_diff = self.k/self.mu/self.phi/self.ct  # hydraulic diffusivity constant

        # Initial and boundary conditions
        self.Q = Q*self.bbl_to_m3 / self.day_to_sec
        self.p_i = p_i*self.psi_to_pa

        # Time control for simulation
        self.t = 0
        self.dt = dt*self.hour_to_sec
        self.t_final = t_final*self.hour_to_sec
        self.t_all = []

        # Calculation constants
        self.eta = self.n_diff*np.exp(-2*self.y)*self.dt/self.r_w**2/self.dy**2
        self.beta = self.Q*self.mu*self.dy/(2*np.pi*self.k*self.h)

        # defina diagolas  and d vector
        self.upperDiag = -self.eta[1:]
        self.diag = np.ones(self.N)+2*self.eta
        self.lowerDiag = -self.eta[0:self.N-1]
        self.d = np.zeros(self.N)

        # internal boundary
        self.diag[0] = self.diag[0]-self.eta[0]
        self.d[0] = -self.beta*self.eta[0]

        # exterior boundary
        self.diag[-1] = self.diag[-1]+self.eta[-1]
        self.d[-1] = 2*self.p_i*self.eta[-1]

        self.P_ini = np.repeat(self.p_i, self.N)
        self.P_all = np.transpose(self.P_ini)
        self.P_an_all = np.transpose(self.P_ini)
        self.A = self.tri_diag()
        self.b = self.P_ini + self.d

    def tri_diag(self, k1=-1, k2=0, k3=1):
        return np.diag(self.upperDiag, k1) + np.diag(self.diag, k2) + np.diag(self.lowerDiag, k3)

    def A_matrix(self):
        print(self.A)
        return

    def solve(self):
        self.rhs = self.d+self.P_old
        self.rhs[-1] = self.d[-1]+self.p_i  # constant pressure at boudary
        self.P_new = np.linalg.solve(self.A, self.rhs)
        return

    def solve_time(self):
        self.P_old = self.P_ini
        while self.t <= self.t_final:
            self.solve()
            self.P_old = np.copy(self.P_new)
            self.P_all = np.c_[self.P_old, self.P_all]
            self.t_all = np.append(self.t_all, self.t) # collecting time steps for plotting later
            self.t += self.dt
        # adding row of p_i to correspond to y_e /  r_e
        self.P_all = np.r_[self.P_all, [
            np.repeat(self.p_i, len(self.P_all[0]))]]
        return

    def Numeric_BHP(self):
        # extrapolates pressure in node p_0 to p_w and returns pressure vector for all nodes.
        self.solve_time()
        pw_num = np.flip(self.P_all[0, :]-self.beta/2) 
        pw_num[0] = self.p_i  # correcting point at t=0 to be initial pressure
        return pw_num

    def test_data(self):
        # extracts test data from data file and puts it into arrays
        df = pd.read_csv('data/well_bhp.dat', '\t')
        self.test_bhp = np.asarray(df['well_pressure'])*self.psi_to_pa
        self.test_time = np.asarray(df['time'])*self.hour_to_sec
        return

    def match_test_data(self, Test_data = True, Numerical = True, Line_solution = True, x_scale='log'):
        # gathers data from other internal functions and plots Well test data vs Numerical solution in the same plot
        # the plot can be used to tweak unkown calculation input manually to match test data
        pw_num = self.Numeric_BHP()
        self.test_data()
        fig = plt.figure()
        self.t = self.dt
        self.line_solution(self.r_w)
        ax = fig.add_subplot(1, 1, 1)
        if Test_data == True:
            ax.plot(self.test_time/self.day_to_sec, self.test_bhp, '*', label='Test data')
        if Numerical == True:
            ax.plot(self.t_all/self.day_to_sec, pw_num[:-1], '*', label='Numerical')
        if Line_solution == True:
            ax.plot(self.t_all/self.day_to_sec, np.flip(self.P_an_all[0,:]), label='Line solution')
        ax.legend(loc='best')
        ax.set_title(f'Pressure data up to {self.t_final/self.day_to_sec:.0f} days.')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Pressure (pa x 10e+7)')
        plt.xscale(x_scale)
        plt.grid()
        return

    def line_solution(self, r):
        while self.t <= self.t_final:
            self.P_an = self.P_ini+self.Q*self.mu/4/np.pi/self.k / \
                self.h*sc.expi(-r*r/4/self.n_diff/self.t)
            self.P_an_all = np.c_[self.P_an, self.P_an_all]
            self.t += self.dt
        # adding row of p_i to correspond to y_e /  r_e
        self.P_an_all = np.r_[self.P_an_all, [
            np.repeat(self.p_i, len(self.P_an_all[0]))]]
        return

    def Volume_of_water(self):
        return np.pi*self.r_e*self.r_e*self.h*self.phi

    def plot(self, p):
        plt.title('Pressure distribution up to ' + str(self.t_final/self.day_to_sec
                                                       ) + ' days, with dt ' + str(self.dt/self.day_to_sec) + ' days.')
        plt.plot(np.append(self.r, self.r_e), p, '*')
        plt.xscale('log')
        plt.grid()
        plt.show()
        return

    def solution_comparison(self):
        # for a comparison at a single point in time set dt slightly less than t_final, fex dt=199, t_final=200
        self.t = self.dt
        self.solve_time()
        self.t = self.dt
        self.line_solution(self.r)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.title(f'Solution comparison at time {self.t_final/self.day_to_sec:.0f} days.')
        ax.plot(np.append(self.r, self.r_e),
                self.P_all[:, -2], '*', label='Numerical')
        ax.plot(np.append(self.r, self.r_e),
                self.P_an_all[:, -2], label='Analytical')
        # loc=best, Automatically determines the position the place the legend desc.
        ax.legend(loc='best')
        ax.set_xlabel('Radius (m)')
        ax.set_ylabel('Pressure (pa x 10e+7)')
        plt.xscale('log')
        plt.grid()
        return

        
#%%
PressureSolver().A_matrix()