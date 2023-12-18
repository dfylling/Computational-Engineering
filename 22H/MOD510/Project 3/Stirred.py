# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%


class Stirred:
    """This class is designed to be a collection of answers/solutions in code to the project Stirred (not shaken)
    for the course MOD510-1 22H at UiS (University of Stavanger).
    This class uses matplotlib.pyplot, numpy and pandas libraries.
    """

    def __init__(self,
                 N=1,
                 t0=0,
                 dt=0.10,
                 t_final=10,
                 V_tot=2,
                 q=1,
                 inj_prot='zero',
                 method='euler',
                 model='basic',
                 N_A=-4,
                 V_A=1,
                 D_A=1,
                 tau_min=2,
                 tau_max=3,
                 dtau=0.1):

        # conversions
        self.kg_to_g = 1000
        self.min_to_s = 60

        # Grid
        self.N = N
        self.t0 = t0
        self.t_sim = t0
        self.dt = dt  # sec
        self.t_final = t_final  # sec
        self.t_all = np.arange(t0, t_final+dt, dt)
        self.t_sim_all = self.t_all[1:]
        self.V_tot = V_tot  # Liter
        self.q = q/self.min_to_s  # L/sec
        self.tau = V_tot/q*self.min_to_s  # sec
        self.inj_prot = inj_prot
        self.method = method

        # base case is for all tanks to start with zero concentration
        self.c_old = np.zeros(self.N)
        self.c_old[0] = self.c_inj()
        self.y_old = self.c_old

        # Extended model including diffusion to aneurysm
        self.model = model
        self.N_A = N_A
        self.V_A = V_A
        self.D_A = D_A*self.q

        # Curve matching by optimizing tau
        self.SSR_all = []
        self.dtau = dtau
        self.tau_all = np.arange(tau_min, tau_max, dtau)

    def ode_solver(self):
        """Solver for ODE at time interval t0 to t_final.
        It does this by doing steps with the step function which uses the current method to choose stepping method.
        Appends results to t_all.
        """
        self.sol = self.y_old
        for self.t_sim in self.t_sim_all:
            self.y_old = self.y_old + self.step()
            self.sol = np.c_[self.sol, self.y_old]
        return

    def step(self):
        """Solves one time step with the current method.

        Returns:
            Nx1 array: Array of the computed values of step.
        """
        if self.method == 'euler':
            return self.euler_step()
        elif self.method == 'rk2':
            return self.rk2_step()
        elif self.method == 'rk4':
            return self.rk4_step()
        else:
            print('Unknown method')

    def euler_step(self):
        """Calculate a Euler step.

        Returns:
            Nx1 array: Array of the computed values of step.
        """
        return self.dt*self.rhs(self.y_old)

    def rk2_step(self):
        """Calculate a Runge-Kutta 2nd order step.

        Returns:
            Nx1 array: Array of the computed values of step.
        """
        k1 = self.dt*self.rhs(self.y_old)
        k2 = self.dt*self.rhs(self.y_old + 0.5*k1)
        return k2

    def rk4_step(self):
        """Calculate a Runge-Kutta 4th order step.

        Returns:
            Nx1 array: Array of the computed values of step.
        """
        k1 = self.dt*self.rhs(self.y_old)
        k2 = self.dt*self.rhs(self.y_old + 0.5*k1)
        k3 = self.dt*self.rhs(self.y_old + 0.5*k2)
        k4 = self.dt*self.rhs(self.y_old + k3)
        return (k1+2*k2+2*k3+k4)/6

    def rhs(self, c):
        """Function that calls to the appropriate RHS based on model chosen.
        This is the function that is implicitly passed to the solver by being in the same class.

        Args:
            c (Numpy array): Current state values.

        Returns:
            Numpy array: Derivatives.
        """
        if self.model == 'basic':
            return self.rhs_basic(c)
        elif self.model == 'extended':
            return self.rhs_extended(c)
        else:
            print('Unknown model')

    def rhs_basic(self, c):
        """The right hand side of the basic model.
        See the function at Line 123, rhs(self, c), to see context.

        Args:
            c (Numpy array): Current state values.

        Returns:
            Numpy array: Derivatives.
        """
        rhs = []
        rhs = np.append(self.c_inj(), c)[:-1] - c
        return rhs*self.N/self.tau

    def rhs_extended(self, c):
        """The right hand side of the extended model.
        See the function at Line 123, rhs(self, c), to see context.

        Args:
            c (Numpy array): Current state values.

        Returns:
            Numpy array: Derivatives.
        """
        rhs = []
        rhs = (self.N-1)*self.q/self.V_tot * (np.append(self.c_inj(), c)[:self.N_A-1] - c[:self.N_A])
        Ci = (self.N-1)/self.V_tot*(self.q *(c[self.N_A-1]-c[self.N_A]) - self.D_A*(c[self.N_A]-c[self.N_A+1]))
        Ca = self.D_A/self.V_A*(c[self.N_A]-c[self.N_A+1])
        rhs = np.append(rhs, Ci)
        rhs = np.append(rhs, Ca)
        rhs = np.append(rhs, (self.N-1)*self.q/self.V_tot * ((c[self.N_A]-c[self.N_A+2])))
        rhs = np.append(rhs, (self.N-1)*self.q/self.V_tot * (c[self.N_A+2:-1] - c[self.N_A+3:]))
        return rhs

    def c_inj(self):
        """Reproduces several injection protocols.

        Returns:
            float: The value interpolated at current time.
        """
        if self.inj_prot == 'zero':
            self.ti = [0, 1., 1., 2., 2., 3., 3., 4., 4., 5., 5.]
            self.ci = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif self.inj_prot == 'rect_1s':
            self.ti = [0, 1., 1., 10.]
            self.ci = [1, 1, 0, 0]
        elif self.inj_prot == 'rect_2s':
            self.ti = [0, 2., 2., 10.]
            self.ci = [1, 1, 0, 0]
        elif self.inj_prot == 'rect_3s':
            self.ti = [0, 3., 3., 10.]
            self.ci = [1, 1, 0, 0]
        elif self.inj_prot == 'ramp_1s':
            self.ti = [0, 1., 1., 10.]
            self.ci = [0, 1, 0, 0]
        elif self.inj_prot == 'ramp_2s':
            self.ti = [0, 2., 2., 10.]
            self.ci = [0, 1, 0, 0]
        elif self.inj_prot == 'biphasic':
            self.ti = [0, 1., 1., 2., 2., 3., 3., 10.]
            self.ci = [0.7, 0.7, 0, 0, 1, 1, 0, 0]
        elif self.inj_prot == 'biphasic_2':
            self.ti = [0, 0.9, 0.9, 2., 2., 3., 3., 10.]
            self.ci = [0.7, 0.7, 0, 0, 1, 1, 0, 0]
        else:
            print('Unknown injection protocol')
        return np.interp(self.t_sim, self.ti, self.ci)

    def test_implementation(self):  # Exercise 1 Part 2
        """Tests the current method of the class by comparing to the analytical solutions of the last tank with N=1,2,3
        and plots them. (Combination of plotting and execution since this is in essence just a loop of plotting another functions
        result)
        """
        self.tau = 2
        self.c_old[0] = 1
        if self.N in (1, 2, 3):
            self.ode_solver()
            plt.plot(self.t_all, self.sol[-1, :], '*',
                     label=f'Numerical, {self.method} N={self.N}')
            plt.plot(self.t_all, self.analytic(self.t_all),
                     '-', label=f'Analytical N={self.N}')
            plt.title('Output from tank 1, 2, and 3 ')
            plt.legend()
            plt.xlabel('Time [sec]')
            plt.ylabel('Concentration [kg/L]')
        else:
            print('Value of N is not valid')
        return

    def analytic(self, t):
        """Returns the analytical solution at time(s) t related to the current N.

        Args:
            t (Numpy array / Scalar): Time point(s).

        Returns:
            Numpy array: Analytical solution at time(s)
        """
        if self.N == 1:
            return np.exp(-np.array(t)/self.tau)
        elif self.N == 2:
            return 2*np.array(t)/self.tau*np.exp(-2*np.array(t)/self.tau)
        elif self.N == 3:
            return 9*np.array(t)*np.array(t)/2/self.tau/self.tau*np.exp(-3*np.array(t)/self.tau)

    def error_scaling(self):
        """Plots error at t=1 in comparison to analytical solution at t=1, with the different methods and
        different step sizes.
        """
        # hard coding some parameters below as provided in Exercise 1 part 3
        # Comparing numeric output from last tank to analytic solution at t=1sec
        self.tau = 2
        self.t_final = 1
        self.c_old[0] = 1
        dt_all = np.logspace(-1,-4,4)
        error_all = []
        c_an = self.analytic(self.t_final)
        for self.dt in dt_all:
            self.t_all = np.arange(self.t0, self.t_final+self.dt, self.dt)
            self.t_sim_all = self.t_all[1:]
            self.y_old = self.c_old
            self.ode_solver()
            error = np.abs(self.sol[-1, -1]-c_an)
            error_all = np.append(error_all, error)
            # print(f'Numerical error {error:.2e} for dt={self.dt}')
        plt.plot(dt_all, error_all, '*-', label=f'{self.method}')
        plt.title('Comparison of error scaling at t=1sec')
        plt.legend()
        plt.xlabel('Time step, dt [sec]')
        plt.ylabel('Error [kg/L]')
        plt.yscale('log')
        plt.xscale('log')

    def flowrate(self, M_inj=83.33, file='Healthy_rect_1s'):  # Exercise 2 Part 1
        """Calculates flowrate.

        Args:
            M_inj (float, optional): Injected mass. Defaults to 83.33.
            file (str, optional): File name. Defaults to 'Healthy_rect_1s'.
        """
        # Calculates flowrate based on total injected dye/mass and specified CFD file.
        self.data_CFD(file)
        int = self.integral(self.t_CFD, self.c_CFD)
        flow = M_inj/int*self.min_to_s/self.kg_to_g
        print(f'Calculated flow rate from {file} is {flow:.3f} L/min')

    def data_CFD(self, file):
        """Reads specified file and extracts CFD data from data file and puts it into arrays

        Args:
            file (String): Filename, given that it is in $working-dir/data, and is a .csv file.
        """
        df = pd.read_csv(f'data/{file}.csv')
        self.c_CFD = np.asarray(df['Concentration'])
        self.t_CFD = np.asarray(df['Time'])

    def integral(self, x, y):
        """Find integral by solving each smaller integral between points.
        Each smaller integral k = 0.5*stepsize*(y(k)-y(k-1))

        Args:
            x (Numpy array): Time/X-axis points which is put into f(x)
            y (Numpy array): Height/f(x)/Y-axis which results from f(x)

        Returns:
            Float: The numerical result of integral
        """
        # inputs x and y arrays and returns trapezoidal integral
        return 0.5*np.sum((x[1:]-x[:-1])*(y[:-1]+y[1:]))

    def comparison_plot(self, file='Healthy_rect_1s'):  # Exercise 2 Part 3
        """_summary_

        Args:
            file (str, optional): _description_. Defaults to 'Healthy_rect_1s'.
        """
        self.data_CFD(file)
        self.ode_solver()
        plt.plot(self.t_all, self.sol[-1, :], '*',
                 label=f'Numerical - {self.inj_prot}')
        plt.plot(self.t_CFD, self.c_CFD, '*', label=f'CFD - {file}')
        plt.title('Comparison of numeric output vs CFD-data')
        plt.legend()
        plt.xlabel('Time step, dt [sec]')
        plt.ylabel('Concentration [kg/L]')

    def Optimize_tau(self, condition='Healthy'):
        """Finds the optimal value for tau.

        Args:
            condition (str, optional): Type of aorta. Defaults to 'Healthy'.

        Returns:
            Float: Optimal value for tau found
        """
        # this function has a narrow usage, this is why several variables are hard set inside of it.
        self.inj_prot = 'rect_1s'
        self.c_old = np.zeros(self.N)
        self.c_old[0] = self.c_inj()
        self.y_old = self.c_old
        t0=0
        t_final=5
        self.dt = 0.02  # setting dt to same value as in data file
        self.t_all = np.arange(t0, t_final+self.dt, self.dt)
        self.t_sim_all = self.t_all[1:]
        file = f'{condition}_rect_1s'
        self.data_CFD(file)
        d = self.c_CFD[:250]
        for self.tau in self.tau_all:
            self.ode_solver()
            plt.plot(self.t_all, self.sol[-1, :], '*',label=f'Numerical - {self.inj_prot}')
            plt.plot(self.t_CFD, self.c_CFD, '*', label=f'CFD - {file}')
            plt.show()
            self.y_old = self.c_old
            m = self.sol[-1, :250]
            self.SSR_all = np.append(self.SSR_all, self.SSR(d, m))
        dSSR_dt_all = self.df_dx(self.SSR_all, self.dtau)
        return self.Bisection_numeric(dSSR_dt_all, self.tau_all)

    def SSR(self, d, m):
        """Squared error function.

        Args:
            d (Numpy array): Array of nums
            m (Numpy array): Array of nums

        Returns:
            Float: The squared error.
        """
        return np.sum((d-m)**2)

    def df_dx(self, f, dx):
        """Find derivative of function results with a step size.

        Args:
            f (Numpy array): Function results
            dx (Numpy array): Step size

        Returns:
            Derivative: Derivative of function results
        """
        return (f[1:]-f[:-1])/dx

    def Bisection_numeric(self, f, x):
        """Use the bisection method to find the best tau.

        Args:
            f (function): Function
            x (_type_): Time/X-axis values
        """
        a = 0
        b = len(f)-1
        if f[a]*f[b] < 0:
            while b-a > 1:
                c = int(np.round((a+b)/2, 0))
                if f[c]*f[b] < 0:
                    a = c
                else:
                    b = c
        else:
            print('false starting point')
        # interpolating linearly between point a and b to reach zero
        tau_opt = x[a]+f[a]*(x[b]-x[a])/(f[a]-f[b])
        print(f'Curve fits best at tau = {tau_opt:.3f}')

#%%
Stirred(N=50, tau_min=2.36, tau_max=2.39, dtau=0.01, method='rk4').Optimize_tau(condition='Healthy')
#%%
Stirred(N=50, dt=0.02, q=5, V_tot=2.394*(5/60), t_final=5, method='rk4', inj_prot='rect_1s').comparison_plot(file='Healthy_rect_1s')
#%%