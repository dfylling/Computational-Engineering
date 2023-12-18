import numpy as np
import matplotlib.pyplot as plt


class Reservoir:

    def __init__(self, N=4, r_e=1000, r_w=0.318, alpha=1, p_init=3900, lazy=False):

        self.N = N  # number of nodes
        self.lazy = lazy
        self.r_e = r_e  # ft
        self.r_w = r_w  # ft
        self.alpha = alpha
        self.p_init = p_init  # psi
        self.y_e = np.log(r_e/r_w)  # dimensionless
        self.h = self.y_e/N  # h = \Delta y

        # shifting nodes h/2 away from y_w and y_e
        self.y = np.linspace(self.h/2, self.y_e-self.h/2, self.N)
        self.y = np.append(self.y, self.y_e)  # manually adding y_e
        # transforming y back into radius for plotting
        self.r = self.r_w*np.exp(self.y)

        self.ya = np.linspace(0, self.y_e, 100)
        # transforming ya back into radius for plotting
        self.ra = self.r_w*np.exp(self.ya)

        self.sol = self.pressure_solver()

        self.N_all = np.logspace(0, 1, 2).astype(int)*4
        self.e_lazy = []
        self.e_not_lazy = []

        self.e = 0

    def P(self, y):
        sola = self.alpha*(y-self.y_e)+self.p_init
        return sola

    def pressure_solver(self):

        a = np.ones(self.N-1)
        a2 = np.ones(self.N-1)
        b = np.repeat(-2, self.N)

        # right hand side
        c = np.zeros(self.N)

        # internal boundary
        b[0] = -1
        c[0] = self.alpha*self.h

        # exterior boundary
        if self.lazy:
            c[-1] = -self.p_init
        else:
            # not so lazy
            b[-1] = -3
            c[-1] = -2*self.p_init

        # k=0 diagonal, k neq 0 off diagonal
        A = np.diag(a2, k=-1)+np.diag(b, k=0)+np.diag(a, k=1)
        sol = np.linalg.solve(A, c)

        # manually adding p_init to match y_e in y-array
        sol = np.append(sol, self.p_init)
        return sol

    def plot_pressure(self):
        if self.lazy:
            leg = 'Lazy'
        else:
            leg = 'Not-so-lazy'

        plt.plot(self.r, self.sol, '*', label='Numerical, '+leg)
        plt.plot(self.ra, self.P(self.ya), '-', label='Analytical')
        plt.title('{} method with N={}'.format(leg, self.N))
        plt.grid()
        plt.legend()
        return

        # difference between analytical and numerical solution in p_0
    def single_error(self):
        error = self.P(self.y)[0] - self.pressure_solver()[0]
        return error

def plot_error(N_vals):
    notLazy = []
    lazy = []

    for i in N_vals:
        notLazy.append(Reservoir(N=i).single_error())
        lazy.append(Reservoir(N=i, lazy=True).single_error())

    fig, ax = plt.subplots()
    ax.set_title('Calculation error for lazy and not-so-lazy boundary conditions')
    ax.set_xlabel('Number of nodes (N)')
    ax.set_ylabel('Error (psi)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    x_val = N_vals
    y_vals = [notLazy, lazy]
    labels = ["Not lazy", "Lazy"]
    cols = ['b', 'c']
    points = ['-', '-']
    for y_val, point, col, label in zip(y_vals, points, cols, labels):
        ax.plot(x_val, y_val, point, c=col, label=label)
    plt.grid()
    plt.legend()
    plt.show()

    print(f'N = {N_vals}')
    print(f'Not lazy = {notLazy}')
    print(f'Lazy = {lazy}')

