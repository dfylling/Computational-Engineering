import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import seaborn as sns
import math
import os


class RandomlyViral:
    """
    Class  used  to  model  the  spreading  of  a  contagious  disease  in  a
    population  of  individuals  with  a  2D  random  walk.
    Each  walker  has  a  disease  state  which  is  represented  by  an
    integer  Enum.  Also,  a  set  of  integer  (x,  y)-coordinates  are
    stored  for  each  walker.  The  possible  coordinates  are:
    {0,  1,  ...,  Lx-1}  in  the  x-direction
    {0,  1,  ...,  Ly-1}  in  the  y-direction
    It  is  only  possible  to  move  North,  South,  East,  or  West.
    """

    def __init__(self,
                 simulation_size='full',
                 population_size=683,
                 no_init_infected=10,
                 no_init_recovered=0,
                 nx=50,
                 ny=50,
                 iterations=20,
                 time_steps=300,
                 beta=0.0643,
                 tau=111,
                 model='default',
                 Recovery='off',
                 recovery_probability=0.01,
                 infection_probability=0.9,
                 no_border=False,
                 infectionWaves=False,
                 wave_susc=0.1,
                 wave_reco=0.05,
                 num_waves=3,
                 random_seed=13,
                 death_rate=None,
                 adaptive_recovery_rate=None):
        """
        :param  simulation_size: input 'small' for a small scale simulation for troubleshooting purposes.
        :param  population_size:  The  total  number  of  people  (N).
        :param  no_init_infected:  The  number  of  infected  people  at  t=0.
        :param  no_init_recovered: Number of initial recovered people at t=0
        :param  nx:  The  number  of  lattice  nSIs  in  the  x-direction
        :param  ny:  The  number  of  lattice  nSIs  in  the  y-direction.
        :param  q:  The  probability  of  infection  (0  <=  q  <=  1).
        :param  iterations: Number of successive iterations the simulation will be run.
        :param  time_steps: Number of time steps the simulation will be running for.
        :param  beta: Disease transmission rate for ODE model
        :param  tau: Recovery time for ODE model
        :param  model: Used to enable ODE models: 'ODE_SI' , and 'ODE_SIR'
        :param  Recovery: Used to toggle recovery mechanism 'on' and 'off' for RW model
        :param  recovery_probability: Probability for an infected person to recover per time step
        :param  infection_probability: Probability for an infected person to infect each susceptible person in same location
        :param  no_border: Used to toggle on/off fixed boundary of "looping" grid: True/False
        :param  infectionWaves: Used to toggle on/off addition of infection waves: True/False
        :param  wave_susc: Probability for susceptible person to become infected by a "wave"
        :param  wave_reco: Probability for recovered person to become infected by a "wave"
        :param  num_waves: Number of waves equally spaced through the defined time scale.
        :param  random_seed: Specify seed for replication purpuses
        :param  death_rate: Toggles 'on' death mechanism while not empty, value will define chance for infected person to die per time step.
        :param  adaptive_recovery_rate: Used to adjust recovery
        """
        np.random.seed(random_seed)  # Make it repeatable

        if simulation_size == 'small':
            self.N = 50
            self.nx = 5
            self.ny = 5
            self.iterations = 5
            self.time_steps = 5
        else:
            self.N = population_size
            self.nx = nx
            self.ny = ny
            self.iterations = iterations
            self.time_steps = time_steps

        self.I0 = no_init_infected
        self.R0 = no_init_recovered
        self.beta = beta
        self.tau = tau
        self.model = model
        self.Recovery = Recovery
        self.p_infect = infection_probability
        self.p_recov = recovery_probability
        self.p_recov_init = recovery_probability
        self.iteration = []
        self.time_step = []
        self.t_all = np.arange(self.time_steps)

        self.no_border = no_border
        self.infectionWaves = infectionWaves
        self.wave_susc = wave_susc
        self.wave_reco = wave_reco
        self.num_waves = num_waves        
        self.death_rate = death_rate
        self.adaptive_recovery_rate = adaptive_recovery_rate   

        # States - enumeration of the different states or conditions one walker may have.
        self.Susceptible = 0
        self.Infectious = 1
        self.Recovered = 2
        self.Dead = 3

        # Initial state - as defined by total number of "Walkers" and number of initially infected.
        self.State_0 = np.full(self.N,  self.Susceptible)
        self.State_0[0:self.I0] = self.Infectious
        self.State = self.State_0.copy()

        # Arrays for recording states - the "records" start out as empty arrays of size time_steps x iterations.
        # In the Record_states - function each cell will be filled with information from each time step and iteration.
        empty_record = np.empty([self.time_steps, self.iterations])
        self.Susceptible_all = empty_record.copy()
        self.Infectious_all = empty_record.copy()
        self.Recovered_all = empty_record.copy()
        self.Dead_all = empty_record.copy()

        # Drawing random numbers to populate grid with "Walkers"
        self.Walkers = np.random.randint(
            0, [self.nx,  self.ny], size=(self.N,  2))

        # Used for saving positional info for plotting purposes
        self.avg_final_position = np.zeros([self.nx, self.ny])

        #Variables used for keeping track of number of dead and infected for updating variable recovery rate.
        self.deaths = 0
        self.infected = 0
  

    def move(self):
        proposed_directions = np.random.randint(1, 5, size=self.N)
        for index, dir in enumerate(proposed_directions):
            if dir % 2 == 0:
                self.Walkers[index, 0] += dir-3
            else:
                self.Walkers[index, 1] += dir-2

    def revert_illegal_move(self):
        """After moving, some walkers may have walked out of bounds of the simulations area. This function hunts them down and puts them back where they belong.
        Depending on input the function either reverses the walker back to previous position, or "overflows" the walker to the opposite side of the grid.

        Returns:
            self.Walkers array will be updated so that all walkers are returned into defined grid.
        """
        if self.no_border:
            # X
            tempx = self.Walkers[:, 0]
            for i, x in enumerate(tempx):
                if x < 0:
                    tempx[i] = self.nx - 1
                elif x >= self.nx:
                    tempx[i] = 0
            # Y
            tempy = self.Walkers[:, 1]
            for i, y in enumerate(tempy):
                if y < 0:
                    tempy[i] = self.ny - 1
                elif y >= self.ny:
                    tempy[i] = 0
            # set values
            self.Walkers[:, 0] = tempx
            self.Walkers[:, 1] = tempy
            return

        # X
        tempx = self.Walkers[:, 0]
        for i, x in enumerate(tempx):
            if x < 0:
                tempx[i] = 0
            elif x >= self.nx:
                tempx[i] = self.nx - 1
        # Y
        tempy = self.Walkers[:, 1]
        for i, y in enumerate(tempy):
            if y < 0:
                tempy[i] = 0
            elif y >= self.ny:
                tempy[i] = self.ny - 1
        # set values
        self.Walkers[:, 0] = tempx
        self.Walkers[:, 1] = tempy
        return

    def record_states(self):
        """Records number of Infected, Susceptible and Recovered at a given time step and simulation iteration.

        Returns:
            Each of the self."Condition"_all matrices are updated with number of cases for current time and iteration.
        """
        x, y = self.time_step, self.iteration,
        no_susceptible = np.sum(self.State == self.Susceptible)
        no_infectious = np.sum(self.State == self.Infectious)
        no_recovered = np.sum(self.State == self.Recovered)
        no_dead = np.sum(self.State == self.Dead)
        self.Susceptible_all[x, y] = no_susceptible
        self.Infectious_all[x, y] = no_infectious
        self.Recovered_all[x, y] = no_recovered
        self.Dead_all[x, y] = no_dead
        self.infected = no_infectious
        self.deaths = no_dead

    def infect_susceptible(self):
        """Calculates if an encounter between an Infected and Susceptible walker leads to spread of infection.
        * each susceptible person will have a chance of being infected by each infected person in the same location.

        Returns:
            self.State is updated according to results of encounters.
        """
        Walkers_1d = self.Walkers[:, 1]+self.Walkers[:, 0]*self.ny
        index = np.arange(len(self.State))
        Super = np.c_[index, Walkers_1d, self.State]

        # Get  (x,y)-coordinates  of  susceptibles  &  infectious  people
        S_coord = Super[Super[:, 2] == self.Susceptible]
        I_coord = Super[Super[:, 2] == self.Infectious]

        # Find locations that contain both susceptibles  &  infectious  people
        S_unique = np.unique(S_coord[:, 1])
        I_unique = np.unique(I_coord[:, 1])
        # collision_locations = np.array(S_unique[S_unique == I_unique])
        collision_locations = []
        mask = np.full(len(S_unique), False)
        for loc in I_unique:
            mask = mask + np.equal(S_unique, loc)
        collision_locations = S_unique[mask]

        # trim off all locations without collisions
        mask = np.full(len(self.State), False)
        for loc in collision_locations:
            mask = mask + np.equal(Super[:, 1], loc)
        Super_clean = Super[mask]

        for loc in collision_locations:
            for idx in range(len(Super_clean)):
                if Super_clean[idx, 1] == loc:
                    if Super_clean[idx, 2] == self.Susceptible:
                        q = np.random.uniform(0, 1)
                        exp = np.count_nonzero(np.equal(
                            Super_clean[:, 1], loc) * np.equal(Super_clean[:, 2], self.Infectious))
                        if q < 1-(1-self.p_infect)**exp:
                            self.State[Super_clean[idx, 0]] = self.Infectious

    def recover_infected(self):
        """Calculates if an infected person will recover based on Class input.

        Returns:
            self.State is updated according to results.
        """
        index = np.arange(len(self.State))
        Super = np.c_[index, self.State]

        Super_clean = Super[Super[:, 1] == self.Infectious]
        for idx in range(len(Super_clean)):
            q = np.random.uniform(0, 1)
            if q < self.p_recov:
                self.State[Super_clean[idx, 0]] = self.Recovered

    def single_simulation(self):
        self.infect_susceptible()
        self.plot_grid()
        self.move()
        self.plot_grid()
        self.revert_illegal_move()
        self.plot_grid()

    def plot_grid(self):
        """Plots grid as defined for the class, with location of all susceptible and infected walkers.

        Returns:
            Coorinate grid showing locations of walkers.
        """
        x_s, y_s = np.transpose(self.Walkers[self.State == self.Susceptible])
        x_i, y_i = np.transpose(self.Walkers[self.State == self.Infectious])
        plt.plot(x_s, y_s, 'g.')
        plt.plot(x_i, y_i, 'ro', markersize=12, markerfacecolor='none')
        plt.grid()
        plt.show()

    def plot_grid_heatmap(self, title="Grid heatmap", numbers=False, pretty=True, state=None, ax=None):
        """Plots a heatmap of elements in grid at current iteration and time step.

        Args:
            title (str, optional): Title of plot. Defaults to "Grid heatmap".
            numbers (bool, optional): To use numbers or not for each cell, is bad at large grid sizes. Defaults to False.
            pretty (bool, optional): Use pretty colors or not, not pretty is best for numbers. Defaults to True.
            state (_type_, optional): Which state to plot. Defaults to None, meaning all states.
            ax (_type_, optional): Which ax to plot on. Defaults to None, meaning a new plot.

        Returns:
            _type_: The heatmap values as a nx by ny size numpy array
        """
        if state is None:
            Walkers = self.Walkers
        else:
            Walkers = []
            for i, state in enumerate(self.State):
                if state == self.Infectious:
                    Walkers.append(self.Walkers[i])
        grid = np.zeros([self.nx, self.ny])
        for walker in Walkers:
            grid[walker[0]][walker[1]] += 1
        if pretty:
            y, x = np.meshgrid(np.linspace(0, self.nx, self.nx+1),
                               np.linspace(0, self.ny, self.ny+1))
            if ax is None:
                fig_, ax_ = plt.subplots()
            z_min, z_max = 0, np.abs(grid).max()
            c = ax.pcolormesh(x, y, grid, cmap='Blues', vmin=z_min, vmax=z_max)
            ax.set_title(title)
            ax.axis([x.min(), x.max(), y.min(), y.max()])
        else:  # Does not care / scale with any given axes
            plt.matshow(grid)
            plt.title(title)
            if numbers:
                for (i, j), z in np.ndenumerate(grid):
                    plt.text(j, i, z, ha='center', va='center',
                             bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        if ax is None:
            plt.show()
        return grid

    def add_to_final_position_avg(self, iteration):
        grid = np.zeros([self.nx, self.ny])
        for walker in self.Walkers:
            grid[walker[0]][walker[1]] += 1
        self.avg_final_position = self.avg_final_position + (grid -
                                                             self.avg_final_position)/(iteration+1)
        return self.avg_final_position

    def plot_final_position_avg(self,  title="Avg final position", state=None, ax=None):
        self.plot_grid_heatmap(title=title, state=state, ax=ax)
        return self.avg_final_position

    def simulate_moves(self, moves):
        """Performs X amount of moves and reverting of them, primarily used for testing.

        Args:
            moves (_type_): Number of moves to do (for all members in population)
        """
        for i in range(moves):
            self.move()
            self.revert_illegal_move()
            self.add_to_final_position_avg(i)

    def simulation(self):
        """Main simulation function. Runs the simulation for the number of time steps and iterations as defined for the Class.

        Takes arguments to enable/disable various functions such as recovery and death.

        Returns:
            Coorinate grid showing locations of walkers.
        """
        for self.iteration in range(self.iterations):
            self.Walkers = np.random.randint(
                0, [self.nx,  self.ny], size=(self.N,  2))
            self.State = self.State_0.copy()
            for self.time_step in self.t_all:
                # Infection waves occur at (total time steps)/waves == current time step
                currentTime_div_numWaves = self.time_step % int(
                    (self.t_all[-1]+1)/(self.num_waves+1))
                if self.time_step == self.t_all[0] or self.time_step == self.t_all[-1]:
                    pass
                elif self.infectionWaves and (currentTime_div_numWaves == 0):
                    self.do_infection_wave()
                self.record_states()
                self.infect_susceptible()
                if self.death_rate is not None:
                    self.do_death_check()
                if self.adaptive_recovery_rate is not None:
                    self.update_recovery_rate()
                if self.Recovery == 'on':
                    self.recover_infected()
                self.move()
                self.revert_illegal_move()
                self.add_to_final_position_avg(self.iteration)

    def plot_simulation(self, ax=None, plot_susc=True):
        """Performs the simulation and plots the resulting std and means. 

        Args:
            ax (_type_, optional): Which ax to plot on. Defaults to None, meaning a new plot.
            plot_susc (bool, optional): Whether to plot susceptible elements or not. Defaults to True.
        """
        self.simulation()
        self.crunch_stats()
        if ax is not None:
            ax.plot(self.t_all, self.I_mean, 'r--', label='Infectious_RW')
            ax.fill_between(self.t_all, self.I_mean + self.I_std,
                            self.I_mean - self.I_std, color=(1, 0.9, 0.8, 0.7))
            if plot_susc:
                ax.plot(self.t_all, self.S_mean, 'b--', label='Susceptible_RW')
                ax.fill_between(self.t_all, self.S_mean + self.S_std,
                                self.S_mean - self.S_std, color=(1, 0.9, 0.8, 0.7))

            if self.Recovery == 'on':
                ax.plot(self.t_all, self.R_mean, 'g--', label='Recovered_RW')
                ax.fill_between(self.t_all, self.R_mean + self.R_std,
                                self.R_mean - self.R_std, color=(1, 0.9, 0.8, 0.7))

            if self.death_rate is not None:
                ax.plot(self.t_all, self.D_mean, 'k--', label='Dead_RW')
                ax.fill_between(self.t_all, self.D_mean + self.D_std,
                                self.D_mean - self.D_std, color=(1, 0.9, 0.8, 0.7))
            return
        if plot_susc:
            plt.plot(self.t_all, self.S_mean, 'b--', label='Susceptible_RW')
            plt.fill_between(self.t_all, self.S_mean + self.S_std,
                             self.S_mean - self.S_std, color=(1, 0.9, 0.8, 0.7))
        plt.plot(self.t_all, self.I_mean, 'r--', label='Infectious_RW')
        plt.fill_between(self.t_all, self.I_mean + self.I_std,
                         self.I_mean - self.I_std, color=(1, 0.9, 0.8, 0.7))
        if self.Recovery == 'on':
            plt.plot(self.t_all, self.R_mean, 'g--', label='Recovered_RW')
            plt.fill_between(self.t_all, self.R_mean + self.R_std,
                             self.R_mean - self.R_std, color=(1, 0.9, 0.8, 0.7))

        if self.death_rate is not None:
            plt.plot(self.t_all, self.D_mean, 'k--', label='Dead_RW')
            plt.fill_between(self.t_all, self.D_mean + self.D_std,
                             self.D_mean - self.D_std, color=(1, 0.9, 0.8, 0.7))
        if self.model == 'ODE_SI':
            plt.plot(self.t_all, self.N-self.ODE_Infected(),
                     'b-', label='Susceptible_ODE')
            plt.plot(self.t_all, self.ODE_Infected(),
                     'r-', label='Infectious_ODE')
        elif self.model == 'ODE_SIR':
            self.plot_ODE_SIR()
        plt.xlabel('Time steps')
        plt.ylabel('Population')
        plt.grid()
        plt.legend()
        plt.show()

    def crunch_stats(self):
        self.S_mean = np.mean(self.Susceptible_all, axis=1)
        self.S_std = np.std(self.Susceptible_all, axis=1)
        self.I_mean = np.mean(self.Infectious_all, axis=1)
        self.I_std = np.std(self.Infectious_all, axis=1)
        self.R_mean = np.mean(self.Recovered_all, axis=1)
        self.R_std = np.std(self.Recovered_all, axis=1)
        self.D_mean = np.mean(self.Dead_all, axis=1)
        self.D_std = np.std(self.Dead_all, axis=1)

    def rhs(self, y, t):
        """Right hand side of equation set for solving SIR - ODE model.

        Returns:
            Array of rhs
        """
        S, I, R = y
        return [-self.beta*S*I/self.N, +self.beta*S*I/self.N - I/self.tau, +I/self.tau]

    def SIR_model(self):
        """Using odeint() function to solve SIR equation set.

        Returns:
            Array of number of S-, I-, and R- persons for each time step.
        """
        X0 = [self.N-self.I0-self.R0, self.I0, self.R0]
        self.sol = odeint(self.rhs, X0, self.t_all)
        return

    def calculate_beta(self):
        return np.mean(-(self.Susceptible_all[1:, :]-self.Susceptible_all[:-1, :])*self.N/(self.Susceptible_all[1:, :]*self.Infectious_all[1:, :]))

    def calculate_tau(self):
        step = 1
        return 1/np.mean((self.Recovered_all[step::step, :]-self.Recovered_all[:-step:step, :])/self.Infectious_all[step::step, :])/step

    def print_beta(self):
        self.simulation()
        print(self.calculate_beta())

    def print_tau(self):
        self.simulation()
        print(self.calculate_tau())

    def ODE_Infected(self):
        return self.N/(1+(self.N-self.I0)/self.I0*np.exp(-self.beta*self.t_all))

    def plot_ODE_SIR(self):
        self.SIR_model()
        y_vals = np.transpose(self.sol)
        labels = ['Susceptible_ODE', 'Infectious_ODE', 'Recovered_ODE']
        colors = ['b', 'r', 'g']
        for y_val, color, label in zip(y_vals, colors, labels):
            plt.plot(self.t_all, y_val, color=color, label=label)

    def do_infection_wave(self):
        """Calculates if an susceptible and recovered persons will be infected by a "wave".

        Returns:
            self.State is updated according to results.
        """
        index = np.arange(len(self.State))
        Super = np.c_[index, self.State]
        susceptible = Super[Super[:, 1] == self.Susceptible]
        recovered = Super[Super[:, 1] == self.Recovered]
        for idx in range(len(susceptible)):
            q = np.random.uniform(0, 1)
            if q < self.wave_susc:
                self.State[susceptible[idx, 0]] = self.Infectious
        for idx in range(len(recovered)):
            q = np.random.uniform(0, 1)
            if q < self.wave_reco:
                self.State[recovered[idx, 0]] = self.Infectious

    def do_death_check(self):
        """Calculates if an infected person will die.

        Returns:
            self.State is updated according to results.
        """
        index = np.arange(len(self.State))
        Super = np.c_[index, self.State]
        infectious = Super[Super[:, 1] == self.Infectious]
        for idx in range(len(infectious)):
            q = np.random.uniform(0, 1)
            if q < self.death_rate:
                self.State[infectious[idx, 0]] = self.Dead

    def update_recovery_rate(self):
        """If toggled on, will update recovery rate based on number of deaths and infections.

        Returns:
            self.p_recov is updated according to results.
        """
        self.p_recov = self.p_recov_init + \
            (2*self.deaths*self.adaptive_recovery_rate +
             self.infected*self.adaptive_recovery_rate)/(3*self.N)
