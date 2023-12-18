from anh import solveODE, genericPlot

import numpy as np
import math


def f(c, *args):
    """Given function/formula for change in consentration

    Args:
        c (float): Current tank consentration
        n (int): Number of tanks
        tau (float): Constant for calculation based on physical properties of system

    Returns:
        (float): Change in tank consentrations
    """
    [c_inj, n, tau] = args
    ret = np.zeros_like(c)
    ret[0] = c_inj - c[0]
    for i in range(1, len(c)):
        ret[i] = c[i-1] - c[i]
    return ret * n / tau


def C0_analytical(t): return math.exp(-t/tau)
def C1_analytical(t): return (2*t/tau)*math.exp(-2*t/tau)
def C2_analytical(t): return (9*(t**2)/(2*(tau**2)))*math.exp(-3*t/tau)


tau = 2
y0 = np.array([1, 0, 0])
c_inj = 0
t0 = 0
tf = 10
dt = 0.01

# Exercise 1 part 2, constant tau and injection
n = 1
y_n1_euler = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="euler")
y_n1_rk2 = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="rk2")
y_n1_rk4 = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="rk4")
n = 2
y_n2_euler = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="euler")
y_n2_rk2 = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="rk2")
y_n2_rk4 = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="rk4")
n = 3
y_n3_euler = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="euler")
y_n3_rk2 = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="rk2")
y_n3_rk4 = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="rk4")

[points, _] = np.shape(y_n1_euler)

numPoints = 10000
t = np.linspace(0, 10, numPoints)
t_numeric = np.linspace(0, 10, points)

tank1_analytical = np.zeros_like(t)
tank2_analytical = np.zeros_like(t)
tank3_analytical = np.zeros_like(t)

for i in range(numPoints):
    tank1_analytical[i] = C0_analytical(t[i])
    tank2_analytical[i] = C1_analytical(t[i])
    tank3_analytical[i] = C2_analytical(t[i])
if n == 1:
    cur_analytical = tank1_analytical
if n == 2:
    cur_analytical = tank2_analytical
if n == 3:
    cur_analytical = tank3_analytical
# matplotlib %inline

title = f'Euler\nConcentration in tanks over time, with n={n} & dt={dt}'
xlabel = 'Time'
ylabel = 'Concentration'
x_vals = [t,
          t_numeric, t_numeric, t_numeric]
cols = ['tab:blue', 'tab:purple', 'tab:pink', 'tab:orange']
points = ['-', ':', ':', ':']
linewidths = ['2', '2', '2', '2']

title = f'Euler\nConcentration in tanks over time, with n=1 & dt={dt}'
y_vals = [tank1_analytical,
          y_n1_euler[:, 0], y_n1_euler[:, 1], y_n1_euler[:, 2]]
labels = [f"Analytical 1",
          "Numeric 1", "Numeric 2", "Numeric 3"]
genericPlot(title, xlabel, ylabel, x_vals, y_vals,
            labels, cols, points, linewidths)

title = f'Rk2\nConcentration in tanks over time, with n=2 & dt={dt}'
y_vals = [tank2_analytical,
          y_n2_euler[:, 0], y_n2_euler[:, 1], y_n2_euler[:, 2]]
labels = [f"Analytical 2",
          "Numeric 1", "Numeric 2", "Numeric 3"]
genericPlot(title, xlabel, ylabel, x_vals, y_vals,
            labels, cols, points, linewidths)

title = f'Rk4\nConcentration in tanks over time, with n=3 & dt={dt}'
y_vals = [tank3_analytical,
          y_n3_euler[:, 0], y_n3_euler[:, 1], y_n3_euler[:, 2]]
labels = [f"Analytical 3",
          "Numeric 1", "Numeric 2", "Numeric 3"]
genericPlot(title, xlabel, ylabel, x_vals, y_vals,
            labels, cols, points, linewidths)
# Part 3

tau = 2
y0 = np.array([1, 0, 0])
c_inj = 0
t0 = 0
tf = 2
dt = 0.01
n = 1
depth = 6
timeOne_values = np.zeros([depth, 3])
for i in range(1, depth):
    dt = 10**(-i)
    y_euler = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="euler")
    y_rk2 = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="rk2")
    y_rk4 = solveODE(f, y0, t0, tf, c_inj, n, tau, dt=dt, solver="rk4")
    # Store values at time = 1
    timeOne_index = int(1/dt)
    timeOne_values[i, 0] = y_euler[timeOne_index, n-1]
    timeOne_values[i, 1] = y_rk2[timeOne_index, n-1]
    timeOne_values[i, 2] = y_rk4[timeOne_index, n-1]
    # Find difference at this step size to analytical solution
    timeOne_values[i, :] = timeOne_values[i, :] - C0_analytical(1)
print(f'Errors with: dt = 10^-row\n Euler | Rk2 | Rk4\n{timeOne_values}')
print(f'The analytical value at T=1:\n{C0_analytical(1)}')
