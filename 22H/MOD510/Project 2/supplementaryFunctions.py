import matplotlib.pyplot as plt
import scipy.sparse.linalg
import scipy as sc
from numpy.linalg import solve
import numpy as np

A = np.array([[2, 1, 1, 3], [1, 1, 3, 1],
              [1, 4, 1, 1], [1, 1, 2, 2]], float)
b = np.array([1, -3, 2, 1], float)
N = 4
# Gauss-Jordan Elimination
for i in range(1, N):
    fact = A[i:, i-1]/A[i-1, i-1]
    A[i:, ] -= np.outer(fact, A[i-1, ])
    b[i:] -= b[i-1]*fact

# Back substitution
sol = np.zeros(N, float)
sol[N-1] = b[N-1]/A[N-1, N-1]
for i in range(2, N+1):
    sol[N-i] = (b[N-i]-np.dot(A[(N-i), ], sol))/A[N-i, N-i]

print(f'sol = {sol}')


# # Back substitution - for loop
# sol = np.zeros(N, float)
# for i in range(N-1, -1, -1):
#     sol[i] = b[i]
#     for j in range(i+1, N):
#         sol[i] -= A[i][j]*sol[j]
#     sol[i] /= A[i][i]

# LU Decomposition
x = solve(A, b)
print(f'x   = {x}')


def solve_jacobi(A, b, x=-1, w=1, max_iter=1000, EPS=1e-6):
    """
    Solves the linear system Ax=b using the Jacobian method, stops if
    solution is not found after max_iter or if solution changes less
    than EPS
    """
    if (x == -1):  # default guess
        x = np.zeros(len(b))

    D = np.diag(A)
    R = A-np.diag(D)
    eps = 1
    x_old = x
    iter = 0
    w = 0.1
    while (eps > EPS and iter < max_iter):
        iter += 1
        x = w*(b-np.dot(R, x_old))/D + (1-w)*x_old
        eps = np.sum(np.abs(x-x_old))
        x_old = x
    print('found solution after ' + str(iter) + ' iterations')
    return x


def solve_GS(A, b, x=-1, max_iter=1000, EPS=1e-6):
    """
    Solves the linear system Ax=b using the Gauss-Seidel method, stops if
    solution is not found after max_iter or if solution changes less
    than EPS
    """
    if (x == -1):  # default guess
        x = np.zeros(len(b))

    D = np.diag(A)
    R = A-np.diag(D)
    eps = 1
    iter = 0
    while (eps > EPS and iter < max_iter):
        iter += 1
        eps = 0.
        for i in range(len(x)):
            tmp = x[i]
            x[i] = (b[i] - np.dot(R[i, :], x))/D[i]
            eps += np.abs(tmp-x[i])
    print('found solution after ' + str(iter) + ' iterations')
    return x


def OLS(x, y):
    # returns regression coefficients
    # in ordinary least square
    # x: observations
    # y: response
    # R^2: R-squared
    n = np.size(x)  # number of data points

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    # R^2
    y_pred = b_0 + b_1*x
    S_yy = np.sum(y*y) - n*m_y*m_y
    y_res = y-y_pred
    S_res = np.sum(y_res*y_res)

    return (b_0, b_1, 1-S_res/S_yy)


def OLSM(x, y):
    # returns regression coefficients
    # in ordinary least square using solve function
    # x: observations
    # y: response

    XT = np.array([np.ones(len(x)), x], float)
    X = np.transpose(XT)
    B = np.dot(XT, X)
    C = np.dot(XT, y)
    return solve(B, C)


#
central_difference = False
# set simulation parameters
h = 0.25
L = 1.0
n = int(round(L/h))
Tb = 25  # rhs
sigma = 100
k = 1.65
beta = sigma*L**2/k
y = np.arange(n+1)*h


def analytical(x):
    return beta*(1-x*x)/2+Tb


def tri_diag(a, b, c, k1=-1, k2=0, k3=1):
    """ a,b,c diagonal terms
    default k-values for 4x4 matrix:
    | b0 c0 0 0 |
    | a0 b1 c1 0 |
    | 0 a1 b2 c2|
    | 0 0 a2 b3|
    """
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)


# defina a, b and c vector
a = np.ones(n-1)
b = ...
c = ...
if central_difference:
    c[0] = ...
else:
    b[0] = ...
A = tri_diag(a, b, c)
print(A)  # view matrix - compare with N=4 to make sure no bugs
# define rhs vector
d = ...
# rhs boundary condition
d[-1] = ...
Tn = np.linalg.solve(A, d)
print(Tn)

# Sparse matrix
# right hand side
# rhs vector
d = np.repeat(-h*h*beta, n)
# rhs - constant temperature
Tb = 25
d[-1] = d[-1]-Tb
# Set up sparse matrix
diagonals = np.zeros((3, n))
diagonals[0, :] = 1
diagonals[1, :] = -2
diagonals[2, :] = 1
# No flux boundary condition
diagonals[2, 1] = 2
A_sparse = sc.sparse.spdiags(diagonals, [-1, 0, 1], n, n, format='csc')
# to view matrix - do this and check that it is correct!
print(A_sparse.todense())
# solve matrix
Tb = sc.sparse.linalg.spsolve(A_sparse, d)
# if you like you can use timeit to check the efficiency
# %timeit sc.sparse.linalg.spsolve( ... )

# Solve non-linear equations


def iterative(x, g, prec=1e-8, MAXIT=1000):
    '''Approximate solution of x=g(x) by fixed point iterations.
    x : starting point for iterations
    eps : desired precision
    Returns x when x does not change more than prec
    and number of iterations MAXIT are not exceeded'''

    eps = 1
    n = 0
    while eps > prec and n < MAXIT:
        x_next = g(x)
        eps = np.abs(x-x_next)
        x = x_next
        n += 1
        if (np.isinf(x)):
            print('Quitting .. maybe bad starting point?')
            return x
    if (n < MAXIT):
        print('Found solution: ', x, ' After ', n, 'iterations')
    else:
        print('Max number of iterations exceeded')
    return x


def dvdwEOS(nu, t, p):
    return (1+8*t/(p+3/nu**2))/3


def iterative(x, g, *args, prec=1e-8):
    MAX_ITER = 1000
    eps = 1
    n = 0
    while eps > prec and n < MAX_ITER:
        x_next = g(x, *args)
        eps = np.abs(x-x_next)
        x = x_next
        n += 1
    print('Number of iterations: ', n)
    return x


iterative(1, dvdwEOS, 1.2, 1.5)


def bisection(f, a, b, prec=1e-8, MAXIT=100):
    '''Approximate solution of f(x)=0 on interval [a,b] by bisection.
    f : f(x)=0.
    a,b : brackets the root f(a)*f(b) has to be negative
    eps : desired precision
    Returns the midpoint when it is closer than eps to the root,
    unless MAXIT are not exceeded
    '''
    if f(a)*f(b) >= 0:
        print('You need to bracket the root, f(a)*f(b) >= 0')
        return None
    an = a
    bn = b
    cn = 0.5*(an + bn)
    c_old = cn - 10*prec
    n = 0
    while np.abs(cn-c_old) >= prec and n < MAXIT:
        c_old = cn
        f_cn = f(cn)
        if f(an)*f_cn < 0:
            bn = cn
        elif f(bn)*f_cn < 0:
            an = cn
        elif f_cn == 0:
            print('Found exact solution ', cn,
                  ' after ', n, 'iterations')
            return cn
        else:
            print('Bisection method fails.')
            return None
        cn = 0.5*(an+bn)
        n += 1
    if n < MAXIT-1:
        print('Found solution ', cn, ' after ', n, 'iterations')
        return cn
    else:
        print('Max number of iterations: ', MAXIT, ' reached.')
        print('Try to increase MAXIT or decrease prec')
        print('Returning best guess, value of function is : ', f_cn)
    return None


def newton(f, x, prec=1e-8, MAXIT=500):
    '''Approximate solution of f(x)=0 by Newtons method.
    The derivative of the function is calculated numerically
    f : f(x)=0.
    x : starting point
    eps : desired precision
    Returns x when it is closer than eps to the root,
    unless MAX_ITERATIONS are not exceeded
    '''
    MAX_ITERATIONS = MAXIT
    x_old = x
    h = 1e-4
    for n in range(MAX_ITERATIONS):
        x_new = x_old - 2*h*f(x_old)/(f(x_old+h)-f(x_old-h))
        if (abs(x_new-x_old) < prec):
            print('Found solution:', x_new,
                  ', after:', n, 'iterations.')
            return x_new
        x_old = x_new
    print('Max number of iterations: ', MAXIT, ' reached.')
    print('Try to increase MAXIT or decrease prec')
    print('Returning best guess, value of function is: ', f(x_new))
    return x_new


def gradient_descent(f, x, df, g=.001, prec=1e-8, MAXIT=10):
    '''Minimize f(x) by gradient descent.
    f : min(f(x))
    x : starting point
    df : derivative of f(x)
    g : learning rate
    prec: desired precision
    Returns x when it is closer than eps to the root,
    unless MAXIT are not exceeded
    '''
    x_old = x
    for n in range(MAXIT):
        plot_regression_line(x_old)
        x_new = x_old - g*df(x_old)
        if (abs(np.max(x_new-x_old)) < prec):
            print('Found solution:', x_new,
                  ', after:', n, 'iterations.')
            return x_new
        x_old = x_new
    print('Max number of iterations: ', MAXIT, ' reached.')
    print('Try to increase MAXIT or decrease prec')
    print('Returning best guess, value of function is: ', f(x_new))
    return x_new


x_obs_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_obs_ = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])


def plot_regression_line(b, x=x_obs_, y=y_obs_):
    global N_
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30, label="data")
    # predicted response vector
    y_pred = b[0] + b[1]*x
    # plotting the regression line
    if (len(b) > 1):
        # plt.plot(x, y_pred, color = "g", label = "R-squared = {0:.3f}".format(b[2]))
        plt.plot(x, y_pred, color="g", label="iteration:" + str(N_) +
                 ", (b[0],b[1])= ({0:.3f}".format(b[0]) + ", {0:.3f})".format(b[1]))
        plt.legend()
    else:
        plt.plot(x, y_pred, color="g")
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    # plt.savefig('../fig-nlin/stdec'+str(N_)+'.png', bbox_inches='tight',transparent=True)
    N_ = N_+1
    # function to show plot
    plt.show()


def Jacobian(x, f, dx=1e-5):
    N = len(x)
    x0 = np.copy(x)
    f0 = f(x)
    J = np.zeros(shape=(N, N))
    for j in range(N):
        x[j] = x[j] + dx
        for i in range(N):
            J[i][j] = (f(x)[i]-f0[i])/dx
        x[j] = x[j] - dx
    return J


def newton_rapson(x, f, J=None, jacobian=False, prec=1e-8, MAXIT=100):
    '''Approximate solution of f(x)=0 by Newtons method.
    The derivative of the function is calculated numerically
    f : f(x)=0.
    J : Jacobian
    x : starting point
    eps : desired precision
    Returns x when it is closer than eps to the root,
    unless MAX_ITERATIONS are not exceeded
    '''
    MAX_ITERATIONS = MAXIT
    x_old = np.copy(x)
    for n in range(MAX_ITERATIONS):
        plot_regression_line(x_old)
        if not jacobian:
            J_ = Jacobian(x_old, f)
        else:
            J_ = J(x_old)
        z = np.linalg.solve(J_, -f(x_old))
        x_new = x_old+z
        if (np.sum(abs(x_new-x_old)) < prec):
            print('Found solution:', x_new,
                  ', after:', n, 'iterations.')
            return x_new
        x_old = np.copy(x_new)
    print('Max number of iterations: ', MAXIT, ' reached.')
    print('Try to increase MAXIT or decrease prec')
    print('Returning best guess, value of function is: ', f(x_new))
    return x_new


def S(b, x=x_obs_, y=y_obs_):
    return np.sum((y-b[0]-b[1]*x)**2)


def dS(b, x=x_obs_, y=y_obs_):
    return np.array([-2*np.sum(y-b[0]-b[1]*x),
                     -2*np.sum((y-b[0]-b[1]*x)*x)])


def J(b, x=x_obs_, y=y_obs_):
    N = len(b)
    J = np.zeros(shape=(N, N))
    xs = np.sum(x)
    J[0][0] = 2*len(x)
    J[0][1] = 2*xs
    J[1][0] = 2*xs
    J[1][1] = 2*np.sum(x*x)
    return J


N_ = 0
print('Gradient ')
b = np.array([0, 0])
