import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import pdb

def norm1D(my,Sgm,x):

    [n,d]=np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(0, n):
        p[i] = 1 / (np.sqrt(2 * np.pi) * Sgm) * \
            np.exp(-1 / 2 * np.square((x[i] - my)) / (np.square(Sgm)))

    return p

def norm2D(my, Sgm, x1, x2):
    x1v, x2v = np.meshgrid(x1, x2)
    [n,d]=np.shape(x1v)
    l=2
    p = np.zeros(np.shape(x1v))
    for i in np.arange(0, n):
        for j in np.arange(0, d):
            x = np.array([x1v[i,j], x2v[i,j]])
            p[i, j] = (2 * np.pi)**(-l/2) * np.power(np.linalg.det(Sgm),-1/2) * np.exp(-1 / 2 * np.linalg.multi_dot([ x - my , np.linalg.inv(Sgm) , (x - my).reshape(-1,1)]))
            
    return p, x1v, x2v

def classplot(g, x1, x2, gnan=0, discr='', gsv={'gsv':0, 'figstr':'fig_'}):

    if discr == 'pxw':
        zlb = '$p(\mathbf{x}|\omega_i)$'
    elif discr == 'Pwpxw':
        zlb = '$P(\omega_i)p(\mathbf{x}|\omega_i)$'
    elif discr == 'Pwx':
        zlb = '$P(\omega_i | \mathbf{x})$'
    else:
        zlb = ''

    eps = sys.float_info.epsilon
    X1, X2 = np.meshgrid(x1, x2)
    M = len(g)
    col = ['r', 'b', 'g', 'y', 'k']
    fig = plt.figure(1)
    fig.set_facecolor('gray')
    ax = fig.add_subplot(projection='3d')
    obj = []
    mx = []
    for i in range(0, M):
        if gnan == 0:
            G = g[i]
        else:
            G = np.copy(g[i]) + 0 * 1e100 * eps
            NN = (G < np.amax(g, axis=0)) * 1
            NN = NN.astype(float)
            G = np.copy(g[i])
            np.putmask(G, NN == 1, np.nan)
        obj = ax.plot_surface(X1, X2, G, facecolor=col[i])
        mx.append(g[i].max())
    zm = np.around(1.2 * max(mx), decimals=2)
    xt = (np.linspace(x1[0, 0], x1[-1, 0], 5))
    yt = (np.linspace(x2[0, 0], x2[-1, 0], 5))
    zt = (np.linspace(0, zm, 4))
    zt = np.around(zt, decimals=3)
    ax.set(xticks=xt, yticks=yt, zticks=zt)
    ax.set(xlim=(x1[0, 0], x1[-1, 0]), ylim=(x2[0, 0], x2[-1, 0]))
    ax.set(xlabel='$x_1$', ylabel='$x_2$', zlabel=zlb)
    if gsv['gsv']:
        plt.savefig(gsv['figstr'] + discr + '.png')

    plt.show()