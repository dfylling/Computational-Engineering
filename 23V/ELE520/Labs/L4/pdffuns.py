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



def plot_P(P, x1v, x2v):
    a = P[0]
    b = P[1]
    mask = b<a
    w1 = a.copy()*np.nan
    w2 = a.copy()*np.nan

    w1[mask] = a[mask]
    w2[~mask] = b[~mask]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x1v, x2v, w1, cmap='autumn')
    ax.plot_surface(x1v, x2v, w2, cmap='winter')

def parzen(X, h1, x1, x2):
    d = len(X)
    P = [] 
    for i in range(d):
        g = np.zeros((len(x1),len(x2)))
        N = X[i].shape[1]
        hn = h1 * np.power(N,-0.5)
        Sgm = hn**2 * np.identity(d)

        for k in range(N):
            my = np.array([X[i][0][k],X[i][1][k]])
            p, x1v, x2v = norm2D(my, Sgm, x1, x2)
            g = g + p/N
        P.append(g)
    return P, x1v, x2v

def knn2D(X, kn, x1, x2):
    x1v, x2v = np.meshgrid(x1, x2)
    [n,d]=np.shape(x1v)
    [D, N] = np.shape(X)
    p = np.zeros(np.shape(x1v))
    for i in np.arange(0, n):
        for j in np.arange(0, d):
            x = np.array([x1v[i,j], x2v[i,j]])
            x = np.tile(x,(N, 1)).T
            R = np.sqrt(np.sum(np.square(X-x),0))
            V = np.sort(R)[kn-1]
            p[i, j] = kn / N / V            
    return p, x1v, x2v

def plot_P_post(P, pw, x1v, x2v):
    Pw1x = P[0] * pw[0] / (P[0] * pw[0] + P[1] * pw[1])
    Pw2x = P[1] * pw[1] / (P[0] * pw[0] + P[1] * pw[1])
    fig = plt.figure()
    
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x1v, x2v, Pw1x, cmap='autumn')
    ax.plot_surface(x1v, x2v, Pw2x, cmap='winter')