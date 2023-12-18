import numpy as np
import scipy


def denseSolver(A, b):
    """
    Use dense solver which is a direct solver for a dense system 'Ax=b'.
    :param A: A NumPy 2D array of a tridiagonal matrix-
    :param b: A NumPy 1D array of the b vector of the equation 'Ax=b'.
    :return: A NumPy 1D array of the x vector of the equation 'Ax=b', 
    """
    return np.linalg.solve(A, b)


def sparseSolver(A, b):
    """
    Use sparse solver which accelerates solving speed when many of the elements is 0. Solves systems in form 'Ax=b'.
    :param A: A NumPy 2D array of a tridiagonal matrix-
    :param b: A NumPy 1D array of the b vector of the equation 'Ax=b'.
    :return: A NumPy 1D array of the x vector of the equation 'Ax=b', 
    """
    return scipy.sparse.linalg.spsolve(scipy.sparse.csr_matrix(A), b)


def thomasAlgorithmSolver(A, b):
    """
    Interface function for the thomas_algorithm function below.
    :param A: A NumPy 2D array of a tridiagonal matrix.
    :param b: A NumPy 1D array of the b vector of the equation 'Ax=b'.
    :return: A NumPy 1D array of the x vector of the equation 'Ax=b', 
    """
    [numRows, numCols] = np.shape(A)
    lowerDiag = np.r_[0,   np.array([A[i+1, i] for i in range(numRows-1)])]
    diag = np.array([A[i,  i] for i in range(numRows)])
    upperDiag = np.r_[np.array([A[i,  i+1] for i in range(numRows-1)]),    0]
    return thomas_algorithm(lowerDiag, diag, upperDiag, b)


def thomas_algorithm(l, d, u, r):
    """
    Solves a tridiagonal linear system of equations with the Thomas-algorithm.
    The code is based on pseudo-code from the following reference:
    Cheney, E. W., & Kincaid, D. R.
    Numerical mathematics and computing, 7th edition,
    Cengage Learning, 2013.
    IMPORTANT NOTES:
    - This function modifies the contents of the input vectors l, d, u and rhs.
    - For Numba to work properly, we must input NumPy arrays, and not lists.
    :param l: A NumPy array containing the lower diagonal (l[0] is not used).
    :param d: A NumPy array containing the main diagonal.
    :param u: A NumPy array containing the upper diagonal (u[-1] is not used).
    :param r: A NumPy array containing the system right-hand side vector.
    :return: A NumPy array containing the solution vector.
    """
    # Allocate memory for solution
    solution = np.zeros_like(d)
    n = len(solution)
    # Forward elimination
    for k in range(1, n):
        xmult = l[k] / d[k-1]
        d[k] = d[k] - xmult*u[k-1]
        r[k] = r[k] - xmult*r[k-1]
    # Back-substitution
    solution[n-1] = r[n-1] / d[n-1]
    for k in range(n-2, -1, -1):
        solution[k] = (r[k]-u[k]*solution[k+1])/d[k]
    return solution
