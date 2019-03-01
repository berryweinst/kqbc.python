import numpy as np
from scipy.linalg import schur
from sklearn.metrics.pairwise import rbf_kernel
import matlab.engine
import copy



def hit_n_run(x, A ,T):
    """
    Returns a random point using the hit and run algorithm from the
    convex body defined by Ax>=0 and ||x||<=1.
    The random walks begins from the point x which is assumed
    to be an internal point (i.e. satisfies the constraints Ax>=0
    and ||x||<=1. The number of steps the algorithms will perform
    is T.

    Inputs:
    x - A starting point for the random walk. Must be internal
       point.

    A - A set of constraints defining the convex body: Ax>=0

    T - Number of steps to perform.

    Outputs:
    x - A random point in the convex body {x: Ax>=0, ||x||<=1}.

    """


    dim = len(x)
    x = x[:]
    u = np.random.randn(T, dim) # at step t the algorithm will pick a random point
                               # on the line through x and x+u(t,:)
    # u = np.arange(1, T * dim + 1).reshape(T, dim, order='F')
    Au = u @ A.T
    nu = np.sum(u**2, 1)
    l = np.random.rand(T, 1)
    # l = np.arange(1, T + 1)

    for t in np.arange(0, T):
        Ax = A @ x
        ratio = -np.divide(Ax, Au[t, :][np.newaxis].T)
        I = np.where(Au[t,:] > 0)[0]
        mn = np.append(ratio[I], -np.inf).max()
        I = np.where(Au[t, :] < 0)[0]
        mx = np.append(ratio[I], np.inf).min()

        disc = ((x.T @ u[t, :].T) ** 2 - nu[t] * (np.linalg.norm(x) ** 2 - 1))[0]
        if disc < 0:
            print('negative disc, probably x is not a feasable point.')
            disc = 0

        hl = (-(x.T @ u[t, :][np.newaxis].T) + np.sqrt(disc)) / nu[t]
        ll = (-(x.T @ u[t, :][np.newaxis].T) - np.sqrt(disc)) / nu[t]

        xx = min(hl[0], mx)
        nn = max(ll[0], mn)
        x = x + u[t, :][np.newaxis].T * (nn + l[t] * (xx-nn))
    return x




def KQBC(K, Y_train, T):
    """
    Runs the kernelized version of the Query By Committee (QBC) algorithm.

    Inputs:
        K       - Training instances with kernel applied.
        Y_train - Training labels (values +1 and -1).
        T       - Number of random walk steps to make when selecting a
                 random hypothesis.
        kernel  - Type of kernel to be used. Possible values are
                 'Linear', 'Poly' and 'Gauss'.
        parameter1, parameter2 -
                 parameters for the chosen kernel.

    Outputs:
        selection - The instances for which the algorithm queried for
                   label.
        coefs     - The coefficients of the hypotheses used in each of
                   the steps of the algorithm.
        errors    - The training error at each step.
    """

    tol = 1e-10 #tolerance for the factor function

    coefs = []
    errors = []

    samp_num = len(Y_train)

    selection = [0] # initialization: select the first sample point
    selected = 1

    coef = np.zeros((len(Y_train), 1))
    coef[0] = Y_train[0] / np.sqrt(K[0, 0])
    # preds = K @ coef
    # errate = sum(np.multiply(Y_train.squeeze(), preds.squeeze()) <= 0)

    for ii in np.arange(1, samp_num):
        extension = selection + [ii]
        s, u = schur(K[extension][: , extension])
        s = np.diag(s)
        I = np.where(s > tol)[0]
        A = u[:, I] @ np.diag(s[I] ** -0.5)

        y_select_diag = Y_train[selection] if len(selection) == 1 else np.diag(Y_train[selection, 0])
        restri = y_select_diag @ K[selection][: , extension] @ A

        co1 = np.linalg.pinv(A) @ coef[extension]

        # try:
        co2 = hit_n_run(co1, restri, T)
        co1 = hit_n_run(co1, restri, T)

        pred1 = K[ii, extension] @ (A @ co1)
        pred2 = K[ii, extension] @ (A @ co2)

        if (pred1 * pred2 <= 0): # the classifiers disagree
            selection = extension
            if (Y_train[ii] * pred1 >= 0):
                coef[extension] = A @ co1
            else:
                coef[extension] = A @ co2

            coefs.append(copy.deepcopy(coef))
            errors.append(np.sum(np.multiply(Y_train, K @ coef) <= 0) / len(Y_train))
            # print((len(selection), errors[-1]))

    return selection, coefs, errors