import numpy as np
from scipy.linalg import schur


def hit_n_run(x,A,T):
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


    x = np.array([[478.2635],
                    [464.3478],
                      [770.7708],
                       [54.7719]])


    A = np.array([[-2.43819880e+00, 8.64835984e-01, -8.78624514e-02, 1.18932330e-01],
               [-9.76110184e-01, -5.64887681e-01, -2.76434238e-01, -1.48243176e+00],
               [1.68822239e+00, 6.94140818e-01, 8.81213677e-04, -1.80938297e+00]])

    dim = len(x)
    x = x[:]
    u = np.random.randn(T,dim) # at step t the algorithm will pick a random point
                               # on the line through x and x+u(t,:)
    # u = np.arange(1, T * dim + 1).reshape(T, dim, order='F')
    Au = np.matmul(u, A.T)
    nu = np.sum(u**2, 1)
    l = np.random.randn(T, 1)
    # l = np.arange(1, T + 1)

    for t in np.arange(0, T):
        Ax = np.matmul(A, x)
        ratio = -Ax / np.expand_dims(Au[t,:], 1)
        I = np.where(Au[t,:] > 0)
        mn = np.append(ratio[I], -np.inf).max()
        I = np.where(Au[t,:] < 0)
        mx = np.append(ratio[I], np.inf).min()

        pre = np.matmul(nu[t], np.linalg.norm(x) ** 2 - 1) if nu[t].size > 1 else nu[t] * (np.linalg.norm(x) ** 2 - 1)
        disc = np.matmul(x.T, u[t, :].T) ** 2 - pre
        if disc < 0:
            print('negative disc, probably x is not a feasable point.')
            disc = 0

        hl = (-np.matmul(x.T, u[t,:].T) + np.sqrt(disc)) / nu[t]
        ll = (-np.matmul(x.T, u[t,:].T) - np.sqrt(disc)) / nu[t]

        xx = min(hl, mx)
        nn = max(ll, mn)
        m = xx-nn
        x0 = nn + np.matmul(l[t], m) if m.size > 1 else nn + l[t] * m
        x1_pre = np.expand_dims(u[t,:], 1)
        x1 = np.matmul(x1_pre, x0) if x0.size > 1 else x1_pre * x0
        x = x + x1.reshape(x.shape)
        print('step')
    return x




def KQBC(X_train, Y_train, T, kernel, queries=100, **kwargs):
    """
    Runs the kernelized version of the Query By Committee (QBC) algorithm.

    Inputs:
        X_train - Training instances.
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

    if kernel == 'Linear':
        K = np.matmul(X_train, X_train.T)
    elif kernel == 'Poly':
        K = (np.matmul(X_train, X_train.T) + kwargs['parameter1']) ** kwargs['parameter2']
    elif kernel == 'Gauss':
        nrm = np.sum(X_train ** 2, 1)
        K = np.exp(-(np.tile(nrm, (1, len(Y_train))) - 2 * np.matmul(X_train, X_train.T) +
                     np.tile(nrm.T, (len(Y_train), 1))) / (2 * kwargs['parameter1'] ** 2))
    else:
        print('Unknown kernel')

    coefs = np.empty((Y_train.shape[0],0))
    errors = []
    q_cnt = 0 # query counter

    samp_num = len(Y_train)

    selection = [0] # initialization: select the first sample point
    selected = 1

    coef = np.zeros((len(Y_train), 1))
    coef[0] = Y_train[0] / np.sqrt(K[0, 0])
    preds = np.matmul(K, coef)
    errate = sum(np.multiply(Y_train.squeeze(), preds.squeeze()) <= 0)

    for ii in np.arange(1, samp_num):
        if q_cnt >= queries:
            break

        extension = selection + [ii]
        (s, u) = schur(K[extension][:, extension])
        s = np.diag(s)
        I = (s > tol)
        A = np.matmul(u[:, I], np.diag(s[I] ** -0.5))

        restri = np.matmul(np.matmul(np.diag(Y_train[selection]), K[selection][:, extension]), A)

        co1 = np.matmul(np.linalg.pinv(A), coef[extension])

        # try:
        co2 = hit_n_run(co1, restri, T)
        co1 = hit_n_run(co2, restri, T)
        # except:
        #     print("Noooo")

        pred1 = np.matmul(K[ii][extension], np.matmul(A, co1))
        pred2 = np.matmul(K[ii][extension], np.matmul(A, co2))

        if (pred1 * pred2 <= 0): # the classifiers disagree
            selection = extension
            q_cnt += 1
        if (Y_train[ii] * pred1 >= 0):
            coef[extension] = np.matmul(A, co1)
        else:
            coef[extension] = np.matmul(A, co2)

        coefs = np.append(coefs, coef)
        errors.append(np.sum(np.multiply(Y_train, np.matmul(K, coef).squeeze()) <= 0) / len(Y_train))
        # print((ii, len(selection), errors))

    return selection, coef, errors