from kqbc import KQBC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets
import matlab.engine
import matplotlib.pyplot as plt
import argparse
# from mnist.utils import prepare_mnist_data, get_rbf_data
from sklearn.metrics.pairwise import rbf_kernel
import math


def get_rbf_data(lower, upper, num, num_dims):
    X = np.random.uniform(lower, upper, size=(num,num_dims))
    Y = []
    for x1,x2 in X:
        if x2 < np.sin(10*x1)/5 + 0.3 or ((x2 - 0.8)**2 + (x1 - 0.5)**2) < 0.15**2:
            Y.append(1)
        else:
            Y.append(-1)
    return X, np.asarray(Y)



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Run KQBC algorithm')
  parser.add_argument('--steps', dest='steps',
                      help='Number of steps to run the same experiment in order to take the mean of the results',
                      default=10, type=int)
  parser.add_argument('--dim', dest='dim',
                      help='The dimentionality of the data to synthetically generate',
                      default=3, type=int)
  parser.add_argument('--kernel', dest='kernel',
                      help='Kernel type for the data loading and the KQBC',
                      default='Linear', type=str)
  parser.add_argument('--hnr_iter', dest='hnr_iter',
                      help='Number of iteration inside the hit and run walk',
                      default=200, type=int)
  parser.add_argument('--samples', dest='samples',
                      help='List of the samples to run the svm and the kqbc',
                      default=[5, 8, 10, 13, 18, 20, 23, 28, 30, 33, 38,  40], #, 50, 60, 70, 80, 90, 100, 110, 120, 130],
                      nargs='+', type=int)
  parser.add_argument('--plot', dest='plot',
                      help='Whether to plot the error comparison',
                      action='store_true')
  parser.add_argument('--gamma', dest='gamma',
                      help='rbf gamma value',
                      default=3.0, type=float)

  args = parser.parse_args()
  return args



if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    T = args.hnr_iter
    param_dict = {args.dim: args.samples}

    err_dict_svm = {}
    err_dict_kqbc = {}
    err_dict_svm_kqbc = {}
    eng = matlab.engine.start_matlab()

    # creat data
    if args.kernel == 'Linear':
        mu = [0] * args.dim
        cov = np.eye(args.dim)
        X = np.random.multivariate_normal(mu, cov, size=5000)
        y = np.sign(X[:, 0])
    elif args.kernel == 'Gauss':
        X, y = get_rbf_data(0, 1, 10000, 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # XX, y = gen_linear_sep_data(n_samples=10000, n_features=d)
    # X = np.ones((XX.shape[0], XX.shape[1]+1))
    # X[:, :-1] = XX
    # X, y = eng.GenData(float(d), nargout=2)
    # X = np.array(X, dtype=np.float32)
    # y = np.array(y).squeeze()

    for t in range(args.steps):
        print("Step: %d - running SVM and KQBC" % (t))
        for d, kl in param_dict.items():
            if d not in err_dict_svm.keys():
                err_dict_svm[d] = []
                err_dict_kqbc[d] = []
                err_dict_svm_kqbc[d] = []

                param1, param2 = (0, 0) if args.kernel == 'Linear' else (math.sqrt(1/(2*args.gamma)), 0.0)
                kqbc_selection, coefs = eng.KQBC(matlab.double(X_train.tolist()),
                                                    matlab.double(np.expand_dims(y_train, 1).tolist()), T,
                                                    args.kernel, param1, param2, nargout=2)
                # kqbc_selection, coefs, _ = KQBC(X_train, np.expand_dims(y_train, 1), T, 'Linear')
                kqbc_selection = np.array(kqbc_selection, dtype=np.int32).squeeze()
                kqbc_selection -= 1

            for idx, k in enumerate(kl):
                print("K = %d" % (k))
                # SVM classifier
                svm_kernel_name = 'linear' if args.kernel == 'Linear' else 'rbf'
                svclassifier = SVC(kernel=svm_kernel_name, gamma=args.gamma)
                # svclassifier = SVC(kernel=svm_kernel_name)
                if np.abs(np.sum(y_train[:k])) == k:
                    svclassifier.fit(X_train[k:k+k,:], y_train[k:k+k])
                else:
                    svclassifier.fit(X_train[:k,:], y_train[:k])
                y_pred = svclassifier.predict(X_test)
                svm_err = np.sum(y_pred * y_test <= 0)/len(y_test)*100
                if t == 0:
                    err_dict_svm[d].append(svm_err)
                else:
                    err_dict_svm[d][idx] += svm_err

                # Run KQBC matlab code
                coef = np.array(coefs, dtype=np.float32)[:, k-1]
                # K = rbf_kernel(X_train, X_train, gamma=param1)
                # kqbc_cls = X_train.T @ coef if args.kernel == 'Linear' else K @ coef


                # SVM with KQBC selections
                # kqbc_svm = SVC(kernel='linear')
                # kqbc_svm.fit(X_train[kqbc_selection[:k],:], y_train[kqbc_selection[:k]])
                # y_pred_svm_kqbc = kqbc_svm.predict(X_test)
                # kqbc_svm_err = np.sum(y_pred_svm_kqbc * y_test <= 0) / len(y_test) * 100
                # if t == 0:
                #     err_dict_svm_kqbc[d].append(kqbc_svm_err)
                # else:
                #     err_dict_svm_kqbc[d][idx] += kqbc_svm_err


                # Dot prod with KQBC classifier
                y_pred_kqbc = np.sign(rbf_kernel(X_test, X_train, gamma=args.gamma) @ coef)
                # y_pred_kqbc = np.sign(X_test @ kqbc_cls)
                # if np.sum(y_pred_kqbc != y_test) > np.sum(y_pred_kqbc == y_test):
                #     y_pred_kqbc = -y_pred_kqbc
                kqbc_err = np.sum(y_test * y_pred_kqbc <= 0)/len(y_test)*100
                print("KQBC error = %.2f \t SVM error = %.2f" % (kqbc_err, svm_err))
                if t == 0:
                    err_dict_kqbc[d].append(kqbc_err)
                else:
                    err_dict_kqbc[d][idx] += kqbc_err


    if args.plot:
        df = pd.DataFrame({'x': args.samples,
                           'SVM': np.array([i / args.steps for i in err_dict_svm[args.dim]]),
                           'KQBC': np.array([i / args.steps for i in err_dict_kqbc[args.dim]])})
        df.plot(x='x', logy=True, ylim=(0.1, 1e2), title='Generalization error: SVM vs. KQBC (data dim = %d)' % (args.dim))
