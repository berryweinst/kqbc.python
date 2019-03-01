from kqbc import KQBC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets
import matlab.engine
import matplotlib.pyplot as plt
import argparse
from mnist.utils import prepare_mnist_data, select_one_class_vs_all
from sklearn.metrics.pairwise import rbf_kernel
import math
import random

# random.seed(1234)
# np.random.seed(1234)


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
                      # default=[5, 8, 10, 13, 18, 20, 23, 28, 30, 33, 38,  40],
                      default=[5, 10, 20, 30, 40,  50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
                      nargs='+', type=int)
  parser.add_argument('--plot', dest='plot',
                      help='Whether to plot the error comparison',
                      action='store_true')
  parser.add_argument('--gamma', dest='gamma',
                      help='rbf gamma value',
                      default=3.0, type=float)
  parser.add_argument('--param_c', dest='param_c',
                      help='rbf C value',
                      default=1.0, type=float)

  args = parser.parse_args()
  return args



if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    T = args.hnr_iter
    param_dict = {args.dim: args.samples}

    err_dict_svm = np.zeros((len(args.samples), args.steps))
    err_dict_kqbc = np.zeros((len(args.samples), args.steps))
    err_dict_svm_kqbc = np.zeros((len(args.samples), args.steps))
    # eng = matlab.engine.start_matlab()

    # create data
    if args.kernel == 'Linear':
        mu = [0] * args.dim
        cov = np.eye(args.dim)
        X = np.random.multivariate_normal(mu, cov, size=10000)
        y = np.sign(X[:, 0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    elif args.kernel == 'Gauss':
        X_train, y_train, X_test, y_test = prepare_mnist_data()
        X_train, y_train = select_one_class_vs_all(X_train, y_train)
        X_test, y_test = select_one_class_vs_all(X_test, y_test, samples=200)
        X_train = X_train / 255.0
        X_test = X_test / 255.0


    param1, param2 = (0, 0) if args.kernel == 'Linear' else (args.gamma, 0.0)

    if args.kernel == 'Linear':
        K = X_train @ X_train.T
    elif args.kernel == 'Poly':
        K = (X_train @ X_train.T + param1) ** param2
    elif args.kernel == 'Gauss':
        K = rbf_kernel(X_train, X_train, gamma=param1)
        # nrm = np.sum(X_train ** 2, 1)
        # K = np.exp(-(np.tile(nrm, (1, len(Y_train))) - 2 * np.matmul(X_train, X_train.T) +
        #              np.tile(nrm.T, (len(Y_train), 1))) / (2 * kwargs['parameter1'] ** 2))
    else:
        print('Unknown kernel')


    # XX, y = gen_linear_sep_data(n_samples=10000, n_features=d)
    # X = np.ones((XX.shape[0], XX.shape[1]+1))
    # X[:, :-1] = XX
    # X, y = eng.GenData(float(d), nargout=2)
    # X = np.array(X, dtype=np.float32)
    # y = np.array(y).squeeze()

    for t in range(args.steps):
        print("Step: %d - running SVM and KQBC" % (t))

        # kqbc_selection, coefs = eng.KQBC(matlab.double(X_train.tolist()),
        #                                     matlab.double(np.expand_dims(y_train, 1).tolist()), T,
        #                                     args.kernel, param1, param2, nargout=2)
        kqbc_selection, coefs, _ = KQBC(K, y_train[:, np.newaxis], T)
        # kqbc_selection = np.array(kqbc_selection, dtype=np.int32).squeeze()
        # kqbc_selection -= 1

        for idx, k in enumerate(param_dict[args.dim]):
            print("K = %d" % (k))
            # SVM classifier
            svm_kernel_name = 'linear' if args.kernel == 'Linear' else 'rbf'
            svclassifier = SVC(kernel=svm_kernel_name, gamma=args.gamma if args.kernel == 'Gauss' else 'auto',
                               C=args.param_c)
            # svclassifier = SVC(kernel=svm_kernel_name)

            svm_idx = random.sample(range(0, y_train.shape[0]), k)
            while np.sum(y_train[svm_idx]) in [k, -k]:
                svm_idx = random.sample(range(0, y_train.shape[0]), k)
            svclassifier.fit(X_train[svm_idx,:], y_train[svm_idx])
            y_pred = svclassifier.predict(X_test)
            svm_err = np.sum(y_pred * y_test <= 0)/len(y_test)*100
            err_dict_svm[idx, t] = svm_err

            # coef = np.array(coefs, dtype=np.float32)[:, k-1]
            # K = rbf_kernel(X_train, X_train, gamma=param1)



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
            coef = coefs[k - 1]
            if args.kernel == 'Linear':
                kqbc_cls = X_train.T @ coef
                y_pred_kqbc = np.sign(X_test @ kqbc_cls)
            elif args.kernel == 'Gauss':
                y_pred_kqbc = np.sign(rbf_kernel(X_test, X_train, gamma=args.gamma) @ coef)

            kqbc_err = np.sum(y_test * y_pred_kqbc.squeeze() <= 0)/len(y_test)*100
            print("KQBC error = %.2f \t SVM error = %.2f" % (kqbc_err, svm_err))
            err_dict_kqbc[idx ,t] = kqbc_err




    if args.plot:
        df = pd.DataFrame({'x': args.samples,
                           'SVM': err_dict_svm.mean(axis=1),
                           'KQBC': err_dict_kqbc.mean(axis=1)})
        df.plot(x='x', logy=True, ylim=(0.1, 1e2), yerr=[err_dict_svm.std(axis=1), err_dict_kqbc.std(axis=1)],
                title='Generalization error: SVM vs. KQBC (data dim = %d)' % (
                    args.dim) if args.krenel == 'Linear' else 'Mnist generalization error: SVM vs. KQBC')
