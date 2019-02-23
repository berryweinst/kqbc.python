from kqbc import KQBC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets
import matlab.engine
import matplotlib.pyplot as plt
import argparse




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
  parser.add_argument('--hnr_iter', dest='hnr_iter',
                      help='Number of iteration inside the hit and run walk',
                      default=200, type=int)
  parser.add_argument('--samples', dest='samples',
                      help='List of the samples to run the svm and the kqbc',
                      default=[5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40], nargs='+', type=int)
  parser.add_argument('--plot', dest='plot',
                      help='Whether to plot the error comparison',
                      action='store_true')

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
    mu = [0] * args.dim
    cov = np.eye(args.dim)
    X = np.random.multivariate_normal(mu, cov, size=10000)
    y = np.sign(X[:, 0])
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

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                kqbc_selection, coefs, _ = eng.KQBC(matlab.double(X_train.tolist()),
                                                    matlab.double(np.expand_dims(y_train, 1).tolist()), T,
                                                    'Linear', 0, 0, nargout=3)
                # kqbc_selection, coefs, _ = KQBC(X_train, np.expand_dims(y_train, 1), T, 'Linear')
                kqbc_selection = np.array(kqbc_selection, dtype=np.int32).squeeze()
                kqbc_selection -= 1

            for idx, k in enumerate(kl):
                print("K = %d" % (k))
                # SVM classifier
                svclassifier = SVC(kernel='linear')
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
                kqbc_cls = X_train.T @ coef


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
                y_pred_kqbc = np.sign(X_test @ kqbc_cls)
                # if np.sum(y_pred_kqbc != y_test) > np.sum(y_pred_kqbc == y_test):
                #     y_pred_kqbc = -y_pred_kqbc
                kqbc_err = np.sum(y_test * y_pred_kqbc <= 0)/len(y_test)*100
                if t == 0:
                    err_dict_kqbc[d].append(kqbc_err)
                else:
                    err_dict_kqbc[d][idx] += kqbc_err


    if args.plot:
        df = pd.DataFrame({'x': args.samples,
                           'SVM': np.array([i / args.steps for i in err_dict_svm[args.dim]]),
                           'KQBC': np.array([i / args.steps for i in err_dict_kqbc[args.dim]])})
        df.plot(x='x', logy=True, ylim=(0.1, 1e2), title='Generalization error: SVM vs. KQBC')
