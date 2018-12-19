# from kqbc import KQBC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
                      default=5, type=int)
  parser.add_argument('--hnr_iter', dest='hnr_iter',
                      help='Number of iteration inside the hit and run walk',
                      default=5, type=int)
  parser.add_argument('--samples', dest='samples',
                      help='List of the samples to run the svm and the kqbc',
                      default=[5, 10, 15, 20, 25, 30, 35, 40], nargs='+', type=int)
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

    for t in range(args.steps):
        print("Step: %d - running SVM and KQBC" % (t))
        for d, kl in param_dict.items():
            if d not in err_dict_svm.keys():
                err_dict_svm[d] = []
                err_dict_kqbc[d] = []
                err_dict_svm_kqbc[d] = []
            for idx, k in enumerate(kl):
                mu = [0] * d
                cov = np.eye(d)
                X = np.random.multivariate_normal(mu, cov, size=10000)
                y = np.sign(X[:,0])

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

                # SVM classifier
                svclassifier = SVC(kernel='linear')
                if np.abs(np.sum(y_train[:k])) == k:
                    svclassifier.fit(X_train[k:k+k,:], y_train[k:k+k])
                else:
                    svclassifier.fit(X_train[:k,:], y_train[:k])
                y_pred = svclassifier.predict(X_test)
                svm_err = np.sum(y_pred != y_test)/len(y_test)*100
                if t == 0:
                    err_dict_svm[d].append(svm_err)
                else:
                    err_dict_svm[d][idx] += svm_err

                # Run KQBC matlab code
                eng = matlab.engine.start_matlab()
                kqbc_selection, kqbc_cls = eng.KQBC(matlab.double(X_train.tolist()), matlab.double(y_train.tolist()), T, 'Linear', k, 0, 0, nargout=2)
                kqbc_cls = np.array(kqbc_cls, dtype=np.float).squeeze()
                kqbc_selection = np.array(kqbc_selection, dtype=np.int32).squeeze()


                # run KQBC classifier
                kqbc_svm = SVC(kernel='linear')
                kqbc_svm.fit(X_train[kqbc_selection,:], y_train[kqbc_selection])
                y_pred_svm_kqbc = kqbc_svm.predict(X_test)
                kqbc_svm_err = np.sum(y_pred_svm_kqbc != y_test) / len(y_test) * 100
                if t == 0:
                    err_dict_svm_kqbc[d].append(kqbc_svm_err)
                else:
                    err_dict_svm_kqbc[d][idx] += kqbc_svm_err


                # SVM with KQBC selections
                y_pred_kqbc = np.sign(X_test @ kqbc_cls)
                if np.sum(y_pred_kqbc != y_test) > np.sum(y_pred_kqbc == y_test):
                    y_pred_kqbc = -y_pred_kqbc
                kqbc_err = np.sum(y_pred_kqbc != y_test)/len(y_test)*100
                if t == 0:
                    err_dict_kqbc[d].append(kqbc_err)
                else:
                    err_dict_kqbc[d][idx] += kqbc_err


    if args.plot:
        df = pd.DataFrame({'x': args.samples,
                           'y1': np.array([i / args.steps for i in err_dict_svm[args.dim]]),
                           'y2': np.array([i / args.steps for i in err_dict_kqbc[args.dim]]),
                           'y3': np.array([i / args.steps for i in err_dict_svm_kqbc[args.dim]])})
        plt.plot('x', 'y1', data=df, marker='', color='olive', linewidth=2, label="SVM")
        plt.plot('x', 'y2', data=df, marker='', color='blue', linewidth=2, linestyle='dashed', label="KQBC")
        plt.plot('x', 'y3', data=df, marker='', color='red', linewidth=2, linestyle='dashed', label="SVM with KQBC selection")
        plt.legend()

