from mlxtend.data import loadlocal_mnist
import numpy as np

def prepare_mnist_data():
    X_train, y_train = loadlocal_mnist(
        images_path = '../mnist/train-images-idx3-ubyte',
        labels_path = '../mnist/train-labels-idx1-ubyte')


    X_test, y_test = loadlocal_mnist(
        images_path = '../mnist/t10k-images-idx3-ubyte',
        labels_path = '../mnist/t10k-labels-idx1-ubyte')

    return X_train, y_train, X_test, y_test



def select_one_class_vs_all(X, y, sel_class=6, samples=1000):
    y = y.astype(np.float32)
    one_class_indices = np.where(y == sel_class)[0]
    one_class_indices = np.random.choice(one_class_indices, samples)
    y[one_class_indices] = 1

    other_class_indices = np.where(y != sel_class)[0]
    other_class_indices = np.random.choice(other_class_indices, samples)
    y[other_class_indices] = -1

    indices = np.concatenate((one_class_indices, other_class_indices), axis=0)
    np.random.shuffle(indices)
    return X[indices], y[indices]


def get_rbf_data(lower, upper, num, num_dims):
    X = np.random.uniform(lower, upper, size=(num,num_dims))
    Y = []
    for x1,x2 in X:
        if x2 < np.sin(10*x1)/5 + 0.3 or ((x2 - 0.8)**2 + (x1 - 0.5)**2) < 0.15**2:
            Y.append(1)
        else:
            Y.append(-1)
    return X, np.asarray(Y)


