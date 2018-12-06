'''
Module linear_func.py
************************

Defines a linear function and a linear classifier
based on the function.
'''
import numpy as np
from pylab import *

class Classifier:
    '''Class to represent a linear function and associated linear
    classifier on n-dimension space.'''

    def __init__(self, vect_w=None):
        '''Initializes coefficients, if None then
        must be initialized later.

        :param vect_w: vector of coefficients.'''
        self.vect_w = vect_w

    def init_random_last0(self, dimension, seed=None):
        '''
        Initializes to random vector with last coordinate=0,
        uses seed if provided.

        :params dimension: vector dimension;

        :params seed: random seed.
        '''
        if seed is not None:
            np.random.seed(seed)
        self.vect_w = -1 + 2 * np.random.rand(dimension)
        self.vect_w[-1] = 0  # exclude class coordinate

    def value_on(self, vect_x):
        '''Computes value of the function on vector vect_x.

        :param vect_x: the argument of the linear function.'''
        return sum(p * q for p, q in zip(self.vect_w, vect_x))

    def class_of(self, vect_x):
        '''Computes a class, one of the values {-1, 1} on vector vect_x.

        :param vect_x: the argument of the linear function.'''
        return 1 if self.value_on(vect_x) >= 0 else -1

    def intersect_aabox2d(self, box=None):
        '''Returns two points intersection (if any) of the
        decision line (function value = 0) with axis-aligned
        rectangle.'''
        if box is None:
            box = ((-1, -1), (1, 1))

        minx = min(box[0][0], box[1][0])
        maxx = max(box[0][0], box[1][0])
        miny = min(box[0][1], box[1][1])
        maxy = max(box[0][1], box[1][1])

        intsect_x = []
        intsect_y = []

        for side_x in (minx, maxx):
            ya = -(self.vect_w[0] + self.vect_w[1] * side_x) / self.vect_w[2]
            if ya >= miny and ya <= maxy:
                intsect_x.append(side_x)
                intsect_y.append(ya)
        for side_y in (miny, maxy):
            xb = -(self.vect_w[0] + self.vect_w[2] * side_y) / self.vect_w[1]
            if xb <= maxx and xb >= minx:
                intsect_x.append(xb)
                intsect_y.append(side_y)
        return intsect_x, intsect_y


def separable_2d(seed, n_points, classifier):
    '''Generates n_points which are separable via
    passed in classifier in 2d.

    :params seed: sets random seed for the random
        generator,

    :params n_points: number of points to generate,

    :params classifier: a function returning
        either +1 or -1 for each point in 2d.'''
    np.random.seed(seed)

    dim_x = 2
    data_dim = dim_x + 1 + 1  # leading 1 and class value
    data = np.ones(data_dim * n_points).reshape(n_points, data_dim)

    # fill in random values
    data[:, 1] = -1 + 2 * np.random.rand(n_points)
    data[:, 2] = -1 + 2 * np.random.rand(n_points)

    # TODO: use numpy way of applying a function to rows.
    for idx in range(n_points):
        data[idx, -1] = classifier.class_of(data[idx])

    return data



def get_linear_2d_data(n):
    data_dim = 2
    classifier = Classifier()
    classifier.init_random_last0(data_dim + 2, 20)

    data = separable_2d(10, n, classifier)

    condition = data[:, 3] >= 0
    positive = np.compress(condition, data, axis=0)

    neg_condition = data[:, 3] < 0
    negative = np.compress(neg_condition, data, axis=0)

    x_pos = positive[:, 1]
    y_pos = positive[:, 2]

    x_neg = negative[:, 1]
    y_neg = negative[:, 2]

    X_pos = np.array([i for i in zip(x_pos, y_pos)])
    y_pos = np.ones(X_pos.shape[0])
    X_neg = np.array([i for i in zip(x_neg, y_neg)])
    y_neg = -np.ones(X_neg.shape[0])

    plot_lim = 1.2
    box = ((-plot_lim, -plot_lim), (plot_lim, plot_lim))
    decision_x, decision_y = classifier.intersect_aabox2d(box)

    figure()

    ylim([-plot_lim, plot_lim])
    xlim([-plot_lim, plot_lim])

    plot(X_pos[:,0], X_pos[:,1], 'g+', label="Class=+1")
    plot(X_neg[:,0], X_neg[:,1], 'r.', label="Class=-1")
    plot(decision_x, decision_y, 'b-', label="Decision Boundary")

    plt.legend(bbox_to_anchor=(0., 0.9, 1., .102), ncol=3, mode="expand", borderaxespad=0.)

    xlabel('x')
    ylabel('y')
    title('Artificial Training Data')

    show()





    return np.concatenate((X_pos, X_neg)), np.concatenate((y_pos, y_neg))