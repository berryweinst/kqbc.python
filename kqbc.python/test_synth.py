from kqbc import KQBC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



d = 5
mu = [0] * d
cov = np.eye(d)
k = 40


X = np.random.multivariate_normal(mu, cov, size=10000)
y = np.sign(X[:,0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train[:k,:], y_train[:k])
y_pred = svclassifier.predict(X_test)


kqbc = KQBC(X_train, y_train, T=150, kernel='Linear', queries=40)