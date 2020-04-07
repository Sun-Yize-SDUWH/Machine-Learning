import numpy as np
import matplotlib.pylab as plt


np.random.seed(20)

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

x = np.random.normal(0, 1, 500)
y = np.random.normal(0, 1, 500)
X = np.vstack((x, y)).T

# original data
plt.scatter(X[:, 0], X[:, 1])
plt.title('White Data')
plt.axis('equal')
plt.show()


def cov(x, y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)


def cov_mat(X):
    return np.array([[cov(X[0], X[0]), cov(X[0], X[1])],
                     [cov(X[0], X[1]), cov(X[1], X[1])]])


print(cov_mat(X.T))

# streatched data
X = X - np.mean(X, 0)

sx, sy = 0.7, 3.4
Scale = np.array([[sx, 0], [0, sy]])

Y = X.dot(Scale)

plt.scatter(Y[:, 0], Y[:, 1])
plt.title('Stretched Data')
plt.axis('equal')
plt.show()
# print(cov_mat(Y.T))

# streatched and rotated data
theta = 0.25 * np.pi
c, s = np.cos(theta), np.sin(theta)
Rot = np.array([[c, -s], [s, c]])

T = Scale.dot(Rot)
Y = X.dot(T)

plt.scatter(Y[:, 0], Y[:, 1])
plt.title('Stretched and Rotated Data')
plt.axis('equal')
plt.show()
# print(cov_mat(Y.T))

# draw line
C = cov_mat(Y.T)
eVe, eVa = np.linalg.eig(C)

plt.scatter(Y[:, 0], Y[:, 1])
for e, v in zip(eVe, eVa.T):
    plt.plot([0, 3 * np.sqrt(e) * v[0]], [0, 3 * np.sqrt(e) * v[1]], 'k-', lw=2)
plt.title('Stretched and Rotated Data with Direction')
plt.axis('equal')
plt.show()

# whiten the data back
R, S = eVa, np.diag(np.sqrt(eVe))

T = R.dot(S).T

Z = Y.dot(np.linalg.inv(T))

plt.scatter(Z[:, 0], Z[:, 1])
plt.title('Uncorrelated Data')
plt.axis('equal')
plt.show()
