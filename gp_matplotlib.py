import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score

dataTrain = np.loadtxt('train_data.csv', delimiter=';')

# shuffle data in order not to have points from first generations otherwise they might not represent the entire set
np.random.seed(2)
np.random.shuffle(dataTrain)

number_of_points = 50

# split input variables and labels
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]


phs = XTrain[:, 0]
molarities = XTrain[:, 1]

# plot data

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# my_3d_plot = ax.scatter(phs, molarities, YTrain, zdir='z', s=20, depthshade=True, c=YTrain, cmap='plasma')
# ax.set_xlabel('ph_values')
# ax.set_ylabel('molarity_values')
# ax.set_zlabel('fitness_values')
# plt.colorbar(my_3d_plot)
# plt.title('3D plot of data set')
# plt.show()

# 2D model

x_train_set = XTrain[0:number_of_points+1]
# print 'this is X TRAIN SET' + str(x_train_set)
y_train_set = YTrain[0:number_of_points+1]
y_train_set_reshaped = np.reshape(y_train_set, (len(y_train_set), 1))
print np.shape(y_train_set_reshaped)
# print 'this is Y TRAIN SET' + str(len(y_train_set))

x_test_set = XTrain[number_of_points:len(XTrain)]
# print 'this is X TEST SET' + str(x_test_set)
y_test_set = YTrain[number_of_points:len(YTrain)]
y_test_set_reshaped = np.reshape(y_test_set, (len(y_test_set), 1))

# print 'this is Y TEST SET' + str(len(y_test_set))

# Instantiate a Gaussian Process model
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# kernel = 1**2 * RBF(length_scale=1.) * DotProduct(sigma_0=1.0)**2
kernel = 1**2 * RBF([1.0, 1.0]) + WhiteKernel(noise_level=1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

fitting = gp.fit(x_train_set, y_train_set_reshaped)
print gp.kernel_

# Fit to data using Maximum Likelihood Estimation of the parameters
# gp.fit(x_train_set, y_train_set_reshaped)

y_pred = gp.predict(x_test_set)

print y_pred
print y_test_set_reshaped

print r2_score(y_test_set, y_pred)
print gp.log_marginal_likelihood(gp.kernel_.theta)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = plt.figure()
plt.plot(x_train_set, y_train_set_reshaped, 'r:', label=u'$f(x) = x\,\sin(x)$')
# plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
# plt.plot(x, y_pred, 'b-', label=u'Prediction')
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([y_pred - 1.9600 * sigma,
#                         (y_pred + 1.9600 * sigma)[::-1]]),
#          alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
plt.legend(loc='upper left')
# plt.show()

