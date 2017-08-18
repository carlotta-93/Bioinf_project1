import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import GPy

print 'hello world'

dataTrain = np.loadtxt('train_data2.csv', delimiter=';')

# shuffle data in order not to have points from first generations otherwise they might not represent the entire set
np.random.seed(2)
np.random.shuffle(dataTrain)

number_of_points = 50
# split input variables and labels
XTrain = dataTrain[:, :-1]

print 'THIS IS THE MEAN ' + str(np.mean(XTrain, axis=0))


YTrain = dataTrain[:, -1]

phs = XTrain[:, 0][0:number_of_points]
molarities = XTrain[:, 1][0:number_of_points]

# ph_bin = []
# for ph in phs:
#     ph_bin.append((ph - 7.) / 5.3)
#
# mol_bin = []
# for mol in molarities:
#     mol_bin.append((mol - 5.) / 15.)

# plot data

# my_plot = plt.scatter(phs, molarities, c=YTrain, lw=0.5, label=YTrain, cmap='plasma')
# plt.grid()
# plt.xlabel('ph values')
# plt.ylabel('molarity values')
# plt.title('Training data set, coloured by fitness values')
# plt.colorbar(my_plot)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# my_3d_plot = ax.scatter(phs, molarities, YTrain, zdir='z', s=20, depthshade=True, c=YTrain, cmap='plasma')
ax.set_xlabel('ph_values')
ax.set_ylabel('molarity_values')
ax.set_zlabel('fitness_values')
# plt.colorbar(my_3d_plot)
# plt.show()


#
# lala = np.reshape(phs, (40, 1))
# lalala = np.reshape(molarities, (40, 1))

# 1-D example

GPy.plotting.change_plotting_library('plotly')

X = np.random.uniform(-3., 3., (20, 1))

# data_mean = np.mean(data, axis=0)
# data_centered = data - data_mean

Y = np.sin(X) + np.random.randn(20, 1) * 0.05

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

m = GPy.models.GPRegression(X, Y, kernel)
from IPython.display import display
display(m)
fig = m.plot()
# GPy.plotting.show(fig, filename='basic_gp_regression_notebook')

m.optimize(messages=True)
m.optimize_restarts(num_restarts=10)

fig = m.plot()
# GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')

fig = m.plot(plot_density=True)
# GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')



# 2D example

# # sample inputs and outputs
# X2 = np.random.uniform(-3., 3., (50, 2))
# print X2
# Y2 = np.sin(X[:, 0:1]) * np.sin(X[:, 1:2]) + np.random.randn(50, 1)*0.05
#
# # define kernel
# ker = GPy.kern.Matern52(2, ARD=True) + GPy.kern.White(2)
#
# # create simple GP model
# m = GPy.models.GPRegression(X2, Y2, ker)
#
# # optimize and plot
# m.optimize(messages=True,max_f_eval = 1000)
# fig = m.plot()
# display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
# display(m)

x_train_set = XTrain[0:number_of_points]
# print 'this is X TRAIN SET' + str(len(x_train_set))
y_train_set = YTrain[0:number_of_points]
y_train_set_reshaped = np.reshape(y_train_set, (50, 1))
print np.shape(y_train_set_reshaped)
# print 'this is Y TRAIN SET' + str(len(y_train_set))


x_test_set = XTrain[number_of_points:len(XTrain)]
# print 'this is X TEST SET' + str(len(x_test_set))
y_test_set = YTrain[number_of_points:len(YTrain)]
# print 'this is Y TEST SET' + str(len(y_test_set))


# Fit a GP

kg = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
kb = GPy.kern.Bias(input_dim=2)
k = kg + kb
my_kernel = k.plot()

m = GPy.models.GPRegression(x_train_set, y_train_set_reshaped, k, normalizer=False, noise_var=.1)
plotting_figure = m.plot()
GPy.plotting.show(plotting_figure, filename='3D_example_2')
# m.constrain_bounded('rbf_var', 1e-3, 1e5)
# m.constrain_bounded('bias_var', 1e-3, 1e5)
# m.constrain_bounded('rbf_len', .1, 200.)
# m.constrain_fixed('noise', 1e-5)

m.optimize(messages=True, max_f_eval=1000)
m.optimize_restarts(num_restarts=10)


# y_prediction = m.predict(x_test_set)
# print y_prediction

# print m
plotting_figure_normalized = m.plot()
GPy.plotting.show(plotting_figure_normalized, filename='3D_example_opt_non_rand')