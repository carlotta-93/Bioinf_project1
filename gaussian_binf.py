import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import GPy

GPy.plotting.change_plotting_library('plotly')

print 'hello world'

dataTrain = np.loadtxt('train_data2.csv', delimiter=';')

# shuffle data in order not to have points from first generations otherwise they might not represent the entire set
np.random.seed(2)
np.random.shuffle(dataTrain)

number_of_points = 50
# split input variables and labels
XTrain = dataTrain[:, :-1]

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

# lala = np.reshape(phs, (40, 1))
# lalala = np.reshape(molarities, (40, 1))

# data_mean = np.mean(data, axis=0)
# data_centered = data - data_mean


# 2D example

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

kg = GPy.kern.RBF(input_dim=2, variance=0.5, lengthscale=0.5, ARD=True)
kb = GPy.kern.Bias(input_dim=2)

# k = kg + kb
k = kg
# GPy.plotting.show(k.plot(), filename='Kernel')

print 'THIS IS THE MEAN OF THE OBSERVATIONS: ' + str(np.mean(y_train_set_reshaped))

m = GPy.models.GPRegression(x_train_set, y_train_set_reshaped, k, normalizer=False, noise_var=1.)
print 'This is the regression model object: ' + str(m)

plotting_figure = m.plot()
# GPy.plotting.show(plotting_figure, filename='3D_example')

m.optimize(messages=True, max_f_eval=1000)  # fit the model
m.optimize_restarts(num_restarts=10)

# y_prediction = m.predict(x_test_set)
# print y_prediction

# print m
plotting_figure_normalized = m.plot()

# GPy.plotting.show(plotting_figure_normalized, filename='3D_example_opt_non_rand')

print x_test_set
predicted_values = m.predict(x_test_set)

print 'These are the predictions: ' + str(predicted_values)

