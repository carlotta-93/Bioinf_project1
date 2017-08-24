import numpy as np
import matplotlib.pyplot as plt
import GPy

GPy.plotting.change_plotting_library('plotly')

dataTrain = np.loadtxt('train_data.csv', delimiter=';')
print len(dataTrain)
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
my_3d_plot = plt.scatter(phs, molarities, s=40, c=YTrain, cmap='plasma')
plt.xlabel('ph_values')
plt.ylabel('molarity_values')
plt.grid(True)
# my_3d_plot.set_zlabel('fitness_values')
plt.colorbar(my_3d_plot)
plt.title('Distribution of the dataset')
plt.show()

# 2D model

x_train_set = XTrain[0:number_of_points+1]
# print 'this is X TRAIN SET' + str(x_train_set)
# y_train_set = YTrain[0:number_of_points+1]

y_train_set_reshaped = np.reshape(YTrain, (len(YTrain), 1))
# print np.shape(y_train_set_reshaped)
# print 'this is Y TRAIN SET' + str(len(y_train_set))

x_test_set = XTrain[number_of_points:len(XTrain)]
print 'this is X TEST SET' + str(x_test_set)
y_test_set = YTrain[number_of_points:len(YTrain)]
# print 'this is Y TEST SET' + str(len(y_test_set))

# Fit a GP

kg = GPy.kern.RBF(input_dim=2, variance=0.5, lengthscale=1., ARD=True)
# GPy.plotting.show(k.plot(), filename='Kernel')

# print 'THIS IS THE MEAN OF THE OBSERVATIONS: ' + str(np.mean(y_train_set_reshaped))

m = GPy.models.GPRegression(XTrain, y_train_set_reshaped, kernel=kg, normalizer=False, noise_var=1.)

print 'This is the regression model object: ' + str(m)

plotting_figure = m.plot()
GPy.plotting.show(plotting_figure, filename='GP model')

m.optimize(messages=True, max_f_eval=1000)  # fit the model
m.optimize_restarts(num_restarts=10)

plotting_figure_normalized = m.plot(plot_training_data=True)
GPy.plotting.show(plotting_figure_normalized, filename='GP model fitting')


predicted_values_mean, predicted_values_var = m.predict(x_test_set)

print 'These are the predictions: ' + str(predicted_values_mean)

print 'These are the values of the test set: ' + str(y_test_set)

