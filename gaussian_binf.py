import numpy as np
import matplotlib.pyplot as plt

print 'hello world'


dataTrain = np.loadtxt('train_data2.csv', delimiter=';')

# split input variables and labels
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]

print XTrain
phs = XTrain[:, 0]
molarities = XTrain[:, 1]
print len(YTrain)


# plot data


my_plot = plt.scatter(phs, molarities, c=YTrain, lw=0.5, label=YTrain)

plt.grid()
plt.xlabel('ph values')
plt.ylabel('molarity values')
plt.title('Training data set, coloured by fitness values')
plt.colorbar(my_plot)
plt.show()





