"""
This is the function (script) which generates the simulated data for doing inverse model comparison.
This function (script) generates sinusoidal waves of y as a function of input x
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos
from mpl_toolkits.mplot3d import Axes3D

# Define some hyper-params
x_dimension = 2         # Current version only support 2 dimension due to visualization issue
y_dimension = 1         # Current version only support 2 dimension due to visualization issue
x_low = -1
x_high = 1
num_sample_dimension = 100
f = 3

def plotData(data_x, data_y, save_dir='generated_sinusoidal_scatter.png'):
    """
    Plot the scatter plot of the simulated sinusoidal wave
    :param data_x: The simulated data x
    :param data_y: The simulated data y
    :param save_dir: The save name of the plot
    :return: None
    """
    # Plot one graph for each dimension of y
    for i in range(len(data_y)):
        f = plt.figure()
        ax = Axes3D(f)
        plt.title('scattering plot for dimension {} for sinusoidal data'.format(i+1))
        print(np.shape(data_x[0, :]))
        ax.scatter(data_x[0, :], data_x[1, :], data_y[i, :], s=2)
        f.savefig('dimension_{}'.format(i+1) + save_dir)


def getYfromX(x):
    #y_shape = [num_sample_dimension for i in range(x_dimension + 1)]
    #y_shape[0] = y_dimension
    #data_y = np.zeros(y_shape)
    y_shape = np.array(np.shape(x))
    y_shape[-1] = 1             # y_dimension hard-coded
    data_y = np.zeros(y_shape)
    print("shape of data_y is", np.shape(data_y))
    print("shape of input x is", np.shape(x))
    data_y = sin(f*np.pi*x[:,0])  +  cos(f*np.pi*x[:,1])
    #for i in range(2):
        #data_y[:] += sin(f*np.pi*x[:, i])
        #data_y[:, 1] += cos(f*np.pi*x[:, i])
        # data_y[0, :] += x[i, ::]              # Easy case for validation of architecture
    return data_y 

if __name__ == '__main__':
    xx = []
    for i in range(x_dimension):
        xx.append(np.random.uniform(x_low, x_high, size=num_sample_dimension))         # append each linspace into the list
    x = np.array(np.meshgrid(*xx))                                # shape(x_dim, #point, #point, ...) of data points
    # Reshape the data into one long list
    data_x = np.concatenate([np.reshape(np.ravel(x[i, :]), [-1, 1] ) for i in range(x_dimension)], axis=1)
    print('shape x', np.shape(data_x))
    data_y = getYfromX(data_x)
    print('shape y', np.shape(data_y))
    # Save the data into txt files
    np.savetxt('data_x.csv', data_x, delimiter=',')
    np.savetxt('data_y.csv', data_y, delimiter=',')
