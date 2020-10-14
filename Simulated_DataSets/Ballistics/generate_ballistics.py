"""
This is the function that generates the gaussian mixture for artificial model
The simulated cluster would be similar to the artifical data set from the INN Benchmarking paper
"""
# Define some
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

# Define some constants
k = 0.25
g = 9.81
m = 0.2

##############################
# Previous data distribution #
##############################
#k=0.9
#m=0.5
#g=1

num_samples = 128000

def determine_final_position(x, final_pos_return=False, use_minimizer=False):
    """
    The function to determine the final y position of a ballistic movement which starts at (x1, x2) and throw at angle
    of x3 and initial velocity of x4, y is the horizontal position of the ball when hitting ground
    :param x: (N,4) numpy array
    :param final_pos_return: The flag to return the final position list for each point in each time steps, for debuggin
    purpose and should be turned off during generation
    :param use_minimizer: The flag to use a minimizer to solve the function to get time y
    :return: (N, 1) numpy array of y value
    """
    # Get the shape N
    N = np.shape(x)[0]

    # Initialize the output y
    output = np.zeros([N, 1])

    # Initialize the guess for time steps
    t_max = 100
    t_interval = 0.0001
    time_list = np.arange(0, t_max, t_interval)

    if final_pos_return:
        final_pos_list = []

    for i in range(N):
        print("Solving for the sample number", i)
        # Separate the x into subsets for simplicity
        x1, x2, x3, x4 = x[i, 0], x[i, 1], x[i, 2], x[i, 3]

        final_pos = Position_at_time_T(time_list, x1, x2, x3, x4)

        # Final time step
        time = np.argmin(final_pos[:, 1])
        final_x = final_pos[time, 0]

        final_y = final_pos[time, 1]
        best_y = final_y
        if final_y > 0.001 and not use_minimizer:
            warnings.warn('Your linear spaced time solution is not accurate enough, current accuracy is {} at time step {} and y is {}'.format(final_y, time*t_interval, final_x))
            print('Your linear spaced time solution is not accurate enough, current accuracy is {} at time step {} and y is {}'.format(final_y, time*t_interval, final_x))
        # If the minimizer is used
        if use_minimizer:
            time_minimizer = solve_by_minimizer(x1, x2, x3, x4)
            x_minimizer = Position_at_time_T(time_minimizer, x1, x2, x3, x4)[:, 0]
            y_minimizer = Position_at_time_T(time_minimizer, x1, x2, x3, x4)[:, 1]
            print('Using a minimizer, the time= {} with x={} y={}'.format(time_minimizer, x_minimizer, y_minimizer)) 
            print('In contrast linear, the time= {} with x={} y={}'.format(time*t_interval, final_x, final_y)) 
            print('We choose the best solution among the two, which is {}'.format(best_y))
            best_y = min(final_y, y_minimizer)

            if best_y > 0.001:
                warnings.warn('Your scipy minimizer solution and the linear spaced time solution are both not accurate enough')
                print('Your scipy minimizer solution and the linear spaced time solution are both not accurate enough')
                

        #print("final_x = ", final_x, "final_y= ", final_y)
        if best_y == final_y:
            output[i] = final_x
        else:
            output[i] = x_minimizer
        if final_pos_return:
            final_pos_list.append(final_pos)
    if final_pos_return:
        return output, final_pos_list
    else:
        return output


def solve_by_minimizer(x1, x2, x3, x4):
    """
    This function is to solve the instable function solving for linear spacing method.
    It calls the scipy.minimizer to solve for the Y = 0 point
    """
    from scipy.optimize import minimize
    # Initial gues of the answer
    t0 = 20
    res = minimize(Abs_Pos_y_at_time_T, t0,args=(x1,x2,x3,x4),options={'disp': False},bounds=[(0,None)],tol=1e-4)
    result = np.copy(res.x)
    return result


def Abs_Pos_y_at_time_T(t, x1, x2, x3, x4):
    # Get the initial velocity information
    v1 = x4 * np.cos(x3)
    v2 = x4 * np.sin(x3)
    if len(np.shape(t)) < 1:  # For the case of it is a number input
        print("This t is not a long list")
        N = 1 
    else:
        N = len(t)
    output = np.zeros([N, 2])   # Initialize the output
    exponential_part = np.exp(-k*t/m) - 1
    output[:, 0] = x1 - v1 * m / k * exponential_part
    output[:, 1] = x2 - m / k / k * ((g*m + v2 * k)  * exponential_part + g*t*k)
    # Set all the positions after hitting ground to be 1
    output[:, 1] = np.abs(output[:,1])
    #hit_ground = output[:, 1] < 0
    #output[hit_ground, 1] = 1
    return output[:, 1]


    return Position_at_time_T(t, x1, x2, x3, x4)[:, 1]


def Position_at_time_T(t, x1, x2, x3, x4):
    """
    infer the position of the trajectory at time x given input
    :param x1: x initial position, single number
    :param x2: y initial positio, single numbern
    :param x3: angle of thro, single numberw
    :param x4: velocity of thro, single numberw
    :param t: (N x 1) or (int) numpy array of time steps
    :return: (N X 2) or (int) numpy array of positions
    """
    # Get the initial velocity information
    v1 = x4 * np.cos(x3)
    v2 = x4 * np.sin(x3)
    if len(np.shape(t)) < 1:  # For the case of it is a number input
        print("This t is not a long list")
        N = 1 
    else:
        N = len(t)
    output = np.zeros([N, 2])   # Initialize the output
    exponential_part = np.exp(-k*t/m) - 1
    output[:, 0] = x1 - v1 * m / k * exponential_part
    output[:, 1] = x2 - m / k / k * ((g*m + v2 * k)  * exponential_part + g*t*k)
    # Set all the positions after hitting ground to be 1
    output[:, 1] = np.abs(output[:,1])
    #hit_ground = output[:, 1] < 0
    #output[hit_ground, 1] = 1
    return output


def generate_random_x():
    """
    Generate random X array according to the description of the Benchmarking paper
    :return: (N, 4) random samples
    """
    output = np.zeros([num_samples, 4])
    output[:, 0] = np.random.normal(0, 0.25, size=num_samples)
    output[:, 1] = np.random.normal(1.5, 0.25, size=num_samples)
    output[:, 2] = np.radians(np.random.uniform(9, 72, size=num_samples))
    output[:, 3] = np.random.poisson(15, size=num_samples)
    return output


def plot_trajectory(x):
    """
    Plot the trajectory of a ballistic movement, in order to varify the correctness of this simulation
    :param x: (N, 4) the input x1~x4 param
    :return: one single plot of all the N trajectories
    """
    # Get the trajectory first
    y, final_positions = determine_final_position(x, final_pos_return=True)
    f = plt.figure()
    plt.plot([0, 10], [0, 0],'r--', label="horitonal line")
    for i in range(len(y)):
        before_hit_ground = np.argmin(final_positions[i][:, 1])
        plt.plot(final_positions[i][:before_hit_ground, 0], final_positions[i][:before_hit_ground, 1], label=str(i))
    plt.legend()
    #plt.ylim(bottom=0)
    #plt.xlim([-1, 10])
    plt.title("trajectory plot for ballistic data set")
    plt.savefig('k={} m={} g={} Trajectory_plot.png'.format(k,m,g))



def generate_1000_random_x(save_dir='/work/sr365/multi_eval/Random/ballistics/'):
    """
    Generate the random solutions for comparisons
    """
    for i in range(1000):
        Xpred_save_name = save_dir + 'ballistics_Xpred_random_guess_inference' + str(i) + '.csv'
        Ypred_save_name = Xpred_save_name.replace('Xpred','Ypred')
        Xpred = generate_random_x()
        Ypred = determine_final_position(Xpred, use_minimizer=True)
        np.savetxt(Xpred_save_name, Xpred, delimiter=' ')
        np.savetxt(Ypred_save_name, Ypred, delimiter=' ')

if __name__ == '__main__':
    X = generate_random_x()
    y = determine_final_position(X, use_minimizer=True)
    #plot_trajectory(X)
    np.savetxt('data_x_large.csv', X, delimiter=',')
    np.savetxt('data_y_large.csv', y, delimiter=',')
    #generate_1000_random_x()
