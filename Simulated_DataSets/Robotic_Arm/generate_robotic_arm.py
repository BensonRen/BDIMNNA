"""
This is the function (script) which generates the simulated data for doing inverse model comparison.
This function (script) generates robotic arms positions of y as a function of input x
"""

# Import libraries
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from os.path import join
from numpy import sin, cos

# Define some hyper-parameters
arm_num = 3                                                         # Number of arms
arm_lengths = [0.5, 0.5, 1]                                         # length of the robotic arm
arm_max_length = np.sum(arm_lengths)                                # Drawing limit
num_samples = 10000
sample_vars = [1/16, 1/4, 1/4, 1/4]
eps = 0.2

def determine_final_position(origin_pos, arm_angles, arm_lengths=arm_lengths, evaluate_mode=False):
    """
    The function that computes the final position of the angle. One number input implementation now
    :param origin_pos: [N, 1] array of float , The origin position on the y axis
    :param arm_angles: [N, arm_num] an array of angles [-pi, pi] that states the angle of the arm w.r.t
                        horizontal surface. Horizon to the right is 0, up is positive and below is negative
    :param arm_lengths: [N, arm_num] an array of float that states the lengths of the robotic arms
    :param evaluate_mode: Boolean. The flag indicating whether this is a evaluate mode or not
    :return: current_pos: [N, 2] The final position of the robotic arm, a array of 2 numbers indicating (x, y) co-ordinates
    :return: positions: list of #arm_num of [N, 2], The positions that the arms nodes for plotting purpose
    """
    # First make sure that the angles are legal angles
    # Start computing for positions
    positions = []                                          # Holder for positions to be plotted
    current_pos = np.zeros([len(origin_pos), 2])            # The robotic arm starts from [0, y]
    print("shape of current_pos = ", np.shape(current_pos))
    current_pos[:, 0] = 0                                   # The x axis is 0 for all cases
    current_pos[:, 1] = origin_pos                          # Plug in the
    positions.append(np.copy(current_pos))

    # Set up the arm angle difference matrix for compliance to the original paper
    arm_angles_diff = np.copy(arm_angles)
    arm_angles_diff[:, 1] -=  arm_angles[:, 0]
    arm_angles_diff[:, 2] -=  arm_angles[:, 1] + arm_angles[:, 0]

    for arm_index in range(arm_num):                                # Loop through the arms
        # print("i=",i)
        current_pos[:, 0] += cos(arm_angles_diff[:, arm_index]) * arm_lengths[arm_index]
        current_pos[:, 1] += sin(arm_angles_diff[:, arm_index]) * arm_lengths[arm_index]
        positions.append(np.copy(current_pos))
    if not evaluate_mode:
        plot_arms(np.array(positions))
    return current_pos, np.array(positions)


def plot_arms(positions, save_dir='.', save_name='robotic_arms_plot.png', margin=0.2):
    """
    Plot the position of the robotic arms given positions of the points
    :param positions: [num_arm + 1, N, 2] The position of the joints places
    :param save_dir: The saving directory of the plot
    :param save_name: The saving name of the plot
    :param margin: The margin of x,y limit during drawing
    :return: None
    """
    f = plt.figure()
    shape = np.shape(positions)
    # Draw the verticle line for the origin arm
    plt.plot([0, 0], [-1.5, 1.5], lw=1, c='b')
    # plt.plot(positions[1:, :, 0], positions[1:, :, 1], ms=5, mfc='w')#, mec='b')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 1], lw=3, ls='-', c='k', alpha=0.1)
    print("shape of positions", np.shape(positions))
    # print(np.min(positions[:, :, 0], axis=0) )
    strange_list = np.arange(shape[1])[np.min(positions[:, :, 0], axis=0) < 0]
    print("strange list is", strange_list)
    print("position of those strange ones are", positions[:, strange_list, :])
    # print(positions)
    for i in range(shape[1]):
        #print("i =",i)
        #print("position0:", positions[:, i, 0])
        #print("position1:", positions[:, i, 1])
        plt.plot(positions[:, i, 0], positions[:, i, 1], lw=1, ls='-', c='k', alpha=0.1)
        if i in strange_list:
            print(i)
            plt.plot(positions[:, i, 0], positions[:, i, 1], lw=2, c='b', alpha=0.5)
        #plt.scatter(positions[:, i, 0], positions[:, i, 1], c='b', s=30, marker='o')
        #plt.scatter(positions[:, i, 0], positions[:, i, 1], c='w', s=20, marker='o')

    plt.plot()
    # plt.ylim([-arm_max_length, arm_max_length])
    # plt.xlim([-margin, arm_max_length - origin_limit + margin])
    f .savefig(join(save_dir, save_name))


def Sample_through_space():
    # Initialize the data
    data_x = np.random.randn(num_samples, arm_num + 1)
    for i in range(len(data_x[0, :])):
        data_x[:, i] *= np.sqrt(sample_vars[i])
        print(np.var(data_x[:, i]))
    # print("data_x", data_x)
    print("shape of data_x is:", np.shape(data_x))
    data_y, positions = determine_final_position(data_x[:, 0], data_x[:, 1:])
    # print("data_y", data_y)
    print("shape of data_y is", np.shape(data_y))
    print("shape of positions is:", np.shape(positions))
    # print("position=", positions)
    plot_arms(positions)
    np.savetxt('data_x.csv', data_x, delimiter=',')
    np.savetxt('data_y.csv', data_y, delimiter=',')


if __name__ == '__main__':
    # positions = np.array([np.reshape(np.array([0, 2]), [-1, 2]), np.reshape(np.array([1, 4]), [-1, 2]),
    #                          np.reshape(np.array([2, 3]), [-1, 2]), np.reshape(np.array([3, 0.2]), [-1, 2])])
    # print("original position shape", np.shape(positions))
    # plot_arms(positions)
    Sample_through_space()
