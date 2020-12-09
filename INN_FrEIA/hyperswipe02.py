"""
This .py file is to run train.py for hyper-parameter swipping in a linear fashion.
"""
import train
#os.environ["CUDA_VISIBLE_DEVICE"] = "-1"               #Uncomment this line if you want to use CPU only
import  numpy as np
import flag_reader
if __name__ == '__main__':
    #linear_unit_list = [20, 30, 40, 50]
    #dim_z_list = [2, 3, 4, 5, 6, 7]
    #linear_unit_list = [1000, 500]
    #linear_unit_list = [1000, 500, 300, 150]
    # reg_scale_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    #reg_scale_list = [5e-4]
    #for linear_unit in linear_unit_list:
    #    # Setting the loop for setting the parameter
    #    for i in range(4, 10):
    #        flags = flag_reader.read_flag()  	            # setting the base case
    #        linear = [linear_unit for j in range(i)]        # Set the linear units
    #        linear[0] = flags.dim_y + flags.dim_z           # The start of linear
    #        linear[-1] = flags.dim_x                        # The end of linear
    #        flags.linear_d = linear
    #        for reg_scale in reg_scale_list:
    #            flags.reg_scale = reg_scale
    #            for dim_z in dim_z_list:
    #               flags.dim_z = dim_z
    #               flags.linear_d[0] = flags.dim_y + flags.dim_z
    #                # print("flags.linear_d", flags.linear_d)
    #                # print("flags.linear_e", flags.linear_e)
    #                for j in range(3):
    #                    flags.model_name = flags.data_set + "reg"+ str(flags.reg_scale) + "trail_"+str(j) + "_backward_complexity_swipe_layer" + str(linear_unit) + "_num" + str(i)
    #                    train.training_from_flag(flags)
    
    ###################
    # dim_z & dim_y #
    ###################

    #for dim_z in range(5, 8):
    #    for dim_y in range(dim_z, 8):
    #        flags = flag_reader.read_flag()  	            # setting the base case
    #        flags.dim_tot = dim_z + dim_y + 10              # Give 10 dimension of spare case
    #        flags.dim_z = dim_z
    #        flags.dim_y = dim_y
    #        flags.linear_se[-1] = dim_y
    #        flags.model_name = flags.data_set + "dim_y" + str(dim_y) + "dim_z" + str(dim_z)
    #        train.training_from_flag(flags)
    
    
    ###########################
    # dim_tot & #couple_layer #
    ###########################

    lambda_mse_list = [0.01, 0.03, 0.05,0.08, 0.1, 0.2, 0.3, 0.4]
    # for dim_tot in range(4, 6):
    for z_dim in range(2,3):
        for couple_layer in range(6, 9):
            for lambda_mse in lambda_mse_list:
                for i in range(2):
                    flags = flag_reader.read_flag()  	            # setting the base case\
                    flags.lambda_mse = lambda_mse
                    #flags.dim_tot = dim_tot
                    flags.dim_z = z_dim
                    flags.couple_layer_num = couple_layer
                    flags.model_name = flags.data_set + "couple_layer_num" + str(couple_layer) + "dim_z" + str(z_dim) + "lambda_mse" + str(lambda_mse) + "trail" + str(i)
                    train.training_from_flag(flags)
    
    #################
    # kl_coeff & lr #
    #################
    
    #kl_coeff_list = [1, 0.75, 0.5, 0.25, 0.1]#e-2, 8e-3, 5e-3, 3e-3, 1e-3]
    #lr_list = [1e-3, 5e-3, 1e-2]
    #for kl_coeff in kl_coeff_list:
    #    for lr in lr_list:
    #        flags = flag_reader.read_flag()
    #        flags.learn_rate = lr
    #        flags.kl_coeff = kl_coeff

    ##################
    # #unit & #layer #
    ##################

    #unit_list = [100, 200, 300, 400]
    #for layer_num in range(5, 10):
    #    for unit in unit_list:
    #        flags = flag_reader.read_flag()  	            # setting the base case
    #        linear_d = [unit for j in range(layer_num)]
    #        linear_e = [unit for j in range(layer_num)]
    #        linear_d[0] = flags.dim_y + flags.dim_z
    #        linear_d[-1] = flags.dim_x
    #        linear_e[0] = flags.dim_y + flags.dim_x
    #        linear_e[-1] = flags.dim_z * 2
    #        flags.linear_d = linear_d
    #        flags.linear_e = linear_e
    #        for i in range(3):
    #            flags.model_name = flags.data_set + "layer_num" + str(layer_num) + "unit_" + str(unit) + "reg" + str(flags.reg_scale) + "trail" + str(i)
    #            print(flags.model_name)
    #            train.training_from_flag(flags)


