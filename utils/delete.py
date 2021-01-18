import os
import numpy

# This is the program to delete all the duplicate Xtruth Ytruth files generated
input_dir = '../multi_eval/'
delete_mse_file_mode = False                            # Deleting the mse file for the forward filtering


# For all the architectures
for folders in os.listdir(input_dir):
    #print(folders)
    if os.path.isdir(os.path.join(input_dir,folders)):
        # For all the datasets inside it
        for dataset in os.listdir(os.path.join(input_dir, folders)):
            #print(dataset)
            if os.path.isdir(os.path.join(input_dir, folders, dataset)):
                current_folder = os.path.join(input_dir, folders, dataset)
                print("current folder is:", current_folder)
                for file in os.listdir(current_folder):
                    current_file = os.path.join(current_folder, file)
                    if os.path.getsize(current_file) == 0:
                        print('deleting file {} due to empty'.format(current_file))
                        os.remove(current_file)
                    elif '_Ytruth_' in file:
                        if 'ce0.csv' in file or 'NA' in folders:
                            os.rename(current_file, os.path.join(current_folder, 'Ytruth.csv'))
                        else:
                            os.remove(current_file)
                    elif '_Xtruth_' in file:
                        if 'ce0.csv' in file or 'NA' in folders:
                            os.rename(current_file, os.path.join(current_folder, 'Xtruth.csv'))
                        else:
                            os.remove(current_file)
                    elif '_Ypred_' in file and file.endswith(dataset + '.csv'):
                        os.rename(current_file, os.path.join(current_folder, 'Ypred.csv'))
                    elif '_Xpred_' in file and file.endswith(dataset + '.csv'):
                        os.rename(current_file, os.path.join(current_folder, 'Xpred.csv'))
                    if delete_mse_file_mode and 'mse_' in file:
                        os.remove(current_file)
