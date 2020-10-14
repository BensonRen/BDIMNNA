import os
import numpy

# This is the program to delete all the duplicate Xtruth Ytruth files generated
input_dir = './'

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
                    if '_Ytruth_' in file:
                        if 'ce0.csv' in file or 'NA' in folders:
                            os.rename(os.path.join(current_folder, file), os.path.join(current_folder, 'Ytruth.csv'))
                        else:
                            os.remove(os.path.join(current_folder, file))
                    elif '_Xtruth_' in file:
                        if 'ce0.csv' in file or 'NA' in folders:
                            os.rename(os.path.join(current_folder, file), os.path.join(current_folder, 'Xtruth.csv'))
                        else:
                            os.remove(os.path.join(current_folder, file))
                    

                    
