# This function generates a random geometry x for meta-material dataset. The boundary of the generated data is [-1, 1.273] for first 4 columns and [-1,1] for the last 4 columns. The reason why the first 4 column has a non-1 boundary is due to historic reason and since changing the normalization would change our best behaving model weights, we kept it that way
# This is only the data_x.csv generator, after generating this, please go to NA/predict.py and run the create_mm_dataset() function to get the data_y.csv which is the spectra of the meta-material dataset. Pls be reminded that the neural simulator only has the accuracy of 6e-5 only at the given range above.
# Running this file again would help to create a new set of meta-material dataset, which would help make sure that the models you chose from the 10 trained ones are not biased towards the validataion set instead of the real test performance.
import numpy as np

data_num = 10000
x_dim = 8
# Generate random number
data_x_1 = np.random.uniform(size=(data_num,4), low=-1, high=1.273)
data_x_2 = np.random.uniform(size=(data_num,4), low=-1, high=1)
data_x = np.concatenate([data_x_1, data_x_2],axis=1)
print('data_x now has shape:', np.shape(data_x))
np.savetxt('dataIn/data_x.csv', data_x)

