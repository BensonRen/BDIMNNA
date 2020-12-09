import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

def importData(directory, x_range, y_range):
    # pull data into python, should be either for training set or eval set
    train_data_files = []
    for file in os.listdir(os.path.join(directory)):
        if file.endswith('.csv'):
            train_data_files.append(file)
    print(train_data_files)
    # get data
    ftr = []
    lbl = []
    for file_name in train_data_files:
        # import full arrays
        print(x_range)
        ftr_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',
                                header=None, usecols=x_range)
        lbl_array = pd.read_csv(os.path.join(directory, file_name), delimiter=',',
                                header=None, usecols=y_range)
        # append each data point to ftr and lbl
        for params, curve in zip(ftr_array.values, lbl_array.values):
            ftr.append(params)
            lbl.append(curve)
    ftr = np.array(ftr, dtype='float32')
    lbl = np.array(lbl, dtype='float32')
    for i in range(len(ftr[0, :])):
        print('For feature {}, the max is {} and min is {}'.format(i, np.max(ftr[:, i]), np.min(ftr[:, i])))
    return ftr, lbl


# check that the data we're using is distributed uniformly and generate some plots
def check_data(input_directory, col_range=range(2, 10), col_names=('h1','h2','h3','h4','r1','r2','r3','r4')):
    for file in os.listdir(input_directory):
        if file.endswith('.csv'):
            print('\n histogram for file {}'.format(os.path.join(input_directory, file)))
            with open(os.path.join(input_directory, file)) as f:
                data = pd.read_csv(f, header=None, delimiter=',', usecols=col_range,
                                   names=col_names)
                for name in col_names:
                    print('{} unique values for {}: {}'.format(len(data[name].unique()),
                                                               name,
                                                               np.sort(data[name].unique()))
                          )
                hist = data.hist(bins=13, figsize=(10, 5))
                plt.tight_layout()
                plt.show()
                print('done plotting column data')




# add columns of derived values to the input data
# for now, just ratios of the inputs
def addColumns(input_directory, output_directory, x_range, y_range):
    print('adding columns...')
    print('importing data')
    data_files = []
    for file in os.listdir(os.path.join(input_directory)):
        if file.endswith('.csv'):
            data_files.append(file)
    for file in data_files:
        ftr = pd.read_csv(os.path.join(input_directory, file), delimiter=',', usecols=x_range,
                          names=['id0', 'id1'] + ['ftr' + str(i) for i in range(8)])
        lbl = pd.read_csv(os.path.join(input_directory, file), delimiter=',', usecols=y_range, header=None)

        print('computing new columns')
        newCol = 0  # count the number of new columns added
        for i in range(2, 6):  # first four are heights
            for j in range(6, 10):  # second four are radii
                ftr['ftr{}'.format(j-2)+'/'+'ftr{}'.format(i-2)] = ftr.apply(lambda row: row.iloc[j]/row.iloc[i], axis=1)
                newCol += 1
        print('total new columns added is {}\n'.format(newCol))
        print('exporting')
        data_total = pd.concat([ftr, pd.DataFrame(lbl)], axis=1)
        # data_total['2010'] = data_total.str.replace('\n', ' ')
        with open(os.path.join(output_directory, file[:-4] + '_div01.csv'), 'a') as file_out:
            # for some stupid reason to_csv seems to insert a blank line between every single data line
            # maybe try to fix this issue later
            #data_total.to_csv(file_out, sep=',', index=False, header=False)
            data_out = data_total.values
            np.savetxt(file_out, data_out, delimiter=',', fmt='%f')
    print('done')

# finds simulation files in input_dir and finds + saves the subset that adhere to the geometry contraints r_bound
# and h_bound
def gridShape(input_dir, output_dir, shapeType, r_bounds, h_bounds):

    files_to_filter = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            files_to_filter.append(os.path.join(input_dir, file))

    print('filtering through {} files...'.format(len(files_to_filter)))
    print('bounds on radii: [{}, {}], bounds on heights: [{}, {}]...'.format(r_bounds[0], r_bounds[1],
                                                                       h_bounds[0], h_bounds[1]))
    lengthsPreFilter = []
    lengthsPostFilter = []
    for file in files_to_filter:
        with open(file, 'r') as f:
            geom_specs = pd.read_csv(f, delimiter=',', header=None).values
            geoms_filt = []
            geoms_filtComp = []
            lengthsPreFilter.append(len(geom_specs))
            if shapeType=='corner':
                print('cutting a corner of the data...')
                for geom_spec in geom_specs:
                    hs = geom_spec[2:6]
                    rs = geom_spec[6:10]
                    if (np.all(hs >= h_bounds[0]) and np.all(hs <= h_bounds[1])) or \
                       (np.all(rs >= r_bounds[0]) and np.all(rs <= r_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            elif shapeType=='rCut':
                print('cutting based on r values only...')
                for geom_spec in geom_specs:
                    rs = geom_spec[6:10]
                    if (np.all(rs >= r_bounds[0]) and np.all(rs <= r_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            elif shapeType == 'hCut':
                print('cutting based on h values only...')
                for geom_spec in geom_specs:
                    hs = geom_spec[2:6]
                    if (np.all(hs >= h_bounds[0]) and np.all(hs <= h_bounds[1])):
                        geoms_filt.append(geom_spec)
                    else:
                        geoms_filtComp.append(geom_spec)
            else:
                print('shapeType {} is not valid.'.format(shapeType))
                return
            geoms_filt = np.array(geoms_filt)
            geoms_filtComp = np.array(geoms_filtComp)
            lengthsPostFilter.append(len(geoms_filt))
            print('{} reduced from {} to {}, ({}%)'.format(file, lengthsPreFilter[-1],
                                                           lengthsPostFilter[-1],
                                                           100*np.round(lengthsPostFilter[-1]/lengthsPreFilter[-1], 4)))

        save_file = os.path.join(output_dir, os.path.split(file)[-1][:-4] + '_filt')
        # save the filtered geometries, for training
        with open(save_file + '.csv', 'w+') as f:
            np.savetxt(f, geoms_filt, delimiter=',', fmt='%f')

        # save the all the goemetries filtered out, for evaluation
        with open(save_file + 'Comp.csv', 'w+') as f:
            np.savetxt(f, geoms_filtComp, delimiter=',', fmt='%f')


    print('\nAcross all files: of original {} combos, {} remain ({}%)'.format(sum(lengthsPreFilter),
                                                                              sum(lengthsPostFilter),
                                                                              100*np.round(sum(lengthsPostFilter)/ \
                                                                                       sum(lengthsPreFilter), 4)
                                                                              ))


def read_data_meta_material( x_range, y_range, geoboundary,  batch_size=128,
                 data_dir=os.path.abspath(''), rand_seed=1234, normalize_input = True, test_ratio=0.02,
                             eval_data_all=False):
    """
      :param input_size: input size of the arrays
      :param output_size: output size of the arrays
      :param x_range: columns of input data in the txt file
      :param y_range: columns of output data in the txt file
      :param cross_val: number of cross validation folds
      :param val_fold: which fold to be used for validation
      :param batch_size: size of the batch read every time
      :param shuffle_size: size of the batch when shuffle the dataset
      :param data_dir: parent directory of where the data is stored, by default it's the current directory
      :param rand_seed: random seed
      :param test_ratio: if this is not 0, then split test data from training data at this ratio
                         if this is 0, use the dataIn/eval files to make the test set
      """
    """
    Read feature and label
    :param is_train: the dataset is used for training or not
    :param train_valid_tuple: if it's not none, it will be the names of train and valid files
    :return: feature and label read from csv files, one line each time
    """
    if eval_data_all:
        test_ratio = 0.999
    # get data files
    print('getting data files...')
    ftrTrain, lblTrain = importData(os.path.join(data_dir, 'dataIn'), x_range, y_range)
    if (test_ratio > 0):
        print("Splitting training data into test set, the ratio is:", str(test_ratio))
        ftrTrain, ftrTest, lblTrain, lblTest = train_test_split(ftrTrain, lblTrain,
                                                                test_size=test_ratio, random_state=rand_seed)
    else:
        print("Using separate file from dataIn/Eval as test set")
        ftrTest, lblTest = importData(os.path.join(data_dir, 'dataIn', 'eval'), x_range, y_range)

    print('total number of training samples is {}'.format(len(ftrTrain)))
    print('total number of test samples is {}'.format(len(ftrTest)),
          'length of an input spectrum is {}'.format(len(lblTest[0])))
    print('downsampling output curves')
    # resample the output curves so that there are not so many output points
    # drop the beginning of the curve so that we have a multiple of 300 points
    if len(lblTrain[0]) > 2000:                                 # For Omar data set
        lblTrain = lblTrain[::, len(lblTest[0])-1800::6]
        lblTest = lblTest[::, len(lblTest[0])-1800::6]

    print('length of downsampled train spectra is {} for first, {} for final, '.format(len(lblTrain[0]),
                                                                                       len(lblTrain[-1])),
          'set final layer size to be compatible with this number')
    print('length of downsampled test spectra is {}, '.format(len(lblTest[0]),
                                                         len(lblTest[-1])),
          'set final layer size to be compatible with this number')

    # determine lengths of training and validation sets
    num_data_points = len(ftrTrain)
    #train_length = int(.8 * num_data_points)

    print('generating torch dataset')
    assert np.shape(ftrTrain)[0] == np.shape(lblTrain)[0]
    assert np.shape(ftrTest)[0] == np.shape(lblTest)[0]

    #Normalize the data if instructed using boundary and the current numbers are larger than 1
    if normalize_input and np.max(np.max(ftrTrain)) > 30:
        ftrTrain[:,0:4] = (ftrTrain[:,0:4] - (geoboundary[0] + geoboundary[1]) / 2)/(geoboundary[1] - geoboundary[0]) * 2
        ftrTest[:,0:4] = (ftrTest[:,0:4] - (geoboundary[0] + geoboundary[1]) / 2)/(geoboundary[1] - geoboundary[0]) * 2
        ftrTrain[:,4:] = (ftrTrain[:,4:] - (geoboundary[2] + geoboundary[3]) / 2)/(geoboundary[3] - geoboundary[2]) * 2
        ftrTest[:,4:] = (ftrTest[:,4:] - (geoboundary[2] + geoboundary[3]) / 2)/(geoboundary[3] - geoboundary[2]) * 2

    train_data = MetaMaterialDataSet(ftrTrain, lblTrain, bool_train= True)
    test_data = MetaMaterialDataSet(ftrTest, lblTest, bool_train= False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


def get_data_into_loaders(data_x, data_y, batch_size, DataSetClass, rand_seed=1234, test_ratio=0.05):
    """
    Helper function that takes structured data_x and data_y into dataloaders
    :param data_x: the structured x data
    :param data_y: the structured y data
    :param rand_seed: the random seed
    :param test_ratio: The testing ratio
    :return: train_loader, test_loader: The pytorch data loader file
    """
    # Normalize the input
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_ratio,
                                                        random_state=rand_seed)
    print('total number of training sample is {}, the dimension of the feature is {}'.format(len(x_train), len(x_train[0])))
    print('total number of test sample is {}'.format(len(y_test)))

    # Construct the dataset using a outside class
    train_data = DataSetClass(x_train, y_train)
    test_data = DataSetClass(x_test, y_test)

    # Construct train_loader and test_loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def normalize_np(x):
    """
    Normalize the x into [-1, 1] range in each dimension [:, i]
    :param x: np array to be normalized
    :return: normalized np array
    """
    for i in range(len(x[0])):
        x_max = np.max(x[:, i])
        x_min = np.min(x[:, i])
        x_range = (x_max - x_min ) /2.
        x_avg = (x_max + x_min) / 2.
        x[:, i] = (x[:, i] - x_avg) / x_range
        assert np.max(x[:, i]) == 1, 'your normalization is wrong'
        assert np.min(x[:, i]) == -1, 'your normalization is wrong'
    return x


def read_data_sine_test_1d(flags, eval_data_all=False):
    """
    Data reader function for testing sine wave 1d data (1d to 1d)
    :param flags: Input flags
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Sine_test/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class_1d_to_1d, test_ratio=0.99)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class_1d_to_1d, test_ratio=flags.test_ratio)
  

def read_data_ballistics(flags, eval_data_all=False):
    """
    Data reader function for the ballistic data set
    :param flags: Input flags
    :return train_loader and test_loader in pytorch data set format (unnormalized)
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Ballistics/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_x[:,3] /= 15
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=flags.test_ratio)


def read_data_gaussian_mixture(flags, eval_data_all=False):
    """
    Data reader function for the gaussian mixture data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    # Read the data
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Gaussian_Mixture/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    data_y = np.squeeze(data_y)                             # Squeeze since this is a 1 column label
    data_x = normalize_np(data_x)
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=flags.test_ratio)


def read_data_sine_wave(flags, eval_data_all=False):
    """
    Data reader function for the sine function data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Sinusoidal_Wave/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    #data_x = normalize_np(data_x)
    #data_y = normalize_np(data_y)
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_class, test_ratio=flags.test_ratio)


def read_data_naval_propulsion(flags, eval_data_all=False):
    """
    Data reader function for the naval propulsion data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Naval_Propulsion/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    data_x = normalize_np(data_x)
    data_y = normalize_np(data_y)
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)


def read_data_robotic_arm(flags, eval_data_all=False):
    """
    Data reader function for the robotic arm data set
    :param flags: Input flags
    :return: train_loader and test_loader in pytorch data set format (normalized)
    """
    data_dir = os.path.join(flags.data_dir, 'Simulated_DataSets/Robotic_Arm/')
    data_x = pd.read_csv(data_dir + 'data_x.csv', header=None).astype('float32').values
    data_y = pd.read_csv(data_dir + 'data_y.csv', header=None).astype('float32').values
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)

def read_data_ensemble_MM(flags, eval_data_all=False):
    data_dir = os.path.join('../', 'Simulated_DataSets', 'Meta_material_Neural_Simulator', 'dataIn')
    data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None, sep=',').astype('float32').values
    data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None, sep=',').astype('float32').values
    print("I am reading data from the:", data_dir)
    
    
    #data_x = pd.read_csv(os.path.join(data_dir, 'data_x.csv'), header=None, sep=' ').astype('float32').values
    #data_y = pd.read_csv(os.path.join(data_dir, 'data_y.csv'), header=None, sep=' ').astype('float32').values
    if eval_data_all:
        return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=0.999)
    return get_data_into_loaders(data_x, data_y, flags.batch_size, SimulatedDataSet_regress, test_ratio=flags.test_ratio)

def read_data(flags, eval_data_all=False):
    """
    The data reader allocator function
    The input is categorized into couple of different possibilities
    0. meta_material
    1. gaussian_mixture
    2. sine_wave
    3. naval_propulsion
    4. robotic_arm
    5. ballistics
    :param flags: The input flag of the input data set
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return:
    """
    if flags.data_set == 'meta_material':
        print("This is a meta-material dataset")
        if flags.geoboundary[0] == -1:          # ensemble produced ones
            print("reading from ensemble place")
            train_loader, test_loader = read_data_ensemble_MM(flags, eval_data_all=eval_data_all)
        else:
            train_loader, test_loader = read_data_meta_material(x_range=flags.x_range,
                                                                y_range=flags.y_range,
                                                                geoboundary=flags.geoboundary,
                                                                batch_size=flags.batch_size,
                                                                normalize_input=flags.normalize_input,
                                                                data_dir=flags.data_dir,
                                                                eval_data_all=eval_data_all,
                                                                test_ratio=flags.test_ratio)
            print("I am reading data from:", data_dir)
        # Reset the boundary is normalized
        if flags.normalize_input:
            flags.geoboundary_norm = [-1, 1, -1, 1]
    elif flags.data_set == 'gaussian_mixture':
        train_loader, test_loader = read_data_gaussian_mixture(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'sine_wave':
        train_loader, test_loader = read_data_sine_wave(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'naval_propulsion':
        train_loader, test_loader = read_data_naval_propulsion(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'robotic_arm':
        train_loader, test_loader = read_data_robotic_arm(flags, eval_data_all=eval_data_all)
    elif flags.data_set == 'ballistics':
        train_loader, test_loader = read_data_ballistics(flags,  eval_data_all=eval_data_all)
    elif flags.data_set == 'sine_test_1d':
        train_loader, test_loader = read_data_sine_test_1d(flags, eval_data_all=eval_data_all)
    else:
        sys.exit("Your flags.data_set entry is not correct, check again!")
    return train_loader, test_loader

class MetaMaterialDataSet(Dataset):
    """ The Meta Material Dataset Class """
    def __init__(self, ftr, lbl, bool_train):
        """
        Instantiate the Dataset Object
        :param ftr: the features which is always the Geometry !!
        :param lbl: the labels, which is always the Spectra !!
        :param bool_train:
        """
        self.ftr = ftr
        self.lbl = lbl
        self.bool_train = bool_train
        self.len = len(ftr)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.ftr[ind, :], self.lbl[ind, :]


class SimulatedDataSet_class_1d_to_1d(Dataset):
    """ The simulated Dataset Class for classification purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]


class SimulatedDataSet_class(Dataset):
    """ The simulated Dataset Class for classification purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind]


class SimulatedDataSet_regress(Dataset):
    """ The simulated Dataset Class for regression purposes"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        return self.x[ind, :], self.y[ind, :]

