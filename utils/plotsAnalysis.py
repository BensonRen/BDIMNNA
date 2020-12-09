import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from utils import helper_functions
from utils.evaluation_helper import compare_truth_pred
from sklearn.neighbors import NearestNeighbors
from pandas.plotting import table
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def InferenceAccuracyExamplePlot(model_name, save_name, title, sample_num=10,  fig_size=(15,5), random_seed=1,
                                 target_region=[0,300 ]):
    """
    The function to plot the Inference accuracy and compare with FFDS algorithm.
    It takes the Ypred and Ytruth file as input and plot the first <sample_num> of spectras.
    It also takes a random of 10 points to at as the target points.
    :param model_name: The model name as the postfix for the Ytruth file
    :param save_name:  The saving name of the figure
    :param title:  The saving title of the figure
    :param sample_num: The number of sample to plot for comparison
    :param fig_size: The size of the figure
    :param random_seed:  The random seed value
    :param target_region:  The region that the targets get
    :return:
    """
    # Get the prediction and truth file first
    Ytruth_file = os.path.join('data','test_Ytruth_{}.csv'.format(model_name))
    Ypred_file = os.path.join('data','test_Ypred_{}.csv'.format(model_name))
    Ytruth = pd.read_csv(Ytruth_file, header=None, delimiter=' ').values
    Ypred = pd.read_csv(Ypred_file, header=None, delimiter=' ').values

    # Draw uniform random distribution for the reference points
    np.random.seed(random_seed)     # To make sure each time we have same target points
    targets = target_region[0] + (target_region[1] - target_region[0]) * np.random.uniform(low=0, high=1, size=10) # Cap the random numbers within 0-299
    targets = targets.astype("int")
    # Make the frequency into real frequency in THz
    fre_low = 0.86
    fre_high = 1.5
    frequency = fre_low + (fre_high - fre_low)/len(Ytruth[0, :]) * np.arange(300)

    for i in range(sample_num):
        # Start the plotting
        f = plt.figure(figsize=fig_size)
        plt.title(title)
        plt.scatter(frequency[targets], Ytruth[i,targets], label='S*')
        plt.plot(frequency, Ytruth[i,:], label='FFDS')
        plt.plot(frequency, Ypred[i,:], label='Candidate')
        plt.legend()
        plt.ylim([0,1])
        plt.xlim([fre_low, fre_high])
        plt.grid()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transmittance")
        plt.savefig(os.path.join('data',save_name + str(i) + '.png'))


def RetrieveFeaturePredictionNMse(model_name):
    """
    Retrieve the Feature and Prediciton values and place in a np array
    :param model_name: the name of the model
    return Xtruth, Xpred, Ytruth, Ypred
    """
    # Retrieve the prediction and truth and prediction first
    feature_file = os.path.join('data', 'test_Xtruth_{}.csv'.format(model_name))
    pred_file = os.path.join('data', 'test_Ypred_{}.csv'.format(model_name))
    truth_file = os.path.join('data', 'test_Ytruth_{}.csv'.format(model_name))
    feat_file = os.path.join('data', 'test_Xpred_{}.csv'.format(model_name))

    # Getting the files from file name
    Xtruth = pd.read_csv(feature_file,header=None, delimiter=' ')
    Xpred = pd.read_csv(feat_file,header=None, delimiter=' ')
    Ytruth = pd.read_csv(truth_file,header=None, delimiter=' ')
    Ypred = pd.read_csv(pred_file,header=None, delimiter=' ')
    
    #retrieve mse, mae
    Ymae, Ymse = compare_truth_pred(pred_file, truth_file) #get the maes of y
    
    print(Xtruth.shape)
    return Xtruth.values, Xpred.values, Ytruth.values, Ypred.values, Ymae, Ymse

def ImportColorBarLib():
    """
    Import some libraries that used in a colorbar plot
    """
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib as mpl
    print("import sucessful")
    
    return mpl
  
def UniqueMarkers():
    import itertools
    markers = itertools.cycle(( 'x','1','+', '.', '*','D','v','h'))
    return markers
  
def SpectrumComparisonNGeometryComparison(rownum, colnum, Figsize, model_name, boundary = [-1,1,-1,1]):
    """
    Read the Prediction files and plot the spectra comparison plots
    :param SubplotArray: 2x2 array indicating the arrangement of the subplots
    :param Figsize: the size of the figure
    :param Figname: the name of the figures to save
    :param model_name: model name (typically a list of numebr containing date and time)
    """
    mpl = ImportColorBarLib()    #import lib
    
    Xtruth, Xpred, Ytruth, Ypred, Ymae, Ymse =  RetrieveFeaturePredictionNMse(model_name)  #retrieve features
    print("Ymse shape:",Ymse.shape)
    print("Xpred shape:", Xpred.shape)
    print("Xtrth shape:", Xtruth.shape)
    #Plotting the spectrum comaprison
    f = plt.figure(figsize=Figsize)
    fignum = rownum * colnum
    for i in range(fignum):
      ax = plt.subplot(rownum, colnum, i+1)
      plt.ylabel('Transmission rate')
      plt.xlabel('frequency')
      plt.plot(Ytruth[i], label = 'Truth',linestyle = '--')
      plt.plot(Ypred[i], label = 'Prediction',linestyle = '-')
      plt.legend()
      plt.ylim([0,1])
    f.savefig('Spectrum Comparison_{}'.format(model_name))
    
    """
    Plotting the geometry comparsion, there are fignum points in each plot
    each representing a data point with a unique marker
    8 dimension therefore 4 plots, 2x2 arrangement
    
    """
    #for j in range(fignum):
    pointnum = fignum #change #fig to #points in comparison
    
    f = plt.figure(figsize = Figsize)
    ax0 = plt.gca()
    for i in range(4):
      truthmarkers = UniqueMarkers() #Get some unique markers
      predmarkers = UniqueMarkers() #Get some unique markers
      ax = plt.subplot(2, 2, i+1)
      #plt.xlim([29,56]) #setting the heights limit, abandoned because sometime can't see prediciton
      #plt.ylim([41,53]) #setting the radius limits
      for j in range(pointnum):
        #Since the colored scatter only takes 2+ arguments, plot 2 same points to circumvent this problem
        predArr = [[Xpred[j, i], Xpred[j, i]] ,[Xpred[j, i + 4], Xpred[j, i + 4]]]
        predC = [Ymse[j], Ymse[j]]
        truthplot = plt.scatter(Xtruth[j,i],Xtruth[j,i+4],label = 'Xtruth{}'.format(j),
                                marker = next(truthmarkers),c = 'm',s = 40)
        predplot  = plt.scatter(predArr[0],predArr[1],label = 'Xpred{}'.format(j),
                                c =predC ,cmap = 'jet',marker = next(predmarkers), s = 60)
      
      plt.xlabel('h{}'.format(i))
      plt.ylabel('r{}'.format(i))
      rect = mpl.patches.Rectangle((boundary[0],boundary[2]),boundary[1] - boundary[0], boundary[3] - boundary[2],
																		linewidth=1,edgecolor='r',
                                   facecolor='none',linestyle = '--',label = 'data region')
      ax.add_patch(rect)
      plt.autoscale()
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),
                 mode="expand",ncol = 6, prop={'size': 5})#, bbox_to_anchor=(1,0.5))
    
    cb_ax = f.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = f.colorbar(predplot, cax=cb_ax)
    #f.colorbar(predplot)
    f.savefig('Geometry Comparison_{}'.format(model_name))


class HMpoint(object):
    """
    This is a HeatMap point class where each object is a point in the heat map
    properties:
    1. BV_loss: best_validation_loss of this run
    2. feature_1: feature_1 value
    3. feature_2: feature_2 value, none is there is no feature 2
    """
    def __init__(self, bv_loss, f1, f2 = None, f1_name = 'f1', f2_name = 'f2'):
        self.bv_loss = bv_loss
        self.feature_1 = f1
        self.feature_2 = f2
        self.f1_name = f1_name
        self.f2_name = f2_name
        #print(type(f1))
    def to_dict(self):
        return {
            self.f1_name: self.feature_1,
            self.f2_name: self.feature_2,
            self.bv_loss: self.bv_loss
        }


def HeatMapBVL(plot_x_name, plot_y_name, title,  save_name='HeatMap.png', HeatMap_dir = 'HeatMap',
                feature_1_name=None, feature_2_name=None,
                heat_value_name = 'best_validation_loss'):
    """
    Plotting a HeatMap of the Best Validation Loss for a batch of hyperswiping thing
    First, copy those models to a folder called "HeatMap"
    Algorithm: Loop through the directory using os.look and find the parameters.txt files that stores the
    :param HeatMap_dir: The directory where the checkpoint folders containing the parameters.txt files are located
    :param feature_1_name: The name of the first feature that you would like to plot on the feature map
    :param feature_2_name: If you only want to draw the heatmap using 1 single dimension, just leave it as None
    """
    one_dimension_flag = False          #indication flag of whether it is a 1d or 2d plot to plot
    #Check the data integrity 
    if (feature_1_name == None):
        print("Please specify the feature that you want to plot the heatmap");
        return
    if (feature_2_name == None):
        one_dimension_flag = True
        print("You are plotting feature map with only one feature, plotting loss curve instead")

    #Get all the parameters.txt running related data and make HMpoint objects
    HMpoint_list = []
    df_list = []                        #make a list of data frame for further use
    for subdir, dirs, files in os.walk(HeatMap_dir):
        for file_name in files:
             if (file_name == 'parameters.txt'):
                file_path = os.path.join(subdir, file_name) #Get the file relative path from 
                # df = pd.read_csv(file_path, index_col=0)
                flag = helper_functions.load_flags(subdir)
                flag_dict = vars(flag)
                df = pd.DataFrame()
                for k in flag_dict:
                    df[k] = pd.Series(str(flag_dict[k]), index=[0])
                print(df)
                if (one_dimension_flag):
                    df_list.append(df[[heat_value_name, feature_1_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]), eval(str(df[feature_1_name][0])), 
                                                f1_name = feature_1_name))
                else:
                    if feature_2_name == 'linear_unit':                         # If comparing different linear units
                        df['linear_unit'] = eval(df[feature_1_name][0])[1]
                        df['best_validation_loss'] = get_bvl(file_path)
                    if feature_2_name == 'kernel_second':                       # If comparing different kernel convs
                        print(df['conv_kernel_size'])
                        print(type(df['conv_kernel_size']))
                        df['kernel_second'] = eval(df['conv_kernel_size'][0])[1]
                        df['kernel_first'] = eval(df['conv_kernel_size'][0])[0]
                    df_list.append(df[[heat_value_name, feature_1_name, feature_2_name]])
                    HMpoint_list.append(HMpoint(float(df[heat_value_name][0]),eval(str(df[feature_1_name][0])),
                                                eval(str(df[feature_2_name][0])), feature_1_name, feature_2_name))
    
    print(df_list)
    #Concatenate all the dfs into a single aggregate one for 2 dimensional usee
    df_aggregate = pd.concat(df_list, ignore_index = True, sort = False)
    df_aggregate.astype({heat_value_name: 'float'})
    print("before transformation:", df_aggregate)
    [h, w] = df_aggregate.shape
    for i in range(h):
        for j in range(w):
            if isinstance(df_aggregate.iloc[i,j], str) and (isinstance(eval(df_aggregate.iloc[i,j]), list)):
                # print("This is a list!")
                df_aggregate.iloc[i,j] = len(eval(df_aggregate.iloc[i,j]))

    print("after transoformation:",df_aggregate)
    
    #Change the feature if it is a tuple, change to length of it
    for cnt, point in enumerate(HMpoint_list):
        print("For point {} , it has {} loss, {} for feature 1 and {} for feature 2".format(cnt, 
                                                                point.bv_loss, point.feature_1, point.feature_2))
        assert(isinstance(point.bv_loss, float))        #make sure this is a floating number
        if (isinstance(point.feature_1, tuple)):
            point.feature_1 = len(point.feature_1)
        if (isinstance(point.feature_2, tuple)):
            point.feature_2 = len(point.feature_2)

    
    f = plt.figure()
    #After we get the full list of HMpoint object, we can start drawing 
    if (feature_2_name == None):
        print("plotting 1 dimension HeatMap (which is actually a line)")
        HMpoint_list_sorted = sorted(HMpoint_list, key = lambda x: x.feature_1)
        #Get the 2 lists of plot
        bv_loss_list = []
        feature_1_list = []
        for point in HMpoint_list_sorted:
            bv_loss_list.append(point.bv_loss)
            feature_1_list.append(point.feature_1)
        print("bv_loss_list:", bv_loss_list)
        print("feature_1_list:",feature_1_list)
        #start plotting
        plt.plot(feature_1_list, bv_loss_list,'o-')
    else: #Or this is a 2 dimension HeatMap
        print("plotting 2 dimension HeatMap")
        #point_df = pd.DataFrame.from_records([point.to_dict() for point in HMpoint_list])
        df_aggregate = df_aggregate.reset_index()
        df_aggregate.sort_values(feature_1_name, axis=0, inplace=True)
        df_aggregate.sort_values(feature_2_name, axis=0, inplace=True)
        df_aggregate.sort_values(heat_value_name, axis=0, inplace=True)
        print("before dropping", df_aggregate)
        df_aggregate = df_aggregate.drop_duplicates(subset=[feature_1_name, feature_2_name], keep='first')
        print("after dropping", df_aggregate)
        point_df_pivot = df_aggregate.reset_index().pivot(index=feature_1_name, columns=feature_2_name, values=heat_value_name).astype(float)
        point_df_pivot = point_df_pivot.rename({'5': '05'}, axis=1)
        point_df_pivot = point_df_pivot.reindex(sorted(point_df_pivot.columns), axis=1)
        print("pivot=")
        csvname = HeatMap_dir + 'pivoted.csv'
        point_df_pivot.to_csv(csvname)
        print(point_df_pivot)
        sns.heatmap(point_df_pivot, cmap = "YlGnBu")
    plt.xlabel(plot_y_name)                 # Note that the pivot gives reversing labels
    plt.ylabel(plot_x_name)                 # Note that the pivot gives reversing labels
    plt.title(title)
    plt.savefig(save_name)


def PlotPossibleGeoSpace(figname, Xpred_dir, compare_original = False,calculate_diversity = None):
    """
    Function to plot the possible geometry space for a model evaluation result.
    It reads from Xpred_dir folder and finds the Xpred result insdie and plot that result
    :params figname: The name of the figure to save
    :params Xpred_dir: The directory to look for Xpred file which is the source of plotting
    :output A plot containing 4 subplots showing the 8 geomoetry dimensions
    """
    Xpred = helper_functions.get_Xpred(Xpred_dir)
    
    Xtruth = helper_functions.get_Xtruth(Xpred_dir)

    f = plt.figure()
    ax0 = plt.gca()
    print(np.shape(Xpred))
    if (calculate_diversity == 'MST'):
        diversity_Xpred, diversity_Xtruth = calculate_MST(Xpred, Xtruth)
    elif (calculate_diversity == 'AREA'):
        diversity_Xpred, diversity_Xtruth = calculate_AREA(Xpred, Xtruth)

    for i in range(4):
      ax = plt.subplot(2, 2, i+1)
      ax.scatter(Xpred[:,i], Xpred[:,i + 4],s = 3,label = "Xpred")
      if (compare_original):
          ax.scatter(Xtruth[:,i], Xtruth[:,i+4],s = 3, label = "Xtruth")
      plt.xlabel('h{}'.format(i))
      plt.ylabel('r{}'.format(i))
      plt.xlim(-1,1)
      plt.ylim(-1,1)
      plt.legend()
    if (calculate_diversity != None):
        plt.text(-4, 3.5,'Div_Xpred = {}, Div_Xtruth = {}, under criteria {}'.format(diversity_Xpred, diversity_Xtruth, calculate_diversity), zorder = 1)
    plt.suptitle(figname)
    f.savefig(figname+'.png')

def PlotPairwiseGeometry(figname, Xpred_dir):
    """
    Function to plot the pair-wise scattering plot of the geometery file to show
    the correlation between the geometry that the network learns
    """
    
    Xpredfile = helper_functions.get_Xpred(Xpred_dir)
    Xpred = pd.read_csv(Xpredfile, header=None, delimiter=' ')
    f=plt.figure()
    axes = pd.plotting.scatter_matrix(Xpred, alpha = 0.2)
    #plt.tight_layout()
    plt.title("Pair-wise scattering of Geometery predictions")
    plt.savefig(figname)

def calculate_AREA(Xpred, Xtruth):
    """
    Function to calculate the area for both Xpred and Xtruth under using the segmentation of 0.01
    """
    area_list = np.zeros([2,4])
    X_list = [Xpred, Xtruth]
    binwidth = 0.05
    for cnt, X in enumerate(X_list):
        for i in range(4):
            hist, xedges, yedges = np.histogram2d(X[:,i],X[:,i+4], bins = np.arange(-1,1+binwidth,binwidth))
            area_list[cnt, i] = np.mean(hist > 0)
    X_histgt0 = np.mean(area_list, axis = 1)
    assert len(X_histgt0) == 2
    return X_histgt0[0], X_histgt0[1]

def calculate_MST(Xpred, Xtruth):
    """
    Function to calculate the MST for both Xpred and Xtruth under using the segmentation of 0.01
    """

    MST_list = np.zeros([2,4])
    X_list = [Xpred, Xtruth]
    for cnt, X in enumerate(X_list):
        for i in range(4):
            points = X[:,i:i+5:4]
            distance_matrix_points = distance_matrix(points,points, p = 2)
            csr_mat = csr_matrix(distance_matrix_points)
            Tree = minimum_spanning_tree(csr_mat)
            MST_list[cnt,i] = np.sum(Tree.toarray().astype(float))
    X_MST = np.mean(MST_list, axis = 1)
    return X_MST[0], X_MST[1]


def get_bvl(file_path):
    """
    This is a helper function for 0119 usage where the bvl is not recorded in the pickled object but in .txt file and needs this funciton to retrieve it
    """
    df = pd.read_csv(file_path, delimiter=',')
    bvl = 0
    for col in df:
        if 'best_validation_loss' in col:
            print(col)
            strlist = col.split(':')
            bvl = eval(strlist[1][1:-2])
    if bvl == 0:
        print("Error! We did not found a bvl in .txt.file")
    else:
        return float(bvl)


def MeanAvgnMinMSEvsTry(data_dir):
    """
    Plot the mean average Mean and Min Squared error over Tries
    :param data_dir: The directory where the data is in
    :param title: The title for the plot
    :return:
    """
    # Read Ytruth file
    if not os.path.isdir(data_dir): 
        print("Your data_dir is not a folder in MeanAvgnMinMSEvsTry function")
        print("Your data_dir is:", data_dir)
        return
    Yt = pd.read_csv(os.path.join(data_dir, 'Ytruth.csv'), header=None, delimiter=' ').values
    print("shape of ytruth is", np.shape(Yt))
    # Get all the Ypred into list
    Ypred_list = []
    
    ####################################################################
    # Special handling for NA as it output file structure is different #
    ####################################################################
    if 'NA' in data_dir or 'BP' in data_dir: 
        l, w = np.shape(Yt)
        num_trails = 200
        Ypred_mat = np.zeros([l, num_trails, w])
        check_full = np.zeros(l)                                     # Safety check for completeness
        for files in os.listdir(data_dir):
            if '_Ypred_' in files:
                Yp = pd.read_csv(os.path.join(data_dir, files), header=None, delimiter=' ').values
                if len(np.shape(Yp)) == 1:                          # For ballistic data set where it is a coloumn only
                    Yp = np.reshape(Yp, [-1, 1])
                print("shape of Ypred file is", np.shape(Yp))
                # Truncating to the top num_trails inferences
                if len(Yp) != num_trails:
                    Yp = Yp[:num_trails,:]
                number_str = files.split('inference')[-1][:-4]
                print(number_str)
                number = int(files.split('inference')[-1][:-4])
                Ypred_mat[number, :, :] = Yp
                check_full[number] = 1
        assert np.sum(check_full) == l, 'Your list is not complete'
        # Finished fullfilling the Ypred mat, now fill in the Ypred list as before
        for i in range(num_trails):
            Ypred_list.append(Ypred_mat[:, i, :])
    else:
        for files in os.listdir(data_dir):
            if 'Ypred' in files:
                #print(files)
                Yp = pd.read_csv(os.path.join(data_dir, files), header=None, delimiter=' ').values
                if len(np.shape(Yp)) == 1:                          # For ballistic data set where it is a coloumn only
                    Yp = np.reshape(Yp, [-1, 1])
                #print("shape of Ypred file is", np.shape(Yp))
                Ypred_list.append(Yp)
    # Calculate the large MSE matrix
    mse_mat = np.zeros([len(Ypred_list), len(Yt)])
    print("shape of mse_mat is", np.shape(mse_mat))
    
    for ind, yp in enumerate(Ypred_list):
        if np.shape(yp) != np.shape(Yt):
            print("Your Ypred file shape does not match your ytruth, however, we are trying to reshape your ypred file into the Ytruth file shape")
            print("shape of the Yp is", np.shape(yp))
            print("shape of the Yt is", np.shape(Yt))
            yp = np.reshape(yp, np.shape(Yt))
            if ind == 1:
                print(np.shape(yp))
        # For special case yp = -999, it is out of numerical simulator
        print("shape of np :", np.shape(yp))
        print("shape of Yt :", np.shape(Yt))
        if np.shape(yp)[1] == 1:                        # If this is ballistics
            print("this is ballistics dataset, checking the -999 situation now")
            valid_index = yp[:, 0] != -999
            print("shape of valid flag :", np.shape(valid_index))
            valid_num = np.sum(valid_index)
            yp = yp[valid_index, :]
            Yt_valid = Yt[valid_index, :]
            print("shape of np after valid :", np.shape(yp))
            print("shape of Yt after valid :", np.shape(Yt_valid))
            mse = np.mean(np.square(yp - Yt_valid), axis=1)
            if valid_num == len(valid_index):
                mse_mat[ind, :] = mse
            else:
                mse_mat[ind, :valid_num] = mse
                mse_mat[ind, valid_num:] = np.mean(mse)
        else:
            mse = np.mean(np.square(yp - Yt), axis=1)
            mse_mat[ind, :] = mse
    print("shape of the yp is", np.shape(yp)) 
    print("shape of mse is", np.shape(mse))
    
    # Shuffle array and average results
    shuffle_number = 0
    if shuffle_number > 0:
        # Calculate the min and avg from mat
        mse_min_list = np.zeros([len(Ypred_list), shuffle_number])
        mse_avg_list = np.zeros([len(Ypred_list), shuffle_number])
    
        for shuf in range(shuffle_number):
            rng = np.random.default_rng()
            rng.shuffle(mse_mat)
            for i in range(len(Ypred_list)):
                mse_avg_list[i, shuf] = np.mean(mse_mat[:i+1, :])
                mse_min_list[i, shuf] = np.mean(np.min(mse_mat[:i+1, :], axis=0))
        # Average the shuffled result
        mse_avg_list = np.mean(mse_avg_list, axis=1)
        mse_min_list = np.mean(mse_min_list, axis=1)
    else:               # Currently the results are not shuffled as the statistics are enough
        # Calculate the min and avg from mat
        mse_min_list = np.zeros([len(Ypred_list),])
        mse_avg_list = np.zeros([len(Ypred_list),])
        mse_std_list = np.zeros([len(Ypred_list),])
        mse_quan2575_list = np.zeros([2, len(Ypred_list)])
        if 'NA' in data_dir:            
            cut_front = 0
        else:
            cut_front = 0
        for i in range(len(Ypred_list)-cut_front):
            mse_avg_list[i] = np.mean(mse_mat[cut_front:i+1+cut_front, :])
            mse_min_list[i] = np.mean(np.min(mse_mat[cut_front:i+1+cut_front, :], axis=0))
            mse_std_list[i] = np.std(np.min(mse_mat[cut_front:i+1+cut_front, :], axis=0))
            mse_quan2575_list[0, i] = np.percentile(np.min(mse_mat[cut_front:i+1+cut_front, :], axis=0), 25)
            mse_quan2575_list[1, i] = np.percentile(np.min(mse_mat[cut_front:i+1+cut_front, :], axis=0), 75)

    # Save the list down for further analysis
    np.savetxt(os.path.join(data_dir, 'mse_mat.csv'), mse_mat, delimiter=' ')
    np.savetxt(os.path.join(data_dir, 'mse_avg_list.txt'), mse_avg_list, delimiter=' ')
    np.savetxt(os.path.join(data_dir, 'mse_min_list.txt'), mse_min_list, delimiter=' ')
    np.savetxt(os.path.join(data_dir, 'mse_std_list.txt'), mse_std_list, delimiter=' ')
    np.savetxt(os.path.join(data_dir, 'mse_quan2575_list.txt'), mse_quan2575_list, delimiter=' ')

    # Plotting
    f = plt.figure()
    x_axis = np.arange(len(Ypred_list))
    plt.plot(x_axis, mse_avg_list, label='avg')
    plt.plot(x_axis, mse_min_list, label='min')
    plt.legend()
    plt.xlabel('inference number')
    plt.ylabel('mse error')
    plt.savefig(os.path.join(data_dir,'mse_plot vs time'))
    return None


def MeanAvgnMinMSEvsTry_all(data_dir): # Depth=2 now based on current directory structure
    """
    Do the recursive call for all sub_dir under this directory
    :param data_dir: The mother directory that calls
    :return:
    """
    for dirs in os.listdir(data_dir):
        print("entering :", dirs)
        print("this is a folder?:", os.path.isdir(os.path.join(data_dir, dirs)))
        print("this is a file?:", os.path.isfile(os.path.join(data_dir, dirs)))
        #if this is not a folder 
        if not os.path.isdir(os.path.join(data_dir, dirs)):
            print("This is not a folder", dirs)
            continue
        for subdirs in os.listdir(os.path.join(data_dir, dirs)):
            if os.path.isfile(os.path.join(data_dir, dirs, subdirs, 'mse_min_list.txt')):                               # if this has been done
                continue;
            print("enters folder", subdirs)
            MeanAvgnMinMSEvsTry(os.path.join(data_dir, dirs, subdirs))
    return None


def DrawBoxPlots_multi_eval(data_dir, data_name, save_name='Box_plot'):
    """
    The function to draw the statitstics of the data using a Box plot
    :param data_dir: The mother directory to call
    :param data_name: The data set name
    """
    # Predefine name of mse_mat
    mse_mat_name = 'mse_mat.csv'

    #Loop through directories
    mse_mat_dict = {}
    for dirs in os.listdir(data_dir):
        print(dirs)
        if not os.path.isdir(os.path.join(data_dir, dirs)):# or 'NA' in dirs:
            print("skipping due to it is not a directory")
            continue;
        for subdirs in os.listdir((os.path.join(data_dir, dirs))):
            if subdirs == data_name:
                # Read the lists
                mse_mat = pd.read_csv(os.path.join(data_dir, dirs, subdirs, mse_mat_name),
                                           header=None, delimiter=' ').values
                # Put them into dictionary
                mse_mat_dict[dirs] = mse_mat

    # Get the box plot data
    box_plot_data = []
    for key in sorted(mse_mat_dict.keys()):
        data = mse_mat_dict[key][0, :]
        # data = np.mean(mse_mat_dict[key], axis=1)
        box_plot_data.append(data)
        print('{} avg error is : {}'.format(key, np.mean(data)))

    # Start plotting
    f = plt.figure()
    plt.boxplot(box_plot_data, patch_artist=True, labels=sorted(mse_mat_dict.keys()))
    plt.ylabel('mse')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.savefig(os.path.join(data_dir, data_name + save_name + '.png'))
    return None


def DrawAggregateMeanAvgnMSEPlot(data_dir, data_name, save_name='aggregate_plot', gif_flag=False): # Depth=2 now based on current directory structure
    """
    The function to draw the aggregate plot for Mean Average and Min MSEs
    :param data_dir: The mother directory to call
    :param data_name: The data set name
    :param git_flag: Plot are to be make a gif
    :return:
    """
    # Predefined name of the avg lists
    min_list_name = 'mse_min_list.txt'
    avg_list_name = 'mse_avg_list.txt'
    std_list_name = 'mse_std_list.txt'
    quan2575_list_name = 'mse_quan2575_list.txt'

    # Loop through the directories
    avg_dict, min_dict, std_dict, quan2575_dict = {}, {}, {}, {}
    for dirs in os.listdir(data_dir):
        # Dont include NA for now and check if it is a directory
        print("entering :", dirs)
        print("this is a folder?:", os.path.isdir(os.path.join(data_dir, dirs)))
        print("this is a file?:", os.path.isfile(os.path.join(data_dir, dirs)))
        if not os.path.isdir(os.path.join(data_dir, dirs)):# or dirs == 'NA':# or 'boundary' in dirs::
            print("skipping due to it is not a directory")
            continue;
        for subdirs in os.listdir((os.path.join(data_dir, dirs))):
            if subdirs == data_name:
                # Read the lists
                mse_avg_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, avg_list_name),
                                           header=None, delimiter=' ').values
                mse_min_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, min_list_name),
                                           header=None, delimiter=' ').values
                mse_std_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, std_list_name),
                                           header=None, delimiter=' ').values
                mse_quan2575_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, quan2575_list_name),
                                           header=None, delimiter=' ').values
                print("The quan2575 error range shape is ", np.shape(mse_quan2575_list))
                print("dirs =", dirs)
                print("shape of mse_min_list is:", np.shape(mse_min_list))
                # Put them into dictionary
                avg_dict[dirs] = mse_avg_list
                min_dict[dirs] = mse_min_list
                std_dict[dirs] = mse_std_list
                quan2575_dict[dirs] = mse_quan2575_list
    print(min_dict)
       
    def plotDict(dict, name, data_name=None, logy=False, time_in_s_table=None, plot_points=51, avg_dict=None, resolution=5, err_dict=None):
        """
        :param name: the name to save the plot
        :param dict: the dictionary to plot
        :param logy: use log y scale
        :param time_in_s_table: a dictionary of dictionary which stores the averaged evaluation time
                in seconds to convert the graph
        :param plot_points: Number of points to be plot
        :param resolution: The resolution of points
        :param err_dict: The error bar dictionary which takes the error bar input
        :param avg_dict: The average dict for plotting the starting point
        """
        color_dict = {"NA":"g", "Tandem": "b", "VAE": "r","cINN":"m", 
                        "INN":"k", "Random": "y","MDN": "violet", "Tandem__with_boundary":"orange", "NA__boundary_prior":"violet","NA__no_boundary_prior":"m","INN_new":"violet",
                        "NA__boundary_no_prior": "grey", "NA_noboundary": "olive"}
        f = plt.figure()
        for key in sorted(dict.keys()):
            x_axis = np.arange(len(dict[key])).astype('float')
            x_axis += 1
            if time_in_s_table is not None:
                x_axis *= time_in_s_table[data_name][key]
            print("printing", name)
            print(key)
            #print(dict[key])
            if err_dict is None:
                plt.plot(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution],c=color_dict[key],label=key)
            else:
                print(np.shape(err_dict[key]))
                plt.errorbar(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution],c=color_dict[key],
                        yerr=err_dict[key][:, :plot_points:resolution], label=key.replace('_',' '), capsize=5)#, errorevery=resolution)#,
                        #dash_capstyle='round')#, uplims=True, lolims=True)
        if logy:
            ax = plt.gca()
            ax.set_yscale('log')
        plt.legend(loc=1)
        if time_in_s_table is not None:
            plt.xlabel('inference time (s)')
        else:
            plt.xlabel('# of inference made (T)')
        #plt.ylabel('MSE')
        plt.xlim([-1, plot_points+2])
        if 'ball' in data_name:
            data_name = 'D1: ' + data_name
        elif 'sine' in data_name:
            data_name = 'D2: ' + data_name
        elif 'robo' in data_name:
            data_name = 'D3: ' + data_name
        elif 'meta' in data_name:
            data_name = 'D4: ' + data_name

        plt.title(data_name.replace('_',' '), fontsize=20)
        plt.grid(True, axis='both',which='both',color='b',alpha=0.3)
        plt.savefig(os.path.join(data_dir, data_name + save_name + name), transparent=True)
        plt.close('all')
    plotDict(min_dict,'_minlog_quan2575.png', logy=True, avg_dict=avg_dict, err_dict=quan2575_dict, data_name=data_name)
    #plotDict(min_dict,'_min_quan2575.png', logy=False, avg_dict=avg_dict, err_dict=quan2575_dict)
    #plotDict(min_dict,'_minlog_std.png', logy=True, avg_dict=avg_dict, err_dict=std_dict)
    #plotDict(min_dict,'_min_std.png', logy=False, avg_dict=avg_dict, err_dict=std_dict)

    # if plot gifs
    if not gif_flag:
        return
    else:
        for i in range(2,20,1):
            plotDict(min_dict, str(i), logy=True, plot_points=i)
        for i in range(20,1000,20):
            plotDict(min_dict, str(i), logy=True, plot_points=i)



def DrawEvaluationTime(data_dir, data_name, save_name='evaluation_time', logy=False, limit=1000):
    """
    This function is to plot the evaluation time behavior of different algorithms on different data sets
    :param data_dir: The mother directory where all the results are put
    :param data_name: The specific dataset to analysis
    :param save_name: The saving name of the plotted figure
    :param logy: take logrithmic at axis y
    :param limit: the limit of x max
    :return:
    """
    eval_time_dict = {}
    for dirs in os.listdir(data_dir):
        print("entering :", dirs)
        print("this is a folder?:", os.path.isdir(os.path.join(data_dir, dirs)))
        print("this is a file?:", os.path.isfile(os.path.join(data_dir, dirs)))
        if not os.path.isdir(os.path.join(data_dir, dirs)):
            print("skipping due to it is not a directory")
            continue;
        for subdirs in os.listdir((os.path.join(data_dir, dirs))):
            if subdirs == data_name:
                # Read the lists
                eval_time = pd.read_csv(os.path.join(data_dir, dirs, subdirs, 'evaluation_time.txt'),
                                           header=None, delimiter=',').values[:, 1]
                # Put them into dictionary
                eval_time_dict[dirs] = eval_time

    # Plotting
    f = plt.figure()
    for key in sorted(eval_time_dict.keys()):
        average_time = eval_time_dict[key][-1] / len(eval_time_dict[key])
        plt.plot(np.arange(len(eval_time_dict[key])), eval_time_dict[key], label=key + 'average_time={0:.2f}s'.format(average_time))
    plt.legend()
    plt.xlabel('#inference trails')
    plt.ylabel('inference time taken (s)')
    plt.title(data_name + 'evaluation_time')
    plt.xlim([0, limit])
    if logy:
        ax = plt.gca()
        ax.set_yscale('log')
        plt.savefig(os.path.join(data_dir, data_name + save_name + 'logy.png'))
    else:
        plt.savefig(os.path.join(data_dir, data_name + save_name + '.png'))

if __name__ == '__main__':
    MeanAvgnMinMSEvsTry_all('../multi_eval')
    datasets = ['meta_material', 'robotic_arm','sine_wave','ballistics']
    for dataset in datasets:
        DrawAggregateMeanAvgnMSEPlot('../multi_eval', dataset)
