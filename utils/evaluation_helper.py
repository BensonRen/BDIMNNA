"""
This is the helper functions for evaluation purposes

"""
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from utils.helper_functions import simulator


def get_test_ratio_helper(flags):
    """
    The unified place for getting the test_ratio the same for all methods for the dataset,
    This is for easier changing for multi_eval
    """
    if flags.data_set == 'ballistics':
        #return 0.01                       # 100 in total
        return 0.02                        # 200 in total
    elif flags.data_set == 'sine_wave':
        #return 0.01                        # 100 in total
        return 0.02                        # 200 in total
    elif flags.data_set == 'robotic_arm':
        return 0.02                          # 200 in total
        #return 0.01                          # 100 in total
    elif flags.data_set == 'meta_material':
        return 0.01                        # 20k in total for Meta material original, ramdom has 10k
    else:
        print("Your dataset is none of the artificial datasets")
        return None

def compare_truth_pred(pred_file, truth_file, cut_off_outlier_thres=None, quiet_mode=False):
    """
    Read truth and pred from csv files, compute their mean-absolute-error and the mean-squared-error
    :param pred_file: full path to pred file
    :param truth_file: full path to truth file
    :return: mae and mse
    """
    if isinstance(pred_file, str):      # If input is a file name (original set up)
        pred = np.loadtxt(pred_file, delimiter=' ')
        truth = np.loadtxt(truth_file, delimiter=' ')
    elif isinstance(pred_file, np.ndarray):
        pred = pred_file
        truth = truth_file
    else:
        print('In the compare_truth_pred function, your input pred and truth is neither a file nor a numpy array')
    if not quiet_mode:
        print("in compare truth pred function in eval_help package, your shape of pred file is", np.shape(pred))
    if len(np.shape(pred)) == 1:
        # Due to Ballistics dataset gives some non-real results (labelled -999)
        valid_index = pred != -999
        if (np.sum(valid_index) != len(valid_index)) and not quiet_mode:
            print("Your dataset should be ballistics and there are non-valid points in your prediction!")
            print('number of non-valid points is {}'.format(len(valid_index) - np.sum(valid_index)))
        pred = pred[valid_index]
        truth = truth[valid_index]
        # This is for the edge case of ballistic, where y value is 1 dimensional which cause dimension problem
        pred = np.reshape(pred, [-1,1])
        truth = np.reshape(truth, [-1,1])
    mae = np.mean(np.abs(pred-truth), axis=1)
    mse = np.mean(np.square(pred-truth), axis=1)

    if cut_off_outlier_thres is not None:
        mse = mse[mse < cut_off_outlier_thres]
        mae = mae[mae < cut_off_outlier_thres]

        
    return mae, mse


def plotMSELossDistrib(pred_file, truth_file, flags, save_dir='data/'):
    if (flags.data_set == 'gaussian_mixture'):
        # get the prediction and truth array
        pred = np.loadtxt(pred_file, delimiter=' ')
        truth = np.loadtxt(truth_file, delimiter=' ')
        # get confusion matrix
        cm = confusion_matrix(truth, pred)
        cm = cm / np.sum(cm)
        # Calculate the accuracy
        accuracy = 0
        for i in range(len(cm)):
            accuracy += cm[i,i]
        print("confusion matrix is", cm)
        # Plotting the confusion heatmap
        f = plt.figure(figsize=[15,15])
        plt.title('accuracy = {}'.format(accuracy))
        sns.set(font_scale=1.4)
        sns.heatmap(cm, annot=True)
        eval_model_str = flags.eval_model.replace('/','_')
        f.savefig(save_dir + '{}.png'.format(eval_model_str),annot_kws={"size": 16})

    else:
        mae, mse = compare_truth_pred(pred_file, truth_file)
        plt.figure(figsize=(12, 6))
        plt.hist(mse, bins=100)
        plt.xlabel('Mean Squared Error')
        plt.ylabel('cnt')
        plt.suptitle('(Avg MSE={:.4e})'.format(np.mean(mse)))
        eval_model_str = flags.eval_model.replace('/','_')
        plt.savefig(os.path.join(save_dir,
                             '{}.png'.format(eval_model_str)))
        print('(Avg MSE={:.4e})'.format(np.mean(mse)))


def eval_from_simulator(Xpred_file, flags):
    """
    Evaluate using simulators from pred_file and return a new file with simulator results
    :param Xpred_file: The prediction file with the Xpred in its name
    :param data_set: The name of the dataset
    """
    Xpred = np.loadtxt(Xpred_file, delimiter=' ')
    Ypred = simulator(flags.data_set, Xpred)
    Ypred_file = Xpred_file.replace('Xpred', 'Ypred_Simulated')
    np.savetxt(Ypred_file, Ypred)
    Ytruth_file = Xpred_file.replace('Xpred','Ytruth')
    plotMSELossDistrib(Ypred_file, Ytruth_file, flags)
