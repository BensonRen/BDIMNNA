"""
This file serves as a prediction interface for the network
"""
# Built in
import os
import sys
sys.path.append('../utils/')
# Torch

# Own
from ensemble_mm import flag_reader_ensemble
from ensemble_mm.class_wrapper_ensemble import Network
from ensemble_mm.model_maker_ensemble import Forward
from utils import data_reader

# Libs
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from utils.evaluation_helper import plotMSELossDistrib
import seaborn as sns
import matplotlib.pyplot as plt

def load_flags(save_dir, save_file="flags.obj"):
    """
    This function inflate the pickled object to flags object for reuse, typically during evaluation (after training)
    :param save_dir: The place where the obj is located
    :param save_file: The file name of the file, usually flags.obj
    :return: flags
    """
    with open(os.path.join(save_dir, save_file), 'rb') as f:     # Open the file
        flags = pickle.load(f)                                  # Use pickle to inflate the obj back to RAM
    return flags


def predict_from_model(pre_trained_model, Xpred_file, save_mode=False):
    """
    Predicting interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :return: None
    """
    # Retrieve the flag object
    print("This is doing the prediction for file", Xpred_file)
    print("Retrieving flag object for parameters")
    if (pre_trained_model.startswith("models")):
        eval_model = pre_trained_model[7:]
        print("after removing prefix models/, now model_dir is:", eval_model)
    
    flags = load_flags(pre_trained_model)                       # Get the pre-trained model
    flags.eval_model = pre_trained_model                    # Reset the eval mode

    # Get the data, this part is useless in prediction but just for simplicity
    #train_loader, test_loader = data_reader.read_data(flags)
    train_loader, test_loader = None, None
    print("Making network now")

    # Make Network
    ntwk = Network(Forward, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # Evaluation process
    print("Start eval now:")
    pred_file, truth_file = ntwk.predict(Xpred_file, save_mode=False)
     
    if save_mode:
        # Plot the MSE distribution
        flags.eval_model = pred_file.replace('.','_') # To make the plot name different
        plotMSELossDistrib(pred_file, truth_file, flags)
        print("Evaluation finished")

    return pred_file, truth_file, flags

def ensemble_predict(model_list, Xpred_file, plot_mode=True):
    """
    This predicts the output from an ensemble of models
    :param model_list: The list of model names to aggregate
    :param Xpred_file: The Xpred_file that you want to predict
    :return: The prediction Ypred_file
    """
    print("this is doing ensemble prediction for models :", model_list)
    pred_list = []
    # Get the predictions into a list of np array
    for pre_trained_model in model_list:
        pred_file, truth_file, flags = predict_from_model(pre_trained_model, Xpred_file, save_mode=False)
        #pred = np.loadtxt(pred_file, delimiter=' ')
        pred = pred_file
        pred_list.append(np.copy(np.expand_dims(pred, axis=2)))
    # Take the mean of the predictions
    pred_all = np.concatenate(pred_list, axis=2)
    pred_mean = np.mean(pred_all, axis=2)
    save_name = Xpred_file.replace('Xpred', 'Ypred_ensemble')
    np.savetxt(save_name, pred_mean)

    # saving the plot down
    save_name_ensemble = Xpred_file.split('/')[-1][:-4]
    print('save name of emsemble = ', save_name_ensemble)
    flags.eval_model = 'ensemble_model' + save_name_ensemble
    if plot_mode:
        plotMSELossDistrib(save_name, truth_file, flags)




def predict_all(models_dir="data"):
    """
    This function predict all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if 'Xpred' in file and 'meta_material' in file:                     # Only meta material has this need currently
            print("predicting for file", file)
            predict_from_model("models/meta_materialreg0.0005trail_2_complexity_swipe_layer1000_num6", 
            os.path.join(models_dir,file))
    return None


def ensemble_predict_master(model_dir, Xpred_file, plot_mode=True):
    model_list = []
    for model in os.listdir(model_dir):
        if os.path.isdir(os.path.join(model_dir,model)):
            model_list.append(os.path.join(model_dir, model))
    print('model_list', model_list)
    ensemble_predict(model_list, Xpred_file, plot_mode=plot_mode)

def ensemble_predict_all(model_dir, Xpred_file_dir, plot_mode=True):
    """
    use ensemble predict a bunch of Xpred file
    """
    for file in os.listdir(Xpred_file_dir):
        if 'Xpred' in file and 'meta_material' in file:
            ensemble_predict_master(model_dir, os.path.join(Xpred_file_dir, file), plot_mode=plot_mode)

if __name__ == '__main__':
    #predict_all('/work/sr365/multi_eval/Random/meta_material')
    #ensemble_predict_master('/work/sr365/MM_ensemble/models/','datapool/Xpred.csv')
    #ensemble_predict_all('/work/sr365/MM_ensemble/models/','/hpc/home/sr365/NIPS/idlm_Pytorch/INN_FrEIA/data')
    #model_list = ['cINN','MDN','NA','Tandem','VAE','NA_noboundary']
    model_list = ['NA_noboundary']
    for model in model_list:
        ensemble_predict_all('/work/sr365/MM_ensemble/models/',
                             '/work/sr365/multi_eval/' + model + '/meta_material/',
                             plot_mode=False)
