"""
This file serves as a evaluation interface for the network
"""
# Built in
import os
# Torch

# Own
import flag_reader
from class_wrapper import Network
from model_maker import INN
from utils import data_reader
from utils import helper_functions
from utils.evaluation_helper import plotMSELossDistrib
from utils.evaluation_helper import get_test_ratio_helper

# Libs

def evaluate_from_model(model_dir, multi_flag=False, eval_data_all=False):
    """
    Evaluating interface. 1. Retreive the flags 2. get data 3. initialize network 4. eval
    :param model_dir: The folder to retrieve the model
    :param eval_data_all: The switch to turn on if you want to put all data in evaluation data
    :return: None
    """
    # Retrieve the flag object
    print("Retrieving flag object for parameters")
    if (model_dir.startswith("models")):
        model_dir = model_dir[7:]
        print("after removing prefix models/, now model_dir is:", model_dir)
    flags = helper_functions.load_flags(os.path.join("models", model_dir))
    flags.eval_model = model_dir                    # Reset the eval mode

    flags.test_ratio = get_test_ratio_helper(flags)

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags, eval_data_all=eval_data_all)
    print("Making network now")

    # Make Network
    ntwk = Network(INN, flags, train_loader, test_loader, inference_mode=True, saved_model=flags.eval_model)
    print(ntwk.ckpt_dir)
    print("number of trainable parameters is :")
    pytorch_total_params = sum(p.numel() for p in ntwk.model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # Evaluation process
    print("Start eval now:")
    if multi_flag:
        ntwk.evaluate_multiple_time()
    else:
        pred_file, truth_file = ntwk.evaluate()

    # Plot the MSE distribution
    if flags.data_set != 'meta_material' and not multi_flag: 
        plotMSELossDistrib(pred_file, truth_file, flags)
    print("Evaluation finished")
   
def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None

def evaluate_different_dataset(multi_flag, eval_data_all):
     """
     This function is to evaluate all different datasets in the model with one function call
     """
     #data_set_list = []
     data_set_list = ["meta_material","sine_wave","ballistics","robotic_arm"]
     for eval_model in data_set_list:
        for j in range(10):
            useless_flags = flag_reader.read_flag()
            useless_flags.eval_model = "retrain" + str(j) + eval_model
            evaluate_from_model(useless_flags.eval_model, multi_flag=multi_flag, eval_data_all=eval_data_all)


if __name__ == '__main__':
    # Read the flag, however only the flags.eval_model is used and others are not used
    useless_flags = flag_reader.read_flag()

    print(useless_flags.eval_model)
    #evaluate_from_model(useless_flags.eval_model)
    #evaluate_from_model(useless_flags.eval_model, multi_flag=True)
    #evaluate_from_model(useless_flags.eval_model, multi_flag=False, eval_data_all=True)
    
    ##############################################
    # evaluate multiple dataset at the same time!#
    ##############################################
    evaluate_different_dataset(multi_flag=False, eval_data_all=False)
    #evaluate_different_dataset(multi_flag=True, eval_data_all=False)
    
    
    
    # Call the evaluate function from model
    # evaluate_from_model(useless_flags.eval_model)

