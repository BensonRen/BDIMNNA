"""
Hyper-parameters of the Tandem model
"""
# Define which data set you are using
DATA_SET = 'meta_material'
# DATA_SET = 'gaussian_mixture'
# DATA_SET = 'sine_wave'
# DATA_SET = 'naval_propulsion'
# DATA_SET = 'robotic_arm'
# DATA_SET = 'ballistics'
TEST_RATIO = 0.2

# Model Architecture parameters
#LOAD_FORWARD_CKPT_DIR = 'pre_trained_forward/'
LOAD_FORWARD_CKPT_DIR = None
#LINEAR_F = [4, 500, 500, 500, 1]
LINEAR_F = [8, 1000, 1000, 1000, 1000, 150]
CONV_OUT_CHANNEL_F = [4, 4, 4]
CONV_KERNEL_SIZE_F = [8, 5, 5]
CONV_STRIDE_F = [2, 1, 1]

LINEAR_B = [150, 1000, 1000, 1000, 1000, 1000, 8]
CONV_OUT_CHANNEL_B = [4, 4, 4]
CONV_KERNEL_SIZE_B = [5, 5, 8]
CONV_STRIDE_B = [1, 1, 2]

"""
# Model Architectural Params for gaussian mixture dataset
#LINEAR_F = [3, 1000, 1000, 1000, 1000, 1000, 1000, 2]
LINEAR_F = [4, 500, 500, 500, 500,  1]
CONV_OUT_CHANNEL_F = []
CONV_KERNEL_SIZE_F = []
CONV_STRIDE_F = []

#LINEAR_B = [2, 500, 500, 500, 500, 500, 3]
LINEAR_B = [1, 500, 500, 500, 500, 4]
CONV_OUT_CHANNEL_B = []
CONV_KERNEL_SIZE_B = []
CONV_STRIDE_B = []
"""

# Optimizer parameters
OPTIM = "Adam"
REG_SCALE = 5e-4
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 1024
EVAL_STEP = 20
TRAIN_STEP = 500
VERB_STEP = 20
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-3 #-1 means dont stop

# Running specific parameter
USE_CPU_ONLY = False
DETAIL_TRAIN_LOSS_FORWARD = True
# Data-specific parameters
X_RANGE = [i for i in range(2, 10 )]
#Y_RANGE = [i for i in range(10 , 2011 )]                       # Real Meta-material dataset range
Y_RANGE = [i for i in range(10 , 310 )]                         # Artificial Meta-material dataset
MODEL_NAME = None 
DATA_DIR = '../'                                               # All simulated simple dataset
#DATA_DIR = '/work/sr365/'                                      # real Meta-material dataset
#DATA_DIR = '/work/sr365/NN_based_MM_data/'                      # Artificial Meta-material dataset
# DATA_DIR = '/home/omar/PycharmProjects/github/idlm_Pytorch-master/forward/'
GEOBOUNDARY = [-1,1,-1,1]
#GEOBOUNDARY = [30, 52, 42, 52]
NORMALIZE_INPUT = True


#EVAL_MODEL = 'robotic_armreg0.0001trail_1_complexity_swipe_layer500_num5'
#EVAL_MODEL = 'ballistics'
EVAL_MODEL = 'sine_wave'
#EVAL_MODEL = 'retrain_without_boundaryrobotic_armreg0.0001trail_1_complexity_swipe_layer500_num5'
#EVAL_MODEL = 'retrain_without_boundayballistics'
#EVAL_MODEL = 'meta_materialreg0.0005trail_2_complexity_swipe_layer250_num6'
#EVAL_MODEL = 'sine_wavereg0.0005trail_0_complexity_swipe_layer500_num8'
