"""
The parameter file storing the parameters for INN Model
"""

# Define which data set you are using
# DATA_SET = 'meta_material'
# DATA_SET = 'gaussian_mixture'
# DATA_SET = 'sine_wave'
# DATA_SET = 'naval_propulsion'
DATA_SET = 'robotic_arm'
# DATA_SET = 'ballistics'
TEST_RATIO = 0.2

# Architectural Params
DIM_Z = 2
DIM_X = 4
DIM_Y = 2
DIM_TOT = 4
COUPLE_LAYER_NUM = 6
DIM_SPEC = None
SUBNET_LINEAR = []                                          # Linear units for Subnet FC layer
#LINEAR_SE = [150, 150, 150, 150, DIM_Y]                                              # Linear units for spectra encoder
LINEAR_SE = []                                              # Linear units for spectra encoder
CONV_OUT_CHANNEL_SE = []
CONV_KERNEL_SIZE_SE = []
CONV_STRIDE_SE = []
#CONV_OUT_CHANNEL_SE = [4, 4, 4]
#CONV_KERNEL_SIZE_SE = [5, 5, 8]
#CONV_STRIDE_SE = [1, 1, 2]

# Loss ratio
LAMBDA_MSE = 0.1             # The Loss factor of the MSE loss (reconstruction loss)
LAMBDA_Z = 300.             # The Loss factor of the latent dimension (converging to normal distribution)
LAMBDA_REV = 400.           # The Loss factor of the reverse transformation (let x converge to input distribution)
ZEROS_NOISE_SCALE = 5e-2          # The noise scale to add to
Y_NOISE_SCALE = 1e-2


# Optimization params
KL_COEFF = 1
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 20
GRAD_CLAMP = 15
TRAIN_STEP = 500
VERB_STEP = 50
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.9
STOP_THRESHOLD = -float('inf')
CKPT_DIR = 'models/'

# Data specific params
X_RANGE = [i for i in range(2, 10 )]
#Y_RANGE = [i for i in range(10 , 2011 )]                       # Real Meta-material dataset range
Y_RANGE = [i for i in range(10 , 310 )]                         # Artificial Meta-material dataset
FORCE_RUN = True
MODEL_NAME  = None
# MODEL_NAME  = 'dim_z_2 + wBN + 100 + lr1e-3 + reg5e-3'
DATA_DIR = '../'                                               # All simulated simple dataset
#DATA_DIR = '/work/sr365/'                                      # real Meta-material dataset
#DATA_DIR = '/work/sr365/NN_based_MM_data/'                      # Artificial Meta-material dataset
GEOBOUNDARY =[-1,1,-1,1]
#GEOBOUNDARY =[30, 52, 42, 52]
NORMALIZE_INPUT = True

# Running specific params
USE_CPU_ONLY = False
#EVAL_MODEL = "ballistics_Jakob"
#EVAL_MODEL = "robotic_armcouple_layer_num5dim_total4"
EVAL_MODEL = "retrain_time_evalsine_wavecouple_layer_num4dim_total4lambda_mse0.01trail1"
#EVAL_MODEL = "sine_wavecouple_layer_num4dim_total4lambda_mse0.01trail1"
#EVAL_MODEL = "sine_wavecouple_layer_num6dim_total5"
