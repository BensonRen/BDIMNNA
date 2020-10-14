"""
The parameter file storing the parameters for VAE Model
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
DIM_Z = 3
DIM_X = 4
DIM_Y = 2
DIM_SPEC = None
LINEAR_D = [DIM_Y + DIM_Z, 500, 500, 500, 500, 500, 500, 500,    DIM_X]           # Linear units for Decoder
LINEAR_E = [DIM_Y + DIM_X, 500, 500, 500, 500, 500, 500, 500, 2*DIM_Z]                   # Linear units for Encoder
LINEAR_SE = []                      # Linear units for spectra encoder
CONV_OUT_CHANNEL_SE = []
CONV_KERNEL_SIZE_SE = []
CONV_STRIDE_SE = []
#LINEAR_SE = [150, 500, 500, 500, 500, DIM_Y]                      # Linear units for spectra encoder
#CONV_OUT_CHANNEL_SE = [4, 4, 4]
#CONV_KERNEL_SIZE_SE = [5, 5, 8]
#CONV_STRIDE_SE = [1, 1, 2]

# Optimization params
KL_COEFF = 0.005
OPTIM = "Adam"
REG_SCALE = 5e-3
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096
EVAL_STEP = 20
TRAIN_STEP = 500
VERB_STEP = 1
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = -float('inf')

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
#GEOBOUNDARY =[30, 52, 42, 52]
GEOBOUNDARY =[-1,1,-1,1]
NORMALIZE_INPUT = True

# Running specific params
USE_CPU_ONLY = False
#EVAL_MODEL = "ballistics_3M_mse_best"
#EVAL_MODEL = "robotic_arm"
EVAL_MODEL = "retrain_time_evalsine_wavekl_coeff0.02lr0.001reg0.005"
#EVAL_MODEL = "sine_wavekl_coeff0.02lr0.001reg0.005"
#EVAL_MODEL = "meta_materiallayer_num12unit_1000reg0.005trail3"
