import os

# Directories
DIR_HOME        = os.path.expanduser('~')
DIR_CKPT        = 'checkpoints'
DIR_LST         = 'lists'
DIR_MODEL       = 'models'
DIR_SUMMARY     = 'summary'
DIR_LOG         = 'logs'
DIR_BESTCKPT    = 'bestckpt'
DIR_SCORE       = 'score'

DIR_DATA_RAW      = os.path.join(DIR_HOME, 'data', 'rgbd-dataset')
DIR_DATA_NOFRINGE = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed-nofringe')
DIR_DATA_FRINGE   = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed-fringe')
DIR_DATA_MASKED   = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed-masked')
DIR_DATA_4D       = os.path.join(DIR_HOME, 'data', 'rgbd-dataset-processed4d')
#DIR_DATA_EVAL_RAW = os.path.join(DIR_HOME, 'data', 'rgbd-dataset_eval')
#DIR_DATA_EVAL     = os.path.join(DIR_HOME, 'data', 'rgbd-dataset_eval-processed')

N_FEAT_RAND = 5
DIR_DATA_MASKED_FEAT = [os.path.join(DIR_HOME, 'data', 'rgbd-feat-masked-'+str(i+1)) for i in range(N_FEAT_RAND)]

DIR_DATA_AUX      = os.path.join(DIR_HOME, 'data', 'rgbd-aux')
if not os.path.exists(DIR_DATA_AUX): os.makedirs(DIR_DATA_AUX)

# Lists
#PTH_FULLTRAIN_LST    = os.path.join(DIR_LST, 'fulltrain.lst')
#PTH_FULLEVAL_LST     = os.path.join(DIR_LST, 'fulleval.lst')
PTH_FULL_LST         = os.path.join(DIR_LST, 'fulllist.lst')
PTH_TRAIN_LST        = [os.path.join(DIR_LST, 'train_'+str(trial+1)+'.lst') for trial in range(10)]
PTH_TRAIN_SHORT_LST  = [os.path.join(DIR_LST, 'train_short_'+str(trial+1)+'.lst') for trial in range(10)]
PTH_EVAL_LST         = [os.path.join(DIR_LST, 'eval_'+str(trial+1)+'.lst') for trial in range(10)]
PTH_TESTINSTANCE_IDS = os.path.join(DIR_LST, 'testinstance_ids.txt')
PTH_DICT             = os.path.join(DIR_LST, 'dictionary.lst')
PTH_TRIAL_SPLIT      = os.path.join(DIR_LST, 'trial_split.npy')

# Model
PTH_WEIGHT_ALEX = os.path.join(DIR_MODEL, 'bvlc_alexnet.npy') # AlexNet's pretrained model
PTH_MEAN_IMG    = os.path.join(DIR_MODEL, 'ilsvrc_2012_mean.npy') # mean image of imagenet dataset
PTH_RGB_MEAN    = os.path.join(DIR_MODEL, 'rgb_mean_BGR.npy')
PTH_DEP_MEAN    = os.path.join(DIR_MODEL, 'dep_mean_BGR.npy')
PTH_RGB_MODEL   = os.path.join(DIR_MODEL, 'rgb_model.npy') # rgb model trained in phase 1
PTH_DEP_MODEL   = os.path.join(DIR_MODEL, 'dep_model.npy') # depth model trained in phase 1
PTH_FUS_MODEL   = os.path.join(DIR_MODEL, 'fus_model.npy') # fusion model trained in phase 2

# Classes
CLASSES         = open(PTH_DICT, 'r').read().splitlines()

# Parameters
N_TRIALS        = 10
IMG_S           = 227 # size of a square image
IMG_RAW_S       = 256

# Extensions of RGBD dataset
EXT_RGB         = '_crop.png'
EXT_D           = '_depthcrop.png'
EXT_MASK        = '_maskcrop.png'
EXT_RGB_FEAT    = '_crop.npy'
EXT_D_FEAT      = '_depthcrop.npy'
