# HYPERPARAMETERS

# Training
MODEL_SAVE_DIR = 'models/exp3'
DATA_TRAIN_STEM_LBL = 'stem_lbl_cropped_container'
DATA_TRAIN_STEM = 'stem_data_cropped_container'
DATA_TRAIN_SEG_LBL = 'seg_lbl_cropped_container'
DATA_TRAIN_SEG = 'seg_data_cropped_container'
SAVE_STEPS = 100 #after how many saving steps the model and a progress garphic is saved
BATCH_SIZE = 2
CLOCK = False # printing duration of a trining step

# Testing
DATA_TEST = 'test_imgs'
DATA_TEST_LBL = 'test_imgs'
BATCH_SIZE_TEST = 1
NUM_TESTS = 1
MODEL_TEST_DIR = "models/model_lmachine_1/_step_17799"

# Data Preparation
ORIGIN_LBL_DIRECTORY = 'segmentation_data/train/lbl'
ORIGIN_DATA_DIRECTORY = 'segmentation_data/train/img'
SUBPICS = 200
CROP_SIZE = (256, 256, 3)
# please create following directories, take for training later
SAVE_LBL = 'stem_lbl_cropped_container/stem_lbl_cropped/'
SAVE_DATA = 'stem_data_cropped_container/stem_data_cropped/'
