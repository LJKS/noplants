# HYPERPARAMETERS

# Training
MODEL_SAVE_DIR = 'models/exp2'
DATA_TRAIN_STEM_LBL = 'stem_lbl_cropped_container'
DATA_TRAIN_STEM = 'stem_data_cropped_container'
DATA_TRAIN_SEG_LBL = 'seg_lbl_cropped_container'
DATA_TRAIN_SEG = 'seg_data_cropped_container'
EPOCHS = 100
SAVE_STEPS = 100
BATCH_SIZE = 2
CLOCK=False

# Testing
DATA_TEST = 'test_imgs'
#DATA_TEST = 'seg_data_cropped_container/seg_data_cropped'
DATA_TEST_LBL = 'test_imgs'
#DATA_TEST_LBL = 'seg_lbl_cropped_container/seg_lbl_cropped'
BATCH_SIZE_TEST = 1
NUM_TESTS = 1
MODEL_TEST_DIR = "models/model_lmachine_1/_step_17799"

# Data Preparation
ORIGIN_LBL_DIRECTORY = 'segmentation_data/train/lbl'
ORIGIN_DATA_DIRECTORY = 'segmentation_data/train/img'
SUBPICS = 200
CROP_SIZE = (256, 256, 3)
# please create following directory
SAVE_LBL = 'stem_lbl_cropped_container/stem_lbl_cropped/'
SAVE_DATA = 'stem_data_cropped_container/stem_data_cropped/'
