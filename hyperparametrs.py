# HYPERPARAMETERS

# Training
MODEL_SAVE_DIR = 'models/exp1'
DATA_TRAIN_LBL = 'stem_lbl_cropped_container'
DATA_TRAIN = 'stem_data_cropped_container'
EPOCHS = 100
SAVE_STEPS = 100
BATCH_SIZE = 6
CLOCK=False

# Testing
DATA_TEST = 'stem_data_cropped_container/stem_data_cropped'
DATA_TEST_LBL = 'stem_lbl_cropped_container/stem_lbl_cropped'
BATCH_SIZE_TEST = 1
NUM_TESTS = 1
MODEL_TEST_DIR = "models/ad_one_pic_only_flip/model_epoch0_step_5899"

# Data Preparation
ORIGIN_LBL_DIRECTORY = 'stem_lbl_human'
ORIGIN_DATA_DIRECTORY = 'stem_data'
SUBPICS = 200
CROP_SIZE = (256, 256, 3)
# please create following directory
SAVE_LBL = 'stem_lbl_cropped_container/stem_lbl_cropped/'
SAVE_DATA = 'stem_data_cropped_container/stem_data_cropped/'
