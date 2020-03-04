# HYPERPARAMETERS

# Training
MODEL_SAVE_DIR = 'models/ad_one_pic_only_flip'
DATA_TRAIN_LBL = 'stem_lbl_cropped_container'
DATA_TRAIN = 'stem_data_cropped_container'
EPOCHS = 100
SAVE_STEPS = 100


# Testing
DATA_TEST = 'stem_data_cropped_container/stem_data_cropped'
DATA_TEST_LBL = 'stem_lbl_cropped_container/stem_lbl_cropped'
BATCH_SIZE_TEST = 1
NUM_TESTS = 1
MODEL_TEST_DIR = "models/ad_one_pic_only_flip/model_epoch0_step_7899"
