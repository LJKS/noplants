# noplants

## Setting up a conda environment
In ourder to use our code certain dependencies must be met. We propose setting up a conda environment. 
Pleas install Miniconda on Linux according to the following link: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

All the needed dependencies are listed in the environment.yml file and a respective environment can be created with it.
In order to do so, please first clone the git environment and then open a terminal and go to the respective directory. Then type:

```console
(base) username@dev:~/noplants$ conda env create -f environment.yml # creates the sepecified environment
(base) username@dev:~/noplants$ conda actvate killingplants # now code can be executed
(killingplants) username@dev:~/noplants$ conda deactivate # deactivates environment
(base) username@dev:~/noplans$
```

## Prepare targets
In order to make most of the data availabe we set up a script preparing the data for our network implementation.
The prepare_targets.py takes high resolution pitures and their high resolution labes, crobs them into small subimages and transform the targets so that the three colour channels of resulting RGB image give the probability for belonging to a certan class (good plant, weed, ground). 

To run the script please adjust the following parameter in the hyperparameters.py: 
```python
# Data Preparation
ORIGIN_LBL_DIRECTORY = 'stem_lbl_human' # folder with rare data
ORIGIN_DATA_DIRECTORY = 'stem_data' # folder with labeled data
SUBPICS = 200
CROP_SIZE = (256, 256, 3)
# please create following directory 
SAVE_LBL = 'stem_lbl_cropped_container/stem_lbl_cropped/' 
SAVE_DATA = 'stem_data_cropped_container/stem_data_cropped/'
```
After creating the saving paths as specified run the sript.
```console
(killingplants) usr@dev:~/noplants$ python prepare_targets.py
```
