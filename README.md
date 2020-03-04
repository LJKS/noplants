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
