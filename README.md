# Learning Prehensile Dexterity by Imitating and Emulating State-only Observations
This is a repository containing the code for the paper "Learning Prehensile Dexterity by Imitating and Emulating State-only Observations".

Project webpage: [CIMER](https://sites.google.com/view/cimer-2024/)

Paper link: [Learning Prehensile Dexterity by Imitating and Emulating State-only Observations](https://arxiv.org/abs/2404.05582)

## Environment Setup
**1. Install mujoco2.1.0**:
```
# Download binary file of mujoco2.1.0
cd ~
mkdir .mujoco
cd .mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mv mujoco210-linux-x86_64.tar.gz mujoco210
tar -zxvf mujoco210
# Add Environment variables to ~/.bashrc
sudo gedit ~/.bashrc
# Add the following 4 commands at the end of .bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
# Source ~/.bashrc to commit the changes
source ~/.bashrc
```
**2. Create a conda environment for mjrl and install mjrl**:
```
cd mjrl
# !!! Delete line 21 (mujoco-py) in mjrl/setup/env.yml, later we will install it manually !!!
conda update conda
conda env create -f setup/env.yml
source activate mjrl-env
pip install -e .
# Now install mujoco-py
pip install mujoco-py==2.1.2.14
# We need to check whether mujoco-py is installed successfully. Run python in current conda environment (mjrl-env) and import mujoco_py.
# If mujoco_py is installed successfully, it should be (compiled and) imported without errors.
python3
import mujoco_py
# If mujoco_py is imported for the first time, it will be compiled automatically.
# If a Cython related error occurs, try changing the version of gcc and Cython
conda install -c conda-forge gcc=12.1.0
pip install Cython==3.0.0a10
```
**3. Install mj_envs**:
```
cd mj_envs
conda activate mjrl-env
pip install -e .
```

## Policy Visualization
We provide with several trained policies for quick visualization. Under `CIMER` folder, run the follow commands:
### Hammer task
**CIMER**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/controller_training/visualize.py --eval_data Samples/Hammer/Hammer_task.pickle --visualize True --save_fig True --config Samples/Hammer/CIMER/job_config.json --policy Samples/Hammer/CIMER/best_eval_sr_policy.pickle
```
**SOIL**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/SOIL/visualize_policy_on_demos.py --config Samples/Hammer/SOIL/job_config.json --policy Samples/Hammer/SOIL/best_policy.pickle --demos Samples/Hammer/Hammer_task.pickle
```
**Pure RL**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/SOIL/visualize_policy_on_demos.py --config Samples/Hammer/Pure_RL/job_config.json --policy Samples/Hammer/Pure_RL/best_policy.pickle --demos Samples/Hammer/Hammer_task.pickle
```
### Relocate task
**CIMER**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/controller_training/visualize.py --eval_data Samples/Relocate/Relocate_task.pickle --visualize True --save_fig True --config Samples/Relocate/CIMER/job_config.json --policy Samples/Relocate/CIMER/best_eval_sr_policy.pickle
```
**SOIL**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/SOIL/visualize_policy_on_demos.py --config Samples/Relocate/SOIL/job_config.json --policy Samples/Relocate/SOIL/best_policy.pickle --demos Samples/Relocate/Relocate_task.pickle
```
**Pure RL**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/SOIL/visualize_policy_on_demos.py --config Samples/Relocate/Pure_RL/job_config.json --policy Samples/Relocate/Pure_RL/best_policy.pickle --demos Samples/Relocate/Relocate_task.pickle
```
### Door task
**CIMER**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/controller_training/visualize.py --eval_data Samples/Door/Door_task.pickle --visualize True --save_fig True --config Samples/Door/CIMER/job_config.json --policy Samples/Door/CIMER/best_eval_sr_policy.pickle
```
**SOIL**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/SOIL/visualize_policy_on_demos.py --config Samples/Door/SOIL/job_config.json --policy Samples/Door/SOIL/best_policy.pickle --demos Samples/Door/Door_task.pickle
```
**Pure RL**:
```
conda activate mjrl-env
MJPL python3 hand_dapg/dapg/SOIL/visualize_policy_on_demos.py --config Samples/Door/Pure_RL/job_config.json --policy Samples/Door/Pure_RL/best_policy.pickle --demos Samples/Door/Door_task.pickle
```

## Policy Training
We also provide codes to train new policies. Under `CIMER` folder, run the follow commands:
```
mkdir -p Training/Hammer
mkdir -p Training/Relocate
mkdir -p Training/Door
```
### Hammer task
**CIMER**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/controller_training/job_script.py --output Training/Hammer/CIMER --config hand_dapg/dapg/controller_training/dapg-hammer_PPO.txt --eval_data Samples/Hammer/Hammer_task.pickle
```
**SOIL**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/SOIL/job_script.py --output Training/Hammer/SOIL --config hand_dapg/dapg/SOIL/soil-hammer.txt --eval_data Samples/Hammer/Hammer_task.pickle
```
**Pure RL**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/SOIL/job_script.py --output Training/Hammer/PureRL --config hand_dapg/dapg/SOIL/purerl-hammer.txt --eval_data Samples/Hammer/Hammer_task.pickle
```
### Relocate task
**CIMER**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/controller_training/job_script.py --output Training/Relocate/CIMER --config hand_dapg/dapg/controller_training/dapg-relocate_PPO.txt --eval_data Samples/Relocate/Relocate_task.pickle
```
**SOIL**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/SOIL/job_script.py --output Training/Relocate/SOIL --config hand_dapg/dapg/SOIL/soil-relocate.txt --eval_data Samples/Relocate/Relocate_task.pickle
```
**Pure RL**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/SOIL/job_script.py --output Training/Relocate/PureRL --config hand_dapg/dapg/SOIL/purerl-relocate.txt --eval_data Samples/Relocate/Relocate_task.pickle
```
### Door task
**CIMER**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/controller_training/job_script.py --output Training/Door/CIMER --config hand_dapg/dapg/controller_training/dapg-door_PPO.txt --eval_data Samples/Door/Door_task.pickle
```
**SOIL**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/SOIL/job_script.py --output Training/Door/SOIL --config hand_dapg/dapg/SOIL/soil-door.txt --eval_data Samples/Door/Door_task.pickle
```
**Pure RL**:
```
conda activate mjrl-env
python3 hand_dapg/dapg/SOIL/job_script.py --output Training/Door/PureRL --config hand_dapg/dapg/SOIL/purerl-door.txt --eval_data Samples/Door/Door_task.pickle
```
## Bibtex
```
@misc{han2024CIMER,
      title={Learning Prehensile Dexterity by Imitating and Emulating State-only Observations}, 
      author={Yunhai Han and Zhenyang Chen and Harish Ravichandar},
      year={2024},
      eprint={2404.05582},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```