# Learning Prehensile Dexterity by Imitating and Emulating State-only Observations
This is a repository containing the code for the paper "Learning Prehensile Dexterity by Imitating and Emulating State-only Observations".

Project webpage: [CIMER](https://sites.google.com/view/cimer-2024/)

Paper link: [Learning Prehensile Dexterity by Imitating and Emulating State-only Observations](https://arxiv.org/abs/2404.05582)

## Environment Setup
### Install mujoco2.0.0
Adapted from [Ben's tutorial 2.1](https://www.cs.cmu.edu/~cga/controls-intro-22/kantor/How_to_Install_MuJoCo_on_Ubuntu_V1.pdf)

1. Setup the directory
```
cd ~
mkdir .mujoco
cd .mujoco
```
2. Download the MuJoCo version [2.0 binaries](https://www.roboti.us/index.html) and select the mujoco200 Linux option. Or if you are feeling adventurous here’s the direct download link: https://www.roboti.us/download/mujoco200_linux.zip.
3. Get the license: Go to https://www.roboti.us/license.html
4. Unzip the zipped file and place it in the directory `∼/.mujoco/mujoco200` and place
your license key (mjkey.txt) in `∼/.mujoco/mujoco200/bin/mjkey.txt` and `~/.mujoco/mjkey.txt`.
5. Test this installation by navigating to `∼/.mujoco/mujoco200/bin` and executing `./simulate ../model/humanoid.xml.`
7. Add the following 4 commands at the end of `~/.bashrc`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
```
6. Source ~/.bashrc to commit the changes
```
source ~/.bashrc
```
### Setup repo and install conda environment
1. Navigate to your installation directory and run:
```
git clone git@github.com:GT-STAR-Lab/CIMER.git
cd mjrl
conda update conda
conda env create -f setup/env.yml
conda activate mjrl-env
pip install -e .
pip install mujoco-py==2.0.2.8
```
2. Verify the installation of mujoco-py by running `python` in current conda environment (mjrl-env) in terminal and `import mujoco_py`. If mujoco_py is installed successfully, it should be (compiled and) imported without errors.
```
python3
import mujoco_py
```
3. Install mj_envs:
```
cd ../mj_envs
pip install -e .
```
4. Troubleshooting:
- Missing GL version: install GLEW by `sudo apt-get install -y libglew-dev`
- If a Cython related error occurs when compiling (`import mujoco_py`, check the version of gcc and Cython
```
conda install -c conda-forge gcc=12.1.0
pip install "Cython<3"
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
## Additional notes
We indeed provide the learned Koopman Matrix for the Motion Generation policy (Under `CIMER/hand_dapg/dapg/controller_training/koopman_without_vel` folder). If you would like to learn the Motion Generation policy yourself, please refer to our previous project ([KODex](https://sites.google.com/view/kodex-corl)) for more details.
## Bibtex
```
@article{han2024CIMER,
  title={Learning Prehensile Dexterity by Imitating and Emulating State-only Observations},
  author={Han, Yunhai and Chen, Zhenyang and Ravichandar, Harish},
  journal={arXiv preprint arXiv:2404.05582},
  year={2024}
}
```
