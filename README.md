Half-Cheetah PPO gait policy trained in Pybullet. Different sets of configurations were experimented with to determine the best hyperparameters for the Half-Cheetah environment.
# How to install dependencies in project
--------------------------------------
```console
conda create --name half-cheetah --file requirements.txt
cd pybullet-gym
pip install -e .
```console
This should install all what needed in the project then run,
```console
conda activate half-cheetah
```
Afterwards, you can run the test enviroment by calling,
```console
python test.py
```
