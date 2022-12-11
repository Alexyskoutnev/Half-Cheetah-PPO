How to install dependencies in project
--------------------------------------
1. conda create --name half-cheetah --file requirements.txt
2. cd pybullet-gym
3. pip install -e .
This should install all what needed in the project then run,
conda activate half-cheetah
Afterwards, you can run the test enviroment by calling,
python test.py