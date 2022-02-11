# Reinforcement Learning for Intelligent Control

Final project of the subject *Intelligent Control*, which is part of the *Masters degree in Systems and Control Engineering*.

<p align="center">
  <img src="https://github.com/a-r2/RL4IC/blob/main/Environments.png?raw=true"/>
</p>

## 1. Description

The objective of this project is to implement several state-of-the-art reinforcement learning algorithms in environments that are considered interesting from the point of view of control theory due to their continuous action space.

Reinforcement learning is particularly useful in those cases when a full model of the system is not known, it is very complex or the uncertainty of such model is too high to succesfully apply conventional control techniques.

For this purpose, a software tool has been created, following a user-friendly approach, that allows the user to play around with these algorithms, by tweaking and testing them in different scenarios.

## 2. Requirements

* [Python 3.8 and pip](https://www.python.org/downloads/)
* [Git](https://desktop.github.com/)
* [Virtualenv 20.3.1](https://pypi.org/project/virtualenv/)
* [Stable-baselines3 0.10.0](https://pypi.org/project/stable-baselines3/)
* [Box2D 2.3.10](https://pypi.org/project/Box2D/)
* [Gym-cartpole-swingup 0.1.0](https://pypi.org/project/gym-cartpole-swingup/)

## 3. Installation

Tested on Ubuntu 20.10 and Windows 10:

1. Clone this project's Git repository inside any directory: ```git clone https://github.com/zkubixz/RL4IC```
2. Create a Python virtual environment (in order to avoid conflicts between modules): ```virtualenv venv```
3. Activate the virtual environment: ```source venv/bin/activate```
4. Install the necessary modules inside the virtual environment (verify ```(venv)``` appears on the commands window beforehand): ```pip install -r requirements.txt``` on Linux or ```pip install -r requirements_windows.txt``` on Windows. Alternatively, these modules can be installed separatedly: ```pip install stable-baselines3 box2d gym-cartpole-swingup```

## 4. Execution

Just run it! (```python3 run.py```)
