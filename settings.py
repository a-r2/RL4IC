""" GLOBAL CONSTANTS """

import sys

if sys.platform == 'win32':

    SLASH = '\\'

else:

    SLASH = '/'

ALGS = ('A2C', 'DDPG', 'PPO', 'SAC', 'TD3') #algorithms
ENVS = ('Cart Pole', 'Inverted Pendulum', 'Lunar Lander', 'Mountain Car') #environments
ENVSPECS = ((5, 1, 1.0, -1.0), (3, 1, 2.0, -2.0), (8, 2, 1.0, -1.0), (2, 1, 1.0, -1.0))
#(observations space size, actions space size, high action, low action)
ENVSTITLES = ('TorchCartPoleSwingUp-v0', 'Pendulum-v0', 'LunarLanderContinuous-v2', \
             'MountainCarContinuous-v0') #environments gym titles
NOISETYPES = ('Default', 'Normal', 'Ornstein-Uhlenbeck') #types of noise
MODES = ('Deploy', 'Plot', 'Train') #operational modes
RECDEPTIME = (6, 10, 4, 4) #recommended deploy times
YES = ["y", "Y", "yes", "Yes", "yEs", "yeS", "YEs", "yES", "YES"] #valid affirmative answers
NO = ["n", "N", "no", "No", "nO", "NO"] #valid negative answers
MODELSDIR = 'Trained Models'
LOGDIR = 'Log'
BESTSUFFIX = '_Best.zip'
EVALSUFFIX = '_Eval.npz'
