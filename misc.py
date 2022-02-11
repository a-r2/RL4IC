import gym
import gym_cartpole_swingup
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.ddpg import MlpPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.td3 import MlpPolicy

from settings import *
from hyperparams import *

class RenderCallback(BaseCallback):
    """render gym environment during training"""

    def _on_step(self) -> bool:

        self.training_env.render()
        
        return True

def create_callback(UserInputsDict, Env):
    """create callbacks used during training"""

    TrainingSteps = UserInputsDict.get('TrainingSteps') 
    WatchAgent = UserInputsDict.get('WatchAgent') 

    PathLog = path_log(UserInputsDict)
    MaxStepsPerEpisode = Env.env.spec.max_episode_steps

    CallbacksList = list()

    if WatchAgent:
    
        CallbacksList.append(RenderCallback())

    CallbacksList.append(EvalCallback(Env, log_path=PathLog, best_model_save_path=PathLog, \
                         eval_freq=MaxStepsPerEpisode))

    return CallbackList(CallbacksList)

def create_env(UserInputsDict):
    """create gym environment"""

    SelEnvInd = UserInputsDict.get('SelEnvInd')

    env = gym.make(ENVSTITLES[SelEnvInd])

    Env = Monitor(env, filename=None) #wrap gym environment in monitor mode

    return Env

def create_path(UserInputsDict):
    """create algorithm-environment and log directories"""

    SelAlgInd = UserInputsDict.get('SelAlgInd') 
    SelEnvInd = UserInputsDict.get('SelEnvInd') 

    try: #list directories in algorithm-environment path

        DirList = os.listdir(path_dir(UserInputsDict))

    except: #create algorithm-environment path

        DirList = []
        os.makedirs(path_dir(UserInputsDict), exist_ok=True)
   
    # List log directories
    LogDirList = []
    
    for LogDir in DirList:
       
        if LogDir.find(LOGDIR) == 0:

            LogDirList.append(int(LogDir.split(LOGDIR)[-1]))

        else:

            continue

    LogDirList = sorted(LogDirList)
    
    # Create new log directory
    if len(LogDirList) == 0:

        PathLog = path_dir(UserInputsDict) + LOGDIR + '1' + SLASH

    else:

        LastInd = LogDirList[-1]
        PathLog = path_dir(UserInputsDict) + LOGDIR + str(LastInd + 1) + SLASH

    os.makedirs(PathLog)

def create_model(UserInputsDict, Env):
    """create reinforcement learning model"""

    NoiseLevel = UserInputsDict.get('NoiseLevel')
    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')
    SelNoiseInd = UserInputsDict.get('SelNoiseInd')

    ParamsDict = DEFPARAMS[ALGS[SelAlgInd]][ENVS[SelEnvInd]] #default parameters

    # Create action noise
    try:

        if SelNoiseInd == 0: #default

            noise_type = ParamsDict['noise_type']
            noise_std = ParamsDict['noise_std']

            if noise_type == 'normal':

                ActionNoise = NormalActionNoise(mean=np.zeros(ENVSPECS[SelEnvInd][1]), \
                              sigma=noise_std  * np.ones(ENVSPECS[SelEnvInd][1]))

            elif noise_type == 'ornstein-uhlenbeck':

                ActionNoise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(ENVSPECS[SelEnvInd][1]), \
                              sigma=noise_std * np.ones(ENVSPECS[SelEnvInd][1]))

        elif SelNoiseInd == 1: #normal

            ActionNoise = NormalActionNoise(mean=np.zeros(ENVSPECS[SelEnvInd][1]), sigma=NoiseLevel * np.ones(ENVSPECS[SelEnvInd][1]))

        elif SelNoiseInd == 2: #Ornstein-Uhlenbeck

            ActionNoise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(ENVSPECS[SelEnvInd][1]), sigma=NoiseLevel * np.ones(ENVSPECS[SelEnvInd][1]))

    except:

        ActionNoise = None

    # Create model
    if SelAlgInd == 0:

        policy = ParamsDict['policy']
        learning_rate = ParamsDict['learning_rate']
        n_steps = ParamsDict['n_steps']
        gamma = ParamsDict['gamma']
        gae_lambda = ParamsDict['gae_lambda']
        ent_coef = ParamsDict['ent_coef']
        vf_coef = ParamsDict['vf_coef']
        max_grad_norm = ParamsDict['max_grad_norm']
        rms_prop_eps = ParamsDict['rms_prop_eps']
        use_rms_prop = ParamsDict['use_rms_prop']
        use_sde = ParamsDict['use_sde']
        sde_sample_freq = ParamsDict['sde_sample_freq']
        normalize_advantage = ParamsDict['normalize_advantage']
        tensorboard_log = ParamsDict['tensorboard_log']
        create_eval_env = ParamsDict['create_eval_env']
        policy_kwargs = ParamsDict['policy_kwargs']
        verbose = ParamsDict['verbose']
        seed = ParamsDict['seed']
        device = ParamsDict['device']
        _init_setup_model = ParamsDict['_init_setup_model']

        Model = A2C(policy=policy, env=Env, learning_rate=learning_rate, n_steps=n_steps, \
                    gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, \
                    max_grad_norm=max_grad_norm, rms_prop_eps=rms_prop_eps, \
                    use_rms_prop=use_rms_prop, use_sde=use_sde, sde_sample_freq=sde_sample_freq, \
                    normalize_advantage=normalize_advantage, tensorboard_log=tensorboard_log, \
                    create_eval_env=create_eval_env, policy_kwargs=policy_kwargs, verbose=verbose, \
                    seed=seed, device=device, _init_setup_model=_init_setup_model)
        
    elif SelAlgInd == 1:

        policy = ParamsDict['policy']
        learning_rate = ParamsDict['learning_rate']
        buffer_size = ParamsDict['buffer_size']
        learning_starts = ParamsDict['learning_starts']
        batch_size = ParamsDict['batch_size']
        tau = ParamsDict['tau']
        gamma = ParamsDict['gamma']
        train_freq = ParamsDict['train_freq']
        gradient_steps = ParamsDict['gradient_steps']
        n_episodes_rollout = ParamsDict['n_episodes_rollout']
        tensorboard_log = ParamsDict['tensorboard_log']
        create_eval_env = ParamsDict['create_eval_env']
        policy_kwargs = ParamsDict['policy_kwargs']
        verbose = ParamsDict['verbose']
        seed = ParamsDict['seed']
        device = ParamsDict['device']
        _init_setup_model = ParamsDict['_init_setup_model']

        Model = DDPG(policy=policy, env=Env, learning_rate=learning_rate, buffer_size=buffer_size, \
                     learning_starts=learning_starts, batch_size=batch_size, tau=tau, \
                     gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, \
                     n_episodes_rollout=n_episodes_rollout, action_noise=ActionNoise, \
                     tensorboard_log=tensorboard_log, create_eval_env=create_eval_env, \
                     policy_kwargs=policy_kwargs, verbose=verbose, seed=seed, device=device, \
                     _init_setup_model=_init_setup_model)

    elif SelAlgInd == 2:

        policy = ParamsDict['policy']
        learning_rate = ParamsDict['learning_rate']
        n_steps = ParamsDict['n_steps']
        batch_size = ParamsDict['batch_size']
        n_epochs = ParamsDict['n_epochs']
        gamma = ParamsDict['gamma']
        gae_lambda = ParamsDict['gae_lambda']
        clip_range = ParamsDict['clip_range']
        clip_range_vf = ParamsDict['clip_range_vf']
        ent_coef = ParamsDict['ent_coef']
        vf_coef = ParamsDict['vf_coef']
        max_grad_norm = ParamsDict['max_grad_norm']
        use_sde = ParamsDict['use_sde']
        sde_sample_freq = ParamsDict['sde_sample_freq']
        target_kl = ParamsDict['target_kl']
        tensorboard_log = ParamsDict['tensorboard_log']
        create_eval_env = ParamsDict['create_eval_env']
        policy_kwargs = ParamsDict['policy_kwargs']
        verbose = ParamsDict['verbose']
        seed = ParamsDict['seed']
        device = ParamsDict['device']
        _init_setup_model = ParamsDict['_init_setup_model']

        Model = PPO(policy=policy, env=Env, learning_rate=learning_rate, n_steps=n_steps, \
                    batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, \
                    clip_range=clip_range, clip_range_vf=clip_range_vf, ent_coef=ent_coef, \
                    vf_coef=vf_coef, max_grad_norm=max_grad_norm, use_sde=use_sde, \
                    sde_sample_freq=sde_sample_freq, target_kl=target_kl, \
                    tensorboard_log=tensorboard_log, create_eval_env=create_eval_env, \
                    policy_kwargs=policy_kwargs, verbose=verbose, seed=seed, device=device, \
                    _init_setup_model=_init_setup_model)

    elif SelAlgInd == 3:

        policy = ParamsDict['policy']
        learning_rate = ParamsDict['learning_rate']
        buffer_size = ParamsDict['buffer_size']
        learning_starts = ParamsDict['learning_starts']
        batch_size = ParamsDict['batch_size']
        tau = ParamsDict['tau']
        gamma = ParamsDict['gamma']
        train_freq = ParamsDict['train_freq']
        gradient_steps = ParamsDict['gradient_steps']
        n_episodes_rollout = ParamsDict['n_episodes_rollout']
        optimize_memory_usage = ParamsDict['optimize_memory_usage']
        ent_coef = ParamsDict['ent_coef']
        target_update_interval = ParamsDict['target_update_interval']
        target_entropy= ParamsDict['target_entropy']
        use_sde = ParamsDict['use_sde']
        sde_sample_freq = ParamsDict['sde_sample_freq']
        use_sde_at_warmup = ParamsDict['use_sde_at_warmup']
        tensorboard_log = ParamsDict['tensorboard_log']
        create_eval_env = ParamsDict['create_eval_env']
        policy_kwargs = ParamsDict['policy_kwargs']
        verbose = ParamsDict['verbose']
        seed = ParamsDict['seed']
        device = ParamsDict['device']
        _init_setup_model = ParamsDict['_init_setup_model']

        Model = SAC(policy=policy, env=Env, learning_rate=learning_rate, buffer_size=buffer_size, \
                    learning_starts=learning_starts, batch_size=batch_size, tau=tau, \
                    gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, \
                    n_episodes_rollout=n_episodes_rollout, action_noise=ActionNoise, \
                    optimize_memory_usage=optimize_memory_usage, ent_coef=ent_coef, \
                    target_update_interval=target_update_interval, target_entropy=target_entropy, \
                    use_sde=use_sde, sde_sample_freq=sde_sample_freq, \
                    use_sde_at_warmup=use_sde_at_warmup, tensorboard_log=tensorboard_log, \
                    create_eval_env=create_eval_env, policy_kwargs=policy_kwargs, \
                    verbose=verbose, seed=seed, device=device, _init_setup_model=_init_setup_model)

    elif SelAlgInd == 4:

        policy = ParamsDict['policy']
        learning_rate = ParamsDict['learning_rate']
        buffer_size = ParamsDict['buffer_size']
        learning_starts = ParamsDict['learning_starts']
        batch_size = ParamsDict['batch_size']
        tau = ParamsDict['tau']
        gamma = ParamsDict['gamma']
        train_freq = ParamsDict['train_freq']
        gradient_steps = ParamsDict['gradient_steps']
        n_episodes_rollout = ParamsDict['n_episodes_rollout']
        optimize_memory_usage = ParamsDict['optimize_memory_usage']
        policy_delay = ParamsDict['policy_delay']
        target_policy_noise = ParamsDict['target_policy_noise']
        target_noise_clip= ParamsDict['target_noise_clip']
        tensorboard_log = ParamsDict['tensorboard_log']
        create_eval_env = ParamsDict['create_eval_env']
        policy_kwargs = ParamsDict['policy_kwargs']
        verbose = ParamsDict['verbose']
        seed = ParamsDict['seed']
        device = ParamsDict['device']
        _init_setup_model = ParamsDict['_init_setup_model']

        Model = TD3(policy=policy, env=Env, learning_rate=learning_rate, buffer_size=buffer_size, \
                    learning_starts=learning_starts, batch_size=batch_size, tau=tau, \
                    gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, \
                    n_episodes_rollout=n_episodes_rollout, action_noise=ActionNoise, \
                    optimize_memory_usage=optimize_memory_usage, policy_delay=policy_delay, \
                    target_policy_noise=target_policy_noise, target_noise_clip=target_noise_clip, \
                    tensorboard_log=tensorboard_log, create_eval_env=create_eval_env, \
                    policy_kwargs=policy_kwargs, verbose=verbose, seed=seed, device=device, \
                    _init_setup_model=_init_setup_model)

    return Model

def deploy_model(UserInputsDict, Model, Env):
    """compute model deployment"""

    DeployTime = UserInputsDict.get('DeployTime')
    NoiseLevel = UserInputsDict.get('NoiseLevel')
    SelEnvInd = UserInputsDict.get('SelEnvInd')
    WatchAgent = UserInputsDict.get('WatchAgent')

    # Initialize actions array
    NumActions = ENVSPECS[SelEnvInd][1]
    Actions = np.empty([1, NumActions])

    # Initialize observations array
    NumObservations = ENVSPECS[SelEnvInd][0]
    Observations = np.empty([1, NumObservations])

    Observation = Env.env.reset() #initial observation

    # Select time step
    try:

        TimeStep = Env.env.dt

    except:

        TimeStep = 0.01

    NumIter = int(DeployTime / TimeStep) #number of iterations

    if WatchAgent:

        Env.env.render()
        time.sleep(1)
    
    # Deploy loop
    for _ in range(NumIter - 1):

        Action, _ = Model.predict(Observation) #get actions
        Actions = np.vstack([Actions, np.array(Action)]) #store actions
        Observation, _, _, _ = Env.env.step(Action) #get observations
        Observation = Observation + np.random.normal(0, NoiseLevel / 6, NumObservations)
        #add observation noise (NoiseLevel = 6 * sigma)
        Observations = np.vstack([Observations, Observation]) #store observations

        if WatchAgent:

            Env.env.render() #render environment

    Env.env.close()
    Actions = np.delete(Actions, 0, 0) #remove first action (empty) 
    Observations = np.delete(Observations, 0, 0) #remove first observation (empty)
    Times = np.linspace(0, DeployTime, num=len(Observations)) #create uniform times array

    return Times, Observations, Actions

def load_model(UserInputsDict, Env):
    """load previously trained models"""
    
    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')
    SelModelStr = UserInputsDict.get('SelModelStr')

    if SelAlgInd == 0 and ALGS[SelAlgInd] == 'A2C':

        Model = A2C.load(path_log(UserInputsDict) + SelModelStr, Env, verbose=0)

    elif SelAlgInd == 1 and ALGS[SelAlgInd] == 'DDPG':

        Model = DDPG.load(path_log(UserInputsDict) + SelModelStr, Env, verbose=0)
    
    elif SelAlgInd == 2 and ALGS[SelAlgInd] == 'PPO':

        Model = PPO.load(path_log(UserInputsDict) + SelModelStr, Env, verbose=0)
    
    elif SelAlgInd == 3 and ALGS[SelAlgInd] == 'SAC':

        Model = SAC.load(path_log(UserInputsDict) + SelModelStr, Env, verbose=0)
    
    elif SelAlgInd == 4 and ALGS[SelAlgInd] == 'TD3':

        Model = TD3.load(path_log(UserInputsDict) + SelModelStr, Env, verbose=0)
    
    return Model

def path_log(UserInputsDict):
    """path to log directory"""

    LogStr = UserInputsDict.get('LogStr') 
    SelAlgInd = UserInputsDict.get('SelAlgInd') 
    SelEnvInd = UserInputsDict.get('SelEnvInd') 

    if LogStr: #log dir exists (resume training or deploy)

        PathLog = path_dir(UserInputsDict) + LogStr + SLASH

    else: #log dir does not exist (new training)
        
        DirList = os.listdir(path_dir(UserInputsDict))
        
        LogDirList = []
        
        for LogDir in DirList:
           
            if LogDir.find(LOGDIR) == 0:

                LogDirList.append(int(LogDir.split(LOGDIR)[-1]))

            else:

                continue

        LogDirList = sorted(LogDirList)
        LastInd = LogDirList[-1]
        PathLog = path_dir(UserInputsDict) + LOGDIR + str(LastInd) + SLASH

    return PathLog

def path_dir(UserInputsDict):
    """path to algorithm-environment directory"""

    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')

    PathDir = os.getcwd() + SLASH + MODELSDIR + SLASH + ALGS[SelAlgInd] + \
              SLASH + ENVS[SelEnvInd] + SLASH

    return PathDir

def plot_deploy(UserInputsDict, Times, Observations, Actions):
    """plot deployment results"""

    DeployTime = UserInputsDict.get('DeployTime')
    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')

    if SelEnvInd == 0:

        Position = Observations[:,0] #horizontal position
        Velocity = Observations[:,1] #horizontal velocity
        Theta = np.arctan2(Observations[:,3], Observations[:,2]) * 180 / np.pi #angle
        ThetaDot = Observations[:,4] * 180 / np.pi #angular rate

        Force = Actions #horizontal force

        plt.figure('Observations')
        plt.suptitle('Evolution of the observations (' + ALGS[SelAlgInd] + ' - ' + \
                     ENVS[SelEnvInd] + ')')
        plt.subplot(2,2,1)
        plt.plot(Times, Position, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Position x [m]')
        plt.subplot(2,2,2)
        plt.plot(Times, Velocity, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Velocity $\.x$ [m/s]")
        plt.subplot(2,2,3)
        plt.plot(Times, Theta, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.ylim(-180, 180)
        plt.yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Angle $\theta$ [º]")
        plt.subplot(2,2,4)
        plt.plot(Times, ThetaDot, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Angular rate $\.\theta$ [º/s]")
        plt.show()

        plt.figure('Actions')
        plt.suptitle('Evolution of the actions (' + ALGS[SelAlgInd] + ' - ' + \
                      ENVS[SelEnvInd] + ')')
        plt.plot(Times, Force, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.ylim(-1, 1)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Force F [N]')
        plt.show()

    elif SelEnvInd == 1: 

        Theta = np.arctan2(Observations[:,1], Observations[:,0]) * 180 / np.pi #angle
        ThetaDot = Observations[:,2] * 180 / np.pi #angular rate

        Torque = Actions

        plt.figure('Observations')
        plt.suptitle('Evolution of the observations (' + ALGS[SelAlgInd] + ' - ' + \
                     ENVS[SelEnvInd] + ')')
        plt.subplot(2,1,1)
        plt.plot(Times, Theta, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.ylim(-180, 180)
        plt.yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Angle $\theta$ [º]")
        plt.subplot(2,1,2)
        plt.plot(Times, ThetaDot, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Angular rate $\.\theta$ [º/s]")
        plt.show()

        plt.figure('Actions')
        plt.suptitle('Evolution of the actions (' + ALGS[SelAlgInd] + ' - ' + \
                      ENVS[SelEnvInd] + ')')
        plt.plot(Times, Torque, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.ylim(-2, 2)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Torque $\tau$ [N·m]")
        plt.show()
        
    elif SelEnvInd == 2:

        HorPosition = Observations[:,0] #horizontal position
        VerPosition = Observations[:,1] #vertical position
        HorVelocity = Observations[:,2] #horizontal velocity
        VerVelocity = Observations[:,3] #vertical velocity
        Theta = Observations[:,4] * 180 / np.pi #angle
        ThetaDot = Observations[:,5] * 180 / np.pi #angular rate
        
        HorForce = Actions[:,1] #horizontal force
        VerForce = Actions[:,0] #vertical force

        plt.figure('Observations')
        plt.suptitle('Evolution of the observations (' + ALGS[SelAlgInd] + ' - ' + \
                     ENVS[SelEnvInd] + ')')
        plt.subplot(3,2,1)
        plt.plot(Times, HorPosition, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Horizontal position x [m]')
        plt.subplot(3,2,2)
        plt.plot(Times, VerPosition, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Vertical position y [m]')
        plt.subplot(3,2,3)
        plt.plot(Times, HorVelocity, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Horizontal velocity $\.x$ [m/s]")
        plt.subplot(3,2,4)
        plt.plot(Times, VerVelocity, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Vertical velocity $\.y$ [m/s]")
        plt.subplot(3,2,5)
        plt.plot(Times, Theta, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.ylim(-180, 180)
        plt.yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Angle $\theta$ [º]")
        plt.subplot(3,2,6)
        plt.plot(Times, ThetaDot, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel(r"Angular rate $\.\theta$ [º/s]")
        plt.show()

        plt.figure('Actions')
        plt.suptitle('Evolution of the actions (' + ALGS[SelAlgInd] + ' - ' + \
                      ENVS[SelEnvInd] + ')')
        plt.subplot(2,1,1)
        plt.plot(Times, HorForce, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.ylim(-1, 1)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Horizontal force Fx [N]')
        plt.subplot(2,1,2)
        plt.plot(Times, VerForce, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.ylim(-1, 1)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Vertical force Fy [N]')
        plt.show()

    if SelEnvInd == 3:

        Position = Observations[:,0]
        Velocity = Observations[:,1]

        Force = Actions

        plt.figure('Observations')
        plt.suptitle('Evolution of the observations (' + ALGS[SelAlgInd] + ' - ' + \
                     ENVS[SelEnvInd] + ')')
        plt.subplot(2,1,1)
        plt.plot(Times, Position, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.subplot(2,1,2)
        plt.plot(Times, Velocity, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.show()

        plt.figure('Actions')
        plt.suptitle('Evolution of the actions (' + ALGS[SelAlgInd] + ' - ' + \
                      ENVS[SelEnvInd] + ')')
        plt.plot(Times, Force, marker='.', markersize=4)
        plt.xlim(0, DeployTime)
        plt.ylim(-1, 1)
        plt.grid()
        plt.xlabel('Time [s]')
        plt.ylabel('Force [N]')
        plt.show()

def plot_train(UserInputsDict):
    """plot training results"""

    ResumeTraining = UserInputsDict.get('ResumeTraining')
    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')
    SelModeInd = UserInputsDict.get('SelModeInd')
    SelModelStr = UserInputsDict.get('SelModelStr')
    TrainingSteps = UserInputsDict.get('TrainingSteps')

    PathLog = path_log(UserInputsDict)
    Files = os.listdir(PathLog)

    for File in Files:

        try:

            if File[-9:] == EVALSUFFIX:
                
                if SelModeInd == 1:

                    FileStr = PathLog + SelModelStr + '.npz'

                elif (SelModeInd == 2 and ResumeTraining):

                    PrevTrainingSteps = int(SelModelStr.split('_')[-1])
                    FileStr = PathLog + ALGS[SelAlgInd] + '_' + ENVS[SelEnvInd].replace(' ','') + \
                              '_' + str(PrevTrainingSteps + TrainingSteps) + EVALSUFFIX

                else:

                    FileStr = PathLog + ALGS[SelAlgInd] + '_' + ENVS[SelEnvInd].replace(' ','') + \
                              '_' + str(TrainingSteps) + EVALSUFFIX

                Evaluations = np.load(FileStr)

                EpisodeLen = Evaluations.get('ep_lengths')[0]
                Rewards = Evaluations.get('results')
                TimeSteps = Evaluations.get('timesteps')

                EpisodesNum = Rewards.shape[0]
                RewardsPerEpisodeNum  = Rewards.shape[1]
                RewardsNum = Rewards.size
                LastEpisodeTimeStep = TimeSteps[-1]

                MeanRewardsPerEpisode = np.mean(Rewards, 1).reshape(EpisodesNum, )
                TimeSteps = np.linspace(LastEpisodeTimeStep / RewardsNum , \
                                        LastEpisodeTimeStep, RewardsNum)
                Episodes = np.linspace(1, EpisodesNum, EpisodesNum, dtype=int)
                Rewards = Rewards.reshape(RewardsNum, )

                plt.figure('Rewards')
                plt.suptitle('Evolution of the reward during the training (' + \
                              ALGS[SelAlgInd] + ' - ' + ENVS[SelEnvInd] + ')')
                plt.subplot(2,1,1)
                plt.plot(TimeSteps, Rewards, marker='.', markersize=4)
                plt.xlabel('Time step [-]')
                plt.ylabel('Reward [-]')
                plt.xlim(TimeSteps[0], TimeSteps[-1])
                plt.grid()
                plt.subplot(2,1,2)
                plt.plot(Episodes, MeanRewardsPerEpisode, marker='.', markersize=4)
                plt.xlabel('Episode [-]')
                plt.ylabel('Mean reward [-]')
                plt.xlim(Episodes[0], Episodes[-1])
                plt.grid()
                plt.show()

                break

        except:

            exit("\nToo few training steps to plot anything.\n")

def rename_files(UserInputsDict):
        """rename training output files (best_model and evaluations)"""

        ResumeTraining = UserInputsDict.get('ResumeTraining')
        SelAlgInd = UserInputsDict.get('SelAlgInd')
        SelEnvInd = UserInputsDict.get('SelEnvInd')
        SelModelStr = UserInputsDict.get('SelModelStr')
        TrainingSteps = UserInputsDict.get('TrainingSteps')

        PathLog = path_log(UserInputsDict)
        FilesList = os.listdir(PathLog)

        # Rename 'best_model.zip'
        if 'best_model.zip' in FilesList:

            PrevFileStr = PathLog + 'best_model.zip'

            if ResumeTraining:

                PrevTrainingSteps = int(SelModelStr.split('_')[-1])
                PostFileStr = PathLog + ALGS[SelAlgInd] + '_' + \
                              ENVS[SelEnvInd].replace(' ','') + '_' + \
                              str(PrevTrainingSteps + TrainingSteps) + \
                              BESTSUFFIX

            else:

                PostFileStr = PathLog + ALGS[SelAlgInd] + '_' + \
                              ENVS[SelEnvInd].replace(' ','') + '_' + \
                              str(TrainingSteps) + BESTSUFFIX

            os.rename(PrevFileStr, PostFileStr)

        # Rename 'evaluations.npz'
        if 'evaluations.npz' in FilesList:

            PrevFileStr = PathLog + 'evaluations.npz'

            if ResumeTraining:

                PrevTrainingSteps = int(SelModelStr.split('_')[-1])
                PostFileStr = PathLog + ALGS[SelAlgInd] + '_' + \
                              ENVS[SelEnvInd].replace(' ','') + '_' + \
                              str(PrevTrainingSteps + TrainingSteps) + \
                              EVALSUFFIX

            else:

                PostFileStr = PathLog + ALGS[SelAlgInd] + '_' + \
                              ENVS[SelEnvInd].replace(' ','') + '_' + \
                              str(TrainingSteps) + EVALSUFFIX

            os.rename(PrevFileStr, PostFileStr)

def save_model(UserInputsDict, Model, Env):
    """save trained models"""

    ResumeTraining = UserInputsDict.get('ResumeTraining')
    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')
    SelModelStr = UserInputsDict.get('SelModelStr')
    TrainingSteps = UserInputsDict.get('TrainingSteps')

    if ResumeTraining: #retrained model

        PrevTrainingSteps = int(SelModelStr.split('_')[-1])
        Model.save(path_log(UserInputsDict) + ALGS[SelAlgInd] + '_' + \
                   ENVS[SelEnvInd].replace(' ','') + '_' + \
                   str(PrevTrainingSteps + TrainingSteps))

    else: #new model

        Model.save(path_log(UserInputsDict) + ALGS[SelAlgInd] + '_' + \
                   ENVS[SelEnvInd].replace(' ','') + '_' + \
                   str(TrainingSteps))

def select_algorithm(UserInputsDict):
    """select algorithm model-free reinforcement learning algorithm"""

    # Input algorithm
    AlgLen = len(ALGS)
    InputStr = "\nPlease, introduce the index of the desired algorithm [1, " + \
               str(AlgLen) + "], then press ENTER:\n\n"
    
    for AlgInd in range(AlgLen):
    
        InputStr = InputStr + "\t[" + str(AlgInd+1) + "] " + ALGS[AlgInd] + "\n"
        
    SelAlgInd = input(InputStr + "\n")
    
    while not SelAlgInd.isnumeric() or int(SelAlgInd) < 1 or int(SelAlgInd) > AlgLen:
    
        SelAlgInd = input("\nThe selected index is not correct. Please, select an index " \
                          "value within the range [1, " + str(AlgLen) + "].\n\n")  
        
    SelAlgInd = int(SelAlgInd) - 1
    SelAlgStr = str(ALGS[SelAlgInd])
    UserInputsDict.update(SelAlgInd=SelAlgInd) #0 = A2C, 1 = DDPG, 2 = PPO, 3 = SAC, 4 = TD3
    print("\nThe selected environment is '" + SelAlgStr + "'.")

    return UserInputsDict

def select_deploy_time(UserInputsDict):
    """select deployment time of trained model"""

    DeployTime = UserInputsDict.get('DeployTime')
    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')

    ParamsDict = DEFPARAMS[ALGS[SelAlgInd]][ENVS[SelEnvInd]]

    # Input deploy time
    DeployTime = input("\nPlease, introduce the desired duration of deployment in " + \
                       "seconds (recommended: " + str(ParamsDict['deploy_time']) + "), " + \
                       "then press ENTER:\n\n")

    while not DeployTime.isnumeric() or int(DeployTime) <= 0:
    
        DeployTime = input("\nThe input is not valid. Please, select " + \
                           "a value within the range (0, Inf).\n\n")
        
    DeployTime = int(DeployTime)
    UserInputsDict.update(DeployTime=DeployTime) #seconds
    print("\nThe deployment will end after " + str(DeployTime) + " seconds.")

    return UserInputsDict

def select_environment(UserInputsDict):
    """select gym environment in which the agent will learn"""

    # Input environment
    EnvLen = len(ENVS)
    InputStr = "\nPlease, introduce the index of the desired environment [1, " + \
               str(EnvLen) + "], then press ENTER:\n\n"
    
    for EnvInd in range(EnvLen):
    
        InputStr = InputStr + "\t[" + str(EnvInd+1) + "] " + ENVS[EnvInd] + "\n"
        
    SelEnvInd = input(InputStr + "\n")
    
    while not SelEnvInd.isnumeric() or int(SelEnvInd) < 1 or int(SelEnvInd) > EnvLen:
    
        SelEnvInd = input("\nThe selected index is not correct. Please, select an index " \
                          "value within the range [1, " + str(EnvLen) + "].\n\n")  
        
    SelEnvInd = int(SelEnvInd) - 1
    SelEnvStr = str(ENVS[SelEnvInd])
    UserInputsDict.update(SelEnvInd=SelEnvInd)
    #0 = Cart Pole, 1 = Inverted Pendulum, 2 = Lunar Lander, 3 = Mountain Car
    print("\nThe selected environment is '" + SelEnvStr + "'.")
    
    return UserInputsDict

def select_noise_level_action(UserInputsDict):
    """select level of noise in the agent's actions during training"""

    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')

    ParamsDict = DEFPARAMS[ALGS[SelAlgInd]][ENVS[SelEnvInd]]

    # Input action noise level
    NoiseLevel = input("\nPlease, introduce the desired level of action noise as " + \
                       "a real number within the range [0, 1] (recommended: " + \
                       str(ParamsDict['noise_std']) + "), then press ENTER:\n\n")

    while not NoiseLevel.replace('.', '').isnumeric() or float(NoiseLevel) < 0 or \
          float(NoiseLevel) > 1:
    
        NoiseLevel = input("\nThe input is not valid. Please, select " + \
                           "a value within the range [0, 1].\n\n")
        
    NoiseLevel = float(NoiseLevel)
    UserInputsDict.update(NoiseLevel=NoiseLevel) #equals to standard deviation
    print("\nThe selected action noise level is '" + str(NoiseLevel) + "'.")

    return UserInputsDict

def select_noise_level_observation(UserInputsDict):
    """select level of noise in the observations during deployment"""

    # Input observations noise level
    NoiseLevel = input("\nPlease, introduce the desired level of observation noise as " + \
                       "a real number within the range [0, 1], then press ENTER:\n\n")

    while not NoiseLevel.replace('.', '').isnumeric() or float(NoiseLevel) < 0 or \
          float(NoiseLevel) > 1:
    
        NoiseLevel = input("\nThe input is not valid. Please, select " + \
                           "a value within the range [0, 1].\n\n")
        
    NoiseLevel = float(NoiseLevel)
    UserInputsDict.update(NoiseLevel=NoiseLevel) #equals to  6 * standard deviation
    print("\nThe selected observation noise level is '" + str(NoiseLevel) + "'.")

    return UserInputsDict

def select_noise_type(UserInputsDict):
    """select type of noise"""

    SelAlgInd = UserInputsDict.get('SelAlgInd')
    SelEnvInd = UserInputsDict.get('SelEnvInd')

    ParamsDict = DEFPARAMS[ALGS[SelAlgInd]][ENVS[SelEnvInd]]

    # Input noise type
    NoiseTypesLen = len(NOISETYPES)

    InputStr = "\nPlease, introduce the index of the desired type of noise [1, " + \
               str(NoiseTypesLen) + "], then press ENTER:\n\n"
    
    for SelNoiseInd in range(NoiseTypesLen):
    
        InputStr = InputStr + "\t[" + str(SelNoiseInd+1) + "] " + NOISETYPES[SelNoiseInd] + "\n"
        
    SelNoiseInd = input(InputStr + "\n")
    
    while not SelNoiseInd.isnumeric() or int(SelNoiseInd) < 1 or int(SelNoiseInd) > NoiseTypesLen:
    
        SelNoiseInd = input("\nThe selected index is not correct. Please, select an index " + \
                               "value within the range [1, " + str(NoiseTypesLen) + "].\n\n")
        
    SelNoiseInd = int(SelNoiseInd) - 1
    UserInputsDict.update(SelNoiseInd=SelNoiseInd) #0 = Default, 1 = Normal, 2 = Ornstein-Uhlenbeck
    print("\nThe selected type of noise is '" + NOISETYPES[SelNoiseInd]  + "'.")
    
    return UserInputsDict
    
def select_resume_training(UserInputsDict):
    """select whether to resume the training of a previously trained model or start a new one"""

    # Decide if resume training
    ResumeTrainingStr = []

    while (ResumeTrainingStr not in YES) and (ResumeTrainingStr not in NO):
    
        ResumeTrainingStr = input("\nWould you like to resume training an " + \
                                    "already existing model? [Y/N]:\n\n")
        
        if ResumeTrainingStr in YES:
        
            ResumeTraining = True
            print("\nA previous training will be resumed.") 
        
        elif ResumeTrainingStr in NO:
        
            ResumeTraining = False
            print("\nA new training will start.") 
        
        else:
        
            print("\nPlease, try again.")

    UserInputsDict.update(ResumeTraining=ResumeTraining) #False = New, True = Resume

    return UserInputsDict

def select_trained_model(UserInputsDict):
    """select previously trained model"""

    SelModeInd = UserInputsDict.get('SelModeInd')
    ResumeTraining = UserInputsDict.get('ResumeTraining')
    SelAlgInd = UserInputsDict.get('SelAlgInd') 
    SelEnvInd = UserInputsDict.get('SelEnvInd') 
   
    PathDir = path_dir(UserInputsDict)

    try: #list directories inside algorithm-environment directory

        DirList = os.listdir(PathDir)
    
    except: #no algorithm-environment directory

        exit("\nThere are no trained models to deploy for the selected " + \
             "algorithm-environment pair. Please, train a model or " + \
             "add one into '" + PathDir + "'.\n" )

    # List trained models
    DirFilesDict = dict()

    for Dir in DirList:
       
        DirFilesDict[Dir] = os.listdir(PathDir + Dir)

    CommonFilePath = ALGS[SelAlgInd] + "_" + ENVS[SelEnvInd].replace(' ','') + "_"
    ModelDirDict = dict()
    
    for (Dir, Files) in DirFilesDict.items():

        if isinstance(Files, list):

            for File in Files:

                if SelModeInd == 1: #plot

                    if File[-9:] == EVALSUFFIX:

                        ModelDirDict[File.split('.')[0]] = Dir

                elif File.find(CommonFilePath) == 0 and File[-9:] != EVALSUFFIX:

                    if (ResumeTraining and File[-9:] != BESTSUFFIX) or \
                       (not ResumeTraining):

                        ModelDirDict[File.split('.')[0]] = Dir

                else:

                    continue

        else:

            ModelDirDict[Files.split('.')[0]] = Dir

    ModelsNum = len(ModelDirDict.values())

    if ModelsNum == 0:

        exit("\nThere are no trained models to deploy for the selected " + \
             "algorithm-environment pair. Please, train a model or " + \
             "add one into '" + path_dir(UserInputsDict) + "'.\n" )

    # Input trained model
    InputStr = "\nPlease, introduce the index of the desired trained model " + \
               "[1, " + str(ModelsNum) + "], then press ENTER:\n\n"

    ModelInd = 0

    for (Model, Dir) in ModelDirDict.items():
        
        InputStr = InputStr + "\t[" + str(ModelInd+1) + "] " + Model + " (" + Dir  + ")\n"
        ModelInd += 1

    SelModelInd = input(InputStr + "\n")
    
    while not SelModelInd.isnumeric() or int(SelModelInd) < 1 or int(SelModelInd) > ModelsNum:
    
        SelModelInd = input("\nThe selected index is not correct. Please, select an index" \
                            "value within the range [1, " + str(ModelsNum) + "].\n\n")  
    
    ModelsList = list(ModelDirDict.keys())
    SelModelInd = int(SelModelInd) - 1
    SelModelStr = str(ModelsList[SelModelInd])
    UserInputsDict.update(SelModelStr=SelModelStr) #model name
    LogStr = str(ModelDirDict.get(SelModelStr))
    UserInputsDict.update(LogStr=LogStr) #log directory name
    print("\nThe selected trained model is '" + SelModelStr + "'.")

    return UserInputsDict

def select_training_steps(UserInputsDict):
    """select environment steps of training"""

    SelAlgInd = UserInputsDict.get('SelAlgInd') 
    SelEnvInd = UserInputsDict.get('SelEnvInd')

    ParamsDict = DEFPARAMS[ALGS[SelAlgInd]][ENVS[SelEnvInd]]

    # Input training steps
    TrainingSteps = input("\nPlease, introduce the desired number of training steps " + \
                          "(recommended: " + str(ParamsDict['training_steps']) + "), then "
                           "press ENTER:\n\n")

    while not TrainingSteps.isnumeric() or int(TrainingSteps) < 1:
    
        TrainingSteps = input("\nThe input is not valid. Please, select a " + \
                              "value within the range [1, Inf).\n\n")
        
    TrainingSteps = int(TrainingSteps)
    UserInputsDict.update(TrainingSteps=TrainingSteps)
    print("\nThe training will end after " + str(TrainingSteps) + " training steps.")

    return UserInputsDict
 
def select_mode(UserInputsDict):
    """select operational mode"""

    # Input mode
    ModesLen = len(MODES)
    InputStr = "\nPlease, introduce the index of the desired process [1, " + \
               str(ModesLen) + "], then press ENTER:\n\n"
    
    for SelModeInd in range(ModesLen):
    
        InputStr = InputStr + "\t[" + str(SelModeInd+1) + "] " + MODES[SelModeInd] + "\n"
        
    SelModeInd = input(InputStr + "\n")
    
    while not SelModeInd.isnumeric() or int(SelModeInd) < 1 or int(SelModeInd) > ModesLen:
    
        SelModeInd = input("\nThe selected index is not correct. Please, select an index " \
                           "value within the range [1, " + str(ModesLen) + "].\n\n")  
        
    SelModeInd = int(SelModeInd) - 1
    SelModeStr = str(MODES[SelModeInd])
    UserInputsDict.update(SelModeInd=SelModeInd) #0 = Deploy, 1 = Plot, 2 = Train
    print("\nThe selected mode is '" + MODES[SelModeInd]  + "'.")

    return UserInputsDict

def select_watch_agent(UserInputsDict):
    """decide whether to watch the agent or not"""

    # Decide if the environment will be rendered
    WatchAgentStr = []

    while (WatchAgentStr not in YES) and (WatchAgentStr not in NO):
    
        WatchAgentStr = input("\nWould you like to watch the agent? [Y/N]:\n\n")
        
        if WatchAgentStr in YES:
        
            WatchAgent = True
            print("\nThe environment will be rendered.\n")
        
        elif WatchAgentStr in NO:
        
            WatchAgent = False
            print("\nThe environment will not be rendered.\n")
        
        else:
        
            print("\nPlease, try again.\n")

    UserInputsDict.update(WatchAgent=WatchAgent) #False = Not watch, True = Watch
    
    return UserInputsDict

def user_inputs():
    """user input flow"""

    UserInputsDict = dict() #dictionary containing all user input parameters

    UserInputsDict = select_mode(UserInputsDict)
    UserInputsDict = select_algorithm(UserInputsDict)
    SelAlgInd = UserInputsDict.get('SelAlgInd')    
    UserInputsDict = select_environment(UserInputsDict)
    SelModeInd = UserInputsDict.get('SelModeInd')    
    
    if SelModeInd == 0: #deploy
    
        UserInputsDict = select_trained_model(UserInputsDict)
        UserInputsDict = select_deploy_time(UserInputsDict)
        UserInputsDict = select_noise_level_observation(UserInputsDict)
        UserInputsDict = select_watch_agent(UserInputsDict)

    elif SelModeInd == 1: #plot

        UserInputsDict = select_trained_model(UserInputsDict)

    elif SelModeInd == 2: #train
    
        UserInputsDict = select_resume_training(UserInputsDict)
        ResumeTraining = UserInputsDict.get('ResumeTraining')    

        if ResumeTraining: #resume

            UserInputsDict = select_trained_model(UserInputsDict)
            UserInputsDict = select_training_steps(UserInputsDict)
        
        else: #new

            UserInputsDict = select_training_steps(UserInputsDict)

            if SelAlgInd in [1, 3, 4]: #[DDPG, SAC, TD3]

                UserInputsDict = select_noise_type(UserInputsDict)
                SelNoiseInd = UserInputsDict.get('SelNoiseInd')    

                if SelNoiseInd != 0: #not default noise

                    UserInputsDict = select_noise_level_action(UserInputsDict)

        UserInputsDict = select_watch_agent(UserInputsDict)

    else:

        exit("\nUnknown operational mode.\n")

    return UserInputsDict

