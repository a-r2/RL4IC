from settings import *

""" HYPERPARAMETERS """

DEFPARAMS_A2C_CARTPOLE =            {
        
                                    'deploy_time': RECDEPTIME[0], \
                                    'training_steps': 1000000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.0001, \
                                    'n_steps': 5, \
                                    'gamma': 0.99, \
                                    'gae_lambda': 1.0, \
                                    'ent_coef': 0.0, \
                                    'vf_coef': 0.5, \
                                    'max_grad_norm': 0.5, \
                                    'rms_prop_eps': 1e-5, \
                                    'use_rms_prop': False, \
                                    'use_sde': False, \
                                    'sde_sample_freq': -1, \
                                    'normalize_advantage': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    }

DEFPARAMS_A2C_INVERTEDPENDULUM =    {
        
                                    'deploy_time': RECDEPTIME[1], \
                                    'training_steps': 500000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.0007, \
                                    'n_steps': 8, \
                                    'gamma': 0.99, \
                                    'gae_lambda': 0.9, \
                                    'ent_coef': 0.0, \
                                    'vf_coef': 0.4, \
                                    'max_grad_norm': 0.5, \
                                    'rms_prop_eps': 1e-05, \
                                    'use_rms_prop': True, \
                                    'use_sde': True, \
                                    'sde_sample_freq': -1, \
                                    'normalize_advantage': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_A2C_LUNARLANDER =         {
        
                                    'deploy_time': RECDEPTIME[2], \
                                    'training_steps': 3000000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.0003, \
                                    'n_steps': 5, \
                                    'gamma': 0.999, \
                                    'gae_lambda': 0.98, \
                                    'ent_coef': 0.01, \
                                    'vf_coef': 0.2, \
                                    'max_grad_norm': 0.5, \
                                    'rms_prop_eps': 1e-05, \
                                    'use_rms_prop': False, \
                                    'use_sde': False, \
                                    'sde_sample_freq': -1, \
                                    'normalize_advantage': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_A2C_MOUNTAINCAR =         {
        
                                    'deploy_time': RECDEPTIME[3], \
                                    'training_steps': 300000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.0007, \
                                    'n_steps': 5, \
                                    'gamma': 0.99, \
                                    'gae_lambda': 0.99, \
                                    'ent_coef': 0.02, \
                                    'vf_coef': 0.25, \
                                    'max_grad_norm': 0.5, \
                                    'rms_prop_eps': 1e-05, \
                                    'use_rms_prop': False, \
                                    'use_sde': True, \
                                    'sde_sample_freq': -1, \
                                    'normalize_advantage': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_DDPG_CARTPOLE =           {
        
                                    'deploy_time': RECDEPTIME[0], \
                                    'training_steps': 100000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 200000, \
                                    'learning_starts': 10000, \
                                    'batch_size': 100, \
                                    'tau': 0.005, \
                                    'gamma': 0.98, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'ornstein-uhlenbeck', \
                                    'noise_std': 0.8, \
                                    'optimize_memory_usage': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    }

DEFPARAMS_DDPG_INVERTEDPENDULUM =   {
        
                                    'deploy_time': RECDEPTIME[1], \
                                    'training_steps': 20000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 200000, \
                                    'learning_starts': 10000, \
                                    'batch_size': 100, \
                                    'tau': 0.005, \
                                    'gamma': 0.98, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'normal', \
                                    'noise_std': 0.1, \
                                    'optimize_memory_usage': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_DDPG_LUNARLANDER =        {
        
                                    'deploy_time': RECDEPTIME[2], \
                                    'training_steps': 300000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 200000, \
                                    'learning_starts': 10000, \
                                    'batch_size': 100, \
                                    'tau': 0.005, \
                                    'gamma': 0.98, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'normal', \
                                    'noise_std': 0.1, \
                                    'optimize_memory_usage': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_DDPG_MOUNTAINCAR =        {
        
                                    'deploy_time': RECDEPTIME[3], \
                                    'training_steps': 50000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 300000, \
                                    'learning_starts': 100, \
                                    'batch_size': 100, \
                                    'tau': 0.005, \
                                    'gamma': 0.99, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'ornstein-uhlenbeck', \
                                    'noise_std': 0.5, \
                                    'optimize_memory_usage': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_PPO_CARTPOLE =            {
        
                                    'deploy_time': RECDEPTIME[0], \
                                    'training_steps': 500000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'n_steps': 32, \
                                    'batch_size': 256, \
                                    'n_epochs': 20, \
                                    'gamma': 0.98, \
                                    'gae_lambda': 0.8, \
                                    'clip_range': 0.2, \
                                    'clip_range_vf': None, \
                                    'ent_coef': 0.01, \
                                    'vf_coef': 0.5, \
                                    'max_grad_norm': 0.5, \
                                    'use_sde': False, \
                                    'sde_sample_freq': -1, \
                                    'target_kl': None, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    }

DEFPARAMS_PPO_INVERTEDPENDULUM =    {
        
                                    'deploy_time': RECDEPTIME[1], \
                                    'training_steps': 500000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.0007, \
                                    'n_steps': 8, \
                                    'batch_size': 32, \
                                    'n_epochs': 10, \
                                    'gamma': 0.99, \
                                    'gae_lambda': 0.9, \
                                    'clip_range': 0.2, \
                                    'clip_range_vf': None, \
                                    'ent_coef': 0.01, \
                                    'vf_coef': 0.4, \
                                    'max_grad_norm': 0.5, \
                                    'use_sde': True, \
                                    'sde_sample_freq': -1, \
                                    'target_kl': 0.1, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_PPO_LUNARLANDER =         {
        
                                    'deploy_time': RECDEPTIME[2], \
                                    'training_steps': 1000000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.0003, \
                                    'n_steps': 1024, \
                                    'batch_size': 64, \
                                    'n_epochs': 4, \
                                    'gamma': 0.999, \
                                    'gae_lambda': 0.98, \
                                    'clip_range': 0.2, \
                                    'clip_range_vf': None, \
                                    'ent_coef': 0.01, \
                                    'vf_coef': 0.5, \
                                    'max_grad_norm': 0.5, \
                                    'use_sde': False, \
                                    'sde_sample_freq': -1, \
                                    'target_kl': None, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_PPO_MOUNTAINCAR =         {
        
                                    'deploy_time': RECDEPTIME[3], \
                                    'training_steps': 20000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.0000777, \
                                    'n_steps': 8, \
                                    'batch_size': 256, \
                                    'n_epochs': 10, \
                                    'gamma': 0.9999, \
                                    'gae_lambda': 0.9, \
                                    'clip_range': 0.1, \
                                    'clip_range_vf': None,\
                                    'ent_coef': 0.00429, \
                                    'vf_coef': 0.19, \
                                    'max_grad_norm': 5, \
                                    'use_sde': True, \
                                    'sde_sample_freq': -1, \
                                    'target_kl': None, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_SAC_CARTPOLE =            {
        
                                    'deploy_time': RECDEPTIME[0], \
                                    'training_steps': 200000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.00073, \
                                    'buffer_size': 300000, \
                                    'learning_starts': 10000, \
                                    'batch_size': 256, \
                                    'tau': 0.02, \
                                    'gamma': 0.98, \
                                    'train_freq': 64, \
                                    'gradient_steps': 64, \
                                    'n_episodes_rollout': -1, \
                                    'action_noise': None, \
                                    'noise_type': 'ornstein-uhlenbeck', \
                                    'noise_std': 0.5, \
                                    'optimize_memory_usage': False, \
                                    'ent_coef': 'auto', \
                                    'target_update_interval': 1, \
                                    'target_entropy': 'auto', \
                                    'use_sde': True, \
                                    'sde_sample_freq': -1, \
                                    'use_sde_at_warmup': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    }

DEFPARAMS_SAC_INVERTEDPENDULUM =    {
        
                                    'deploy_time': RECDEPTIME[1], \
                                    'training_steps': 20000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 1000000, \
                                    'learning_starts': 100, \
                                    'batch_size': 256, \
                                    'tau': 0.05, \
                                    'gamma': 0.99, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'ornstein-uhlenbeck', \
                                    'noise_std': 0.5, \
                                    'optimize_memory_usage': False, \
                                    'ent_coef': 'auto', \
                                    'target_update_interval': 1, \
                                    'target_entropy': 'auto', \
                                    'use_sde': True, \
                                    'sde_sample_freq': -1, \
                                    'use_sde_at_warmup': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_SAC_LUNARLANDER =         {
        
                                    'deploy_time': RECDEPTIME[2], \
                                    'training_steps': 300000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.00073, \
                                    'buffer_size': 300000, \
                                    'learning_starts': 1000, \
                                    'batch_size': 256, \
                                    'tau': 0.02, \
                                    'gamma': 0.98, \
                                    'train_freq': 64, \
                                    'gradient_steps': 64, \
                                    'n_episodes_rollout': -1, \
                                    'action_noise': None, \
                                    'noise_type': 'ornstein-uhlenbeck', \
                                    'noise_std': 0.5, \
                                    'optimize_memory_usage': False, \
                                    'ent_coef': 'auto', \
                                    'target_update_interval': 1, \
                                    'target_entropy': 'auto', \
                                    'use_sde': True, \
                                    'sde_sample_freq': -1, \
                                    'use_sde_at_warmup': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_SAC_MOUNTAINCAR =         {
        
                                    'deploy_time': RECDEPTIME[3], \
                                    'training_steps': 50000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.0003, \
                                    'buffer_size': 50000, \
                                    'learning_starts': 0, \
                                    'batch_size': 512, \
                                    'tau': 0.01, \
                                    'gamma': 0.9999, \
                                    'train_freq': 32, \
                                    'gradient_steps': 32, \
                                    'n_episodes_rollout': -1, \
                                    'action_noise': None, \
                                    'noise_type': 'ornstein-uhlenbeck', \
                                    'noise_std': 0.5, \
                                    'optimize_memory_usage': False, \
                                    'ent_coef': 0.1, \
                                    'target_update_interval': 1, \
                                    'target_entropy': 'auto', \
                                    'use_sde': True, \
                                    'sde_sample_freq': -1, \
                                    'use_sde_at_warmup': False, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_TD3_CARTPOLE =            {
        
                                    'deploy_time': RECDEPTIME[0], \
                                    'training_steps': 300000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 200000, \
                                    'learning_starts': 10000, \
                                    'batch_size': 100, \
                                    'tau': 0.005, \
                                    'gamma': 0.98, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'normal', \
                                    'noise_std': 0.1, \
                                    'optimize_memory_usage': False, \
                                    'policy_delay': 2, \
                                    'target_policy_noise': 0.2, \
                                    'target_noise_clip': 0.5, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    }

DEFPARAMS_TD3_INVERTEDPENDULUM =    {
        
                                    'deploy_time': RECDEPTIME[1], \
                                    'training_steps': 25000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 200000, \
                                    'learning_starts': 10000, \
                                    'batch_size': 100, \
                                    'tau': 0.005, \
                                    'gamma': 0.98, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'normal', \
                                    'noise_std': 0.1, \
                                    'optimize_memory_usage': False, \
                                    'policy_delay': 2, \
                                    'target_policy_noise': 0.2, \
                                    'target_noise_clip': 0.5, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_TD3_LUNARLANDER =         {
        
                                    'deploy_time': RECDEPTIME[2], \
                                    'training_steps': 200000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 200000, \
                                    'learning_starts': 10000, \
                                    'batch_size': 100, \
                                    'tau': 0.005, \
                                    'gamma': 0.98, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'normal', \
                                    'noise_std': 0.1, \
                                    'optimize_memory_usage': False, \
                                    'policy_delay': 2, \
                                    'target_policy_noise': 0.2, \
                                    'target_noise_clip': 0.5, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_TD3_MOUNTAINCAR =         {
        
                                    'deploy_time': RECDEPTIME[3], \
                                    'training_steps': 300000, \
                                    'policy': 'MlpPolicy', \
                                    'learning_rate': 0.001, \
                                    'buffer_size': 200000, \
                                    'learning_starts': 10000, \
                                    'batch_size': 100, \
                                    'tau': 0.005, \
                                    'gamma': 0.98, \
                                    'train_freq': -1, \
                                    'gradient_steps': -1, \
                                    'n_episodes_rollout': 1, \
                                    'action_noise': None, \
                                    'noise_type': 'ornstein-uhlenbeck', \
                                    'noise_std': 0.5, \
                                    'optimize_memory_usage': False, \
                                    'policy_delay': 2, \
                                    'target_policy_noise': 0.2, \
                                    'target_noise_clip': 0.5, \
                                    'tensorboard_log': None, \
                                    'create_eval_env': False, \
                                    'policy_kwargs': None, \
                                    'verbose': 0, \
                                    'seed': None, \
                                    'device': 'auto', \
                                    '_init_setup_model': True
                                    
                                    } 

DEFPARAMS_A2C =     {

                    ENVS[0]: DEFPARAMS_A2C_CARTPOLE, \
                    ENVS[1]: DEFPARAMS_A2C_INVERTEDPENDULUM, \
                    ENVS[2]: DEFPARAMS_A2C_LUNARLANDER, \
                    ENVS[3]: DEFPARAMS_A2C_MOUNTAINCAR
        
                    }

DEFPARAMS_DDPG=     {

                    ENVS[0]: DEFPARAMS_DDPG_CARTPOLE, \
                    ENVS[1]: DEFPARAMS_DDPG_INVERTEDPENDULUM, \
                    ENVS[2]: DEFPARAMS_DDPG_LUNARLANDER, \
                    ENVS[3]: DEFPARAMS_DDPG_MOUNTAINCAR
        
                    }

DEFPARAMS_PPO =     {

                    ENVS[0]: DEFPARAMS_PPO_CARTPOLE, \
                    ENVS[1]: DEFPARAMS_PPO_INVERTEDPENDULUM, \
                    ENVS[2]: DEFPARAMS_PPO_LUNARLANDER, \
                    ENVS[3]: DEFPARAMS_PPO_MOUNTAINCAR
        
                    }

DEFPARAMS_SAC =     {

                    ENVS[0]: DEFPARAMS_SAC_CARTPOLE, \
                    ENVS[1]: DEFPARAMS_SAC_INVERTEDPENDULUM, \
                    ENVS[2]: DEFPARAMS_SAC_LUNARLANDER, \
                    ENVS[3]: DEFPARAMS_SAC_MOUNTAINCAR
        
                    }

DEFPARAMS_TD3 =     {

                    ENVS[0]: DEFPARAMS_TD3_CARTPOLE, \
                    ENVS[1]: DEFPARAMS_TD3_INVERTEDPENDULUM, \
                    ENVS[2]: DEFPARAMS_TD3_LUNARLANDER, \
                    ENVS[3]: DEFPARAMS_TD3_MOUNTAINCAR
        
                    }

DEFPARAMS =     {

                ALGS[0]: DEFPARAMS_A2C, \
                ALGS[1]: DEFPARAMS_DDPG, \
                ALGS[2]: DEFPARAMS_PPO, \
                ALGS[3]: DEFPARAMS_SAC, \
                ALGS[4]: DEFPARAMS_TD3
                
                }
