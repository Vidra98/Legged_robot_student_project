"""
Run stable baselines 3 on quadruped env 
Check the documentation! https://stable-baselines3.readthedocs.io/en/master/
"""
import os
from datetime import datetime
# stable baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
# utils
from utils.utils import CheckpointCallback
from utils.file_utils import get_latest_model
# gym environment
from env.quadruped_gym_env import QuadrupedGymEnv


LEARNING_ALG = "PPO" # or "SAC"
LOAD_NN = False # if you want to initialize training with a previous model 
NUM_ENVS = 1    # how many pybullet environments to create for data collection
USE_GPU = False # make sure to install all necessary drivers 

# after implementing, you will want to test how well the agent learns with your MDP: 
env_configs = {"motor_control_mode":"CARTESIAN_PD",
               "task_env": "LR_COURSE_TROT",                #LR_COURSE_FWD,LR_COURSE_BACKWARD,LR_COURSE_SIDEWAY,LR_COURSE_TROT
               "observation_space_mode": "LR_COURSE_OBS_FEET_SENSOR_ADDED"}   #LR_COURSE_OBS | LR_COURSE_OBS_FEET_SENSOR_ADDED
#env_configs = {}

if USE_GPU and LEARNING_ALG=="SAC":
    gpu_arg = "auto" 
else:
    gpu_arg = "cpu"

if LOAD_NN:
    interm_dir = "./logs/intermediate_models/"
    log_dir = interm_dir + 'First_trot' # add path
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    model_name = get_latest_model(log_dir)

# directory to save policies and normalization parameters
SAVE_PATH = './logs/intermediate_models/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
os.makedirs(SAVE_PATH, exist_ok=True)
# checkpoint to save policy network periodically
checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=SAVE_PATH,name_prefix='rl_model', verbose=2)
# create Vectorized gym environment
env = lambda: QuadrupedGymEnv(**env_configs)  
env = make_vec_env(env, monitor_dir=SAVE_PATH,n_envs=NUM_ENVS)
# normalize observations to stabilize learning (why?)
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

if LOAD_NN:
    env = lambda: QuadrupedGymEnv()
    env = make_vec_env(env, monitor_dir=SAVE_PATH,n_envs=NUM_ENVS)
    env = VecNormalize.load(stats_path, env)

# Multi-layer perceptron (MLP) policy of two layers of size _,_ 
policy_kwargs = dict(net_arch=[256,256])
# What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
n_steps = 4096 
learning_rate = lambda f: 1e-4 
ppo_config = {  "gamma":0.99,                       # Discount factor
                "n_steps": int(n_steps/NUM_ENVS),   # number of steps to run for each environment per update
                "ent_coef":0.0,                     # Entropy coefficient for the loss calculation
                "learning_rate":learning_rate, 
                "vf_coef":0.5,                      # Value function coefficient for the loss calculation
                "max_grad_norm":0.5,                # The maximum value for the gradient clipping
                "gae_lambda":0.95,                  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                "batch_size":128,
                "n_epochs":10, 
                "clip_range":0.2,                   # Clipping parameter
                "clip_range_vf":1,                  # Clipping parameter for the value function IMPORTANT: this clipping depends on the reward scaling.
                "verbose":1,                        # the verbosity level: 0 no output, 1 info, 2 debug
                "tensorboard_log":None, 
                "_init_setup_model":True,           # Whether or not to build the network at the creation of the instance
                "policy_kwargs":policy_kwargs,      # additional arguments to be passed to the policy on creation
                "device": gpu_arg}

# What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
sac_config={"learning_rate":1e-4,
            "buffer_size":300000,
            "batch_size":256,
            "ent_coef":'auto', 
            "gamma":0.99, 
            "tau":0.005,
            "train_freq":1, 
            "gradient_steps":1,
            "learning_starts": 10000,
            "verbose":1, 
            "tensorboard_log":None,
            "policy_kwargs": policy_kwargs,
            "seed":None, 
            "device": gpu_arg}

if LEARNING_ALG == "PPO":
    model = PPO('MlpPolicy', env, **ppo_config)
elif LEARNING_ALG == "SAC":
    model = SAC('MlpPolicy', env, **sac_config)
else:
    raise ValueError(LEARNING_ALG + 'not implemented')

print(env.observation_space)
if LOAD_NN:
    if LEARNING_ALG == "PPO":
        model = PPO.load(model_name, env)
    elif LEARNING_ALG == "SAC":
        model = SAC.load(model_name, env)
    print("\nLoaded model", model_name, "\n")

# Learn and save (may need to train for longer)
model.learn(total_timesteps=1000000, log_interval=1,callback=checkpoint_callback)
# Don't forget to save the VecNormalize statistics when saving the agent
model.save( os.path.join(SAVE_PATH, "rl_model" ) ) 
env.save(os.path.join(SAVE_PATH, "vec_normalize.pkl" )) 
if LEARNING_ALG == "SAC": # save replay buffer 
    model.save_replay_buffer(os.path.join(SAVE_PATH,"off_policy_replay_buffer"))

