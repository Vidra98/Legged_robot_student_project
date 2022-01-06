import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg") 
else: # linux
  matplotlib.use('TkAgg')

# stable baselines
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.cmd_util import make_vec_env

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"#"PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '111121133812'
log_dir = interm_dir + 'Fast_boi'#SAC
plot_fig=True
save_plot=True
# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode":"CARTESIAN_PD",       #"PD" | "TORQUE" | "CARTESIAN_PD"
               "task_env": "LR_COURSE_FWD",    #LR_COURSE_FWD,LR_COURSE_BACKWARD,LR_COURSE_SIDEWAY,LR_COURSE_TROT 
               "observation_space_mode": "LR_COURSE_OBS"} # LR_COURSE_OBS , LR_COURSE_OBS_FEET_SENSOR_ADDED
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
env_config['test_env']=False
env_config['competition_env'] = False

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(model_name)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + 'basics ')
#test added : plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
#
time_step=0.001
EPISODE_LENGTH = 10
for i in range(10000):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
      steps=info[0]['steps']
      print('---------------------------------------------------------------------')
      velocity=info[0]['base_veloctiy']
      print('episode_reward', episode_reward)
      print('Final base position', info[0]['base_pos'])
      print('Total_time', info[0]['Total_time'])
      print('Total_movement', info[0]['Total_movement'])

      print('mean veloctiy x: ',np.mean(velocity[250:,0]), 'y :',np.mean(velocity[250:,1]), 'z :',np.mean(velocity[250:,2]))
      #time_steps=[x*time_step for x in range(1,len(info[0]['CoT'] ))]

      if plot_fig:
        CoT=info[0]['CoT']
        mean_CoT=np.mean(CoT[50:])
        CoT_plt = plt.figure()
        print('---------------------------------------------------------------------')
        print('Average CoT', mean_CoT)
        plt.plot(CoT[50:])
        plt.xlabel('timesteps [ms]')
        plt.ylabel('CoT')
        plt.title('Cost of transport')
        
        step_start=50
        step_end=200
        Stepping_plt = plt.figure()
        plt.plot(steps[step_start:step_end,0])
        #plt.plot(steps[:,1])
        plt.plot(steps[step_start:step_end,2])
        #plt.plot(steps[:,3])
        plt.legend(['FR', 'RR'])
        #plt.legend(['FR', 'FL', 'RR', 'RL'])
        plt.title('Stepping time')

        Stepp_plt_tot = plt.figure()
        plt.plot(steps[step_start:step_end,0])
        plt.plot(steps[step_start:step_end,1])
        plt.plot(steps[step_start:step_end,2])
        plt.plot(steps[step_start:step_end,3])
        #plt.legend(['FR', 'RR'])
        plt.legend(['FR', 'FL', 'RR', 'RL'])
        plt.title('Stepping time')

        Velocity_plt = plt.figure()
        plt.plot(velocity[:,0]  )
        plt.plot(velocity[:,1])
        plt.plot(velocity[:,2])
        plt.legend(['x', 'y', 'z'])
        plt.xlabel('timesteps [ms]')
        plt.ylabel('Velocity [m/s]')
        plt.title('Velocity')
        if save_plot:
          CoT_plt.savefig('Figures/CoT.eps')
          Stepping_plt.savefig('Figures/Stepping_plt.eps')
          Velocity_plt.savefig('Figures/Velocity_plt.eps')

        plt.show()
      episode_reward = 0
sns.set()


# same plotting code as above!
#plt.plot(velocity[:,0], velocity[:,1],velocity[:,2])
plt.legend('XYZ', ncol=1, loc='upper left')


# [TODO] make plots:
