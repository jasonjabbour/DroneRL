import numpy as np
import time
#import gym
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
import matplotlib.pyplot as plt
from Droneenv import DroneEnv
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback

"""
#checkpoints=[1e6, 10e6, 30e6]
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, checkpoints=[10000, 50000, 100000]):
        super(SaveOnBestTrainingRewardCallback, self).__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_reward = float('-inf')
        self.checkpoints = set(checkpoints)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get episode rewards from the monitor wrapper
            #ep_infos = self.model.get_env().envs[0].get_original_env().get_episode_rewards()
            ep_infos = self.model.get_env().get_attr('get_episode_rewards')[0]()
            if len(ep_infos) > 100:
                current_reward = sum([info['r'] for info in ep_infos[-100:]]) / 100
                current_step = self.model.num_timesteps

                # Check if current reward is greater than previous best and we're at a desired checkpoint
                if current_reward > self.best_reward and current_step in self.checkpoints:
                    self.best_reward = current_reward
                    self.model.save(os.path.join(self.save_path, f"best_model_{current_step}"))
                    print(f"Best model saved with reward {self.best_reward} at step {current_step}")

        return True
"""
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, checkpoints=[100000, 500000, 1000000]):
        super(SaveOnBestTrainingRewardCallback, self).__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_reward = float('-inf')
        self.checkpoints = sorted(checkpoints)
        self.next_checkpoint = self.checkpoints.pop(0)
        self.best_model_path = None

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Accessing the Monitor's method through VecNormalize, DummyVecEnv
            ep_infos = self.model.get_env().get_attr('get_episode_rewards')[0]()
            #print(ep_infos)
            if len(ep_infos) >= 100:
                current_reward = sum(ep_infos[-100:]) / 100
                current_step = self.model.num_timesteps

                # Check if current reward is greater than previous best
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    # Temporarily save this as the best model so far
                    #self.best_model_path = os.path.join(self.save_path, f"temp_best_model_until_{current_step}")
                    self.best_model_path = self.save_path+ f"/temp_best_model_until_{self.next_checkpoint}.zip"
                    self.model.save(self.best_model_path)

                # Check if we have reached the next checkpoint
                if current_step >= self.next_checkpoint:
                    if self.best_model_path:
                        # Rename the saved model to reflect the checkpoint
                        os.rename(self.best_model_path, self.save_path+ f"/best_model_until_{self.next_checkpoint}.zip") #+.zip after first bestmodel
                        print(f"Model saved with best reward {self.best_reward} up to step {self.next_checkpoint}")
                        # Reset best_reward for the next interval
                        self.best_reward = float('-inf')
                        # Shift to the next checkpoint
                        if self.checkpoints:
                            self.next_checkpoint = self.checkpoints.pop(0)
                        self.best_model_path = None
                       
        return True


def test():
    timesteps = 10000
    env = DroneEnv()
    loaded_model = PPO.load('./policies/policy19/policy19.zip')
    observation, _ = env.reset()
    for timestep in range(timesteps):
        #print(timestep)
        #action = env.action_space.sample()
        action, _ = loaded_model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        #print(f'Input Action: {action} Observation: {observation} Reward: {reward} Done: {done}')
        print(f'Reward: {reward}')

        #if done:
        #    observation = env.reset()

def train():
    env = DroneEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda:env])
    env = VecNormalize(env)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="policies/policy15")
    #checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="policies/policy9", name_prefix="policy9")
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, save_path="policies/policy15")
    model.learn(total_timesteps=1100000, callback=callback)
    model.save("policies/policy15/policy15")

def train_simple():
    env = DroneEnv()
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="policies/policy22")
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path="policies/policy22", name_prefix="policy22")
    model.learn(total_timesteps=100000)
    model.save("policies/policy21/policy22")

def profile():
    timesteps = 3000
    env = DroneEnv()
    loaded_model = SAC.load('./policies/policy21/policy22.zip')
    observation, _ = env.reset()
    stored_obs = []
    for timestep in range(timesteps):
        action, _ = loaded_model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        stored_obs.append(observation)
    inference_time_list = []
    for obs in stored_obs:
        start_time = time.time()
        action, _ = loaded_model.predict(obs, deterministic=True)
        end_time = time.time()
        total_time = end_time - start_time
        inference_time_list.append(total_time)
    avg_time = sum(inference_time_list) / len(inference_time_list)
    print(f"Policy average inference time: {avg_time}")

def calc_reward():
    timesteps = 3000
    env = DroneEnv()
    loaded_model = PPO.load('./policies/policy19/policy19.zip')
    observation, _ = env.reset()
    total_reward = []
    
    for timestep in range(timesteps):
        action, _ = loaded_model.predict(observation, deterministic=False)
        observation, reward, done, truncated, info = env.step(action)
        total_reward.append(reward)
    
    avg_reward = sum(total_reward) / timesteps
    print(f"Policy average reward: {avg_reward}")
    x = list(range(timesteps))
    y = total_reward

    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Reward over Timesteps - PPO 1M Training")
    plt.savefig("Reward over Timesteps - PPO 1M Training.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    mode = 'reward'

    if mode == 'train':
        train()
    elif mode == 'train_simple':
        train_simple()
    elif mode == "test":
        test()
    elif mode == "profile":
        profile()
    elif mode == "reward":
        calc_reward()
    else:
        print("unacceptable mode")