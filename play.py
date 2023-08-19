import numpy as np
import time
import gym
import matplotlib.pyplot as plt
from Droneenv import DroneEnv
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback


def test():
    timesteps = 10000
    env = DroneEnv()
    loaded_model = SAC.load('./policies/policy9/policy9.zip')
    observation = env.reset()
    for timestep in range(timesteps):
        #print(timestep)
        #action = env.action_space.sample()
        action, _ = loaded_model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        #print(f'Input Action: {action} Observation: {observation} Reward: {reward} Done: {done}')
        print(f'Reward: {reward}')

        #if done:
        #    observation = env.reset()

def train():
    env = DroneEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="policies/policy9")
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path="policies/policy9", name_prefix="policy9")
    model.learn(total_timesteps=100000)
    model.save("policies/policy9/policy9")

def profile():
    timesteps = 3000
    env = DroneEnv()
    loaded_model = SAC.load('./policies/policy9/policy9.zip')
    observation = env.reset()
    stored_obs = []
    for timestep in range(timesteps):
        action, _ = loaded_model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
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
    loaded_model = SAC.load('./policies/policy9/policy9.zip')
    observation = env.reset()
    total_reward = []
    for timestep in range(timesteps):
        action, _ = loaded_model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        total_reward.append(reward)
    avg_reward = sum(total_reward) / timesteps
    print(f"Policy average reward: {avg_reward}")
    x = list(range(timesteps))
    y = total_reward
    plt.plot(x, y)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Reward over Timesteps - SAC 100K Training")
    plt.savefig("Reward over Timesteps - SAC 100K Training.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    mode = 'reward'

    if mode == 'train':
        train()
    elif mode == "test":
        test()
    elif mode == "profile":
        profile()
    elif mode == "reward":
        calc_reward()
    else:
        print("unacceptable mode")