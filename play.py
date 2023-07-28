import numpy as np
import gym
from Droneenv import DroneEnv

def test():
    timesteps = 10000
    env = DroneEnv()
    env.reset()
    for timestep in range(timesteps):
        #print(timestep)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print(f'Input Action: {action} Observation: {observation} Reward: {reward} Done: {done}')
        print(f'Reward: {reward}')

        if done:
            observation = env.reset()

def train():


    return


if __name__ == "__main__":
    test()