import time
import numpy as np
import gymnasium
from gymnasium import spaces

import mujoco
import mujoco.viewer


class DroneEnv(gymnasium.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    """
    Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space 
    specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.

    Example:
    >>> EnvTest = FooEnv()
    >>> EnvTest.observation_space=spaces.Box(low=-1, high=1, shape=(3,4))
    >>> EnvTest.action_space=spaces.Discrete(2)
    """
    self.eps_timestep = 3000

    self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1, ), dtype=np.float32)
    self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6, ), dtype=np.float32)
    
    self.mujocomodel = mujoco.MjModel.from_xml_path('Models/dronesimple.xml')
    self.mujocodata = mujoco.MjData(self.mujocomodel)

    #self.viewer = mujoco.viewer.launch_passive(self.mujocomodel, self.mujocodata)


  def step(self, action):
    """
    This method is the primary interface between environment and agent.

    Paramters: 
        action: int
                the index of the respective action (if action space is discrete)

    Returns:
        output: (array, float, bool)
                information provided by the environment about its current state:
                (observation, reward, done)
    """
    step_start = time.time()

    mujoco.mj_step(self.mujocomodel, self.mujocodata)
    control_signals = np.array([(3 + action[0]), 0, 0, 0])
    self.mujocodata.ctrl[:] = control_signals
    #print(action[0])

    #with self.viewer.lock():
    #  self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.mujocodata.time % 2)
    #self.viewer.sync()

    # Timestep = 0.01
    #time_until_next_step = self.mujocomodel.opt.timestep - (time.time() - step_start)
    #if time_until_next_step > 0:
    #  time.sleep(time_until_next_step)

    x, y, z = self.mujocodata.qpos[0:3]
    roll, pitch, yaw = self.mujocodata.qpos[3:6]

    #Returns data values
    observation = np.array([x, y, z, roll, pitch, yaw])
    reward = self.reward(observation)
    done = False
    info = {}
    truncated = False
  
    self.step_count +=1

    if self.step_count > self.eps_timestep:
      done = True
    
    return observation, reward, done, truncated, info
  

  def reset(self, seed=0):
    """
    This method resets the environment to its initial values.

    Returns:
        observation:    array
                        the initial state of the environment
    """
    self.step_count = 0

    self.mujocodata.qpos[:] = [0, 0, 0, 0, 0, 0, 0]
    self.mujocodata.qvel[:] = [0, 0, 0, 0, 0, 0]
    self.mujocodata.qacc[:] = [0, 0, 0, 0, 0, 0]
    self.mujocodata.ctrl[:] = [0.0, 0.0, 0.0, 0.0]
    

    x, y, z = self.mujocodata.qpos[0:3]
    roll, pitch, yaw = self.mujocodata.qpos[3:6]

    observation = np.array([x, y, z, roll, pitch, yaw])

    return observation, {}
  
  def reward(self, observation):
    x = observation[0]
    y = observation[1]
    z = observation[2]

    if (x > -0.1) and (x < 0.1) and (y > -0.1) and (y < 0.1) and (z > 0.5) and (z < 0.75):
      return 1
    elif (x > -0.1) and (x < 0.1) and (y > -0.1) and (y < 0.1) and (z > 0.25) and (z < 1):
      return 0.1
    elif (x > -0.1) and (x < 0.1) and (y > -0.1) and (y < 0.1) and (z < 0.1):
      return -0.1
    else:
      return 0
    



  def render(self, mode='human', close=False):
    """
    This methods provides the option to render the environment's behavior to a window 
    which should be readable to the human eye if mode is set to 'human'.
    """
    pass

  def close(self):
    """
    This method provides the user with the option to perform any necessary cleanup.
    """
    pass