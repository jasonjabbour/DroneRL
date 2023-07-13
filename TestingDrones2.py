import time
import numpy as np
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('Models/dronesimple.xml')
d = mujoco.MjData(m)


with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 360 wall-seconds.
  start = time.time()
  timestep = 0
  timesteplist = []
  i = 0
  xlist =[]
  ylist =[]
  zlist =[]
  while viewer.is_running() and timestep < 2000:
    i += 1
    timestep += 1
    timesteplist.append(timestep)
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)
    if i > 100 and i < 500:
      control_signals = np.array([4, 0, 0, 0])
    elif i > 500:
      i = 0
    else:
      control_signals = np.array([0, 0, 0, 0])
    
    
    d.ctrl[:] = control_signals
    # Retrieve position (x, y, z) and orientation (roll, pitch, yaw) of the drone
    x, y, z = d.qpos[0:3]
    roll, pitch, yaw = d.qpos[3:6]

    # Print the position and orientation
    print(f"Position: ({x}, {y}, {z})")
    xlist.append(x)
    ylist.append(y)
    zlist.append(z)
    print(f"Orientation: ({roll}, {pitch}, {yaw})")
    print(timestep)
    


    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

print(xlist)
print(ylist)
print(zlist)
print("Time step list:")
print(timesteplist)

# Create a figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# Plot x position over time
axs[0].plot(timesteplist, xlist)
axs[0].set_ylabel('X Position')
axs[0].set_ylim([-2, 2])

# Plot y position over time
axs[1].plot(timesteplist, ylist)
axs[1].set_ylabel('Y Position')
axs[1].set_ylim([-2, 2])

# Plot z position over time
axs[2].plot(timesteplist, zlist)
axs[2].set_ylabel('Z Position')
axs[2].set_xlabel('Time')
axs[2].set_ylim([-2, 2])

# Adjust the layout to avoid overlapping labels
#plt.tight_layout()

# Show the plot
plt.subplots_adjust(top=0.9)
fig.suptitle('Position of Drone While Hovering', fontsize=16)
plt.savefig('hovering_position_plot.png', dpi=300)
plt.show()