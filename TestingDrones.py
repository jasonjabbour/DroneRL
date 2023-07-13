import mujoco
import mediapy as media
import numpy as np
import matplotlib.pyplot as plt
"""
# Load the MuJoCo model
model = mujoco_py.load_model_from_path("Models/testxml.xml")

# Create a MuJoCo simulator
sim = mujoco_py.MjSim(model)

# Create a MuJoCo renderer
renderer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)

# Set the size of the renderer
width, height = 480, 480
renderer.vopt.framebuffer_size = (width, height)

# Load and set the image
image_path = "image.png"
image = mujoco_py.load_rgba(image_path)
sim.data.mocap_pos[0] = np.array([0, 0, 0])  # Set the position of the image in the world
sim.data.mocap_quat[0] = np.array([0, 0, 0, 0])  # Set the orientation of the image in the world
sim.data.mocap_quat[0] /= np.linalg.norm(sim.data.mocap_quat[0])  # Normalize the quaternion

# Render the scene
renderer.render(sim)
data = renderer.read_pixels(width, height, depth=False)
image = np.flipud(data)

# Display the image using Matplotlib
plt.imshow(image)
plt.axis("off")
plt.show()
"""
xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)
media.show_image(renderer.render())

mujoco.mj_forward(model, data)
renderer.update_scene(data)

media.show_image(renderer.render())
