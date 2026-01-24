import time
import numpy as np
import zmq
import robosuite as suite

from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper
from robosuite.devices import Keyboard

from franka_spacemouse import SpaceMouseInput

# -----------------------
# Controller configuration
# -----------------------
controller_config = load_composite_controller_config(
    controller=None,   # use robot default (OSC)
    robot="Panda",
)

# -----------------------
# Create environment
# -----------------------
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="agentview",
    use_camera_obs=False,
    control_freq=20,
    ignore_done=True,
    hard_reset=False,
)

env = VisualizationWrapper(env)

# -----------------------
# Keyboard device
# -----------------------
device = Keyboard(
    env=env,
    pos_sensitivity=1.0,
    rot_sensitivity=1.0,
)
env.viewer.add_keypress_callback(device.on_press)

# -----------------------
# Spacemouse device
# -----------------------
mouse = SpaceMouseInput()




if __name__ == '__main__':

    # Desired initial joint configuration (Panda arm, 7 joints)
    q2 = np.array([-0.1514, 0.2535, 0.0831, -2.0308, -0.0671, 2.1418, 0.2937], dtype=np.float64)

    # -----------------------
    #  set up communications
    # -----------------------

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    # sock.connect("tcp://127.0.0.1:5555")

    SERVER_IP = '128.30.29.23'
    sock.connect(f"tcp://{SERVER_IP}:5555")

    obs = env.reset()
    # Initialize robot arm pose in simulation before starting teleop.
    robot = env.robots[0]
    robot.set_robot_joint_positions(q2)
    env.sim.forward()
    env.render()

    device.start_control()

    # -----------------------
    # Main loop
    # -----------------------

    while True:
        start = time.time()
        state = robot._joint_positions

        action_dict = device.input2action()
        
        print(action_dict)


        actions = mouse.get_input() # spacemouse x, y, z, roll, pitch yaw, button 
        robot_action = np.array([actions[0], -actions[1], actions[5], 0 ,0 ,0, actions[6]]) # robot's x,y,z,roll,pitch,yaw,gripper


        env.step(robot_action)
        env.render()

        state = np.hstack([robot._joint_positions, robot._joint_velocities, [robot_action[6]]]) # robot's joint state, joint vel, gripper state
        # print(state)
        
        sock.send(state.tobytes())             # blocking send
        # print(f'sent: {state}')
        reply = sock.recv()                # blocking receive

        # ~20 Hz
        dt = time.time() - start
        if dt < 1 / 20:
            time.sleep(1 / 20 - dt)



