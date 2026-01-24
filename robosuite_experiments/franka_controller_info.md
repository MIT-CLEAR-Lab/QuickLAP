# README for Franka Controller

The two files are meant to be a lightweight controller to the robot's joint state in robosuite to the physical robot. At a high level, it (1) reads the joint state of the robosuite robot on Nathan's computer, then (2) sends that joint state as a network packet to a script listening to "tcp://0.0.0.0:5555" on Minyoung's Laptop, and (3) sends the desired joint state to an impedance controller running on the real robot (The Franka NUC).

I have attempted to put some safety features on the controller side, e.g., moving to the set point if it is too far from the current position.


Before doing anything, verify that the Franka NUC is running both the robot arm server and the gripper server. Also that the Franka is in execution mode with the FCI enabled.


To use test the controller, you will first need to run the franka controller in the `quicklap-polymetis` conda environment on Minyoung's Laptop:
```
conda activate quicklap-polymetis
cd ~/Desktop/QuickLAP/robosuite_experiments
python franka_controller.py
```

second, you will need to run a demo teleop controller in the `QuickLap` conda environment on Nathan's Desktop:
```
conda activate QuickLap
cd ~/Desktop/QuickLAP/robosuite_experiments
python franka_send_command_example.py
```

TROUBLESHOOTING:
if nothing is moving in the simulation on Nathan's computer, it is possible that the space mouse is not connected. Try running `franka_spacemouse.py` to see if any inputs are registering. If not, check the suggestion at the top of that python file to connect to the space mouse

If you are getting networking errors on Nathan's computer, double check that the IP address of Minyoung's Laptop is correct, as hard-coded in the `franka_send_command_example.py`. 

....list more help here as we encounter errors


----

To adapt a regular robosuite environment to work with this controller, you will first need to set up the communication channel using zmq:

```
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://127.0.0.1:5555")
```

Next, you will want to send the commands in every update of the simulated environment. The reply waits for the controller on the physical robot to reach its position before advancing the simulation:

```
state = robot._joint_positions # robosuite robot joint positions

sock.send(state.tobytes())
reply = sock.recv()  
```