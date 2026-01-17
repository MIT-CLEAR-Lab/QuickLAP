# README for Franka Controller

The two files are meant to be a lightweight controller to the robot's joint state in robosuite to the physical robot. At a high level, it (1) reads the joint state of the robosuite robot, then (2) sends that joint state as a network packet to a script listening to "tcp://127.0.0.1:5555", and (3) sends the desired joint state to an impedance controller running on the real robot.

I have attempted to put some safety features on the controller side, e.g., moving to the set point if it is too far from the current position.

To use test this, you will first need to run the franka controller in the `quicklap-polymetis` conda environment:

```
conda activate quicklap-polymetis
cd ~/Desktop/QuickLAP/robosuite_experiments
python franka_controller.py
```

second, you will need to run a demo teleop controller in the `quicklap` conda environment:

```
conda activate quicklap
cd ~/Desktop/QuickLAP/robosuite_experiments
python franka_send_command_example.py
```

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