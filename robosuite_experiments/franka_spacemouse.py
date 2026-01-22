import select
import time
from evdev import InputDevice, categorize, ecodes

buttons = {256:0, 257:0}


class SpaceMouseInput():

    def __init__(self):
        self.device =InputDevice("/dev/input/event23")
        self.scale_trans = 0.003
        self.scale_rot = 0.003
        self.axes = {ecodes.ABS_X:0, ecodes.ABS_Y:0, ecodes.ABS_Z:0,
            ecodes.ABS_RX:0, ecodes.ABS_RY:0, ecodes.ABS_RZ:0}
    
    def get_input(self):
        '''
        Returns the space mouse reading as a 6-dimensional array.
        '''
        r, w, x = select.select([self.device.fd], [], [], 0)  # timeout=0 → non-blocking
        if r:
            for event in self.device.read():
                if event.type == ecodes.EV_ABS:
                    self.axes[event.code] = event.value
        
        command = [
            self.axes[ecodes.ABS_X] * self.scale_trans,
            self.axes[ecodes.ABS_Y] * self.scale_trans,
            self.axes[ecodes.ABS_Z] * self.scale_trans,
            self.axes[ecodes.ABS_RX] * self.scale_rot,
            self.axes[ecodes.ABS_RY] * self.scale_rot,
            self.axes[ecodes.ABS_RZ] * self.scale_rot,
        ]
        
        return command


if __name__ == '__main__':
    try:
        mouse = SpaceMouseInput()

        while True:
            

            # Map axes to velocity command
            command = mouse.get_input()
            print(f'{command[0]:.2f}, {command[1]:.2f}, {command[5]:.2f}')
            
            # Small sleep to avoid busy loop
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("Exiting teleop")