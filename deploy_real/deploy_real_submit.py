import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import time
import numpy as np
import os

from common.path_config import PROJECT_ROOT
from common.ctrlcomp import *
from FSM.FSM import *
from common.command_helper import MotorMode
from common.ctrlcomp import *
from common.rotation_helper import get_gravity_orientation_real
from common.remote_controller import RemoteController, KeyMap
from deploy_real.config.hardware_config import RobotConfig
from deploy_real.config.unitree_cpp_env import UnitreeCppEnv 

import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactory
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

# Sentinel function (responsible for listening, not loading the C++ environment)
def run_sentinel_mode():
    print("==========================================")
    print(">>> [Sentinel Mode] activated")
    print(">>> The DDS listener is being initialized...")
    
    cfg = RobotConfig()
    net_if = cfg.unitree.net_if
    
    try:
        channel_factory = ChannelFactory()
        channel_factory.Init(0, net_if)
    except Exception as e:
        print(f"DDS initialization failed: {e}")
        return

    remote_keys_container = {'keys': 0}
    
    def remote_callback(msg: WirelessController_):
        remote_keys_container['keys'] = msg.keys

    subscriber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
    subscriber.Init(remote_callback, 10)

    print(f">>> [Sentinel] Monitoring network card: {net_if}")
    print(">>> [Sentinel] Please press【L2 + R2】to activate the taiji action...")
    

    while True:
        time.sleep(0.1)
        keys = remote_keys_container['keys']
        
        # L2=32, R2=16.
        if (keys & 32) and (keys & 16): 
            print(f">>> [Sentinel] Detected L2+R2 (Keys: {keys})!")   # 48
            break
            
    subscriber.Close()
    time.sleep(1)
    print(">>> [Sentinel] Preparing to restart into main control mode...")
    
    # Restart itself and clear the DDS contamination
    python_exe = sys.executable
    script_file = os.path.abspath(__file__)
    
    # 执行命令： python deploy_real_submit.py run_controller
    os.execv(python_exe, [python_exe, script_file, "run_controller"])

class Controller:
    def __init__(self):

        print(">>> [Main Control Mode] Starting Unitree Cpp interface...")
        self.cfg = RobotConfig()

        self.env = UnitreeCppEnv(self.cfg)
        
        self.remote_controller = RemoteController()
        self.num_joints = self.cfg.num_dofs
        self.control_dt = self.cfg.unitree.control_dt

        # # Wait for the initial state
        # self.wait_for_start()

        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        self.FSM_controller = FSM(self.state_cmd, self.policy_output)
        
        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32) # w, x, y, z
        self.ang_vel = np.zeros(3, dtype=np.float32)
        
        self.running = True
        print("The controller initialization is complete. Wait for the main loop...")

    # def wait_for_start(self):
    #     print("Wait for the start signal...")
    #     self.env.update()
    #     remote_data = self.env.get_wireless_remote()
    #     if remote_data:
    #         self.remote_controller.set(remote_data)
            
    #     while not self.remote_controller.is_button_pressed(KeyMap.start):
    #         time.sleep(self.control_dt)
            
    #         self.env.update()
    #         remote_data = self.env.get_wireless_remote()
    #         if remote_data:
    #             self.remote_controller.set(remote_data)
    #     print("Received the start signal and began to run...")

    def shutdown(self):
        print("Turn off after sending the damping command...")
        damping_cmd = np.zeros(self.num_joints)
        self.env.set_gains([0]*self.num_joints, [2.0]*self.num_joints)  # kd=2.0
        for _ in range(20):  # Send the damping command several times to ensure its effectiveness
            self.env.step(damping_cmd)
            time.sleep(0.02)
        self.env.shutdown()
        print("The robot has been safely shut down.")

    def run(self):
        try:
            while self.running:
                loop_start_time = time.time()

                self.env.update()

                remote_data = self.env.get_wireless_remote()
                if remote_data:
                    self.remote_controller.set(remote_data)
                
                self.qj = self.env._joint_positions
                self.dqj = self.env._joint_velocities
                
                # Processing IMU: The SDK provides [x,y,z,w], and we need [w,x,y,z]
                imu_raw = self.env._imu_quaternion # x,y,z,w
                self.quat = np.array([imu_raw[3], imu_raw[0], imu_raw[1], imu_raw[2]])
                
                self.ang_vel = self.env._imu_angular_velocity

                if self.remote_controller.is_button_pressed(KeyMap.select):
                    break
                
                if self.remote_controller.is_button_pressed(KeyMap.F1): 
                    self.state_cmd.skill_cmd = FSMCommand.PASSIVE
                if self.remote_controller.is_button_pressed(KeyMap.start):
                    self.state_cmd.skill_cmd = FSMCommand.POS_RESET
                if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
                    self.state_cmd.skill_cmd = FSMCommand.LOCO
                if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_12 # taiji
                if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.L1):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_11 # 45s dance
                if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.R1):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_14 # dance2
                if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_15 # zoo dance

                self.state_cmd.vel_cmd[0] = self.remote_controller.ly
                self.state_cmd.vel_cmd[1] = self.remote_controller.lx * -1
                self.state_cmd.vel_cmd[2] = self.remote_controller.rx * -1
                
                self.state_cmd.q = self.qj
                self.state_cmd.dq = self.dqj
                self.state_cmd.gravity_ori = get_gravity_orientation_real(self.quat)
                self.state_cmd.ang_vel = self.ang_vel
                self.state_cmd.base_quat = self.quat

                self.FSM_controller.run()

                target_q = self.policy_output.actions
                target_kp = self.policy_output.kps
                target_kd = self.policy_output.kds

                self.env.set_gains(target_kp, target_kd)
                self.env.step(target_q)

                loop_end_time = time.time()
                delta_time = loop_end_time - loop_start_time
                if delta_time < self.control_dt:
                    time.sleep(self.control_dt - delta_time)

        except KeyboardInterrupt:
            pass
        finally:
            print("The connection is being closed...")
            self.shutdown()


# Decide whether to run the sentinel or the main control based on the parameters
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_controller":
        controller = Controller()
        controller.run()
    else:
        run_sentinel_mode()