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
from hardware_config import RobotConfig
from unitree_cpp_env import UnitreeCppEnv 

class Controller:
    def __init__(self):
        self.cfg = RobotConfig()
        self.net = self.cfg.unitree.net_if
        
        print("The Unitree Cpp interface is being started...")
        self.env = UnitreeCppEnv(self.cfg)
        
        self.remote_controller = RemoteController()
        self.num_joints = self.cfg.num_dofs
        self.control_dt = self.cfg.unitree.control_dt

        # # 等待初始状态(若启用按两次 start)
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

    def shutdown(self):
        print("Turn off after sending the damping command...")
        damping_cmd = np.zeros(self.num_joints)
        self.env.set_gains([0]*self.num_joints, [2.0]*self.num_joints)  # kd=2.0
        for _ in range(20): 
            self.env.step(damping_cmd)
            time.sleep(0.02)
        self.env.shutdown()
        print("The robot has been safely shut down.")

    def run(self):
        try:
            while self.running:
                loop_start_time = time.time()

                self.env.update()

                # 读取手柄数据，更新遥控器状态
                remote_data = self.env.get_wireless_remote()
                if remote_data:
                    self.remote_controller.set(remote_data)
                
                self.qj = self.env._joint_positions   # 关节位置（角度）
                self.dqj = self.env._joint_velocities   # 关节角速度
                
                imu_raw = self.env._imu_quaternion # x,y,z,w
                self.quat = np.array([imu_raw[3], imu_raw[0], imu_raw[1], imu_raw[2]])  # 转为 w,x,y,z
                
                self.ang_vel = self.env._imu_angular_velocity   # 机身角速度

                if self.remote_controller.is_button_pressed(KeyMap.select):
                    break
                
                if self.remote_controller.is_button_pressed(KeyMap.F1): 
                    self.state_cmd.skill_cmd = FSMCommand.PASSIVE
                if self.remote_controller.is_button_pressed(KeyMap.start):
                    self.state_cmd.skill_cmd = FSMCommand.POS_RESET
                if self.remote_controller.is_button_pressed(KeyMap.A) and self.remote_controller.is_button_pressed(KeyMap.R1):
                    self.state_cmd.skill_cmd = FSMCommand.LOCO

                # ================== 起身 ==================
                # if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.GET_UP
                # if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.GET_UP_BACK

                # ================== 舞蹈 ==================
                # if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_10  # Penguin_dance2
                # if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_9  # Penguin_dance4
                if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_12  # taiji
                if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.R1):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_7  # SKILL_guofucheng_dance
                # if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_11  # 45s dance
                if self.remote_controller.is_button_pressed(KeyMap.B) and self.remote_controller.is_button_pressed(KeyMap.R1):
                    self.state_cmd.skill_cmd = FSMCommand.SKILL_16   # SKILL_dahuajiao
                # # if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.R1):
                # #     self.state_cmd.skill_cmd = FSMCommand.SKILL_17  # SKILL_xinglian_dance
                # if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_19  # jiangnanstyle
                # if self.remote_controller.is_button_pressed(KeyMap.Y) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_20  # gaigeshunfeng
                # if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_15  # zoo dance
                # if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_14  # chuxue dance

                # ================== 剧烈动作 ==================
                # if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_2  # Kongfan（韦伯斯特和侧手翻都有，dance_deploy/policy/Kongfan/config/Kongfan.yaml中修改）
                # if self.remote_controller.is_button_pressed(KeyMap.X) and self.remote_controller.is_button_pressed(KeyMap.R1):
                #     self.state_cmd.skill_cmd = FSMCommand.SKILL_13  # Roundhouse_kick


                self.state_cmd.vel_cmd[0] = self.remote_controller.ly
                self.state_cmd.vel_cmd[1] = self.remote_controller.lx * -1
                self.state_cmd.vel_cmd[2] = self.remote_controller.rx * -1
                
                
                self.state_cmd.q = self.qj
                self.state_cmd.dq = self.dqj
                self.state_cmd.gravity_ori = get_gravity_orientation_real(self.quat)
                self.state_cmd.ang_vel = self.ang_vel
                self.state_cmd.base_quat = self.quat
                # print("self.state_cmd.base_quat:", self.state_cmd.base_quat)
                # print("self.quat:", self.quat)

                self.FSM_controller.run()

                target_q = self.policy_output.actions
                target_kp = self.policy_output.kps
                target_kd = self.policy_output.kds

                self.env.set_gains(target_kp, target_kd)   # 设置每个关节的 PD 参数
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

if __name__ == "__main__":
    controller = Controller()
    controller.run()