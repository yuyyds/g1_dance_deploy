from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
import numpy as np
import yaml
import torch
import os

# 继承 FSMState 并把自身标记为某个 FSMStateName
class SkillCooldown(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_COOLDOWN
        self.name_str = "skill_cooldown"

        self.alpha = 0.    # 插值系数
        self.cur_step = 0   # 当前步数
        self.control_dt = 0.02

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "SkillCooldown.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.policy_path = os.path.join(current_dir, "model", config["policy_path"])
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)
            self.upper_body_motor_idx = np.array(config["upper_body_motor_idx"], dtype=np.int32)
            self.lower_body_motor_idx = np.array(config["lower_body_motor_idx"], dtype=np.int32)
            self.tau_limit =  np.array(config["tau_limit"], dtype=np.float32)
            
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.total_time = config["total_time"]
            self.period = config["period"]  # 相位编码（sin/cos）的周期（秒）
            
            # 预分配缓存(指定维度的零数组)
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.obs = np.zeros(self.num_obs)
            self.action = np.zeros(self.num_actions)
            
            # load policy
            self.policy = torch.jit.load(self.policy_path)
            
            # 用 50 次零输入在 torch.inference_mode() 下跑一遍，避免第一次真推理的 JIT/缓存开销
            for _ in range(50):
                with torch.inference_mode():
                    obs_tensor = self.obs.reshape(1, -1)
                    obs_tensor = obs_tensor.astype(np.float32)
                    self.policy(torch.from_numpy(obs_tensor))
                    
            print("SkillCooldown policy initializing ...")
                
    
    def enter(self):
        # 计算本状态总步数
        self.num_step = int(self.total_time / self.control_dt)  # 计算总步数：total_time (1.0) / control_dt (0.02) = 50 步
        self.upper_dof_size = len(self.upper_body_motor_idx)    # 上半身关节/电机个数
        self.upper_init_dof_pos = np.zeros(self.upper_dof_size, dtype=np.float32)   # 为上半身初始姿态缓存分配空间
        # 重置插值系数 alpha 和当前步数 cur_step 为 0
        self.alpha = 0.
        self.cur_step = 0
        # 用当前机器人状态把上半身初始位置记录下来
        for i in range(self.upper_dof_size):
            self.upper_init_dof_pos[i] = self.state_cmd.q[self.upper_body_motor_idx[i]]
            
    
    def run(self):
        # 读取当前传感器状态
        self.gravity_orientation = self.state_cmd.gravity_ori
        self.qj = self.state_cmd.q.copy()
        self.dqj = self.state_cmd.dq.copy()
        self.ang_vel = self.state_cmd.ang_vel.copy()
        self.cmd = np.zeros(3)  # 线速度(x,y)和角速度(z)
        
        # 构造观测中的下半身部分
        self.qj_obs = (self.qj[self.lower_body_motor_idx] - self.default_angles[self.lower_body_motor_idx]) * self.dof_pos_scale    # 位置残差归一化
        self.dqj_obs = self.dqj[self.lower_body_motor_idx] * self.dof_vel_scale     # 速度归一化
        self.ang_vel = self.ang_vel * self.ang_vel_scale    # 机身角速度缩放
        
        # 周期相位编码
        count = self.cur_step * self.control_dt
        phase = count % self.period / self.period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        
        # 拼装完整 obs
        self.obs[:3] = self.ang_vel.copy()  # ang_vel
        self.obs[3:6] = self.gravity_orientation.copy()  # gravity_orientation
        self.obs[6:9] = self.cmd.copy() # cmd
        self.obs[9: 9 + self.num_actions] = self.qj_obs.copy()
        self.obs[9 + self.num_actions: 9 + self.num_actions * 2] = self.dqj_obs.copy()
        self.obs[9 + self.num_actions * 2: 9 + self.num_actions * 3] = self.action.copy()
        self.obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = np.array([sin_phase, cos_phase])
        
        obs_tensor = self.obs.reshape(1, -1)
        obs_tensor = obs_tensor.astype(np.float32)
        self.action = self.policy(torch.from_numpy(obs_tensor)).detach().numpy().squeeze()
        loco_action = self.action * self.action_scale + self.default_angles[self.lower_body_motor_idx]

        self.policy_output.actions[self.lower_body_motor_idx] = loco_action[self.lower_body_motor_idx].copy()
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()
        
        ###########################################################
        # 上半身“回正插值”
        self.cur_step += 1
        self.alpha = min(self.cur_step / self.num_step, 1.0)
        for j in range(self.upper_dof_size):
            motor_idx = self.upper_body_motor_idx[j]
            target_pos = self.default_angles[motor_idx]
            self.policy_output.actions[motor_idx] = self.upper_init_dof_pos[j] * (1 - self.alpha) + target_pos * self.alpha
        
    
    def exit(self):
        pass
    
    def checkChange(self):
        # 冷却时间到 → 清空 skill_cmd → 切回 FSMStateName.LOCOMODE
        if(self.cur_step >= self.num_step):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        