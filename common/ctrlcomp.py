from common.path_config import PROJECT_ROOT

import numpy as np
from common.utils import FSMCommand


class StateAndCmd:
    def __init__(self, num_joints):
        # robot state
        self.num_joints = num_joints
        self.q = np.zeros(num_joints, dtype=np.float32)  # 关节位置数组
        self.dq = np.zeros(num_joints, dtype=np.float32)    # 关节速度数组(dq)
        self.ddq = np.zeros(num_joints, dtype=np.float32)   # 关节加速度数组(ddq)
        
        # 估计力矩数组(tau_est)，可能用于力矩反馈或分析（未在主代码中使用）
        self.tau_est = np.zeros(num_joints, dtype=np.float32)
        self.gravity_ori = np.array([0., 0., 1.])   # 重力方向向量, 沿 z 轴正方向（0, 0, 1）
        self.ang_vel = np.zeros(3)  # 角速度向量（3D）

        # # 箱子状态（新增）
        # self.box_pos = np.zeros(3, dtype=np.float32)  # 箱子位置 (x, y, z)
        # self.box_vel = np.zeros(3, dtype=np.float32)  # 箱子速度 (vx, vy, vz)
        
        # joy cmd
        self.vel_cmd = np.zeros(3)
        self.skill_cmd = FSMCommand.INVALID  # 初始化技能命令为无效状态（INVALID）
        # skill change cmd
        # self.skill_set = FSMCommand.SKILL_1


class PolicyOutput:
    def __init__(self, num_joints):
        # actions
        self.actions = np.zeros(num_joints, dtype=np.float32)
        self.kps = np.zeros(num_joints, dtype=np.float32)
        self.kds = np.zeros(num_joints, dtype=np.float32)

        # interpolation related
        self.upper_body_indices = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], dtype=np.int32)  # 上半身关节索引
        self.lower_body_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)  # 下半身关节索引
        # self.default_upper_pose = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0], dtype=np.float32)  # 上半身默认姿势

        
        