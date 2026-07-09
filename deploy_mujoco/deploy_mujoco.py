import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
from common.joystick import JoyStick, JoystickButton

import random
def generate_terrain(model):
    """
    地形逻辑：
    - 关闭课程学习 (随机分布)
    - 4行 x 5列 网格
    - 比例：平地 55%, 草地 35%, 缓坡 10%
    """
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
    if hfield_id == -1:
        print("Warning: No hfield named 'terrain' found in XML.")
        return

    # 获取高度场的行数和列数，确定地形的分辨率
    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    
    # 获取 Z 轴缩放比例 (为了把物理高度转为 0-1 的数据)
    # XML: <hfield size="10 10 1 0.1"/> -> 第3个参数是 max_height (scale)
    # 注意：MuJoCo 的 hfield_data 是归一化的，物理高度 = data * size[2]
    # 我们假设 size[2] = 1.0 (最常见配置)，如果你的 xml 里是其他值，需要按比例调整
    z_scale = model.hfield_size[hfield_id][2] 

    data = np.zeros((nrow, ncol), dtype=np.float32)   # 高度数据数组，存储地形的高度信息

    # 网格为 4 行 5 列，20个区块
    grid_rows = 4
    grid_cols = 5
    
    # 每个逻辑区块在高度场数组中占据的像素数量
    pixels_per_block_row = nrow // grid_rows
    pixels_per_block_col = ncol // grid_cols

    # 准备地形类型池 (共 20 个块)
    # Flat(55%) = 11块, Grass(35%) = 7块, Slope(10%) = 2块
    terrain_types = (
        ["flat"] * 11 + 
        ["grass"] * 7 + 
        ["slope"] * 2
    )
    
    # 打乱顺序 (模拟 curriculum=False)
    random.seed(42) # 固定种子方便复现
    random.shuffle(terrain_types)

    # 确保出生点区域平坦
    mid_col = grid_cols // 2          # 2
    mid_row_start = grid_rows // 2 - 1 # 1 (分割线前一行)
    mid_row_end = grid_rows // 2       # 2 (分割线后一行)
    
    # 这样以 (0,0) 为中心，前后各 5 米都是平地
    idx_1 = mid_row_start * grid_cols + mid_col
    idx_2 = mid_row_end * grid_cols + mid_col
    
    terrain_types[idx_1] = "flat"
    terrain_types[idx_2] = "flat"
    
    print(f"Fixed spawn area to flat at Grid Indices: {idx_1} (Row {mid_row_start}) & {idx_2} (Row {mid_row_end})")
    print(f"Terrain Grid: {grid_rows}x{grid_cols}, Types distributed: {terrain_types}")

    # 填充各个区块
    for r in range(grid_rows):
        for c in range(grid_cols):
            # 当前块的类型
            idx = r * grid_cols + c
            t_type = terrain_types[idx]

            # 计算当前块在 data 数组中的切片范围
            r_start = r * pixels_per_block_row
            r_end = (r + 1) * pixels_per_block_row
            c_start = c * pixels_per_block_col
            c_end = (c + 1) * pixels_per_block_col
            
            # 处理边界溢出 (处理无法整除的情况)
            if r == grid_rows - 1: r_end = nrow
            if c == grid_cols - 1: c_end = ncol

            # 块的大小
            block_h = r_end - r_start
            block_w = c_end - c_start

            if t_type == "flat":    # 平地
                data[r_start:r_end, c_start:c_end] = 0.0    # 高度为0

            elif t_type == "grass":   # 草地
                # 0-3cm 的随机噪声模拟草地
                noise_max = 0.03 / z_scale
                noise = np.random.uniform(0, noise_max, (block_h, block_w))
                data[r_start:r_end, c_start:c_end] = noise

            elif t_type == "slope":  # 缓坡
                # 最大6%坡度的线性斜坡
                slope_ratio = 0.06
                
                # 创建一个线性梯度 0 -> 1
                x = np.linspace(0, 1, block_h)
                slope_data = x * slope_ratio * 4.0 # 假设块长约4米(根据size=10即总长20米算)
                
                # 归一化并调整维度
                slope_data = slope_data / z_scale
                data[r_start:r_end, c_start:c_end] = slope_data[:, np.newaxis]

    # 应用地形数据
    start_addr = model.hfield_adr[hfield_id]
    model.hfield_data[start_addr : start_addr + nrow*ncol] = data.flatten()
    print("Dance/Easy Terrain generated successfully.")

# PD 控制
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""

    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    generate_terrain(m) # 地形生成
    

    m.opt.timestep = simulation_dt
    # 计算每次控制步的持续时间（时间步长 × 降频因子）
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    policy_output_action = np.zeros(num_joints, dtype=np.float32)   # 初始化关节动作数组
    
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0 
    
    state_cmd = StateAndCmd(num_joints)     # 存储机器人状态和控制命令
    policy_output = PolicyOutput(num_joints)    # 存储控制策略的输出
    FSM_controller = FSM(state_cmd, policy_output)   # 管理不同的控制策略
    
    joystick = JoyStick()
    Running = True
    with mujoco.viewer.launch_passive(m, d) as viewer:
        sim_start_time = time.time()
        while viewer.is_running() and Running:
            try:
                if(joystick.is_button_pressed(JoystickButton.SELECT)):
                    Running = False

                joystick.update()
                if joystick.is_button_released(JoystickButton.L1):
                    state_cmd.skill_cmd = FSMCommand.PASSIVE    # 0 力矩
                if joystick.is_button_released(JoystickButton.START):
                    state_cmd.skill_cmd = FSMCommand.POS_RESET  # 默认姿态
                if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.R1):
                    state_cmd.skill_cmd = FSMCommand.LOCO

                # if joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.GET_UP  # getup
                # if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.GET_UP_BACK  # getup_back

                # if joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_10    # Penguin_dance2（拍视频版）
                # if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_11   # guma_45s_dance
                # if joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_16   # SKILL_dahuajiao
                # if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_12   # guma_taiji
                # if joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_13   # Roundhouse_kick
                # if joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_15   # zoo_dance
                # if joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_9   # Penguin_dance4
                # if joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_7    # SKILL_guofucheng_dance
                # if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_17    # SKILL_xinglian_dance1
                # if joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_19    # jiangnanstyle
                # if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
                #     state_cmd.skill_cmd = FSMCommand.SKILL_20   # gaigeshunfeng
                if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
                    state_cmd.skill_cmd = FSMCommand.SKILL_14   # chuxue
                
                
                if joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.R1):
                    state_cmd.skill_cmd = FSMCommand.SKILL_2    # Kongfan
                
                state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
                state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
                state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
                
                step_start = time.time()
                
                # PD 控制器
                tau = pd_control(policy_output_action, d.qpos[7:], kps, np.zeros_like(kps), d.qvel[6:], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)

                sim_counter += 1
                
                # 是否需要执行控制更新
                if sim_counter % control_decimation == 0:
                    
                    qj = d.qpos[7:]    # 关节位置
                    dqj = d.qvel[6:]    # 关节速度
                    quat = d.qpos[3:7]  # 四元数姿态

                    # # 获取箱子的位置和速度
                    # box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
                    # box_pos = d.qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_XPOS, "box_center")]
                    # box_vel = d.qvel[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_XVEL, "box_center")]
                    
                    omega = d.qvel[3:6]   # 角速度
                    gravity_orientation = get_gravity_orientation(quat)  # 重力方向
                    
                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()

                    # # 添加箱子状态到state_cmd（需要在StateAndCmd类中添加相应属性）
                    # state_cmd.box_pos = box_pos.copy()
                    # state_cmd.box_vel = box_vel.copy()
                    
                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()
            
            except ValueError as e:
                print(str(e))
            
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        