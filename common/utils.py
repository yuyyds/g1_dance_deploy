from common.path_config import PROJECT_ROOT

import numpy as np
from enum import Enum, unique

@unique
class FSMStateName(Enum):
    INVALID = -1
    PASSIVE = 1
    FIXEDPOSE = 2
    SKILL_COOLDOWN = 3
    LOCOMODE = 4
    SKILL_CAST = 5
    GET_UPMODE = 6
    GET_UP_BACK_MODE = 7
    SKILL_Kongfan = 8
    SKILL_Penguin_dance1 = 14
    SKILL_Penguin_dance4 = 15
    SKILL_Penguin_dance2 = 16
    SKILL_loco_cooldown = 17
    SKILL_45s_dance = 18
    SKILL_taiji = 19
    SKILL_Roundhouse_kick = 20
    SKILL_dance2 = 21
    SKILL_zoo_dance = 22
    SKILL_guofucheng_dance = 23
    SKILL_dahuajiao = 24
    SKILL_xinglian_dance1 = 25
    SKILL_pick_up_box = 26
    SKILL_jiangnanstyle = 27
    SKILL_gaigeshunfeng = 28
   

@unique
class FSMCommand(Enum):
    INVALID = -1
    POS_RESET = 1 
    LOCO = 2
    GET_UP = 6    # SKILL_getup
    GET_UP_BACK = 7
    PASSIVE = 4
    SKILL_1 = 5   # SKILL_CAST
    SKILL_2 = 8   # SKILL_Kongfan
    SKILL_7 = 11    # SKILL_guofucheng_dance
    SKILL_8 = 12    # SKILL_Penguin_dance1
    SKILL_9 = 13    # SKILL_Penguin_dance4
    SKILL_10 = 14   # SKILL_Penguin_dance2
    SKILL_11 = 15   # SKILL_45s_dance
    SKILL_12 = 16   # SKILL_taiji
    SKILL_13 = 17   # SKILL_Roundhouse_kick
    SKILL_14 = 18   # SKILL_dance2
    SKILL_15 = 19   # SKILL_zoo_dance
    SKILL_16 = 20   # SKILL_dahuajiao
    SKILL_17 = 21   # SKILL_xinglian_dance1
    SKILL_18 = 22   # SKILL_pick_up_box
    SKILL_19 = 23   # SKILL_jiangnanstyle
    SKILL_20 = 24   # SKILL_gaigeshunfeng
    

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def progress_bar(current, total, length=50):
    percent = current / total
    filled = int(length * percent)
    bar = "█" * filled + "-" * (length - filled)
    return f"\r|{bar}| {percent:.1%} [{current:.3f}s/{total:.3f}s]"

def scale_values(values, target_ranges):
    scaled = []
    for val, (new_min, new_max) in zip(values, target_ranges):
        scaled_val = (val + 1) * (new_max - new_min) / 2 + new_min
        scaled.append(scaled_val)
    return np.array(scaled)


