from common.path_config import PROJECT_ROOT

from policy.passive.PassiveMode import PassiveMode
from policy.fixedpose.FixedPose import FixedPose
from policy.loco_mode.LocoMode import LocoMode
from policy.skill_cooldown.SkillCooldown import SkillCooldown
# from policy.loco_cooldown.loco_cooldown import loco_cooldown
from policy.skill_cast.SkillCast import SkillCast
from policy.Getup.Getup import Getup
from policy.Getup_back.Getup_back import Getup_back
from policy.Kongfan.Kongfan import Kongfan
# from policy.Penguin_dance1.Penguin_danc1 import Penguin_dance1
from policy.Penguin_dance2.Penguin_danc2 import Penguin_dance2
from policy.Penguin_dance4.Penguin_danc4 import Penguin_dance4
from policy.guofucheng_dance.guofucheng_dance import guofucheng_dance
from policy.dahuajiao.dahuajiao import dahuajiao
from policy.Roundhouse_kick.Roundhouse_kick import Roundhouse_kick
from policy.guma_45s_dance.guma_45s_dance import guma_45s_dance
from policy.guma_dance2.guma_dance2 import guma_dance2
from policy.guma_taiji.guma_taiji import guma_taiji
from policy.zoo_dance.zoo_dance import zoo_dance
from policy.xinglian_dance1.xinglian_dance1 import xinglian_dance1
from policy.jiangnanstyle.jiangnanstyle import jiangnanstyle
from policy.gaigeshunfeng.gaigeshunfeng import gaigeshunfeng
# from policy.pick_up_box.pick_up_box import pick_up_box
from FSM.FSMState import *
import time
from common.ctrlcomp import *
from enum import Enum, unique
import os

@unique
class FSMMode(Enum):
    CHANGE = 1  # 状态变更模式
        # elif((policy_name == FSMStateName.SKILL
    NORMAL = 2  # 正常运行模式


class FSM:
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.cur_policy : FSMState
        self.next_policy : FSMState
        
        self.FSMmode = FSMMode.NORMAL
        self.passive_mode = PassiveMode(state_cmd, policy_output)
        self.fixed_pose_1 = FixedPose(state_cmd, policy_output)
        self.loco_policy = LocoMode(state_cmd, policy_output)        
        self.getup_policy = Getup(state_cmd, policy_output)
        self.getup_back_policy = Getup_back(state_cmd, policy_output)

        self.skill_cooldown_policy = SkillCooldown(state_cmd, policy_output)
        self.skill_cast_policy = SkillCast(state_cmd, policy_output)

        # self.Penguin_dance1_policy = Penguin_dance1(state_cmd, policy_output)
        self.Penguin_dance2_policy = Penguin_dance2(state_cmd, policy_output)
        self.Penguin_dance4_policy = Penguin_dance4(state_cmd, policy_output)
        
        self.Kongfan_policy = Kongfan(state_cmd, policy_output)
        self.Roundhouse_kick_policy = Roundhouse_kick(state_cmd, policy_output)
        self.guma_45s_dance_policy = guma_45s_dance(state_cmd, policy_output)
        self.guofucheng_dance_policy = guofucheng_dance(state_cmd, policy_output)
        self.guma_dance2_policy = guma_dance2(state_cmd, policy_output)
        self.zoo_dance_policy = zoo_dance(state_cmd, policy_output)
        self.dahuajiao_policy = dahuajiao(state_cmd, policy_output)
        self.xinglian_dance1_policy = xinglian_dance1(state_cmd, policy_output)
        # self.pick_up_box_policy = pick_up_box(state_cmd, policy_output)
        self.jiangnanstyle_policy = jiangnanstyle(state_cmd, policy_output)
        self.gaigeshunfeng_policy = gaigeshunfeng(state_cmd, policy_output)
        
        self.guma_taiji_policy = guma_taiji(state_cmd, policy_output)
        
        # self.interp_policy = Interp(state_cmd, policy_output)

        print("initalized all policies!!!")
        
        # 设置初始策略为被动模式
        self.cur_policy = self.passive_mode

        print("current policy is ", self.cur_policy.name_str)

        
    def run(self):
        start_time = time.time()

        if(self.FSMmode == FSMMode.NORMAL): 
            self.cur_policy.run()
            nextPolicyName = self.cur_policy.checkChange()
            
            # 如果 checkChange 返回的 nextPolicyName 与当前策略的名称（cur_policy.name）不同，表明需要切换策略
            if(nextPolicyName != self.cur_policy.name):
                # change policy
                self.FSMmode = FSMMode.CHANGE  # 标记进入切换状态
                self.cur_policy.exit()
                self.get_next_policy(nextPolicyName)
                print("Switched to ", self.cur_policy.name_str)
        
        elif(self.FSMmode == FSMMode.CHANGE):
            self.cur_policy.enter()
            self.FSMmode = FSMMode.NORMAL
            self.cur_policy.run()
            
        # self.absoluteWait(self.cur_policy.control_horzion,self.start_time)
        end_time = time.time()
        # print("time cusume: ", end_time - start_time)

    # 等待函数，确保控制周期的定时
    def absoluteWait(self, control_dt, start_time):
        end_time = time.time()
        delta_time = end_time - start_time
        if(delta_time < control_dt):
            time.sleep(control_dt - delta_time)
        else:
            print("inference time beyond control horzion!!!")
            
            
    def get_next_policy(self, policy_name:FSMStateName):
        if(policy_name == FSMStateName.PASSIVE):
            self.cur_policy = self.passive_mode
        elif((policy_name == FSMStateName.FIXEDPOSE)):
            self.cur_policy = self.fixed_pose_1
        elif((policy_name == FSMStateName.LOCOMODE)):
            self.cur_policy = self.loco_policy
        elif((policy_name == FSMStateName.SKILL_COOLDOWN)):
            self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_CAST)):
            self.cur_policy = self.skill_cast_policy

        # 起身
        elif((policy_name == FSMStateName.GET_UPMODE)):
            self.cur_policy = self.getup_policy
        elif(policy_name == FSMStateName.GET_UP_BACK_MODE):
            self.cur_policy = self.getup_back_policy

        # 爆发动作
        elif((policy_name == FSMStateName.SKILL_Kongfan)): 
            self.cur_policy = self.Kongfan_policy

        # 舞蹈
        elif((policy_name == FSMStateName.SKILL_guofucheng_dance)): 
            self.cur_policy = self.guofucheng_dance_policy
        # elif((policy_name == FSMStateName.SKILL_Penguin_dance1)): 
        #     self.cur_policy = self.Penguin_dance1_policy
        elif((policy_name == FSMStateName.SKILL_Penguin_dance2)): 
            self.cur_policy = self.Penguin_dance2_policy
        elif((policy_name == FSMStateName.SKILL_Penguin_dance4)): 
            self.cur_policy = self.Penguin_dance4_policy
        elif((policy_name == FSMStateName.SKILL_45s_dance)): 
            self.cur_policy = self.guma_45s_dance_policy
        elif((policy_name == FSMStateName.SKILL_dahuajiao)): 
            self.cur_policy = self.dahuajiao_policy
        elif((policy_name == FSMStateName.SKILL_taiji)): 
            self.cur_policy = self.guma_taiji_policy
        elif((policy_name == FSMStateName.SKILL_Roundhouse_kick)): 
            self.cur_policy = self.Roundhouse_kick_policy
        elif((policy_name == FSMStateName.SKILL_dance2)): 
            self.cur_policy = self.guma_dance2_policy
        elif((policy_name == FSMStateName.SKILL_zoo_dance)): 
            self.cur_policy = self.zoo_dance_policy
        elif((policy_name == FSMStateName.SKILL_xinglian_dance1)): 
            self.cur_policy = self.xinglian_dance1_policy
        elif((policy_name == FSMStateName.SKILL_loco_cooldown)):
            self.cur_policy = self.loco_cooldown_policy
        # elif((policy_name == FSMStateName.SKILL_pick_up_box)):
        #     self.cur_policy = self.pick_up_box_policy
        elif((policy_name == FSMStateName.SKILL_jiangnanstyle)):
            self.cur_policy = self.jiangnanstyle_policy
        elif((policy_name == FSMStateName.SKILL_gaigeshunfeng)):
            self.cur_policy = self.gaigeshunfeng_policy
        else:
            pass