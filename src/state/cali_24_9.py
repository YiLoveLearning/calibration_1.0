# -*- coding: UTF-8 -*-
"""
模型初始状态确定, 并行版本(四条异步, 所以暂时没有改成服用sumo模拟脚本而是自己写了个(到时候可以先测试sumo并行和mp并行的速度和精度))
改自state_cali_24_7.py, 适用于24_7后的新.sumo模型

[ ] TODO: 调什么减小误差
"""

import datetime
import math
import pandas as pd
import numpy as np
import shutil
import multiprocessing as mp
import os
import time
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import sys
import copy
sys.path.append('..')
sys.path.append('D:/cmm/DT_Self_Calibration/src')
import dynamita.scheduler as ds
import dynamita.tool as dtool
from utils.yaml_cfg import load_yaml
from sumo_sim.module_AAO_SingleSimulate import AAO_Singlemulation

WATER_LINE = 4

class State_Identify:
    def __init__(self, model, 
                 init_state_file_format: str,  # format str, e.g. 'OfflineStates/RandState{}.xml', 任意的初始状态
                 freq: int=120):
        """
        parameter:
            model - 模拟的模型文件List[str]
            initstate - 要生成的初始状态文件
            ct - 要确定初始状态的时刻
            freq - 运行频率
        """
        # 设置模拟的模型文件
        self.model = model
        # 设置随机状态文件的路径
        self.sub_randstate = init_state_file_format
        # scada系统的运行频率
        self.frequency = freq

        # 这个变量用于决定是否更新状态，作为调节初始状态MLSS和TN值的参照基准
        # 将每个并行单元MLSS和TN的实际仪表值从SUMO的输出设置中导出
        self.MLSS = ["Sumo__Plant__aao_cstr7_1_1__XTSS",
                     "Sumo__Plant__aao_cstr7_1_2__XTSS",
                     "Sumo__Plant__aao_cstr7_2_1__XTSS",
                     "Sumo__Plant__aao_cstr7_2_2__XTSS"]
        self.is_save = True

        self.variables=[[    # 常量不可修改
                    'Sumo__Plant__aao_effluent_1_1__SNHx',     # 出水NH4
                    'Sumo__Plant__aao_effluent_1_1__TP',       # 出水TP
                    'Sumo__Plant__aao_effluent_1_1__XTSS',     # 出水SS
                    'Sumo__Plant__aao_effluent_1_1__TCOD',     # 出水TCOD
                    'Sumo__Plant__aao_effluent_1_1__TN',
                    'Sumo__Plant__aao_cstr7_1_1__XTSS',      # MLSS
                    'Sumo__Plant__aao_cstr7_1_1__XVSS',      # MVSS
            ],[
                    'Sumo__Plant__aao_effluent_1_2__SNHx',
                    'Sumo__Plant__aao_effluent_1_2__TP',
                    'Sumo__Plant__aao_effluent_1_2__XTSS',
                    'Sumo__Plant__aao_effluent_1_2__TCOD',
                    'Sumo__Plant__aao_effluent_1_2__TN',
                    'Sumo__Plant__aao_cstr7_1_2__XTSS',
                    'Sumo__Plant__aao_cstr7_1_2__XVSS',
            ],[
                    'Sumo__Plant__aao_effluent_2_1__SNHx',
                    'Sumo__Plant__aao_effluent_2_1__TP',
                    'Sumo__Plant__aao_effluent_2_1__XTSS',
                    'Sumo__Plant__aao_effluent_2_1__TCOD',
                    'Sumo__Plant__aao_effluent_2_1__TN',
                    'Sumo__Plant__aao_cstr7_2_1__XTSS',
                    'Sumo__Plant__aao_cstr7_2_1__XVSS',
            ],[
                    'Sumo__Plant__aao_effluent_2_2__SNHx',
                    'Sumo__Plant__aao_effluent_2_2__TP',
                    'Sumo__Plant__aao_effluent_2_2__XTSS',
                    'Sumo__Plant__aao_effluent_2_2__TCOD',
                    'Sumo__Plant__aao_effluent_2_2__TN',
                    'Sumo__Plant__aao_cstr7_2_2__XTSS',
                    'Sumo__Plant__aao_cstr7_2_2__XVSS',
            ]]
        
        #调用讯模
        self.sumo = AAO_Singlemulation("cali",model,True,self.frequency,None,False)

        self.map_table = {
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__SNHx':'Sumo__Plant__aao_effluent__SNHx' for i in range(1,WATER_LINE+1)},
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__TP':'Sumo__Plant__aao_effluent__TP' for i in range(1,WATER_LINE+1)},
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__XTSS':'Sumo__Plant__aao_effluent__XTSS' for i in range(1,WATER_LINE+1)},
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__TCOD':'Sumo__Plant__aao_effluent__TCOD' for i in range(1,WATER_LINE+1)},
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__TN':'Sumo__Plant__aao_effluent__TN' for i in range(1,WATER_LINE+1)},
        }

        for j in range(1, WATER_LINE+1):   # 分支号
            # [ ] TODO: 所有溶氧都要校准？
            self.variables[j-1] = self.variables[j-1]+[
                f'Sumo__Plant__aao_effluent_{int((j+1)//2)}_{int(2-(j&1))}__Q',
                f'Sumo__Plant__aao_cstr7_{int((j+1)//2)}_{int(2-(j&1))}__SO2',
                *[f'Sumo__Plant__aao_cstr{i}_{int((j+1)//2)}_{int(2-(j&1))}__XTSS' for i in range(3, 8)],
                *[f'Sumo__Plant__aao_cstr{i}_{int((j+1)//2)}_{int(2-(j&1))}__SCCOD' for i in range(3, 8)]]

        self.balance_point = [None]*WATER_LINE

    
    def bisection(self, xl, xr, func, x_tol, y_tol):
        """
        function:
            二分法求根函数(假设func是单调的, 根在xl和xr之间). 求 x 使 -y_tol<func(x)<y_tol 或 x收敛到x_tol
        parameter:
            xl - 根的取值范围的左边界
            xr - 根的取值范围的右边界
            func - 函数表达式(回调函数)
            x_tol - 收敛的x的容差
            y_tol - 收敛的y的容差
        """
        count = -5
        while True:
            xm = (xl+xr)/2
            yl, ym = func(xl), func(xm)
            if yl*ym < 0:   # 根在lm间
                xr = xm
            elif yl*ym > 0:
                xl = xm
            try:
                if abs(xm-xm_last) < x_tol or abs(ym) < y_tol:  # x收敛或y和0偏差小于容差
                    break
            except:
                pass
            finally:
                xm_last = xm
            count += 0.5
        return xm

    def run_Sim_2(self, n, task_index, flow3_qpump=None, inf_tn=None):
        # ysq: 不是随便什么文件取个数据输入就行了, __main__指定了用4.8的数据, 你这里用的8.27的数据, model_cleaning_data是utils/inf_cvt.py生成的, 可以学着转换一个4.8的数据
        cleaning_data = pd.read_excel("../../data/OfflineData/2024-08-27-2024-09-23_model_cleaning_data.xlsx", 
                                  parse_dates=['sampling_time']).iloc[0].to_dict()
        inf_args = cleaning_data
        state_list = [
        '../../data/OfflineStates/22.1.0dll_24_10/state1_init.xml',
        '../../data/OfflineStates/22.1.0dll_24_10/state2_init.xml',
        '../../data/OfflineStates/22.1.0dll_24_10/state3_init.xml',
        '../../data/OfflineStates/22.1.0dll_24_10/state4_init.xml',
        ]
        

        cmdline = [[
            f"set Sumo__Plant__aao_flowdivider4_1_1_sludge__param__Qpumped_target {flow3_qpump if flow3_qpump else inf_args['aao_flowdivider4_1_1_sludge_q']*86.4}",
            f"set Sumo__Plant__aao_influent_1_1__param__TKN {inf_tn if inf_tn else inf_args['aao_influent_1_1_tn']}",
            # f"set Sumo__Plant__shx_mbr1_influent__param__TCOD {inf_cod if inf_cod else inf_args['shx_mbr1_influent_tcod']}",
            f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
            f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
        ],[
            f"set Sumo__Plant__aao_flowdivider4_1_2_sludge__param__Qpumped_target {flow3_qpump if flow3_qpump else inf_args['aao_flowdivider4_1_2_sludge_q']*86.4}",
            f"set Sumo__Plant__aao_influent_1_2__param__TKN {inf_tn if inf_tn else inf_args['aao_influent_1_2_tn']}",
            # f"set Sumo__Plant__shx_mbr1_influent__param__TCOD {inf_cod if inf_cod else inf_args['shx_mbr1_influent_tcod']}",
            f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
            f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
        ],[
            f"set Sumo__Plant__aao_flowdivider4_2_1_sludge__param__Qpumped_target {flow3_qpump if flow3_qpump else inf_args['aao_flowdivider4_2_1_sludge_q']*86.4}",
            f"set Sumo__Plant__aao_influent_2_1__param__TKN {inf_tn if inf_tn else inf_args['aao_influent_2_1_tn']}",
            # f"set Sumo__Plant__shx_mbr1_influent__param__TCOD {inf_cod if inf_cod else inf_args['shx_mbr1_influent_tcod']}",
            f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
            f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
        ],[
            f"set Sumo__Plant__aao_flowdivider4_2_2_sludge__param__Qpumped_target {flow3_qpump if flow3_qpump else inf_args['aao_flowdivider4_2_2_sludge_q']*86.4}",
            f"set Sumo__Plant__aao_influent_2_2__param__TKN {inf_tn if inf_tn else inf_args['aao_influent_2_2_tn']}",
            # f"set Sumo__Plant__shx_mbr1_influent__param__TCOD {inf_cod if inf_cod else inf_args['shx_mbr1_influent_tcod']}",
            f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
            f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
        ]]
        res = self.sumo.run_sim(state_file=state_list, influent_data=cleaning_data, cmdline=cmdline, water_line_no=task_index)
        return res

    def find_Balance(self, task_index, inf_tn=None):
        """
        找到mlss的平衡点,即让mlss不变的排泥量
        运行10天的动态模拟,观察MLSS模拟值的趋势,通过调节排泥量的常态数据输入值使MLSS模拟值趋近于MLSS实测初始值
        更大的排泥量将导致mlss减小, 更小的排泥量将导致mlss增加
        """
        def diff(x):
            """排泥量为x时, mlss 10d减少的量(减少为0则找到平衡点)"""
            # 拷贝随机的状态文件
            shutil.copy(src=self.sub_randstate.format(task_index+1), 
                        dst=self.sub_out_state.format(str(task_index+1)))
            
            mlss_last = self.run_Sim_2(n=10, flow3_qpump=x, task_index=task_index, inf_tn=inf_tn)[self.MLSS[task_index]]
            mlss_now = self.run_Sim_2(n=0.1, flow3_qpump=x, task_index=task_index, inf_tn=inf_tn)[self.MLSS[task_index]]
            return mlss_now - mlss_last

        # 使用二分法找到污泥的平衡点
        balance_point = self.bisection(0, 5000, diff, 1, 10e-4)

        # 恢复到原始状态
        shutil.copy(src=self.sub_randstate.format(task_index+1), 
                    dst=self.sub_out_state.format(str(task_index+1)))

        return balance_point


    def set_Mlss(self, goal_mlss, task_index, inf_tn, all_balance_point):
        """
        function:
            确定MLSS的初始状态,使得mlss达到给定的目标值
            根据MLSS模拟值的趋势调整排泥量使模拟值呈平行于时间轴的一条直线
            继续增加动态模拟的时间并微调改变后的排泥量，当满足
            1、使得模拟10天的污泥浓度改变小于10mg/L,此时可以说明MLSS模拟值基本持平
            2、模拟值与实测值的差距绝对值也小于10mg/L,此时可以说明模拟值与实测值基本一致
        parameter:
            goal_mlss - 目标mlss数值
        """
        # 先模拟5个点, 跳过模拟前段可能出现的突变点(一般在前三个点)
        res = self.run_Sim_2(n=2*12, task_index=task_index, inf_tn=inf_tn)
        mlss = res[self.MLSS[task_index]]

        MLSS_draw = []
        MLSS_draw.append(mlss)
        default_k = 20
        iter_time = 0
        iter_type = 0
        while abs(goal_mlss - float(mlss)) > 10:    # 差值超过10m3/d
            iter_time += 1
            if iter_time > 10:
                if abs(iter_type) < 2:   # 振荡了
                    default_k -= 5
                elif abs(iter_type) > 8:    # 收敛了
                    default_k += 5
                iter_time = 0
                iter_type = 0
            if float(mlss) - goal_mlss  < -10:
                iter_type += 1
                count = -5
                if self.balance_point[task_index]==None:
                    flow3_qpump = max(0, all_balance_point[task_index]-default_k*abs(goal_mlss - float(mlss)))
                else:
                    flow3_qpump = max(0, self.balance_point[task_index]-default_k*abs(goal_mlss - float(mlss)))
                # 设置排泥量为0, 让模拟mlss快速上升
                res = self.run_Sim_2(n=10, flow3_qpump=flow3_qpump, task_index=task_index, inf_tn=inf_tn)
                mlss = res[self.MLSS[task_index]]
                MLSS_draw.append(mlss)
                logger.info(f"count={count}, {task_index+1}号AAO的mlss={mlss}")
                count += 0.5
                progress = 0.33/(1+math.exp(-1*count)) + 0.33   # [ ] TODO: 目前的进度条很糟糕, 未来可改成基于精度的
                plt.plot(range(len(MLSS_draw)),MLSS_draw,'-', color='b', alpha=0.8, linewidth=1, label='current mlss')
                plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
                plt.legend(loc="best") # upper right
                plt.title("MLSS iter")
                plt.savefig(f'../state/mlss/mlss_output_{task_index}.png')
            elif float(mlss) - goal_mlss > 10:  # 目标mlss小于模拟mlss超过200mg/L
                iter_type -= 1
                count = -5
                if self.balance_point[task_index]==None:
                    flow3_qpump = all_balance_point[task_index]+default_k*abs(goal_mlss - float(mlss))
                else:
                    flow3_qpump = self.balance_point[task_index]+default_k*abs(goal_mlss - float(mlss))
                res = self.run_Sim_2(n=10, flow3_qpump=flow3_qpump, task_index=task_index, inf_tn=inf_tn)
                mlss = res[self.MLSS[task_index]]
                MLSS_draw.append(mlss)
                logger.info(f"count={count}, {task_index+1}号AAO的mlss={mlss}")
                count += 0.5
                progress = (0.66-0.33)/(1+math.exp(-1*count)) + 0.33
                plt.plot(range(len(MLSS_draw)),MLSS_draw,'-', color='b', alpha=0.8, linewidth=1, label='current mlss')
                plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
                plt.legend(loc="best") # upper right
                plt.title("MLSS iter")
                plt.savefig(f'../state/mlss/mlss_output_{task_index}.png')
            else:  # 目标mlss和模拟mlss偏差不超过100mg/L
                break

        plt.plot(range(len(MLSS_draw)),MLSS_draw,'-', color='b', alpha=0.8, linewidth=1, label='current mlss')
        plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
        plt.legend(loc="best") # upper right
        plt.title("MLSS iter")
        plt.savefig(f'mlss_output_{task_index}.png')
        return res  # 返回该池子出水值


    # [ ] TODO: 未拓展WATER_LINE
    def effluent_tn(self, aao1_output, aao2_output, aao3_output, aao4_output):
        eff_Q1 = aao1_output['Sumo__Plant__aao_effluent_1_1__Q']
        eff_Q2 = aao2_output['Sumo__Plant__aao_effluent_1_2__Q']
        eff_Q3 = aao3_output['Sumo__Plant__aao_effluent_2_1__Q']
        eff_Q4 = aao4_output['Sumo__Plant__aao_effluent_2_2__Q']
        all_Q = eff_Q1+eff_Q2+eff_Q3+eff_Q4
        # 假设相同的key是汇合到一个effluent的
        same_keys = set(aao1_output.keys()) & set(aao2_output.keys()) & set(aao3_output.keys()) & set(aao4_output.keys())
        all_keys = set(aao1_output.keys()) | set(aao2_output.keys()) | set(aao3_output.keys()) | set(aao4_output.keys())
        ret = {k:0.0 for k in all_keys}
        logger.debug(f'{same_keys=}')
        for k, v in aao1_output.items():
            if k in same_keys:
                ret[k] += v * eff_Q1 / all_Q
                # ret[k] += v * Influ_Q1 / 2 / (Influ_Q1 + Influ_Q2)
            else:
                ret[k] = v
        for k, v in aao2_output.items():
            if k in same_keys:
                ret[k] += v * eff_Q2 / all_Q
                # ret[k] += v * Influ_Q1 / 2 / (Influ_Q1 + Influ_Q2)
            else:
                ret[k] = v
        for k, v in aao3_output.items():
            if k in same_keys:
                ret[k] += v * eff_Q3 / all_Q
                # ret[k] += v * Influ_Q2 / 2 / (Influ_Q1 + Influ_Q2)
            else:
                ret[k] = v
        for k, v in aao4_output.items():
            if k in same_keys:
                ret[k] += v * eff_Q4 / all_Q
                # ret[k] += v * Influ_Q2 / 2 / (Influ_Q1 + Influ_Q2)
            else:
                ret[k] = v
        return ret
    
    def set_Eff_Tn(self, inf_args, goal_eff_tn, eff_tn, balance_point, inf_tn):
        # 去掉task_index，直接校准总出水
        """
        function:
            进行总氮初始状态的确定,使得出水TN达到给定的目标值
            根据实际出水总氮和模拟值的差距适当调整模型常态输入进水的总氮值:
                若实际值-模拟值为正则,增加常态输入的总氮值;
                若实际值-模拟值为负则,减少常态输入的总氮值。
            直至总氮在经过10天,需满足
                1、动态模拟后值的变化小于等于0.1mg/L,此时可以说明总氮模拟值基本持平
                2、并且模拟值与实测初始值之差绝对值小于等于0.1mg/L即可视为总氮初始状态调节完成。
        parameter:
            goal_eff_tn - 目标出水TN值
            res_list - 获取模拟出水TN值
            flow3_qpump - 排泥量
        """
        
        
        name = f"Sumo__Plant__aao_effluent__TN"
        eff_mlss= ['' for i in range(WATER_LINE)]
        if abs(goal_eff_tn-float(eff_tn)) > float(0.1):
            self.is_save = False

            def diff(x):   
                res_list = [{} for _ in range(WATER_LINE)] 
                for task_index in range(WATER_LINE):
                    res = self.run_Sim_2(n=24, inf_tn=x, flow3_qpump=balance_point[task_index],task_index=task_index)
                    res_list[task_index]=res
                eff_tn = self.effluent_tn(*res_list)[name]
                return float(eff_tn) - goal_eff_tn

            inf_tn = self.bisection(0, 50, diff, 0.1, 10e-2, start_progress=0.66, reach_progress=0.98)

            self.is_save = True
            res_list = [{} for _ in range(WATER_LINE)] 
            for task_index in range(WATER_LINE):
                res = self.run_Sim_2(n=24, inf_tn=inf_tn, flow3_qpump=balance_point[task_index],task_index=task_index)
                res_list[task_index]=res
                eff_mlss[task_index] = res[self.MLSS[task_index]]
            eff_tn = self.effluent_tn(*res_list)[name]
            logger.info(f'校准TN后的结果为: ')
            logger.info(f"res={res}, 进水tn: inf_tn={inf_tn}, 出水tn: Eff_TN={eff_tn}")
            logger.info(f"mlss: mlss_7_1_1={eff_mlss[0]},mlss_7_1_2={eff_mlss[1]},mlss_7_2_1={eff_mlss[2]},mlss_7_2_2={eff_mlss[3]}")
        return inf_tn, eff_tn, eff_mlss

    def data_Callback(self, job, data):
        """
        function:
            数据回调函数
        """
        jobData = ds.sumo.getJobData(job)
        jobData["data"] = data  

    def msg_Callback(self, i):
        def msg_Callback_i(job, msg):
            """消息回调函数"""
            if self.is_save:
            # print(f'#{job} {msg}')  # 打印jobID和消息
                if ds.sumo.isSimFinishedMsg(msg):
                    ds.sumo.sendCommand(job, f"save {self.sub_out_state.format(str(i+1))}")
                if msg.startswith('530045'):
                    ds.sumo.finish(job)
            else:
                if ds.sumo.isSimFinishedMsg(msg):
                    ds.sumo.finish(job)
            logger.debug(f"job={job} : {msg}")
        return msg_Callback_i

    # ysq: 没用的删掉
    def run_Sim(self, n, task_index, flow3_qpump=None, inf_tn=None, inf_cod=None):
        """
        function:
            运行模拟函数
        parameter:
            n - 表示n次单频率运行
            flow3_qpump - 表示调整的排泥量
        """
        pos = None
        same_pos = ['Sumo__Plant__param__Sumo2__YOHO_SB_anox',
            'Sumo__Plant__param__Sumo2__YOHO_SB_ox',
            'Sumo__Plant__param__Sumo2__muAOB',
            'Sumo__Plant__param__Sumo2__bAOB',
            'Sumo__Plant__param__Sumo2__KNHx_AOB_AS',
            'Sumo__Plant__param__Sumo2__KO2_AOB_AS']
        alpha_commands = [[f'set Sumo__Plant__aao_cstr{i}_{int((j+1)//2)}_{int(2-(j&1))}__param__alpha 0.9'
                           for i in range(3,8)] for j in range(1, WATER_LINE+1)]
        if pos==None:   
            pos = {
                'Sumo__Plant__param__Sumo2__YOHO_SB_anox': 0.6,
                'Sumo__Plant__param__Sumo2__YOHO_SB_ox': 0.7,
                'Sumo__Plant__param__Sumo2__muAOB':1,    # 硝化菌最大比生长率
                'Sumo__Plant__param__Sumo2__bAOB': 0.16,
                'Sumo__Plant__param__Sumo2__KNHx_AOB_AS': 0.45,
                'Sumo__Plant__param__Sumo2__KO2_AOB_AS': 0.23,
                'Sumo__Plant__PAC1__param__G': 100,
                'Sumo__Plant__PAC__param__G': 120,
                'Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base': 0.97,
            }

        # 2. 设置sumo运行参数
        # 设置sumo并行模式
        ds.sumo.setParallelJobs(1)
        # 绑定消息回调函数
        ds.sumo.message_callback = self.msg_Callback(task_index)
        # 绑定数据回调函数
        ds.sumo.datacomm_callback = self.data_Callback
       
        # 4. 设定时间段内运行sumo进行模拟, 获取输出和loss

        inf_args = self.inf_args
        opt_args = self.inf_args

        # 模型输入变量
        Influ_Q1 = inf_args['influent_q1_handled'] * 24  # 进水流量, 采集数据(m3/h)到sumo(m3/d)的单位转换
        Influ_Q2 = inf_args['influent_q2_handled'] * 24  # 进水流量
        Influ_TCOD          = inf_args['influent_tcod_handled']  # 进水总COD
        Influ_TKN           = inf_args['influent_tkn_handled']  # 进水总氮
        Influ_frSNHx_TKN    = inf_args['influent_frsnhx_tkn_handled']  # 进水氨氮/进水总氮
        Influ_TP            = inf_args['influent_tp_handled']  # 进水总磷
        Influ_pH            = inf_args['influent_ph_handled']  # 进水PH
        Influ_T             = inf_args['influent_t']  # 进水温度
        # [ ]TODO: 碳源和PAC都没有数据
        # Carbon1_Q   = opt_args['carbon1_1'] * 0.024  # 1-1#AAO 碳源投加量, 采集数据(L/h)到sumo(m3/d)的单位转换
        # Carbon2_Q   = opt_args['carbon1_2'] * 0.024  # 1-2#AAO 碳源投加量
        # Carbon3_Q   = opt_args['carbon2_1'] * 0.024  # 1-1#AAO 碳源投加量
        # Carbon4_Q   = opt_args['carbon2_2'] * 0.024  # 1-2#AAO 碳源投加量
        # PAC1_Q      = opt_args['pac1'] * 0.024  # 1#加药间 PAC 投加量
        # PAC2_Q      = opt_args['pac2'] * 0.024  # 2#加药间 PAC 投加量
        # PAC3_Q      = opt_args['pac3'] * 0.024  # 2#加药间 PAC 投加量

        '''十个好氧池的曝气参数''' 
        # 采集数据(m3/h)到sumo(m3/d)的单位转换(再分到两根支管CSTR1_4_Qair和CSTR1_5_Qair)
        # 注意2-1曝气测量有问题'airflow1_2_2'为0
        CSTR1_4_Qair = opt_args['airflow1_1_1'] * 12 * 2    # 新数据都已经拆分过了
        CSTR1_5_Qair = opt_args['airflow1_1_1'] * 12 * 2
        CSTR1_6_Qair = opt_args['airflow1_1_2'] * 12 * 2
        CSTR1_7_Qair = opt_args['airflow1_1_2'] * 12 * 2
        CSTR1_8_Qair = opt_args['airflow1_1_3'] * 24
        CSTR2_4_Qair = opt_args['airflow1_2_1'] * 12 * 2
        CSTR2_5_Qair = opt_args['airflow1_2_1'] * 12 * 2
        CSTR2_6_Qair = opt_args['airflow1_2_2'] * 12 * 2
        CSTR2_7_Qair = opt_args['airflow1_2_2'] * 12 * 2
        CSTR2_8_Qair = opt_args['airflow1_2_3'] * 24
        CSTR3_4_Qair = opt_args['airflow2_1_1'] * 12 * 2
        CSTR3_5_Qair = opt_args['airflow2_1_1'] * 12 * 2
        CSTR3_6_Qair = opt_args['airflow2_1_2'] * 12 * 2
        CSTR3_7_Qair = opt_args['airflow2_1_2'] * 12 * 2
        CSTR3_8_Qair = opt_args['airflow2_1_3'] * 24
        CSTR4_4_Qair = opt_args['airflow2_2_1'] * 12 * 2
        CSTR4_5_Qair = opt_args['airflow2_2_1'] * 12 * 2
        CSTR4_6_Qair = opt_args['airflow2_2_2'] * 12 * 2
        CSTR4_7_Qair = opt_args['airflow2_2_2'] * 12 * 2
        CSTR4_8_Qair = opt_args['airflow2_2_3'] * 24

        Flow1_3_Qpump   = opt_args['internalreflux1_1'] * 24  # 1-1#AAO 内回流量
        Flow2_3_Qpump   = opt_args['internalreflux1_2'] * 24  # 1-2#AAO 内回流量
        Flow3_3_Qpump   = opt_args['internalreflux2_1'] * 24  # 2-1#AAO 内回流量
        Flow4_3_Qpump   = opt_args['internalreflux2_2'] * 24  # 2-2#AAO 内回流量
        Flow3_Qpump     = opt_args['excesssludge1'] * 24  # 1#二沉池剩余污泥量(排泥量)
        Flow6_Qpump     = opt_args['excesssludge2'] * 24  # 2#二沉池剩余污泥量
        Clarif1_Qs      = opt_args['clarifierunderflow1'] * 24  # 1#二沉池底流流量(排泥量+外回流量)
        Clarif2_Qs      = opt_args['clarifierunderflow2'] * 24  # 2#二沉池底流流量

        # 4.2. 设置sumo运行commands
        # 设置参数commands
        set_commands_same = [f'set {iname} {pos[iname]}' for iname in same_pos]
        pos_commands = [[
            *set_commands_same,
            # f'set Sumo__Plant__PAC1__param__G {pos["Sumo__Plant__PAC1__param__G"]}',  # PAC1的G值
            # f'set Sumo__Plant__PAC__param__G {pos["Sumo__Plant__PAC__param__G"]}',   # PAC的G值
            f'set Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base {pos["Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base"]}',    # 深度处理沉淀池无聚合物去除率
            f'set Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_polymer {pos["Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base"]}', # 深度处理沉淀池有聚合物去除率
        ],[
            *set_commands_same,
            # f'set Sumo__Plant__PAC2__param__G {pos["Sumo__Plant__PAC1__param__G"]}',  # PAC2的G值
            # f'set Sumo__Plant__Metal2__param__G {pos["Sumo__Plant__PAC__param__G"]}',    # PAC的G值
            f'set Sumo__Plant__aao_clarifier_4__param__fXTSS_sludge_base {pos["Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base"]}',    # 深度处理沉淀池无聚合物去除率
            f'set Sumo__Plant__aao_clarifier_4__param__fXTSS_sludge_polymer {pos["Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base"]}',     # 深度处理沉淀池有聚合物去除率
        ],[
            *set_commands_same,
            # f'set Sumo__Plant__PAC3__param__G {pos["Sumo__Plant__PAC1__param__G"]}',  # PAC3的G值
            # f'set Sumo__Plant__Metal2__param__G {pos["Sumo__Plant__PAC__param__G"]}',    # PAC的G值
            f'set Sumo__Plant__aao_clarifier_5__param__fXTSS_sludge_base {pos["Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base"]}',        # 深度处理沉淀池无聚合物去除率
            f'set Sumo__Plant__aao_clarifier_5__param__fXTSS_sludge_polymer {pos["Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base"]}',     # 深度处理沉淀池有聚合物去除率
        ],[
            *set_commands_same,
            # f'set Sumo__Plant__PAC4__param__G {pos["Sumo__Plant__PAC1__param__G"]}',  # PAC4的G值
            # f'set Sumo__Plant__Metal2__param__G {pos["Sumo__Plant__PAC__param__G"]}',    # PAC的G值
            f'set Sumo__Plant__aao_clarifier_6__param__fXTSS_sludge_base {pos["Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base"]}',        # 深度处理沉淀池无聚合物去除率
            f'set Sumo__Plant__aao_clarifier_6__param__fXTSS_sludge_polymer {pos["Sumo__Plant__aao_clarifier_3__param__fXTSS_sludge_base"]}',     # 深度处理沉淀池有聚合物去除率
        ]]
            
        end_commands = [
            'maptoic',  # 映射
            f'set Sumo__StopTime {int(n * self.frequency * dtool.minute)}',  # 模拟的时长
            f'set Sumo__DataComm {int(n * self.frequency * dtool.minute)}',  # 通讯的间隔
            'mode dynamic',  # 设置为动态模式
            'start'  # 开始模拟
        ]
        Qsludge_k=0.62
        # print(Influ_Q1/24, Influ_frSNHx_TKN)
        # 总commands设置
        commands=[
            [  # 要执行的sumo执行列表
                *alpha_commands[0],
                *pos_commands[0],
                # 设置其他进水参数
                f'set Sumo__Plant__aao_influent_1_1__param__Q {Influ_Q1 / 2}',
                f'set Sumo__Plant__aao_influent_1_1__param__TCOD {inf_cod if inf_cod!=None else Influ_TCOD}',
                f'set Sumo__Plant__aao_influent_1_1__param__TKN {inf_tn if inf_tn!=None else Influ_TKN}',
                f'set Sumo__Plant__aao_influent_1_1__param__frSNHx_TKN {Influ_frSNHx_TKN}',
                f'set Sumo__Plant__aao_influent_1_1__param__TP {Influ_TP}',
                #f'set Sumo__Plant__aao_influent_1_1__param__pH {Influ_pH}',
                f'set Sumo__Plant__aao_influent_1_1__param__T {Influ_T}',

                # f'set Sumo__Plant__Carbon1__param__Q {Carbon1_Q}',
                # f'set Sumo__Plant__PAC1__param__Q {PAC1_Q / 2}',
                # f'set Sumo__Plant__PAC__param__Q {PAC2_Q / (Influ_Q1 + Influ_Q2) * Influ_Q1 / 2}',

                f'set Sumo__Plant__aao_cstr3_1_1__param__Qair_NTP {CSTR1_4_Qair}',
                f'set Sumo__Plant__aao_cstr4_1_1__param__Qair_NTP {CSTR1_5_Qair}',
                f'set Sumo__Plant__aao_cstr5_1_1__param__Qair_NTP {CSTR1_6_Qair}',
                f'set Sumo__Plant__aao_cstr6_1_1__param__Qair_NTP {CSTR1_7_Qair}',
                f'set Sumo__Plant__aao_cstr7_1_1__param__Qair_NTP {CSTR1_8_Qair}',

                f'set Sumo__Plant__aao_flowdivider3_1_1_influx__param__Qpumped_target {Flow1_3_Qpump}',
                f'set Sumo__Plant__aao_flowdivider4_1_1_sludge__param__Qpumped_target {Qsludge_k*flow3_qpump/2 if flow3_qpump!=None else Qsludge_k*Flow3_Qpump/2}',
                f'set Sumo__Plant__aao_clarifier_1_1__param__Qsludge_target {Clarif1_Qs / 2}',
                *end_commands,
            ], [
                *alpha_commands[1],
                *pos_commands[1],
                # 设置其他进水参数
                f'set Sumo__Plant__aao_influent_1_2__param__Q {Influ_Q1 / 2}',
                f'set Sumo__Plant__aao_influent_1_2__param__TCOD {inf_cod if inf_cod!=None else Influ_TCOD}',
                f'set Sumo__Plant__aao_influent_1_2__param__TKN {inf_tn if inf_tn!=None else Influ_TKN}',
                f'set Sumo__Plant__aao_influent_1_2__param__frSNHx_TKN {Influ_frSNHx_TKN}',
                f'set Sumo__Plant__aao_influent_1_2__param__TP {Influ_TP}',
                #f'set Sumo__Plant__aao_influent_1_2__param__pH {Influ_pH}',
                f'set Sumo__Plant__aao_influent_1_2__param__T {Influ_T}',

                # f'set Sumo__Plant__Carbon1__param__Q {Carbon1_Q}',
                # f'set Sumo__Plant__PAC1__param__Q {PAC1_Q / 2}',
                # f'set Sumo__Plant__PAC__param__Q {PAC2_Q / (Influ_Q1 + Influ_Q2) * Influ_Q1 / 2}',

                f'set Sumo__Plant__aao_cstr3_1_2__param__Qair_NTP {CSTR2_4_Qair}',
                f'set Sumo__Plant__aao_cstr4_1_2__param__Qair_NTP {CSTR2_5_Qair}',
                f'set Sumo__Plant__aao_cstr5_1_2__param__Qair_NTP {CSTR2_6_Qair}',
                f'set Sumo__Plant__aao_cstr6_1_2__param__Qair_NTP {CSTR2_7_Qair}',
                f'set Sumo__Plant__aao_cstr7_1_2__param__Qair_NTP {CSTR2_8_Qair}',

                f'set Sumo__Plant__aao_flowdivider3_1_2_influx__param__Qpumped_target {Flow2_3_Qpump}',
                f'set Sumo__Plant__aao_flowdivider4_1_2_sludge__param__Qpumped_target {Qsludge_k*flow3_qpump/2 if flow3_qpump!=None else Qsludge_k*Flow3_Qpump/2}',
                f'set Sumo__Plant__aao_clarifier_1_2__param__Qsludge_target {Clarif1_Qs / 2}',
                # *alpha_commands[1],
                *end_commands,
            ], [
                *alpha_commands[2],
                *pos_commands[2],
                # 设置其他进水参数
                f'set Sumo__Plant__aao_influent_2_1__param__Q {Influ_Q2 / 2}',
                f'set Sumo__Plant__aao_influent_2_1__param__TCOD {inf_cod if inf_cod!=None else Influ_TCOD}',
                f'set Sumo__Plant__aao_influent_2_1__param__TKN {inf_tn if inf_tn!=None else Influ_TKN}',
                f'set Sumo__Plant__aao_influent_2_1__param__frSNHx_TKN {Influ_frSNHx_TKN}',
                f'set Sumo__Plant__aao_influent_2_1__param__TP {Influ_TP}',
                #f'set Sumo__Plant__aao_influent_2_1__param__pH {Influ_pH}',
                f'set Sumo__Plant__aao_influent_2_1__param__T {Influ_T}',

                # f'set Sumo__Plant__Carbon1__param__Q {Carbon1_Q}',
                # f'set Sumo__Plant__PAC1__param__Q {PAC1_Q / 2}',
                # f'set Sumo__Plant__PAC__param__Q {PAC2_Q / (Influ_Q1 + Influ_Q2) * Influ_Q1 / 2}',

                f'set Sumo__Plant__aao_cstr3_2_1__param__Qair_NTP {CSTR3_4_Qair}',
                f'set Sumo__Plant__aao_cstr4_2_1__param__Qair_NTP {CSTR3_5_Qair}',
                f'set Sumo__Plant__aao_cstr5_2_1__param__Qair_NTP {CSTR3_6_Qair}',
                f'set Sumo__Plant__aao_cstr6_2_1__param__Qair_NTP {CSTR3_7_Qair}',
                f'set Sumo__Plant__aao_cstr7_2_1__param__Qair_NTP {CSTR3_8_Qair}',

                f'set Sumo__Plant__aao_flowdivider3_2_1_influx__param__Qpumped_target {Flow3_3_Qpump}',
                f'set Sumo__Plant__aao_flowdivider4_2_1_sludge__param__Qpumped_target {flow3_qpump/2 if flow3_qpump!=None else Flow3_Qpump/2}',
                f'set Sumo__Plant__aao_clarifier_2_1__param__Qsludge_target {Clarif2_Qs / 2}',
                # *alpha_commands[2],
                *end_commands,
            ], [
                *alpha_commands[3],
                *pos_commands[3],
                # 设置其他进水参数
                f'set Sumo__Plant__aao_influent_2_2__param__Q {Influ_Q2 / 2}',
                f'set Sumo__Plant__aao_influent_2_2__param__TCOD {inf_cod if inf_cod!=None else Influ_TCOD}',
                f'set Sumo__Plant__aao_influent_2_2__param__TKN {inf_tn if inf_tn!=None else Influ_TKN}',
                f'set Sumo__Plant__aao_influent_2_2__param__frSNHx_TKN {Influ_frSNHx_TKN}',
                f'set Sumo__Plant__aao_influent_2_2__param__TP {Influ_TP}',
                #f'set Sumo__Plant__aao_influent_2_2__param__pH {Influ_pH}',
                f'set Sumo__Plant__aao_influent_2_2__param__T {Influ_T}',

                # f'set Sumo__Plant__Carbon1__param__Q {Carbon1_Q}',
                # f'set Sumo__Plant__PAC1__param__Q {PAC1_Q / 2}',
                # f'set Sumo__Plant__PAC__param__Q {PAC2_Q / (Influ_Q1 + Influ_Q2) * Influ_Q1 / 2}',

                f'set Sumo__Plant__aao_cstr3_2_2__param__Qair_NTP {CSTR4_4_Qair}',
                f'set Sumo__Plant__aao_cstr4_2_2__param__Qair_NTP {CSTR4_5_Qair}',
                f'set Sumo__Plant__aao_cstr5_2_2__param__Qair_NTP {CSTR4_6_Qair}',
                f'set Sumo__Plant__aao_cstr6_2_2__param__Qair_NTP {CSTR4_7_Qair}',
                f'set Sumo__Plant__aao_cstr7_2_2__param__Qair_NTP {CSTR4_8_Qair}',
                
                f'set Sumo__Plant__aao_flowdivider3_2_2_influx__param__Qpumped_target {Flow4_3_Qpump}',
                f'set Sumo__Plant__aao_flowdivider4_2_2_sludge__param__Qpumped_target {flow3_qpump/2 if flow3_qpump!=None else Flow3_Qpump/2}',
                f'set Sumo__Plant__aao_clarifier_2_2__param__Qsludge_target {Clarif2_Qs / 2}',
                *end_commands,
            ]
        ]
        

        # 4.3. 设置sumo schedule
        job = ds.sumo.schedule(
                model=self.model[task_index],  # 载入模型
                commands=[f'load {self.sub_out_state.format(str(task_index+1))}']+commands[task_index],
                variables=self.variables[task_index],
                jobData={
                    ds.sumo.persistent: True  # 保留历史数据
                }
            )
        
        # 4.4. 阻塞进程直到模拟结束
        while ds.sumo.scheduledJobs > 0:
            time.sleep(0.1)
        res = {}
        logger.debug(ds.sumo.jobData)
        for k in self.variables[task_index]:
            # 把出水要汇总的改成同名的
            nk = self.map_table.get(k, k)
            res[nk] = ds.sumo.jobData[job]["data"][k]
        ds.sumo.cleanup()  # 清除sumo的任务规划器
        time.sleep(0.2)

        return res
    
    def get_Initial_State(self, task_index, all_res, all_balance_point, inf_tn=None):
        """
        function:
            获取模型初始状态文件
        注意：
            总氮初始状态调整完后MLSS模拟值可能出现微小偏差以至超出规定的标准
            应该略微调整排泥量使得MLSS模拟值满足标准,此时初始状态的调整全部完成
        """
        trace_sumo = logger.add(f'../state/sumo_process.log', filter=lambda record: record['level'].name =='DEBUG')
        trace_cali = logger.add(f'../state/process.log', filter=lambda record: record['level'].name =='INFO')

        balance_point = self.find_Balance(task_index, inf_tn=None)
        self.balance_point[task_index] = balance_point
        logger.info(f"#1 寻找{task_index+1}号污泥平衡点")
        logger.info(f"{task_index+1}号污泥平衡点为: {balance_point}")
        logger.info(f"#2 设置{task_index+1}号mlss为目标值")
        # 使得mlss达到目标值
        # res :当mlss达到目标值时模拟的出水结果
        res = self.set_Mlss(self.aao_mlss[task_index], task_index, inf_tn=inf_tn, all_balance_point=all_balance_point)  
        eff_tn = res[f"Sumo__Plant__aao_effluent__TN"]
        mlss = res[self.MLSS[task_index]]
        logger.info(f"#2 {task_index+1}号mlss到达目标值, 此时{task_index+1}号出水TN={eff_tn},mlss={mlss}")
        all_res[task_index] = res
        all_balance_point[task_index] = balance_point
        

    def run(self, out_state_file_format: str, inf_args: dict, eff_dis_tn):
        """
        用进程池并行执行四个通道状态确定任务
        out_state_file_format: str, 存储校准后的状态文件
        inf_args: dict, 运行参数
        eff_dis_tn: 总出水目标值
        self.aao_mlss: 每条aao中mlss目标值
        """
        self.sub_out_state = out_state_file_format
        self.inf_args = inf_args
        self.eff_dis_tn = eff_dis_tn
        self.aao_mlss = [inf_args['aao1_1_mlss'], inf_args['aao1_2_mlss'],
                        inf_args['aao2_1_mlss'], inf_args['aao2_2_mlss']]
        logger.info("开始执行初始状态确定任务")
        logger.info(f"mlss1-1,1-2,2-1,2-2目标值: {self.aao_mlss[0]},{self.aao_mlss[1]},{self.aao_mlss[2]},{self.aao_mlss[3]}, 出水TN目标值: {self.eff_dis_tn}")
        logger.info(f"TN目标值: {self.eff_dis_tn}")

        # 用进程池并行执行四个通道状态确定任务
        # 暂存校准mlss后每条aao的出水结果值，以判断tn是否达到出水标准
        all_res = mp.Manager().list([{}, {}, {}, {}])
        all_balance_point = mp.Manager().list([0, 0, 0, 0])

        p1 = mp.Process(target=self.get_Initial_State,args=(0,all_res,all_balance_point))
        p2 = mp.Process(target=self.get_Initial_State,args=(1,all_res,all_balance_point))
        p3 = mp.Process(target=self.get_Initial_State,args=(2,all_res,all_balance_point))
        p4 = mp.Process(target=self.get_Initial_State,args=(3,all_res,all_balance_point))
        p1.start()
        p2.start()
        p3.start()
        p4.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        
        # pool = mp.Pool()
        # manager = mp.Manager()
        # locker = manager.Lock()
        # share_progress = manager.Value('test',0.0)

        # pool.map(self.get_Initial_State,range(4))、
        # TN_progress_list = mp.Array('f',[0.0,0.0,0.0,0.0],lock=True)
        name = f"Sumo__Plant__aao_effluent__TN"
        logger.info(f"#2 得到的平衡点值为: 1_1={all_balance_point[0]}, 1_2={all_balance_point[1]}, 2_1={all_balance_point[2]}, 2_2={all_balance_point[3]}")
        eff_tn = self.effluent_tn(*all_res)["Sumo__Plant__aao_effluent__TN"]
        logger.info(f"#2 分别校准各条廊道mlss后模拟值总出水TN={eff_tn}")
        # 校准总进水TN
        logger.info(f"#3 开始校准TN, 设置总出水TN为目标值, TN={self.eff_dis_tn}")
        logger.info(f"#3 校准mlss后模拟值出水TN={eff_tn}")
        inf_tn, eff_tn, eff_mlss = self.set_Eff_Tn(self.eff_dis_tn, eff_tn, all_balance_point, inf_tn=None) 
        logger.info(f"#3 校准后模拟值总出水TN={eff_tn},到达目标值,此时mlss 1-1,1-2,2-1,2-2值为: {eff_mlss[0]},{eff_mlss[1]},{eff_mlss[2]},{eff_mlss[3]}")
        
        # tn及mlss是否达到目标值设置flag标志
        # False：表示达到目标值，True即未达到
        diff_mlss = [0,0,0,0]
        for task_index in range(WATER_LINE):
            diff_mlss[task_index]=abs(eff_mlss[task_index]-self.aao_mlss[task_index])
        logger.info(f"差值为: diff_mlss={diff_mlss},diff_tn={abs(eff_tn-self.eff_dis_tn)}")
        
        while abs(eff_tn-self.eff_dis_tn)>0.1 or any(_ > 10 for _ in diff_mlss):
            logger.info(f"差值为: diff_mlss={diff_mlss},diff_tn={abs(eff_tn-self.eff_dis_tn)}")
            for task_index in range(WATER_LINE):
                if abs(eff_mlss[task_index]-self.aao_mlss[task_index])<=10:
                    logger.info(f"#4 {task_index+1}号mlss={eff_mlss[task_index]}达到目标值")
                    diff_mlss[task_index]=abs(eff_mlss[task_index]-self.aao_mlss[task_index])
                else:
                    logger.info(f"#4 {task_index+1}号mlss={eff_mlss[task_index]}未达到目标值")
                    logger.info(f"#4 此时输入校准后的进水TN: {inf_tn}, 重新校准{task_index+1}号mlss")
                    p = mp.Process(target=self.get_Initial_State,args=(task_index,all_res,all_balance_point,inf_tn))
                    p.start()
                    p.join()
                    logger.info(f"#4 {task_index+1}号mlss={all_res[task_index][self.MLSS[task_index]]}到达目标值")
                    eff_mlss[task_index] = all_res[task_index][self.MLSS[task_index]]
                    diff_mlss[task_index]=abs(eff_mlss[task_index]-self.aao_mlss[task_index])
                logger.info(f"#4 得到的平衡点值为: 1_1={all_balance_point[0]}, 1_2={all_balance_point[1]}, 2_1={all_balance_point[2]}, 2_2={all_balance_point[3]}")
            eff_tn = self.effluent_tn(*all_res)["Sumo__Plant__aao_effluent__TN"]
            # 判断出水TN是否达到目标值
            if abs(eff_tn-self.eff_dis_tn)<=0.1:
                logger.info(f"#4 总出水TN={eff_tn},到达目标值") 
            else:    
                # TN未达到目标值则重新校准TN
                logger.info(f"#4 总出水TN={eff_tn}未到达目标值重新校准TN")
                inf_tn, eff_tn, eff_mlss = self.set_Eff_Tn(self.eff_dis_tn, eff_tn, all_balance_point, inf_tn=None) 
                logger.info(f"#3 校准后模拟值总出水TN={eff_tn},到达目标值,进水tn={inf_tn}",
                    f"此时mlss 1-1,1-2,2-1,2-2值为: {eff_mlss[0]},{eff_mlss[1]},{eff_mlss[2]},{eff_mlss[3]}")
                for task_index in range(WATER_LINE):
                    diff_mlss[task_index]=abs(eff_mlss[task_index]-self.aao_mlss[task_index])
                logger.info(f"差值为: diff_mlss={diff_mlss},diff_tn={abs(eff_tn-self.eff_dis_tn)}")

        logger.info("#5 初始状态确定任务执行完毕")
        logger.info(f"#5 模拟值总出水TN={eff_tn}, mlss 1-1,1-2,2-1,2-2值为: {eff_mlss[0]},{eff_mlss[1]},{eff_mlss[2]},{eff_mlss[3]}")
        


if __name__ == "__main__":
    '''
    示例：
    init_state_file_format : 生成初始状态的状态文件
    init_time : 校准初始时间
    out_state_file_format : 校准后初始状态存放位置
    '''
    trace_sumo = logger.add(f'../state/sumo_process.log', filter=lambda record: record['level'].name =='DEBUG')
    trace_cali = logger.add(f'../state/process.log', filter=lambda record: record['level'].name =='INFO')
    model_list = [
        "D:/cmm/DT_Self_Calibration/data/OfflineSumomodels/22.1.0dll_24_8/sumoproject4.3.1.dll",
        "D:/cmm/DT_Self_Calibration/data/OfflineSumomodels/22.1.0dll_24_8/sumoproject4.3.2.dll",
        "D:/cmm/DT_Self_Calibration/data/OfflineSumomodels/22.1.0dll_24_8/sumoproject4.3.3.dll",
        "D:/cmm/DT_Self_Calibration/data/OfflineSumomodels/22.1.0dll_24_8/sumoproject4.3.4.dll",]
    init_state_file_format = 'D:/ysq/DT_Self_Calibration/data/OfflineStates/22.1.0dll_24_8/state{}_init.xml'
    st = State_Identify(model_list, init_state_file_format)
    clean_data = pd.read_excel("D:/cmm/DT_Self_Calibration/data/OfflineData/2024-03-09-2024-06-28_cleaning_data.xlsx", 
                parse_dates=['cleaning_time'])
    opt_data = pd.read_excel("D:/cmm/DT_Self_Calibration/data/OfflineData/2024-03-09-2024-06-28_operating_data.xlsx", 
                parse_dates=['sampling_time'])
    eff_data = pd.read_excel("D:/cmm/DT_Self_Calibration/data/OfflineData/2024-03-09-2024-06-28_effluent_data.xlsx", 
                parse_dates=['sampling_time'])
    inf_args = clean_data.iloc[0].to_dict()
    init_time = datetime.datetime(2024, 4, 8, 1)
    opt_data = opt_data[opt_data['sampling_time'] == init_time]
    logger.info(f'运行数据得到的值为:{opt_data}')
    inf_args.update(opt_data.iloc[0].to_dict())
    out_state_file_format = 'D:/ysq/DT_Self_Calibration/data/OfflineStates/22.1.0dll_24_tmp/init/state{}_init.xml'
    # 将实际出水总氮导入到出水总氮的时间图中作为初始状态总氮的参考
    eff_dis_tn = eff_data[eff_data['sampling_time'] == init_time]['effluent_dis1_tn'].iloc[0]
    st.run(out_state_file_format, inf_args, eff_dis_tn)