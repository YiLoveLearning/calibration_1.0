# -*- coding: UTF-8 -*-
"""
AAO模型初始状态确定, 并行版本(四条异步, 所以暂时没有改成复用sumo模拟脚本而是自己写了个, 
到时候sumo模拟脚本还要升级, 可能可以复用

TODO:
- [ ] 返回成功与否
- [ ] 场景三
    - [o] 结束标志
    - [o] 模拟最后一个点
    - [o] 1点
- [ ] 日志
    - [ ] 分成业务和算法的，算法是单独的，业务是公共的
    - [ ] 增加步骤（老算法可以不用改）
    - [ ] 文档说明__doc__
    - [o] 去掉重复的
"""

import datetime
import logging
import traceback
import math
import os
from multiprocessing import Process
import multiprocessing as mp
import pandas as pd
import shutil
import psutil
import time
from DigitalTwinApp.common import dbpkg
import DigitalTwinApp.common.dynamita.scheduler as ds
import DigitalTwinApp.common.dynamita.tool as dtool
from DigitalTwinApp.common.modules.module_CommonOperation import DBCommomOper
from DigitalTwinApp.common.modules.module_GlobalParam import AAO_StateIdentify_Progress as SP   # 进度条
from DigitalTwinApp.modules.aao.module_AAO_TaskFlag import AAO_Task_Flag
from DigitalTwinApp.modules.aao.module_AAO_SingleSimulate import AAO_Singlemulation

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

WATER_LINE = 4

_logger_instance = None

def setup_logger():
    global _logger_instance  # 引用全局变量
    if _logger_instance is not None:
        return _logger_instance  # 如果日志记录器已存在，直接返回
    logger_sim = logging.getLogger()
    logger_sim.setLevel(logging.INFO)
    
    handler = logging.FileHandler("aao_stateidentify_logging.log",encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(process)s - %(message)s')
    handler.setFormatter(formatter)
    logger_sim.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger_sim.addHandler(console_handler)
    _logger_instance = logger_sim

    return logger_sim

logger = setup_logger()

def kill_process_and_children(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()  # Terminate all child processes
        parent.terminate()  # Terminate the parent process
    except psutil.NoSuchProcess:
        pass

def run_with_timeout(timeout, func, *args):
    process = Process(target=func, args=(*args, ))
    process.start()
    process.join(timeout)  # 等待指定的超时时间

    if process.is_alive():
        kill_process_and_children(process.pid)  # 递归终止子进程
        process.terminate()  # 超时后终止子进程
        process.join()  # 确保进程被正确清理
        logger.info("Function timed out!")
        return False
    return True

class PiStateCali:
    def __init__(self,
        mlss_Kp: float, mlss_Ki: float, mlss_imax: float,
        tn_Kp: float, tn_Ki: float, tn_imax: float,
        model, freq: int = 120,
        delta_mlss=0.01, delta_tn=0.01, method=1,
    ):
        """
        parameter:
            mlss_Kp - MLSS的比例系数, 小于0
            mlss_Ki - MLSS的积分系数, 小于0
            mlss_imax - MLSS的积分抗饱和策略的参数
            tn_Kp - TN的比例系数, 大于0
            tn_Ki - TN的积分系数, 大于0
            tn_imax - TN的抗饱和积分策略的参数
            model - 模拟的模型文件List[str]
            freq - 运行频率
            delta_mlss - MLSS的目标值的容许误差, 相对误差
            delta_tn - TN的目标值的容许误差, 绝对误差
            method - 积分抗饱和策略
        """
        logger.info(f"{delta_mlss=}, {delta_tn=}, {mlss_Kp=}, {mlss_Ki=}, {mlss_imax=}, {tn_Kp=}, {tn_Ki=}, {tn_imax=}, {method=}")
        self.model = model
        # 设置随机状态文件的路径
        self.frequency = freq
        self.out_path = os.getcwd()
        self.mlss_Kp = mlss_Kp
        self.mlss_Ki = mlss_Ki
        self.mlss_imax = mlss_imax
        self.tn_Kp = tn_Kp
        self.tn_Ki = tn_Ki
        self.tn_imax = tn_imax
        self.method = method
        assert mlss_Kp < 0
        assert mlss_Ki < 0
        assert mlss_imax > 0
        assert tn_imax > 0

        # 将每个并行单元MLSS和TN的实际仪表值从SUMO的输出设置中导出
        self.MLSS = ["Sumo__Plant__aao_cstr7_1_1__XTSS", "Sumo__Plant__aao_cstr7_1_2__XTSS", 
                    "Sumo__Plant__aao_cstr7_2_1__XTSS", "Sumo__Plant__aao_cstr7_2_2__XTSS"]
        self.TN = "Sumo__Plant__aao_effluent__TN"

        self.delta_mlss = delta_mlss
        self.delta_tn = delta_tn
        model_param = DBCommomOper.get_Model_Param(run_type="manual",process_type='aao')
        self.sumo = AAO_Singlemulation(taskname='aao_stateidentify',model=model,state_flag=True,
                            freq=self.frequency, model_param=model_param, is_do=False)

    def run(self, in_state: str, out_state: str, inf_args: dict, goal_mlss, goal_tn):
        logger = setup_logger()
        self.sub_out_state = str(out_state)
        for i in range(WATER_LINE):
            shutil.copy(src=in_state.format(i + 1), dst=self.sub_out_state.format(i + 1))
        self.inf_args = inf_args

        logger.info("开始PID校准mlss和tn")
        logger.info(f"mlss目标值: {goal_mlss=}, tn目标值: {goal_tn=}")

        self.set_mlss_TN(goal_mlss=goal_mlss, goal_tn=goal_tn)
        SP.set(1)
        return True

    def run_Sim(self, state, n, qpump, inf_tn):
        logger.info(f"{qpump=}, {inf_tn=}")
        cmdline = [
            [
                f"set Sumo__Plant__aao_flowdivider4_1_1_sludge__param__Qpumped_target {qpump[0] if qpump[0] else self.inf_args['aao_flowdivider4_1_1_sludge_q']}",
                f"set Sumo__Plant__aao_influent_1_1__param__TKN {inf_tn if inf_tn else self.inf_args['aao_influent_1_1_tn']}",
                f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
                f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
            ],
            [
                f"set Sumo__Plant__aao_flowdivider4_1_2_sludge__param__Qpumped_target {qpump[1] if qpump[1] else self.inf_args['aao_flowdivider4_1_2_sludge_q']}",
                f"set Sumo__Plant__aao_influent_1_2__param__TKN {inf_tn if inf_tn else self.inf_args['aao_influent_1_2_tn']}",
                f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
                f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
            ],
            [
                f"set Sumo__Plant__aao_flowdivider4_2_1_sludge__param__Qpumped_target {qpump[2] if qpump[2] else self.inf_args['aao_flowdivider4_2_1_sludge_q']}",
                f"set Sumo__Plant__aao_influent_2_1__param__TKN {inf_tn if inf_tn else self.inf_args['aao_influent_2_1_tn']}",
                f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
                f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
            ],
            [
                f"set Sumo__Plant__aao_flowdivider4_2_2_sludge__param__Qpumped_target {qpump[3] if qpump[3] else self.inf_args['aao_flowdivider4_2_2_sludge_q']}",
                f"set Sumo__Plant__aao_influent_2_2__param__TKN {inf_tn if inf_tn else self.inf_args['aao_influent_2_2_tn']}",
                f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
                f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
            ],
        ]
        state_list = [state.format(i + 1) for i in range(WATER_LINE)]
        res = self.sumo.new_single_process_rum_sim(state_list, self.inf_args, cmdline=cmdline)
        return res

    def set_mlss_TN(self, goal_mlss: list, goal_tn: float):
        """
        校准MLSS和TN

        parameter:
            goal_mlss - 目标MLSS值
            goal_tn - 目标TN值
        """
        delta_tn = self.delta_tn
        delta_mlss = [self.delta_mlss * goal_mlss[i] for i in range(WATER_LINE)]

        draw_data = {"mlss": [[] for _ in range(WATER_LINE)], "qpump": [[] for _ in range(WATER_LINE)], "eff_tn": [], "inf_tn": [], "t": []}
        t = -1
        sum_delta_mlss = [0.0 for _ in range(WATER_LINE)]
        sum_delta_tn = 0.0
        inf_tn = None
        qpump = [None for _ in range(WATER_LINE)]
        method = self.method
        eff_tn = 0
        mlss = [0,0,0,0]
        while (
            t <= 0
            or abs(goal_tn - eff_tn) > delta_tn
            or not (draw_data["inf_tn"][-1] != 0 and abs(draw_data["inf_tn"][-1] - draw_data["inf_tn"][-2]) < 0.2)
            or any([abs(goal_mlss[i] - mlss[i]) > delta_mlss[i] for i in range(WATER_LINE)])
            or not all([draw_data["qpump"][i][-1] != 0.01 and abs(draw_data["qpump"][i][-1] - draw_data["qpump"][i][-2]) < 20 for i in range(WATER_LINE)])
        ):  # 未达目标范围
            t += 1
            res = self.run_Sim(state=self.sub_out_state, n=1, qpump=qpump, inf_tn=inf_tn)
            eff_tn = res[self.TN]
            mlss = [res[self.MLSS[i]] for i in range(WATER_LINE)]
            progress = math.exp(-(
                max(0, abs(eff_tn-goal_tn)-delta_tn)
                +sum([max(0, abs(mlss[i]-goal_mlss[i])-delta_mlss[i]) for i in range(4)]))/100)
            SP.set(progress)
            
            # 积分抗饱和控制策略
            if method == 1:
                # 策略1
                sum_delta_tn += goal_tn - eff_tn
                sum_delta_tn = min(self.tn_imax, max(-self.tn_imax, sum_delta_tn))
                inf_tn = max(0.01, self.tn_Kp * (goal_tn - eff_tn) + self.tn_Ki * sum_delta_tn)
            elif method == 2:
                # 策略2
                if abs(goal_tn - eff_tn)<self.tn_imax:  # 抗饱和: 积分分离
                    sum_delta_tn += goal_tn - eff_tn
                sum_delta_tn = min(self.tn_imax, max(-self.tn_imax, sum_delta_tn))
                inf_tn = max(0.01, self.tn_Kp * (goal_tn - eff_tn) + self.tn_Ki * sum_delta_tn)
            elif method == 3:
                sum_delta_tn += min(self.tn_imax, max(-self.tn_imax, goal_tn - eff_tn))
                inf_tn = max(0.01, self.tn_Kp * (goal_tn - eff_tn) + self.tn_Ki * sum_delta_tn)
            elif method == 4:
                inf_tn = max(0.01, self.tn_Kp * (goal_tn - eff_tn) + self.tn_Ki * sum_delta_tn)
                if not (inf_tn <=0.01 and (goal_tn - eff_tn)*self.tn_Ki<0):   # 控制量到0截断了, 如果积分还要往下走就不积了
                    sum_delta_tn += goal_tn - eff_tn
            elif method == 5:   # 1+4
                inf_tn = max(0.01, self.tn_Kp * (goal_tn - eff_tn) + self.tn_Ki * sum_delta_tn)
                if not (inf_tn <=0.01 and (goal_tn - eff_tn)*self.tn_Ki<0):   # 控制量到0截断了, 如果积分还要往下走就不积了
                    sum_delta_tn += goal_tn - eff_tn
                sum_delta_tn = min(self.tn_imax, max(-self.tn_imax, sum_delta_tn))

            draw_data["eff_tn"].append(eff_tn)
            draw_data["inf_tn"].append(inf_tn)
            draw_data["t"].append(t)
            if t % 20 == 0:
                logger.debug(f"{draw_data=}")
            # 画图
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            ax1.axhline(y=goal_tn, color="r", linestyle="-", alpha=1, linewidth=1, label=f"goal_tn {goal_tn:.2f}")
            (line1,) = ax1.plot(draw_data["t"], draw_data["eff_tn"], label="eff_tn")
            (line2,) = ax2.plot(draw_data["t"], draw_data["inf_tn"], label="inf_tn")
            (line3,) = ax3.plot(draw_data["inf_tn"], draw_data["eff_tn"])
            ax1.set_title("eff_tn-t")
            ax2.set_title("inf_tn-t")
            ax3.set_title("eff_tn-inf_tn")
            plt.close(fig)
            fig.savefig(os.path.join(self.out_path, "tn_output.png"))

            for i in range(WATER_LINE):
                if method == 1:
                    # 策略1
                    sum_delta_mlss[i] += goal_mlss[i]-mlss[i]
                    sum_delta_mlss[i] = min(self.mlss_imax, max(-self.mlss_imax, sum_delta_mlss[i]))
                    qpump[i] = max(0.01, self.mlss_Kp*(goal_mlss[i]-mlss[i])+self.mlss_Ki*sum_delta_mlss[i])
                elif method == 2:
                    # 策略2
                    if abs(goal_mlss[i] - mlss[i]) < self.mlss_imax:    # 抗饱和: 积分分离
                        sum_delta_mlss[i] += goal_mlss[i] - mlss[i]
                    sum_delta_mlss[i] = min(self.mlss_imax, max(-self.mlss_imax, sum_delta_mlss[i]))
                    qpump[i] = max(0.01, self.mlss_Kp*(goal_mlss[i]-mlss[i])+self.mlss_Ki*sum_delta_mlss[i])
                elif method == 3:
                    sum_delta_mlss[i] += min(self.mlss_imax, max(-self.mlss_imax, goal_mlss[i] - mlss[i]))
                    qpump[i] = max(0.01, self.mlss_Kp * (goal_mlss[i] - mlss[i]) + self.mlss_Ki * sum_delta_mlss[i])
                elif method == 4:
                    qpump[i] = max(0.01, self.mlss_Kp * (goal_mlss[i] - mlss[i]) + self.mlss_Ki * sum_delta_mlss[i])
                    if not (qpump[i] <= 0.01 and (goal_mlss[i] - mlss[i]) * self.mlss_Ki < 0):      # 控制量到0截断了, 如果积分还要往下走就不积了
                        sum_delta_mlss[i] += goal_mlss[i] - mlss[i]
                elif method == 5:   # 1+4
                    qpump[i] = max(0.01, self.mlss_Kp * (goal_mlss[i] - mlss[i]) + self.mlss_Ki * sum_delta_mlss[i])
                    if not (qpump[i] <= 0.01 and (goal_mlss[i] - mlss[i]) * self.mlss_Ki < 0):      # 控制量到0截断了, 如果积分还要往下走就不积了
                        sum_delta_mlss[i] += goal_mlss[i] - mlss[i]
                    sum_delta_mlss[i] = min(self.mlss_imax, max(-self.mlss_imax, sum_delta_mlss[i]))

                draw_data["mlss"][i].append(mlss[i])
                draw_data["qpump"][i].append(qpump[i])
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
                ax1.axhline(y=goal_mlss[i], color="r", linestyle="-", alpha=1, linewidth=1, label=f"goal_mlss{i} {goal_mlss[i]:.2f}")
                # ax2.axhline(y=balance_point, color='r', linestyle='-', alpha=1, linewidth=1, label=f'balance_point {balance_point:.2f}')
                (line1,) = ax1.plot(draw_data["t"], draw_data["mlss"][i], label="mlss")
                (line2,) = ax2.plot(draw_data["t"], draw_data["qpump"][i], label="qpump")
                (line3,) = ax3.plot(draw_data["qpump"][i], draw_data["mlss"][i])
                ax1.set_title("mlss-t")
                ax2.set_title("qpump-t")
                ax3.set_title("mlss-qpump")
                plt.close(fig)
                fig.savefig(os.path.join(self.out_path, f"mlss_output_{i}.png"))

        return res  # 返回该池子出水值

class HumanStateCali:
    def __init__(self, model, tf,
                 init_state_file_format: str,  # format str, e.g. 'OfflineStates/RandState{}.xml', 任意的初始状态
                 timeout:int=2400,   # 单位为秒
                 freq: int=120, abs_mlss=0.1, delta_tn=0.5):
        """
        parameter:
            model - 模拟的模型文件List[str]
            init_state_file_format - 待校准的初始状态文件
            ct - 要确定初始状态的时刻
            freq - 运行频率
        """
        logger = setup_logger()
        self.tf = tf
        self.frequency = freq
        self.timeout = timeout
        self.abs_mlss = abs_mlss
        self.delta_tn = delta_tn
        if timeout<=0:
            logger.error("timeout参数错误, 设置为默认值40min")
        if abs_mlss<=0 or delta_tn<=0:
            logger.error("abs_mlss<=0 or delta_tn<=0, 参数错误, 设置为默认值abs_mlss=0.1, delta_tn=0.5")
        # 设置模拟的模型文件 列表
        self.model = model
        # 设置随机状态文件的路径
        self.sub_randstate = init_state_file_format
        for i in range(WATER_LINE):
            if not os.path.exists(self.sub_randstate.format(i+1)):
                logger.error(f"状态文件{self.sub_randstate.format(i+1)}不存在")
                self.tf.set(-1)
                return
        # scada系统的运行频率
        # 设置四个MLSS名称
        self.MLSS = ["Sumo__Plant__aao_cstr7_1_1__XTSS",
                     "Sumo__Plant__aao_cstr7_1_2__XTSS",
                     "Sumo__Plant__aao_cstr7_2_1__XTSS",
                     "Sumo__Plant__aao_cstr7_2_2__XTSS"]
        # 这个变量用于决定是否更新状态
        self.is_save = True

        self.map_table = {
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__SNHx':'Sumo__Plant__aao_effluent__SNHx' for i in range(1,WATER_LINE+1)},
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__TP':'Sumo__Plant__aao_effluent__TP' for i in range(1,WATER_LINE+1)},
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__XTSS':'Sumo__Plant__aao_effluent__XTSS' for i in range(1,WATER_LINE+1)},
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__TCOD':'Sumo__Plant__aao_effluent__TCOD' for i in range(1,WATER_LINE+1)},
            **{f'Sumo__Plant__aao_effluent_{int((i+1)//2)}_{int(2-(i&1))}__TN':'Sumo__Plant__aao_effluent__TN' for i in range(1,WATER_LINE+1)},
        }

        self.balance_point = [None]*WATER_LINE
        self.aao_mlss = []

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
            reach_progress - 到达的进度值(这个主要是用来给进度条显示的)
        """
        #logger = setup_logger\(\)
        count = -5
        while True:
            xm = (xl+xr)/2
            yl, ym = func(xl), func(xm)
            logger.info(f"xl={xl}, xr={xr}, xm={xm}, yl={yl}, ym={ym}")
            if yl*ym < 0:
                xr = xm
            elif yl*ym > 0:
                xl = xm
            try:
                if abs(xm-xm_last) < x_tol or abs(ym) < y_tol:    # x收敛或y和0偏差小于容差
                    break
            except Exception as e:
                logger.info(repr(e))
            finally:
                xm_last = xm
            count += 0.5
        return xm

    def find_Balance(self, task_index, inf_tn=None):
        """找到mlss的平衡点
            让mlss不变的排泥量, 更大的排泥量将导致mlss减小, 更小的排泥量将导致mlss增加
        """
        tmp_state = os.path.join(os.getcwd(), "stateid_tmpstate{}.xml")
        shutil.copy(src=self.sub_out_state.format(task_index+1), 
                        dst=tmp_state.format(str(task_index+1)))
        def diff(x):
            """排泥量为x时, mlss10h减少的量(减少为0则找到平衡点)"""
            # 拷贝随机的状态文件
            shutil.copy(src=tmp_state.format(task_index+1), 
                        dst=self.sub_out_state.format(str(task_index+1)))

            # mlss_last = {}
            # mlss_now = {}
            # self.run_Sim(n=20, flow6_qpump=x, task_index=task_index, inf_tn=inf_tn, result_dic=mlss_last)
            # self.run_Sim(n=10, flow6_qpump=x, task_index=task_index, inf_tn=inf_tn, result_dic=mlss_now)
            mlss_last = self.new_single_process_rum_sim(n=10, flow6_qpump=x, task_index=task_index, inf_tn=inf_tn)
            mlss_now = self.new_single_process_rum_sim(n=1, flow6_qpump=x, task_index=task_index, inf_tn=inf_tn)
            
            
            return mlss_now[self.MLSS[task_index]] - mlss_last[self.MLSS[task_index]]

        # 使用二分法找到污泥的平衡点
        balance_point = self.bisection(0.1, 5000, diff, 1, 10e-3)
        # 恢复到原始状态
        shutil.copy(src=tmp_state.format(task_index+1), 
                    dst=self.sub_out_state.format(str(task_index+1)))

        return balance_point

    def set_Mlss(self, goal_mlss, task_index, inf_tn, all_balance_point):
        """
        function:
            使得mlss达到给定的目标值
        parameter:
            goal_mlss - 目标mlss数值
        """
        #logger = setup_logger\(\)
        # 先模拟5个点, 跳过模拟前段可能出现的突变点(一般在前三个点)
        # res = {}
        # self.run_Sim(n=2 * 12,task_index=task_index, inf_tn=inf_tn, result_dic=res)
        res = self.new_single_process_rum_sim(n=2 * 12,task_index=task_index, inf_tn=inf_tn)
        mlss = res[self.MLSS[task_index]]
        delta_mlss = self.abs_mlss*goal_mlss

        MLSS_draw = []
        MLSS_draw.append(mlss)
        default_k = 20
        iter_time = 0
        iter_type = 0
        while abs(goal_mlss - float(mlss)) > delta_mlss:    # 差值超过10m3/d
            iter_time += 1
            if iter_time > 10:
                if abs(iter_type) < 2:   # 振荡了
                    default_k -= 5
                elif abs(iter_type) > 8:    # 收敛了
                    default_k += 5
                iter_time = 0
                iter_type = 0
            if float(mlss) - goal_mlss  < -delta_mlss:
                iter_type += 1
                count = -5
                # 设置排泥量为0, 让模拟mlss快速上升
                if self.balance_point[task_index]==None:
                    flow3_qpump = max(0, all_balance_point[task_index]-default_k*abs(goal_mlss - float(mlss)))
                else:
                    flow3_qpump = max(0, self.balance_point[task_index]-default_k*abs(goal_mlss - float(mlss))) # [ ] TODO: default_k忘测了, 好尴尬
                # res = {}
                # self.run_Sim(n=10, flow6_qpump=flow3_qpump,task_index=task_index,inf_tn=inf_tn, result_dic=res)
                res = self.new_single_process_rum_sim(n=10, flow6_qpump=flow3_qpump,task_index=task_index,inf_tn=inf_tn)
                mlss = res[self.MLSS[task_index]]
                MLSS_draw.append(mlss)
                logger.debug(f"count={count}, {task_index+1}号AAO的mlss={mlss}")
                count += 0.5
                logger.info(f"{task_index}号 > 10 set_Mlss {goal_mlss=} {mlss=}  差值 { goal_mlss - float(mlss)}")
                # SP.set(min(share_progress))
                plt.plot(range(len(MLSS_draw)),MLSS_draw,'-', color='b', alpha=0.8, linewidth=1, label='current mlss')
                plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
                plt.legend(loc="best") # upper right
                plt.title(f"{task_index} MLSS iter")
                plt.savefig(f'mlss_output_{task_index}.png')
            elif float(mlss) - goal_mlss > delta_mlss:  # 目标mlss小于模拟mlss超过200mg/L
                iter_type -= 1
                count = -5
                # 循环模拟, 下降到与目标mlss相差不超过100mg/L的时候停止
                if self.balance_point[task_index]==None:
                    flow3_qpump = all_balance_point[task_index]+default_k*abs(goal_mlss - float(mlss))
                else:
                    flow3_qpump = self.balance_point[task_index]+default_k*abs(goal_mlss - float(mlss))
                # res = {}
                # self.run_Sim(n=10, flow6_qpump=flow3_qpump, task_index=task_index,inf_tn=inf_tn, result_dic=res)
                res = self.new_single_process_rum_sim(n=10, flow6_qpump=flow3_qpump, task_index=task_index,inf_tn=inf_tn)
                mlss = res[self.MLSS[task_index]]
                MLSS_draw.append(mlss)
                logger.info(f"count={count}, {task_index+1}号AAO的mlss={mlss}")
                count += 0.5
                logger.info(f"{task_index}号 < = 10 set_Mlss mlss - goal_mlss {float(mlss) - goal_mlss}")
                # SP.set(min(share_progress))
                plt.plot(range(len(MLSS_draw)),MLSS_draw,'-', color='b', alpha=0.8, linewidth=1, label='current mlss')
                plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
                plt.legend(loc="best") # upper right
                plt.title(f"{task_index} MLSS iter")
                plt.savefig(f'mlss_output_{task_index}.png')
            else:  # 目标mlss和模拟mlss偏差不超过100mg/L
                break
        
        plt.plot(range(len(MLSS_draw)),MLSS_draw,'-', color='b', alpha=0.8, linewidth=1, label='current mlss')
        plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
        plt.legend(loc="best") # upper right
        plt.title(f"{task_index} MLSS iter")
        plt.savefig(f'mlss_output_{task_index}.png')
        return res  # 返回出水结果


    # [ ] TODO: 未拓展WATER_LINE
    def effluent_tn(self, aao1_output, aao2_output, aao3_output, aao4_output):
        #logger = setup_logger\(\)
        #logger.info(f"#3 此时tn 1-1,1-2,2-1,2-2值为: {aao1_output[name]},{aao2_output[name]},{aao3_output[name]},{aao4_output[name]}")
        eff_Q1 = aao1_output['Sumo__Plant__aao_effluent_1_1__Q']
        eff_Q2 = aao2_output['Sumo__Plant__aao_effluent_1_2__Q']
        eff_Q3 = aao3_output['Sumo__Plant__aao_effluent_2_1__Q']
        eff_Q4 = aao4_output['Sumo__Plant__aao_effluent_2_2__Q']
        all_Q = eff_Q1+eff_Q2+eff_Q3+eff_Q4
        # 假设相同的key是汇合到一个effluent的
        same_keys = set(aao1_output.keys()) & set(aao2_output.keys()) & set(aao3_output.keys()) & set(aao4_output.keys())
        all_keys = set(aao1_output.keys()) | set(aao2_output.keys()) | set(aao3_output.keys()) | set(aao4_output.keys())
        ret = {k:0.0 for k in all_keys}
        logger.info(f'{same_keys=}')
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

    def set_Eff_Tn(self, goal_eff_tn, eff_tn, balance_point, inf_tn, eff_mlss):
        """
        function:
            使得出水TN达到给定的目标值
        parameter:
            goal_eff_tn - 目标出水TN值
            eff_tn - 当前模拟出水TN值
            flow3_qpump - 排泥量
        """
        #logger = setup_logger\(\)
        name = f"Sumo__Plant__aao_effluent__TN"
        eff_mlss= [0,0,0,0]
        if abs(goal_eff_tn-float(eff_tn)) > float(self.delta_tn):
            self.is_save = False

            tmp_state = os.path.join(os.getcwd(), "stateid_tmpstate{}.xml")
            for task_index in range(WATER_LINE):
                shutil.copy(src=self.sub_out_state.format(task_index+1), 
                                dst=tmp_state.format(str(task_index+1)))
            def diff(x):    # 让出水TN达到目标值
                res_list = [{} for _ in range(WATER_LINE)] 
                for task_index in range(WATER_LINE):
                    shutil.copy(src=tmp_state.format(task_index+1), 
                                dst=self.sub_out_state.format(str(task_index+1)))
                    # self.run_Sim(n=24, inf_tn=x, flow6_qpump=balance_point[task_index],task_index=task_index, result_dic=res_list[task_index])
                    res = self.new_single_process_rum_sim(n=24, inf_tn=x, flow6_qpump=balance_point[task_index],task_index=task_index)
                    res_list[task_index]=res
                eff_tn = self.effluent_tn(*res_list)[name]
                return float(eff_tn) - goal_eff_tn

            inf_tn = self.bisection(0.1, 100, diff, 0.1, 10e-2)
            for task_index in range(WATER_LINE):
                shutil.copy(src=tmp_state.format(task_index+1), 
                            dst=self.sub_out_state.format(str(task_index+1)))

            self.is_save = True
            res_list = [{} for _ in range(WATER_LINE)] 
            for task_index in range(WATER_LINE):
                # res = {}
                # self.run_Sim(n=24, inf_tn=inf_tn, flow6_qpump=balance_point[task_index],task_index=task_index, result_dic=res)
                res = self.new_single_process_rum_sim(n=24, inf_tn=inf_tn, flow6_qpump=balance_point[task_index],task_index=task_index)
                res_list[task_index]=res.copy()
                eff_mlss[task_index] = res[self.MLSS[task_index]]
            eff_tn = self.effluent_tn(*res_list)[name]
            # for i in range(4):
            #     share_progress[i] = 1.0
            # SP.set(1)
            logger.info("最终结果为: ")
            logger.info(f"res={res}, 进水tn: inf_tn={inf_tn}, 出水tn: Eff_TN={eff_tn}")
            logger.info(f"mlss: mlss7_1_1={eff_mlss[0]},mlss7_1_2={eff_mlss[1]},mlss7_2_1={eff_mlss[2]},mlss7_2_2={eff_mlss[3]}")
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
            # logger = setup_logger()
            # if job == 1:
            #     print(f'#{job} {msg}')
            if self.is_save:
                #logger.debug(f'#{job} {msg}')  # 打印jobID和消息
                if ds.sumo.isSimFinishedMsg(msg):
                    ds.sumo.sendCommand(job, f"save {self.sub_out_state.format(str(i+1))}")
                if msg.startswith('530045'):
                    ds.sumo.finish(job)
            else:
                if ds.sumo.isSimFinishedMsg(msg):
                    ds.sumo.finish(job)
            #logger.debug(f"job={job} : {msg}")
        return msg_Callback_i

    def run_Sim(self, n, task_index, flow6_qpump=None,inf_tn=None, inf_cod=None,result_dic=None):
        """
        function:
            运行模拟函数
        parameter:
            n - 表示n次单频率运行
            flow3_qpump - 表示调整的排泥量
        """
        # 任务并发数为1
        ds.sumo.setParallelJobs(1)
        # 绑定消息回调函数
        ds.sumo.message_callback = self.msg_Callback(task_index)
        # 绑定数据回调函数
        ds.sumo.datacomm_callback = self.data_Callback

        # 开启一个任务规划器
        job = ds.sumo.schedule(
            model=self.model[task_index],  # 载入模型
            commands=self.get_commands(n, task_index,flow_qpump_param=flow6_qpump,inf_tn_1=inf_tn, inf_cod_1=inf_cod),
            variables=self.get_variables(task_index),
            jobData={
                ds.sumo.persistent: True  # 保留历史数据
            }
        )
      
        # 阻塞进程直到模拟结束
        while ds.sumo.scheduledJobs > 0:
            time.sleep(0.1)

        res = {}
        logger.info(ds.sumo.jobData)
        for k in self.get_variables(task_index):
            # 把出水要汇总的改成同名的
            nk = self.map_table.get(k, k)
            res[nk] = ds.sumo.jobData[job]["data"][k]

        ds.sumo.cleanup()  # 清除任务规划器
        time.sleep(0.2)

        for key,value in res.items():
                result_dic[key] = value

    def new_single_process_rum_sim(self, n, task_index, flow6_qpump=None,inf_tn=None, inf_cod=None): 
        '''
            cmdline默认为空 即只进行输入固定值的模拟
            如果是优化调用的, 因为参数在不断变化, 为了使得单次模拟模块还能应用, 采用解引用的方法
        '''
        logger.info("new_single_process_rum_sim start") 
        result_dic = mp.Manager().dict()
        process = mp.Process(target=self.run_Sim, args=(n, task_index, flow6_qpump, inf_tn,inf_cod,result_dic))
        process.start()
        process.join()
        logger.info("new_single_process_rum_sim process.join()") 
        final_output = result_dic.copy()
        return final_output

    def get_Initial_State(self, task_index, all_res, all_balance_point, inf_tn=None):
        """
        function:
            获取模型初始状态文件
        """
        logger = setup_logger()
        logger.info(f"get_Initial_State")
        balance_point = self.find_Balance(task_index,inf_tn=inf_tn)
        self.balance_point[task_index] = balance_point
        logger.info(f"#1 寻找{task_index+1}号污泥平衡点")
        logger.info(f"{task_index+1}号污泥平衡点为: {balance_point}")
        logger.info(f"#2 设置{task_index+1}号mlss为目标值,此时进水tn={inf_tn}")
        # 使得mlss达到目标值
        # res :当mlss达到目标值时模拟的出水结果
        res = self.set_Mlss(self.aao_mlss[task_index], task_index, inf_tn=inf_tn, all_balance_point=all_balance_point)  
        eff_tn = res[f"Sumo__Plant__aao_effluent__TN"]
        mlss = res[self.MLSS[task_index]]
        logger.info(f"#2 {task_index+1}号mlss到达目标值, 此时出水TN={eff_tn},mlss={mlss}")
        all_res[task_index] = res
        all_balance_point[task_index] = balance_point
    
    def get_commands(self, n, commands_index, flow_qpump_param=None, inf_tn_1=None, inf_cod_1=None):
        # 模型参数未加入,一个是不确定名字,二个是现目前的模型校准参数也不一定符合应用要求
        start_commands = [f"load {self.sub_out_state.format(str(commands_index+1))}"]
        end_commands = [
            'maptoic',  # 映射
            f'set Sumo__StopTime {int(n * self.frequency * dtool.minute)}',  # 模拟的时长
            f'set Sumo__DataComm {int(n * self.frequency * dtool.minute)}',  # 通讯的间隔
            'mode dynamic',  # 设置为动态模式
            'start'  # 开始模拟
        ]
        logger.info(f"start_commands={start_commands}")
        aao_influent_1_1_q_cd = self.inf_args["aao_influent_1_1_q"]
        aao_influent_1_2_q_cd = self.inf_args["aao_influent_1_2_q"]
        aao_influent_2_1_q_cd = self.inf_args["aao_influent_2_1_q"]
        aao_influent_2_2_q_cd = self.inf_args["aao_influent_2_2_q"]
        aao_influent_1_1_tcod_cd = self.inf_args["aao_influent_1_1_tcod"]
        aao_influent_1_2_tcod_cd = self.inf_args["aao_influent_1_2_tcod"]
        aao_influent_1_1_tp_cd = self.inf_args["aao_influent_1_1_tp"]
        aao_influent_1_2_tp_cd = self.inf_args["aao_influent_1_2_tp"]
        aao_influent_1_1_tn_cd = self.inf_args["aao_influent_1_1_tn"]
        aao_influent_1_2_tn_cd = self.inf_args["aao_influent_1_2_tn"]
        aao_influent_1_1_frsnhx_tkn_cd = self.inf_args["aao_influent_1_1_frsnhx_tkn"]
        aao_influent_1_2_frsnhx_tkn_cd = self.inf_args["aao_influent_1_2_frsnhx_tkn"]
        aao_influent_1_1_t_cd = self.inf_args["aao_influent_1_1_t"]
        aao_influent_1_2_t_cd = self.inf_args["aao_influent_1_2_t"]
        aao_influent_2_1_tcod_cd = self.inf_args["aao_influent_2_1_tcod"]
        aao_influent_2_2_tcod_cd = self.inf_args["aao_influent_2_2_tcod"]
        aao_influent_2_1_tp_cd = self.inf_args["aao_influent_2_1_tp"]
        aao_influent_2_2_tp_cd = self.inf_args["aao_influent_2_2_tp"]
        aao_influent_2_1_tn_cd = self.inf_args["aao_influent_2_1_tn"]
        aao_influent_2_2_tn_cd = self.inf_args["aao_influent_2_2_tn"]
        aao_influent_2_1_frsnhx_tkn_cd = self.inf_args["aao_influent_2_1_frsnhx_tkn"]
        aao_influent_2_2_frsnhx_tkn_cd = self.inf_args["aao_influent_2_2_frsnhx_tkn"]
        aao_influent_2_1_t_cd = self.inf_args["aao_influent_2_1_t"]
        aao_influent_2_2_t_cd = self.inf_args["aao_influent_2_2_t"]
        
        aao_carbon_1_1_q_cd = self.inf_args["aao_carbon_1_1_q"]
        aao_carbon_1_2_q_cd = self.inf_args["aao_carbon_1_2_q"]
        aao_carbon_2_1_q_cd = self.inf_args["aao_carbon_2_1_q"]
        aao_carbon_2_2_q_cd = self.inf_args["aao_carbon_2_2_q"]
        aao_pac_1_1_q_cd = self.inf_args["aao_pac_1_1_q"]
        aao_pac_1_2_q_cd = self.inf_args["aao_pac_1_2_q"]
        aao_pac_3_q_cd = self.inf_args["aao_pac_3_q"]
        aao_pac_4_q_cd = self.inf_args["aao_pac_4_q"]
        aao_pac_5_q_cd = self.inf_args["aao_pac_5_q"]
        aao_pac_6_q_cd = self.inf_args["aao_pac_6_q"]
        aao_pac_2_1_q_cd = self.inf_args["aao_pac_2_1_q"]
        aao_pac_2_2_q_cd = self.inf_args["aao_pac_2_2_q"]
        aao_flowdivider3_1_1_influx_q_cd = self.inf_args["aao_flowdivider3_1_1_influx_q"]
        aao_flowdivider3_1_2_influx_q_cd = self.inf_args["aao_flowdivider3_1_2_influx_q"]
        aao_flowdivider3_2_1_influx_q_cd = self.inf_args["aao_flowdivider3_2_1_influx_q"]
        aao_flowdivider3_2_2_influx_q_cd = self.inf_args["aao_flowdivider3_2_2_influx_q"]
        aao_clarifier_1_1_sludge_target_q_cd = self.inf_args["aao_clarifier_1_1_sludge_target_q"]
        aao_clarifier_1_2_sludge_target_q_cd = self.inf_args["aao_clarifier_1_2_sludge_target_q"]
        aao_clarifier_2_1_sludge_target_q_cd = self.inf_args["aao_clarifier_2_1_sludge_target_q"]
        aao_clarifier_2_2_sludge_target_q_cd = self.inf_args["aao_clarifier_2_2_sludge_target_q"]
        aao_flowdivider4_1_1_sludge_q_cd = self.inf_args["aao_flowdivider4_1_1_sludge_q"]
        aao_flowdivider4_1_2_sludge_q_cd = self.inf_args["aao_flowdivider4_1_2_sludge_q"]
        aao_flowdivider4_2_1_sludge_q_cd = self.inf_args["aao_flowdivider4_2_1_sludge_q"]
        aao_flowdivider4_2_2_sludge_q_cd = self.inf_args["aao_flowdivider4_2_2_sludge_q"]
        aao_cstr3_1_1_qair_ntp_cd = self.inf_args["aao_cstr3_1_1_qair_ntp"]
        aao_cstr4_1_1_qair_ntp_cd = self.inf_args["aao_cstr4_1_1_qair_ntp"]
        aao_cstr5_1_1_qair_ntp_cd = self.inf_args["aao_cstr5_1_1_qair_ntp"]
        aao_cstr6_1_1_qair_ntp_cd = self.inf_args["aao_cstr6_1_1_qair_ntp"]
        aao_cstr7_1_1_qair_ntp_cd = self.inf_args["aao_cstr7_1_1_qair_ntp"]
        aao_cstr3_1_2_qair_ntp_cd = self.inf_args["aao_cstr3_1_2_qair_ntp"]
        aao_cstr4_1_2_qair_ntp_cd = self.inf_args["aao_cstr4_1_2_qair_ntp"]
        aao_cstr5_1_2_qair_ntp_cd = self.inf_args["aao_cstr5_1_2_qair_ntp"]
        aao_cstr6_1_2_qair_ntp_cd = self.inf_args["aao_cstr6_1_2_qair_ntp"]
        aao_cstr7_1_2_qair_ntp_cd = self.inf_args["aao_cstr7_1_2_qair_ntp"]
        aao_cstr3_2_1_qair_ntp_cd = self.inf_args["aao_cstr3_2_1_qair_ntp"]
        aao_cstr4_2_1_qair_ntp_cd = self.inf_args["aao_cstr4_2_1_qair_ntp"]
        aao_cstr5_2_1_qair_ntp_cd = self.inf_args["aao_cstr5_2_1_qair_ntp"]
        aao_cstr6_2_1_qair_ntp_cd = self.inf_args["aao_cstr6_2_1_qair_ntp"]
        aao_cstr7_2_1_qair_ntp_cd = self.inf_args["aao_cstr7_2_1_qair_ntp"]
        aao_cstr3_2_2_qair_ntp_cd = self.inf_args["aao_cstr3_2_2_qair_ntp"]
        aao_cstr4_2_2_qair_ntp_cd = self.inf_args["aao_cstr4_2_2_qair_ntp"]
        aao_cstr5_2_2_qair_ntp_cd = self.inf_args["aao_cstr5_2_2_qair_ntp"]
        aao_cstr6_2_2_qair_ntp_cd = self.inf_args["aao_cstr6_2_2_qair_ntp"]
        aao_cstr7_2_2_qair_ntp_cd = self.inf_args["aao_cstr7_2_2_qair_ntp"]
        commands = [
            [
                *start_commands,
                # 设置1-1进水参数
                f"set Sumo__Plant__aao_influent_1_1__param__Q {aao_influent_1_1_q_cd}",  # 基准模拟1#AAO生化池进水流量
                f"set Sumo__Plant__aao_influent_1_1__param__TCOD {inf_cod_1 if inf_cod_1!=None else aao_influent_1_1_tcod_cd}",  # 基准模拟1#AAO进水化学需氧量
                f"set Sumo__Plant__aao_influent_1_1__param__TP {aao_influent_1_1_tp_cd}",  # 基准模拟1#AAO进水总磷
                f"set Sumo__Plant__aao_influent_1_1__param__TKN {inf_tn_1 if inf_tn_1!=None else aao_influent_1_1_tn_cd}",  # 基准模拟1#AAO进水总氮
                f"set Sumo__Plant__aao_influent_1_1__param__frSNHx_TKN {aao_influent_1_1_frsnhx_tkn_cd}",  # 基准模拟1#AAO进水氨氮在总凯氏氮中的比例
                f"set Sumo__Plant__aao_influent_1_1__param__T {aao_influent_1_1_t_cd}",  # 基准模拟1#AAO进水温度
                # 设置1-1运行参数
                f"set Sumo__Plant__aao_carbon_1_1__param__Q {aao_carbon_1_1_q_cd}",  # 基准模拟1-1#AAO生化池碳源投加量
                f"set Sumo__Plant__aao_pac_1_1__param__Q {aao_pac_1_1_q_cd}",  # 基准模拟1#AAO聚合氯化铝投加量
                f"set Sumo__Plant__aao_pac_3__param__Q {aao_pac_3_q_cd}",  # 基准模拟2号加药间聚合氯化铝投加量
                f"set Sumo__Plant__aao_flowdivider3_1_1_influx__param__Qpumped_target {aao_flowdivider3_1_1_influx_q_cd}",  # 基准模拟1-1#AAO生化池内回流量
                f"set Sumo__Plant__aao_clarifier_1_1__param__Qsludge_target {aao_clarifier_1_1_sludge_target_q_cd}",  # 基准模拟1号二沉池底流流量        
                f"set Sumo__Plant__aao_flowdivider4_1_1_sludge__param__Qpumped_target {flow_qpump_param/2*24 if flow_qpump_param != None else aao_flowdivider4_1_1_sludge_q_cd}",  # 基准模拟1号二沉池剩余污泥量
                f"set Sumo__Plant__aao_cstr3_1_1__param__Qair_NTP {aao_cstr3_1_1_qair_ntp_cd}",  # 基准模拟1-1#AAO生化池好氧段1曝气量
                f"set Sumo__Plant__aao_cstr4_1_1__param__Qair_NTP {aao_cstr4_1_1_qair_ntp_cd}",  # 基准模拟1-1#AAO生化池好氧段2曝气量
                f"set Sumo__Plant__aao_cstr5_1_1__param__Qair_NTP {aao_cstr5_1_1_qair_ntp_cd}",  # 基准模拟1-1#AAO生化池好氧段3曝气量
                f"set Sumo__Plant__aao_cstr6_1_1__param__Qair_NTP {aao_cstr6_1_1_qair_ntp_cd}",  # 基准模拟1-1#AAO生化池好氧段4曝气量
                f"set Sumo__Plant__aao_cstr7_1_1__param__Qair_NTP {aao_cstr7_1_1_qair_ntp_cd}",  # 基准模拟1-1#AAO生化池好氧段5曝气量
                *end_commands,
            ], [
                *start_commands,
                # 设置1-2进水参数
                f"set Sumo__Plant__aao_influent_1_2__param__Q {aao_influent_1_2_q_cd}",  # 基准模拟1#AAO生化池进水流量
                f"set Sumo__Plant__aao_influent_1_2__param__TCOD {inf_cod_1 if inf_cod_1!=None else aao_influent_1_2_tcod_cd}",  # 基准模拟1#AAO进水化学需氧量
                f"set Sumo__Plant__aao_influent_1_2__param__TP {aao_influent_1_2_tp_cd}",  # 基准模拟1#AAO进水总磷
                f"set Sumo__Plant__aao_influent_1_2__param__TKN {inf_tn_1 if inf_tn_1!=None else aao_influent_1_2_tn_cd}",  # 基准模拟1#AAO进水总氮
                f"set Sumo__Plant__aao_influent_1_2__param__frSNHx_TKN {aao_influent_1_2_frsnhx_tkn_cd}",  # 基准模拟1#AAO进水氨氮在总凯氏氮中的比例
                f"set Sumo__Plant__aao_influent_1_2__param__T {aao_influent_1_2_t_cd}",  # 基准模拟1#AAO进水温度
                # 设置1-2运行参数
                f"set Sumo__Plant__aao_carbon_1_2__param__Q {aao_carbon_1_2_q_cd}",  # 基准模拟1-2#AAO生化池碳源投加量
                f"set Sumo__Plant__aao_pac_1_2__param__Q {aao_pac_1_2_q_cd}",  # 基准模拟1#AAO聚合氯化铝投加量
                f"set Sumo__Plant__aao_pac_4__param__Q {aao_pac_4_q_cd}",  # 基准模拟2号加药间聚合氯化铝投加量
                f"set Sumo__Plant__aao_flowdivider3_1_2_influx__param__Qpumped_target {aao_flowdivider3_1_2_influx_q_cd}",  # 基准模拟1-2#AAO生化池内回流量
                f"set Sumo__Plant__aao_clarifier_1_2__param__Qsludge_target {aao_clarifier_1_2_sludge_target_q_cd}",  # 基准模拟1号二沉池底流流量        
                f"set Sumo__Plant__aao_flowdivider4_1_2_sludge__param__Qpumped_target {flow_qpump_param/2*24 if flow_qpump_param != None else aao_flowdivider4_1_2_sludge_q_cd}",  # 基准模拟1号二沉池剩余污泥量
                f"set Sumo__Plant__aao_cstr3_1_2__param__Qair_NTP {aao_cstr3_1_2_qair_ntp_cd}",  # 基准模拟1-2#AAO生化池好氧段1曝气量
                f"set Sumo__Plant__aao_cstr4_1_2__param__Qair_NTP {aao_cstr4_1_2_qair_ntp_cd}",  # 基准模拟1-2#AAO生化池好氧段2曝气量
                f"set Sumo__Plant__aao_cstr5_1_2__param__Qair_NTP {aao_cstr5_1_2_qair_ntp_cd}",  # 基准模拟1-2#AAO生化池好氧段3曝气量
                f"set Sumo__Plant__aao_cstr6_1_2__param__Qair_NTP {aao_cstr6_1_2_qair_ntp_cd}",  # 基准模拟1-2#AAO生化池好氧段4曝气量
                f"set Sumo__Plant__aao_cstr7_1_2__param__Qair_NTP {aao_cstr7_1_2_qair_ntp_cd}",  # 基准模拟1-2#AAO生化池好氧段5曝气量
                *end_commands,
            ], [
                *start_commands,
                # 设置2-1进水参数
                f"set Sumo__Plant__aao_influent_2_1__param__Q {aao_influent_2_1_q_cd}",  # 基准模拟2#AAO生化池进水流量
                f"set Sumo__Plant__aao_influent_2_1__param__TCOD {inf_cod_1 if inf_cod_1!=None else aao_influent_2_1_tcod_cd}",  # 基准模拟2#AAO进水化学需氧量
                f"set Sumo__Plant__aao_influent_2_1__param__TP {aao_influent_2_1_tp_cd}",  # 基准模拟2#AAO进水总磷
                f"set Sumo__Plant__aao_influent_2_1__param__TKN {inf_tn_1 if inf_tn_1!=None else aao_influent_2_1_tn_cd}",  # 基准模拟2#AAO进水总氮
                f"set Sumo__Plant__aao_influent_2_1__param__frSNHx_TKN {aao_influent_2_1_frsnhx_tkn_cd}",  # 基准模拟2#AAO进水氨氮在总凯氏氮中的比例    
                f"set Sumo__Plant__aao_influent_2_1__param__T {aao_influent_2_1_t_cd}",  # 基准模拟2#AAO进水温度
                # 设置2-1运行参数
                f"set Sumo__Plant__aao_carbon_2_1__param__Q {aao_carbon_2_1_q_cd}",  # 基准模拟2-1#AAO生化池碳源投加量
                f"set Sumo__Plant__aao_pac_2_1__param__Q {aao_pac_2_1_q_cd}",  # 基准模拟2号加药间聚合氯化铝投加量
                f"set Sumo__Plant__aao_pac_5__param__Q {aao_pac_5_q_cd}",  # 基准模拟2#AAO聚合氯化铝投加量
                f"set Sumo__Plant__aao_flowdivider3_2_1_influx__param__Qpumped_target {aao_flowdivider3_2_1_influx_q_cd}",  # 基准模拟2-1#AAO生化池内回流量     
                f"set Sumo__Plant__aao_clarifier_2_1__param__Qsludge_target {aao_clarifier_2_1_sludge_target_q_cd}",  # 基准模拟2号二沉池底流流量
                f"set Sumo__Plant__aao_flowdivider4_2_1_sludge__param__Qpumped_target {flow_qpump_param/2*24 if flow_qpump_param != None else aao_flowdivider4_2_1_sludge_q_cd}",  # 基准模拟2号二沉池剩余污泥
                f"set Sumo__Plant__aao_cstr3_2_1__param__Qair_NTP {aao_cstr3_2_1_qair_ntp_cd}",  # 基准模拟2-1#AAO生化池好氧段1曝气量
                f"set Sumo__Plant__aao_cstr4_2_1__param__Qair_NTP {aao_cstr4_2_1_qair_ntp_cd}",  # 基准模拟2-1#AAO生化池好氧段2曝气量
                f"set Sumo__Plant__aao_cstr5_2_1__param__Qair_NTP {aao_cstr5_2_1_qair_ntp_cd}",  # 基准模拟2-1#AAO生化池好氧段3曝气量
                f"set Sumo__Plant__aao_cstr6_2_1__param__Qair_NTP {aao_cstr6_2_1_qair_ntp_cd}",  # 基准模拟2-1#AAO生化池好氧段4曝气量
                f"set Sumo__Plant__aao_cstr7_2_1__param__Qair_NTP {aao_cstr7_2_1_qair_ntp_cd}",  # 基准模拟2-1#AAO生化池好氧段5曝气量
                *end_commands,
            ], [
                *start_commands,
                # 设置2-2进水参数
                f"set Sumo__Plant__aao_influent_2_2__param__Q {aao_influent_2_2_q_cd}",  # 基准模拟2#AAO生化池进水流量
                f"set Sumo__Plant__aao_influent_2_2__param__TCOD {inf_cod_1 if inf_cod_1!=None else aao_influent_2_2_tcod_cd}",  # 基准模拟2#AAO进水化学需氧量
                f"set Sumo__Plant__aao_influent_2_2__param__TP {aao_influent_2_2_tp_cd}",  # 基准模拟2#AAO进水总磷
                f"set Sumo__Plant__aao_influent_2_2__param__TKN {inf_tn_1 if inf_tn_1!=None else aao_influent_2_2_tn_cd}",  # 基准模拟2#AAO进水总氮
                f"set Sumo__Plant__aao_influent_2_2__param__frSNHx_TKN {aao_influent_2_2_frsnhx_tkn_cd}",  # 基准模拟2#AAO进水氨氮在总凯氏氮中的比例    
                f"set Sumo__Plant__aao_influent_2_2__param__T {aao_influent_2_2_t_cd}",  # 基准模拟2#AAO进水温度
                # 设置2-2运行参数
                f"set Sumo__Plant__aao_carbon_2_2__param__Q {aao_carbon_2_2_q_cd}",  # 基准模拟2-2#AAO生化池碳源投加量
                f"set Sumo__Plant__aao_pac_2_2__param__Q {aao_pac_2_2_q_cd}",  # 基准模拟2号加药间聚合氯化铝投加量
                f"set Sumo__Plant__aao_pac_6__param__Q {aao_pac_6_q_cd}",  # 基准模拟2#AAO聚合氯化铝投加量
                f"set Sumo__Plant__aao_flowdivider3_2_2_influx__param__Qpumped_target {aao_flowdivider3_2_2_influx_q_cd}",  # 基准模拟2-2#AAO生化池内回流量    
                f"set Sumo__Plant__aao_clarifier_2_2__param__Qsludge_target {aao_clarifier_2_2_sludge_target_q_cd}",  # 基准模拟2号二沉池底流流量
                f"set Sumo__Plant__aao_flowdivider4_2_2_sludge__param__Qpumped_target {flow_qpump_param/2*24 if flow_qpump_param != None else aao_flowdivider4_2_2_sludge_q_cd}",  # 基准模拟2号二沉池剩余污泥量
                f"set Sumo__Plant__aao_cstr3_2_2__param__Qair_NTP {aao_cstr3_2_2_qair_ntp_cd}",  # 基准模拟2-2#AAO生化池好氧段1曝气量
                f"set Sumo__Plant__aao_cstr4_2_2__param__Qair_NTP {aao_cstr4_2_2_qair_ntp_cd}",  # 基准模拟2-2#AAO生化池好氧段2曝气量
                f"set Sumo__Plant__aao_cstr5_2_2__param__Qair_NTP {aao_cstr5_2_2_qair_ntp_cd}",  # 基准模拟2-2#AAO生化池好氧段3曝气量
                f"set Sumo__Plant__aao_cstr6_2_2__param__Qair_NTP {aao_cstr6_2_2_qair_ntp_cd}",  # 基准模拟2-2#AAO生化池好氧段4曝气量
                f"set Sumo__Plant__aao_cstr7_2_2__param__Qair_NTP {aao_cstr7_2_2_qair_ntp_cd}",  # 基准模拟2-2#AAO生化池好氧段5曝气量
                *end_commands,
            ]
        ]
        # logger.info(f'{commands[commands_index]}')
        return commands[commands_index]
           
    def get_variables(self,variables_index):
        variables =[
            [    # 常量不可修改
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
        for j in range(1, WATER_LINE+1):   # 分支号
            # [ ] TODO: 所有溶氧都要校准？
            variables[j-1] = variables[j-1]+[
            f'Sumo__Plant__aao_effluent_{int((j+1)//2)}_{int(2-(j&1))}__Q',
            f'Sumo__Plant__aao_cstr7_{int((j+1)//2)}_{int(2-(j&1))}__SO2',
            *[f'Sumo__Plant__aao_cstr{i}_{int((j+1)//2)}_{int(2-(j&1))}__XTSS' for i in range(3, 8)],
            *[f'Sumo__Plant__aao_cstr{i}_{int((j+1)//2)}_{int(2-(j&1))}__SCCOD' for i in range(3, 8)]]
   
        return variables[variables_index]

    def execute_task(self, out_state_file_format: str, inf_args, goal_mlss, goal_tn):
        """
        用进程池并行执行四个通道状态确定任务
        out_state_file_format: str, 存储校准后的状态文件
        inf_args: dict, 运行参数
        eff_dis_tn: 总出水目标值
        self.aao_mlss: 每条aao中mlss目标值
        """
        logger = setup_logger()

        task_ret = 0
        self.sub_out_state = out_state_file_format  # 临时的输出状态文件
        self.inf_args = inf_args
        self.aao_mlss = goal_mlss
        self.eff_dis_tn = goal_tn
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

        name = f"Sumo__Plant__aao_effluent__TN"
        logger.info(f"#2 得到的平衡点值为: 1_1={all_balance_point[0]}, 1_2={all_balance_point[1]}, 2_1={all_balance_point[2]}, 2_2={all_balance_point[3]}")
        eff_tn = self.effluent_tn(*all_res)[name]
        eff_mlss= ['' for i in range(WATER_LINE)]
        for task_index in range(WATER_LINE):
            eff_mlss[task_index] = all_res[task_index][self.MLSS[task_index]]
        logger.info(f"#2 分别校准各条廊道mlss后模拟值总出水TN={eff_tn}")
        logger.info(f"#2 mlss 1-1,1-2,2-1,2-2值为: {eff_mlss[0]},{eff_mlss[1]},{eff_mlss[2]},{eff_mlss[3]}")
        # 校准总进水TN
        logger.info(f"#3 开始校准TN, 设置总出水TN为目标值, TN={self.eff_dis_tn}")
        logger.info(f"#3 校准mlss后模拟值出水TN={eff_tn}")
        inf_tn, eff_tn, eff_mlss = self.set_Eff_Tn(self.eff_dis_tn, eff_tn, all_balance_point, inf_tn=None, eff_mlss=eff_mlss) 
        logger.info(f"#3 校准后模拟值总出水TN={eff_tn},到达目标值,此时mlss 1-1,1-2,2-1,2-2值为: {eff_mlss[0]},{eff_mlss[1]},{eff_mlss[2]},{eff_mlss[3]}")
        
        # tn及mlss是否达到目标值设置flag标志
        # False：表示达到目标值，True即未达到
        diff_mlss = [0,0,0,0]
        for task_index in range(WATER_LINE):
            diff_mlss[task_index]=abs(eff_mlss[task_index]-self.aao_mlss[task_index])/self.aao_mlss[task_index]
        logger.info(f"mlss相对误差值为: abs_mlss={diff_mlss},绝对误差delta_tn={abs(eff_tn-self.eff_dis_tn)}")
        while abs(eff_tn-self.eff_dis_tn)>self.delta_tn or any(_ > self.abs_mlss for _ in diff_mlss):
            progress = math.exp(-(
                max(0, abs(eff_tn-self.eff_dis_tn)-self.delta_tn)
                +sum([max(0, abs(dmlssi)-self.abs_mlss) for dmlssi in diff_mlss]))/100)
            SP.set(progress)
            logger.info(f"mlss相对误差值为: abs_mlss={diff_mlss},绝对误差delta_tn={abs(eff_tn-self.eff_dis_tn)}")
            for task_index in range(WATER_LINE):
                if abs(eff_mlss[task_index]-self.aao_mlss[task_index])/self.aao_mlss[task_index]<=self.abs_mlss:
                    logger.info(f"#4 {task_index+1}号mlss={eff_mlss[task_index]}达到目标值")
                    diff_mlss[task_index]=abs(eff_mlss[task_index]-self.aao_mlss[task_index])/self.aao_mlss[task_index]
                else:
                    logger.info(f"#4 {task_index+1}号mlss={eff_mlss[task_index]}未达到目标值")
                    logger.info(f"#4 此时输入校准后的进水TN: {inf_tn}, 重新校准{task_index+1}号mlss")
                    logger.info(f"#4 {task_index+1}号mlss={all_res[task_index][self.MLSS[task_index]]}到达目标值")
                    eff_mlss[task_index] = all_res[task_index][self.MLSS[task_index]]
                    diff_mlss[task_index]=abs(eff_mlss[task_index]-self.aao_mlss[task_index])/self.aao_mlss[task_index]
                    eff_tn = self.effluent_tn(*all_res)[name]
                logger.info(f"#4 得到的平衡点值为: 1_1={all_balance_point[0]}, 1_2={all_balance_point[1]}, 2_1={all_balance_point[2]}, 2_2={all_balance_point[3]}")
            # 判断出水TN是否达到目标值
            if abs(eff_tn-self.eff_dis_tn)<=self.delta_tn:
                logger.info(f"#4 总出水TN={eff_tn},到达目标值") 
            else:    
                # TN未达到目标值则重新校准TN
                logger.info(f"#4 总出水TN={eff_tn}未到达目标值重新校准TN")
                inf_tn, eff_tn, eff_mlss = self.set_Eff_Tn(self.eff_dis_tn, eff_tn, all_balance_point, inf_tn=None, eff_mlss=eff_mlss) 
                logger.info(f"#3 校准后模拟值总出水TN={eff_tn},到达目标值,进水tn={inf_tn},\
                            此时mlss 1-1,1-2,2-1,2-2值为: {eff_mlss[0]},{eff_mlss[1]},{eff_mlss[2]},{eff_mlss[3]}")
                for task_index in range(WATER_LINE):
                    diff_mlss[task_index]=abs(eff_mlss[task_index]-self.aao_mlss[task_index])/self.aao_mlss[task_index]
                logger.info(f"mlss相对误差值为: abs_mlss={diff_mlss},绝对误差delta_tn={abs(eff_tn-self.eff_dis_tn)}")
        
        logger.info("#5 初始状态确定任务执行完毕")
        logger.info(f"#5 模拟值总出水TN={eff_tn}, mlss 1-1,1-2,2-1,2-2值为: {eff_mlss[0]},{eff_mlss[1]},{eff_mlss[2]},{eff_mlss[3]}")

class AAO_State_Identify:
    """
    example:
    >>> init_state_file_format = 'OfflineStates/RandState{}.xml'    # 输入的初始状态文件
    >>> out_state_file_format = 'OfflineStates/state{}_init.xml'    # 输出的初始状态文件
    >>> SI_obj = AAO_State_Identify(model=model_file_list, init_state_file_format=randstate, timeout=timeout,freq=frequency, mode=mode)
    >>> SI_obj.do_state_identify(out_state_file_format= None, tag=tag,ct_start =start_time, ct_end=end_time)
    """
    def __init__(self, model,  
                 init_state_file_format: str,  # format str, e.g. 'OfflineStates/RandState{}.xml', 待校准状态量的初始状态
                 freq: int=120, timeout:int=2400,   # 单位为秒
                 delta_mlss=0.01, delta_tn=0.08, mode=0,    # mode=0为离线, mode=1为基准
                 cali_method = 1,   # 0为老方法, 1为新方法
                 # PI方法参数
                 mlss_Kp: float = -5, mlss_Ki: float = -1, mlss_imax: float = 4000,
                 tn_Kp: float = 1, tn_Ki: float = 0.3, tn_imax: float = 40/0.3,
                 a_saturation_method=5):
        """
        parameter:
            model - 模拟的模型文件List[str]
            init_state_file_format - 待校准的初始状态format str
            freq - 运行频率
            timeout - 超时时间
            delta_mlss - MLSS的目标值的容许误差, 相对误差
            delta_tn - TN的目标值的容许误差, 绝对误差
            mode - 模式, 0为离线, 1为基准
            cali_method - 校准方法, 0为老方法, 1为PI新方法
            mlss_Kp - MLSS的比例系数
            mlss_Ki - MLSS的积分系数
            mlss_imax - MLSS的积分抗饱和策略的参数
            tn_Kp - TN的比例系数
            tn_Ki - TN的积分系数
            tn_imax - TN的抗饱和积分策略的参数
            a_saturation_method - 积分抗饱和策略
        """
        logger = setup_logger()
        logger.info(f'============ input: model={model}, freq={freq}, timeout={timeout}, init_state_file_format={init_state_file_format}'
                    f'delta_mlss={delta_mlss}, delta_tn={delta_tn}, mode={mode}, cali_method={cali_method}, '
                    f'mlss_Kp={mlss_Kp}, mlss_Ki={mlss_Ki}, mlss_imax={mlss_imax}, tn_Kp={tn_Kp}, '
                    f'tn_Ki={tn_Ki}, tn_imax={tn_imax}, a_saturation_method={a_saturation_method}')
        # 任务标志位实例
        if mode==0: # 离线模式
            self.tf = AAO_Task_Flag(task_name="aao_offline_state_identify_ts")
        else:       # 基准模式
            self.tf = AAO_Task_Flag(task_name="aao_basic_state_identify_ts")
        if timeout<=0:
            logger.error("timeout参数错误, 设置为默认值40min")
            timeout = 2400
        if delta_mlss<=0 or delta_tn<=0:
            logger.error("delta_mlss<=0 or delta_tn<=0, 参数错误, 设置为默认值delta_mlss=0.01, delta_tn=0.08")
            delta_mlss = 0.01
            delta_tn = 0.08
        if mode not in [0, 1]:
            logger.error("mode参数错误, 设置为默认值0")
            mode = 0
        if cali_method not in [0, 1]:
            logger.error("cali_method参数错误, 设置为默认值1")
            cali_method = 1
        if a_saturation_method not in [1, 2, 3, 4, 5]:
            logger.error("a_saturation_method参数错误, 设置为默认值5")
            a_saturation_method = 5
        self.mode = mode
        self.model = model
        self.sub_randstate = init_state_file_format
        self.frequency = freq
        self.timeout = timeout
        self.delta_mlss = delta_mlss
        self.delta_tn = delta_tn
        self.cali_method = cali_method
        self.mlss_Kp = mlss_Kp
        self.mlss_Ki = mlss_Ki
        self.mlss_imax = mlss_imax
        self.tn_Kp = tn_Kp
        self.tn_Ki = tn_Ki
        self.tn_imax = tn_imax
        self.a_saturation_method = a_saturation_method

        # 方法类
        self.cali0 = HumanStateCali(model=model, tf=self.tf, init_state_file_format=init_state_file_format,
                 timeout=timeout, freq=freq, abs_mlss=delta_mlss, delta_tn=delta_tn)
        self.cali1 = PiStateCali(mlss_Kp, mlss_Ki, mlss_imax, tn_Kp, tn_Ki, tn_imax,
                        model, freq,
                        delta_mlss, delta_tn, method=a_saturation_method)

    def set_attribute(self, attr_name, value):
        """
        通用方法，用于设置 self 对象的属性。
        
        :param attr_name: 属性名 (str)
        :param value: 属性值
        """
        if isinstance(attr_name, str) and attr_name.isidentifier():
            setattr(self, attr_name, value)
        else:
            raise ValueError(f"'{attr_name}' 不是一个有效的属性名称")

    def do_state_identify(self, tag:str, ct_start: datetime.datetime, ct_end: datetime.datetime,
                out_state_file_format: str|None=None):
        """
        执行状态确定任务
        :param tag: str, 场景标签
        :param ct_start: datetime.datetime, 开始时间
        :param ct_end: datetime.datetime, 结束时间
        :param out_state_file_format: str, 输出的状态文件格式
        """
        logger = setup_logger()
        logger.info("开始执行状态确定任务")
        logger.info(f'============ do_state_identify input:  tag={tag}, ct_start={ct_start}, '
                    f'ct_end={ct_end}, out_state_file_format={out_state_file_format}')
        if ct_start >= ct_end:
            self.tf.set(-1)
            logger.error("开始时间大于结束时间")
            return -1
        for i in range(WATER_LINE):
            if not os.path.exists(self.sub_randstate.format(i+1)):
                logger.error(f"状态文件{self.sub_randstate.format(i+1)}不存在")
                self.tf.set(-1)
                return -1
        if out_state_file_format is None:
            if self.mode==0:    # 离线模式
                logger.info("离线模式")
                t = ct_end if ct_end is not None else ct_start
                out_state_file_format = os.path.join(os.getcwd(), t.strftime("DigitalTwinAPI\\AAO_Config\\OfflineStates\\state_%Y-%m-%d-%H-00-00_{}.xml"))
            elif self.mode==1:  # 基准模式
                logger.info("基准模式")
                out_state_file_format = os.path.join(os.getcwd(), "DigitalTwinAPI\\AAO_Config\\OnlineStates\\state{}.xml")
        try:
            # 设置标志位为"确定状态中"
            self.tf.set(2)
            # 获取最初状态
            SP.set(0)   # 进度为0
            logger.info("开始获取数据库数据")
            mean_inf, eff_dis_tn, inf_records=self.get_db_data_on_ct(ct_start, ct_end)
            if mean_inf is None or eff_dis_tn is None:
                logger.error("数据库中没有数据")
                self.tf.set(-2) # 数据缺失
                raise Exception("数据库查询失败")
            mean_inf = self.update_args(mean_inf)
            goal_mlss = [mean_inf["aao_cstr7_1_1_xtss"], mean_inf["aao_cstr7_1_2_xtss"],
                    mean_inf["aao_cstr7_2_1_xtss"], mean_inf["aao_cstr7_2_2_xtss"]]
            if tag =="scene_4":
                tmp_out_state = os.path.join(os.getcwd(), "DigitalTwinAPI\\AAO_Config\\OnlineStates\\aao_tmp_state{}.xml")
                if self.cali_method == 0:  # 老方法
                    if not run_with_timeout(self.timeout, self.cali0.execute_task, tmp_out_state, 
                            mean_inf, goal_mlss, eff_dis_tn):
                        self.tf.set(-1)
                    else:
                        # 设置标志位为"确定状态完成"
                        self.tf.set(1)
                        for i in range(WATER_LINE):
                            shutil.copy(src=tmp_out_state.format(i+1), dst=out_state_file_format.format(i+1))
                else:   # 新方法
                    if not run_with_timeout(self.timeout, self.cali1.run, self.sub_randstate, tmp_out_state, 
                            mean_inf, goal_mlss, eff_dis_tn):
                        self.tf.set(-1)
                    else:
                        # 设置标志位为"确定状态完成"
                        self.tf.set(1)
                        for i in range(WATER_LINE):
                            shutil.copy(src=tmp_out_state.format(i+1), dst=out_state_file_format.format(i+1))
            elif tag =="scene_3":
                tmp_states=[]
                tmp_state = os.path.join(os.getcwd(), "DigitalTwinAPI\\AAO_Config\\OnlineStates\\state{}.xml")
                for i in range(4):
                    tmp_states.append(tmp_state.format(i+1))
                if inf_records is None:
                    logger.error("数据库查询失败")
                    self.tf.set(-2)
                    raise Exception("数据库查询失败")
                inf_records=pd.DataFrame(inf_records)
                def save_last_run_data(count_time):
                    """
                    function:
                        保存每天1:00的运行状态
                    params:
                        count_time: 当前运行时间
                    """
                    tmp_dir="DigitalTwinAPI\\AAO_Config\\OfflineStates"

                    if (count_time.hour == 1):
                        for i in range(WATER_LINE):
                            count_state = os.path.join(os.getcwd(),tmp_dir, f"{count_time.strftime('state_%Y-%m-%d-%H-%M-%S_{}')}")
                            if not os.path.exists(count_state):
                                logger.info(f"scene_3保存1点状态文件{count_state.format(i+1)}.xml")
                                shutil.copy(src=tmp_state.format(i+1), dst=f"{count_state.format(i+1)}.xml")

                simulation = AAO_Singlemulation(taskname='aao_stateidentify',model=self.model,state_flag=True,freq=self.frequency
                                        ,model_param=None,is_do=False)

                for i in range(len(inf_records)):
                    # 运行SUMO模拟
                    args = self.update_args(inf_records.iloc[i])
                    tmp_data = simulation.new_single_process_rum_sim(state_file=tmp_states, input_data=args)
                    count_time = ct_start + datetime.timedelta(milliseconds=(i*self.frequency*dtool.minute))
                    # 如果是1点则是接下来那一天的初始状态, 保存下来以备后面使用, 离线模式
                    save_last_run_data(count_time)
                self.tf.set(1)

        except Exception as e:
            logger.error(traceback.format_exc())
            self.tf.set(-1)
        finally:
            time.sleep(10)
            SP.set(0)

    def get_db_data_on_ct(self, ct_start, ct_end=None):
        """
        function:
            获取清洗数据表中的选择时刻前两周的数据
        """
        # 连接数据库
        self.db = dbpkg.Database()
        if not self.db.connect_default():
            logger.error("数据库连接失败")
            return None, None, None
        
        if ct_end is None:
            inf_conditions = f"""cleaning_time_cd='{ct_start.strftime("%Y-%m-%d %H:%M:00")}'"""
            inf_record = self.db.query("cleaning_data_aao",  
                    fetch_type=dbpkg.DBBase.FETCH_ONE,  conditions=inf_conditions)[0]

            eff_conditions = f"""sampling_time_rd='{ct_start.strftime("%Y-%m-%d %H:%M:00")}'"""
            colum_infos = self.db.describe("realtime_data")
            selected_columns = [col[0] for col in colum_infos if col[0].startswith("aao")] #选择aao 或者选择mbr 工艺相关的列
            selected_columns.append("sampling_time_rd")
            selected_columns.append("ts")
            selected_columns.append("id")
            record = self.db.query("realtime_data", 
                    fetch_type=dbpkg.DBBase.FETCH_ONE, conditions=eff_conditions,select_cols=selected_columns)[0]
            # 将查找的出水数据结果赋值
            eff_dis_tn = record["aao_effluent_dis1_tn_rd"]
            # 关闭与数据库的连接
            inf_records = None

        else:
            inf_conditions = f"""cleaning_time_cd>='{ct_start.strftime("%Y-%m-%d %H:%M:00")}'
                AND cleaning_time_cd<='{ct_end.strftime("%Y-%m-%d %H:%M:00")}'
                AND TIMEDIFF('{ct_start.strftime("%Y-%m-%d %H:%M:00")}', cleaning_time_cd, 1m)%{self.frequency} = 0"""
            inf_records = self.db.query("cleaning_data_aao",
                    fetch_type=dbpkg.DBBase.FETCH_ALL,  conditions=inf_conditions)

            eff_conditions = f"""sampling_time_rd>='{ct_start.strftime("%Y-%m-%d %H:%M:00")}'
                AND sampling_time_rd<='{ct_end.strftime("%Y-%m-%d %H:%M:00")}'
                AND TIMEDIFF('{ct_start.strftime("%Y-%m-%d %H:%M:00")}', sampling_time_rd, 1m)%{self.frequency} = 0"""
            colum_infos = self.db.describe("realtime_data")
            selected_columns = [col[0] for col in colum_infos if col[0].startswith("aao")] #选择aao 或者选择mbr 工艺相关的列
            selected_columns.append("sampling_time_rd")
            selected_columns.append("ts")
            selected_columns.append("id")
            eff_records = self.db.query("realtime_data",
                                        fetch_type=dbpkg.DBBase.FETCH_ALL, conditions=eff_conditions,select_cols=selected_columns)
            # 关闭与数据库的连接
            inf_record = pd.DataFrame(inf_records).mean().to_dict()
            eff_dis_tn = pd.DataFrame(eff_records).mean().to_dict().get("aao_effluent_dis1_tn_rd", None)
            
        if inf_record is None or len(inf_record)==0:
            inf_record = None
        if inf_records is None or len(inf_records)==0:
            inf_records = None
        
        self.db.close()
        # inf_record = {key[:-3] if key.endswith('_cd') else key: value for key, value in inf_record.items()}
        return inf_record, eff_dis_tn, inf_records

    def cut_keys(self, old_dict):
        '''
        
        把实时表或者清洗表的_rd或者_cd删掉 模拟模块没有区分这些 投入运行前先删掉
        :param old_dict: 原始字典
        :return: 一个新字典, 其中的键名删去了末尾几个表关键字
        '''
        new_dict = {}
        for old_key, value in old_dict.items():
            new_dict[old_key[:-3]] = value
        return new_dict
    
    def update_args(self, pending_data):
        '''
        把从清洗表或者实时表中的数据结合手动输入的数据构造成模拟模块需要的数据
        :param update_data: 清洗表或者实时表中的数据(list)
        :return: 模拟模块需要的数据(list)
        '''
        # 先把尾部_cd _rd _pi处理了 保持数据名称一致性
        pending_data = self.cut_keys(pending_data)
      
        pending_data.update(
            {
            'opt1_aao_sel_inf_rat_1':0.1,
            'opt1_aao_ana_inf_rat_1':0.3,
            'opt1_aao_ano_inf_rat_1':0.6,
            
            'opt1_aao_sel_inf_rat_2':0.1,
            'opt1_aao_ana_inf_rat_2':0.3,
            'opt1_aao_ano_inf_rat_2':0.6,
            
            'opt1_aao_sel_inf_rat_3':0.1,
            'opt1_aao_ana_inf_rat_3':0.3,
            'opt1_aao_ano_inf_rat_3':0.6,
            
            'opt1_aao_sel_inf_rat_4':0.1,
            'opt1_aao_ana_inf_rat_4':0.3,
            'opt1_aao_ano_inf_rat_4':0.6
            }                  
            )
        
        return pending_data