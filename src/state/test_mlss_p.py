# -*- coding: UTF-8 -*-


import datetime
import math
import pandas as pd
import numpy as np
import shutil
import multiprocessing as mp
import time
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import sys
import copy

sys.path.append("..")
import dynamita.tool as dtool
from sumo_sim.module_AAO_SingleSimulate import AAO_Singlemulation

WATER_LINE = 4


class MlssCaliP:
    def __init__(
        self,
        mlss_Kp,
        mlss_Ki,
        mlss_imax: float,
        tn_Kp: float,
        tn_Ki: float,
        tn_imax: float,
        model,
        init_state_file_format: str,
        out_path,  # format str, e.g. 'OfflineStates/RandState{}.xml', 任意的初始状态
        freq: int = 120,
        delta_mlss=0.01,
        delta_tn=0.01,
    ):
        """
        parameter:
            model - 模拟的模型文件List[str]
            initstate - 要生成的初始状态文件
            ct - 要确定初始状态的时刻
            freq - 运行频率
        """
        self.model = model
        # 设置随机状态文件的路径
        self.sub_in_state = str(init_state_file_format)
        self.frequency = freq
        self.out_path = out_path
        self.mlss_Kp = mlss_Kp
        self.mlss_Ki = mlss_Ki
        self.mlss_imax = mlss_imax
        self.tn_Kp = tn_Kp
        self.tn_Ki = tn_Ki
        self.tn_imax = tn_imax
        assert mlss_Kp < 0
        assert mlss_Ki < 0
        assert mlss_imax > 0
        assert tn_imax > 0

        # 将每个并行单元MLSS和TN的实际仪表值从SUMO的输出设置中导出
        self.MLSS = ["Sumo__Plant__aao_cstr7_1_1__XTSS", "Sumo__Plant__aao_cstr7_1_2__XTSS", "Sumo__Plant__aao_cstr7_2_1__XTSS", "Sumo__Plant__aao_cstr7_2_2__XTSS"]
        self.TN = "Sumo__Plant__aao_effluent__TN"

        self.delta_mlss = delta_mlss
        self.delta_tn = delta_tn
        self.sumo = AAO_Singlemulation("init_state_cali", model, True, self.frequency, None, is_do=False)

    def run(self, out_state: str, inf_args: dict, goal_mlss, goal_tn):
        self.sub_out_state = str(out_state)
        for i in range(WATER_LINE):
            shutil.copy(src=self.sub_in_state.format(i + 1), dst=self.sub_out_state.format(i + 1))
        self.inf_args = inf_args

        logger.info("开始执行初始状态确定任务")
        logger.info(f"mlss目标值: {goal_mlss=}")

        self.set_mlss_TN(goal_mlss=goal_mlss, goal_tn=goal_tn)

        # all_balance_point = mp.Manager().list([0, 0, 0, 0])
        # p1 = mp.Process(target=self.set_Mlss,args=(goal_mlss[0], 0, all_balance_point))
        # p2 = mp.Process(target=self.set_Mlss,args=(goal_mlss[1], 1, all_balance_point))
        # p3 = mp.Process(target=self.set_Mlss,args=(goal_mlss[2], 2, all_balance_point))
        # p4 = mp.Process(target=self.set_Mlss,args=(goal_mlss[3], 3, all_balance_point))
        # p1.start()
        # p2.start()
        # p3.start()
        # p4.start()

        # p1.join()
        # p2.join()
        # p3.join()
        # p4.join()
        # self.set_TN(goal_tn, all_balance_point)

    def run_Sim(self, state, n, task_index, flow_qpump=None, inf_tn=None):
        qpump = [None for _ in range(WATER_LINE)]
        if task_index == -1:
            for i in range(WATER_LINE):
                qpump[i] = flow_qpump[i]
            logger.info(f"{qpump=}")
        else:
            qpump[task_index] = flow_qpump
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
        if task_index == -1:
            res = self.sumo.cal_effluent(state_list, self.inf_args, cmdline=cmdline)
        else:
            res = self.sumo.run_sim(state_list, self.inf_args, result_dic=None, cmdline=cmdline, water_line_no=task_index)
        return res

    def set_Mlss(self, goal_mlss, task_index, all_balance_point):
        logger.add(self.out_path / f"info.log", filter=lambda record: record["level"].name == "INFO")  # 子进程
        logger.add(self.out_path / f"debug.log", filter=lambda record: record["level"].name == "DEBUG")  # 子进程

        delta_mlss = self.delta_mlss * goal_mlss

        draw_data = {"mlss": [], "qpump": [], "t": []}
        t = -1
        sum_delta = 0
        qpump = None
        balance = 0
        # while t<=0 or abs(goal_mlss - mlss) > delta_mlss or not (draw_data['qpump'][-1]!=0.01 and abs(draw_data['qpump'][-1]-draw_data['qpump'][-2])<2):    # 未达目标范围
        # while t==-1 or abs(goal_mlss - float(mlss)) > delta_mlss:    # 未达目标范围
        while t <= 0 or abs(goal_mlss - mlss) > delta_mlss or balance < 20:  # 未达目标范围
            if t > 0 and draw_data["qpump"][-1] != 0.01 and abs(draw_data["qpump"][-1] - draw_data["qpump"][-2]) < 2:
                balance += 1
            else:
                balance = 0
            t += 1
            res = self.run_Sim(state=self.sub_out_state, n=1, flow_qpump=qpump, task_index=task_index)
            mlss = res[self.MLSS[task_index]]
            # 控制策略
            # qpump = max(0.01, balance_point+self.K*(goal_mlss-mlss))
            sum_delta += goal_mlss - mlss
            sum_delta = min(self.mlss_imax, max(-self.mlss_imax, sum_delta))
            qpump = max(0.01, self.mlss_Kp * (goal_mlss - mlss) + self.mlss_Ki * sum_delta)
            draw_data["mlss"].append(mlss)
            draw_data["qpump"].append(qpump)
            draw_data["t"].append(t)
            logger.debug(f"{draw_data=}")
            # 画图
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            ax1.axhline(y=goal_mlss, color="r", linestyle="-", alpha=1, linewidth=1, label=f"goal_mlss {goal_mlss:.2f}")
            # ax2.axhline(y=balance_point, color='r', linestyle='-', alpha=1, linewidth=1, label=f'balance_point {balance_point:.2f}')
            (line1,) = ax1.plot(draw_data["t"], draw_data["mlss"], label="mlss")
            (line2,) = ax2.plot(draw_data["t"], draw_data["qpump"], label="qpump")
            (line3,) = ax3.plot(draw_data["qpump"], draw_data["mlss"])
            ax1.set_title("mlss-t")
            ax2.set_title("qpump-t")
            ax3.set_title("mlss-qpump")
            plt.close(fig)
            fig.savefig(self.out_path / f"mlss_output_{task_index}.png")

        all_balance_point[task_index] = qpump
        return res  # 返回该池子出水值

    def set_TN(self, goal_tn, qpump):
        delta_tn = self.delta_tn

        draw_data = {"eff_tn": [], "inf_tn": [], "t": []}
        t = -1
        sum_delta = 0
        inf_tn = None
        balance = 0
        while t <= 0 or abs(goal_tn - eff_tn) > delta_tn or not (draw_data["inf_tn"][-1] != 0 and abs(draw_data["inf_tn"][-1] - draw_data["inf_tn"][-2]) < 0.2):  # 未达目标范围
            t += 1
            res = self.run_Sim(state=self.sub_out_state, n=1, flow_qpump=qpump, inf_tn=inf_tn, task_index=-1)
            eff_tn = res[self.TN]
            # 控制策略
            sum_delta += goal_tn - eff_tn
            sum_delta = min(self.tn_imax, max(-self.tn_imax, sum_delta))
            inf_tn = max(0.01, self.tn_Kp * (goal_tn - eff_tn) + self.tn_Ki * sum_delta)
            draw_data["eff_tn"].append(eff_tn)
            draw_data["inf_tn"].append(inf_tn)
            draw_data["t"].append(t)
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
            fig.savefig(self.out_path / f"tn_output.png")

        return res  # 返回该池子出水值

    def set_mlss_TN(self, goal_mlss: list, goal_tn: float):
        delta_tn = self.delta_tn
        delta_mlss = [self.delta_mlss * goal_mlss[i] for i in range(WATER_LINE)]

        draw_data = {"mlss": [[] for _ in range(WATER_LINE)], "qpump": [[] for _ in range(WATER_LINE)], "eff_tn": [], "inf_tn": [], "t": []}
        t = -1
        sum_delta_mlss = [0 for _ in range(WATER_LINE)]
        sum_delta_tn = 0
        inf_tn = None
        qpump = [None for _ in range(WATER_LINE)]
        while (
            t <= 0
            or abs(goal_tn - eff_tn) > delta_tn
            or not (draw_data["inf_tn"][-1] != 0 and abs(draw_data["inf_tn"][-1] - draw_data["inf_tn"][-2]) < 0.2)
            or any([abs(goal_mlss[i] - mlss[i]) > delta_mlss[i] for i in range(WATER_LINE)])
            or not all([draw_data["qpump"][i][-1] != 0.01 and abs(draw_data["qpump"][i][-1] - draw_data["qpump"][i][-2]) < 20 for i in range(WATER_LINE)])
        ):  # 未达目标范围
            t += 1
            res = self.run_Sim(state=self.sub_out_state, n=1, flow_qpump=qpump, inf_tn=inf_tn, task_index=-1)
            eff_tn = res[self.TN]
            mlss = [res[self.MLSS[i]] for i in range(WATER_LINE)]
            # 控制策略
            # if abs(goal_tn - eff_tn)<self.tn_imax:  # 抗饱和: 积分分离
            #     sum_delta_tn += goal_tn - eff_tn
            sum_delta_tn += min(self.tn_imax, max(-self.tn_imax, goal_tn - eff_tn))
            # sum_delta_tn = min(self.tn_imax, max(-self.tn_imax, sum_delta_tn))

            inf_tn = max(0.01, self.tn_Kp * (goal_tn - eff_tn) + self.tn_Ki * sum_delta_tn)

            draw_data["eff_tn"].append(eff_tn)
            draw_data["inf_tn"].append(inf_tn)
            draw_data["t"].append(t)
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
            fig.savefig(self.out_path / f"tn_output.png")

            for i in range(WATER_LINE):
                # if abs(goal_mlss[i] - mlss[i]) < self.mlss_imax:    # 抗饱和: 积分分离
                #     sum_delta_mlss[i] += goal_mlss[i] - mlss[i]
                sum_delta_mlss[i] += min(self.mlss_imax, max(-self.mlss_imax, goal_mlss[i] - mlss[i]))
                # sum_delta_mlss[i] = min(self.mlss_imax, max(-self.mlss_imax, sum_delta_mlss[i]))

                qpump[i] = max(0.01, self.mlss_Kp * (goal_mlss[i] - mlss[i]) + self.mlss_Ki * sum_delta_mlss[i])
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
                fig.savefig(self.out_path / f"mlss_output_{i}.png")

        return res  # 返回该池子出水值


def qpump_pi_constant(
    init_time, out_path, delta_mlss=0.005, delta_tn=0.01, mlss_Kp: float = -5, mlss_Ki: float = -1, mlss_imax: float = 4000, tn_Kp: float = 3, tn_Ki: float = 1, tn_imax: float = 40
):
    """qpump=Kp*delta(mlss)+Ki*sdelta(mlss)dt, 设定值为恒定值"""
    logger.add(out_path / f"info.log", filter=lambda record: record["level"].name == "INFO")
    logger.add(out_path / f"debug.log", filter=lambda record: record["level"].name == "DEBUG")
    model_list = [
        "../../data/OfflineSumomodels/22.1.0dll_24_10/aao1-1qairV1.1.dll",
        "../../data/OfflineSumomodels/22.1.0dll_24_10/aao1-2qairV1.1.dll",
        "../../data/OfflineSumomodels/22.1.0dll_24_10/aao2-1qairV1.1.dll",
        "../../data/OfflineSumomodels/22.1.0dll_24_10/aao2-2qairV1.1.dll",
    ]
    init_state_file_format = "../../data/OfflineStates/state{}_init.xml"
    clean_data = pd.read_excel("../../data/OfflineData/2024-08-27-2024-09-23_model_cleaning_data.xlsx", parse_dates=["cleaning_time"])
    eff_data = pd.read_excel("../../data/OfflineData/2024-08-27-2024-09-23_effluent_data.xlsx", parse_dates=["sampling_time"])
    inf_args = clean_data[clean_data["cleaning_time"] == init_time].iloc[0].to_dict()
    goal_mlss = [inf_args["aao1_1_mlss"], inf_args["aao1_2_mlss"], inf_args["aao2_1_mlss"], inf_args["aao2_2_mlss"]]
    goal_tn = eff_data[eff_data["sampling_time"] == init_time]["effluent_dis1_tn"].iloc[0]

    logger.info(f"{delta_mlss=}, {delta_tn=}, {mlss_Kp=}, {mlss_Ki=}, {mlss_imax=}, {tn_Kp=}, {tn_Ki=}, {tn_imax=}")
    st = MlssCaliP(mlss_Kp, mlss_Ki, mlss_imax, tn_Kp, tn_Ki, tn_imax, model_list, init_state_file_format, out_path=out_path, delta_mlss=delta_mlss, delta_tn=delta_tn)
    # 用于存储校准后的状态文件
    out_state_file_format = out_path / (f"{delta_mlss}_{delta_tn}_" + "state{}_init.xml")
    st.run(str(out_state_file_format), inf_args, goal_mlss, goal_tn)


if __name__ == "__main__":
    start_time = datetime.datetime(2024, 8, 27, 1)
    end_time = datetime.datetime(2024, 9, 20, 1)
    time_series = pd.date_range(start=start_time, end=end_time, freq="5D")
    for init_time in time_series:
        out_path = Path("../../output/experiment_init_state") / Path(__file__).stem / "test" / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if not out_path.exists():  # 判断是否存在文件夹如果不存在则创建为文件夹
            out_path.mkdir(parents=True, exist_ok=True)
        start = datetime.datetime.now()
        qpump_pi_constant(init_time, out_path, tn_Kp=1, tn_Ki=0.3, tn_imax=8, mlss_imax=200)
        # qpump_pi_constant(init_time, out_path, tn_Kp=1, tn_Ki=0.3, tn_imax=40 / 0.3)
        end = datetime.datetime.now()
        logger.info(f"########### cost time: {end-start} #############")
    # start_time = datetime.datetime(2024, 9, 1, 1)
    # out_path = Path('../../output/experiment_init_state') / Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # if not out_path.exists():  # 判断是否存在文件夹如果不存在则创建为文件夹
    #     out_path.mkdir(parents=True, exist_ok=True)
    # start = datetime.datetime.now()
    # qpump_pi_constant(start_time, out_path, tn_Kp=1, tn_Ki=0.3, tn_imax=40/0.3)
    # end = datetime.datetime.now()
    # logger.info(f'########### cost time: {end-start} #############')
