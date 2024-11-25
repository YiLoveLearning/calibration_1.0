# -*- coding: UTF-8 -*-

import datetime
import pandas as pd
import numpy as np
import shutil
from loguru import logger
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from sumo_sim.module_AAO_SingleSimulate import AAO_Singlemulation
# from sumo_sim.sumo_24_8 import SumoOnline
import dynamita.tool as dtool
from utils.my_math import choose_sample_time
from utils.draw import draw_same_column

class TestMlss:
    def __init__(self, model, init_state_file, out_state, out_path, freq: int=120, is_do = False):
        self.model = model

        self.randstate = init_state_file
        self.out_state = out_state
        self.frequency = freq
        self.out_path = out_path 

        self.MLSS = ["Sumo__Plant__aao_cstr7_1_1__XTSS",
                     "Sumo__Plant__aao_cstr7_1_2__XTSS",
                     "Sumo__Plant__aao_cstr7_2_1__XTSS",
                     "Sumo__Plant__aao_cstr7_2_2__XTSS"]
        self.TN = "Sumo__Plant__aao_effluent__TN"
        self.sumo = AAO_Singlemulation("init_state_cali", model, True, self.frequency,None, is_do = is_do)

    def run_Sim(self, inf_args, state, n, flow6_qpump=None, inf_tn=None, inf_cod=None):
        cmdline = [[
            f"set Sumo__Plant__aao_flowdivider4_1_1_sludge__param__Qpumped_target {flow6_qpump if flow6_qpump else inf_args['aao_flowdivider4_1_1_sludge_q']*86.4}",
            f"set Sumo__Plant__aao_influent_1_1__param__TKN {inf_tn if inf_tn else inf_args['aao_influent_1_1_tn']}",
            # f"set Sumo__Plant__shx_mbr1_influent__param__TCOD {inf_cod if inf_cod else inf_args['shx_mbr1_influent_tcod']}",
            f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
            f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
        ],[
            f"set Sumo__Plant__aao_flowdivider4_1_2_sludge__param__Qpumped_target {flow6_qpump if flow6_qpump else inf_args['aao_flowdivider4_1_2_sludge_q']*86.4}",
            f"set Sumo__Plant__aao_influent_1_2__param__TKN {inf_tn if inf_tn else inf_args['aao_influent_1_2_tn']}",
            # f"set Sumo__Plant__shx_mbr1_influent__param__TCOD {inf_cod if inf_cod else inf_args['shx_mbr1_influent_tcod']}",
            f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
            f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
        ],[
            f"set Sumo__Plant__aao_flowdivider4_2_1_sludge__param__Qpumped_target {flow6_qpump if flow6_qpump else inf_args['aao_flowdivider4_2_1_sludge_q']*86.4}",
            f"set Sumo__Plant__aao_influent_2_1__param__TKN {inf_tn if inf_tn else inf_args['aao_influent_2_1_tn']}",
            # f"set Sumo__Plant__shx_mbr1_influent__param__TCOD {inf_cod if inf_cod else inf_args['shx_mbr1_influent_tcod']}",
            f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
            f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
        ],[
            f"set Sumo__Plant__aao_flowdivider4_2_2_sludge__param__Qpumped_target {flow6_qpump if flow6_qpump else inf_args['aao_flowdivider4_2_2_sludge_q']*86.4}",
            f"set Sumo__Plant__aao_influent_2_2__param__TKN {inf_tn if inf_tn else inf_args['aao_influent_2_2_tn']}",
            # f"set Sumo__Plant__shx_mbr1_influent__param__TCOD {inf_cod if inf_cod else inf_args['shx_mbr1_influent_tcod']}",
            f"set Sumo__StopTime {int(n*self.frequency*dtool.minute)}",  # 设置模拟时长
            f"set Sumo__DataComm {int(n*self.frequency*dtool.minute)}",  # 设置模拟通讯间隔
        ]]
        res = self.sumo.cal_effluent(state, inf_args, cmdline=cmdline)
        return res

    def run(self, inf, flow6_qpump=None, inf_tn=None, n=1, step=10):
        for i in range(4):
            shutil.copy(src=self.randstate[i], dst=self.out_state[i])
        res_list = []
        for i in range(step):
            res = self.run_Sim(inf, self.out_state, n=n, inf_tn=inf_tn, flow6_qpump=flow6_qpump)
            res_list.append(res)
        res_list = pd.DataFrame(res_list)
        # res_list.set_index('sampling_time')
        # for i in range(4):
        #     plt.plot(range(len(mlss_list[i])),mlss_list[i],'-', color='b', alpha=0.8, linewidth=1, label='current mlss')
        #     # plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
        #     plt.legend(loc="best") # upper right
        #     plt.title("MLSS iter")
        #     plt.savefig(self.out_path/f'mlss{i}_{x}_{n}_{step}.png')
        #     plt.clf()
        # plt.plot(range(len(tn_list)),tn_list,'-', color='b', alpha=0.8, linewidth=1, label='current mlss')
        # # plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
        # plt.legend(loc="best") # upper right
        # plt.title("TN iter")
        # plt.savefig(self.out_path/f'tn_{x}_{n}_{step}.png')
        # plt.clf()
        # logger.debug(mlss_list)
        # return mlss_list, tn_list
        return [res_list[self.MLSS[i]] for i in range(4)], res_list[self.TN]

def test_flow6_qpump(test_type, left, right, step, sim_step):
    if test_type == "flow6_qpump":
        prefix = "flow6_qpump"
    else: 
        prefix = "inf_tn"
    model = [
        '../../data/OfflineSumomodels/22.1.0dll_24_10/aao1-1qairV1.1.dll', 
        '../../data/OfflineSumomodels/22.1.0dll_24_10/aao1-2qairV1.1.dll', 
        '../../data/OfflineSumomodels/22.1.0dll_24_10/aao2-1qairV1.1.dll', 
        '../../data/OfflineSumomodels/22.1.0dll_24_10/aao2-2qairV1.1.dll', 
    ]
    state_list = [
        '../../data/OfflineStates/22.1.0dll_24_10/state1_init.xml',
        '../../data/OfflineStates/22.1.0dll_24_10/state2_init.xml',
        '../../data/OfflineStates/22.1.0dll_24_10/state3_init.xml',
        '../../data/OfflineStates/22.1.0dll_24_10/state4_init.xml',
    ]
    start_time = datetime.datetime(2024, 8, 27, 1)
    end_time = datetime.datetime(2024, 9, 22, 23)
    time_series = pd.date_range(start=start_time, end=end_time, freq='2H')
    out_path = Path('../../output/human/test_mlss') / Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if not out_path.exists():  #判断是否存在文件夹如果不存在则创建为文件夹
        out_path.mkdir(parents=True, exist_ok=True)
    logger.add(out_path / 'info.log', filter=lambda record: record['level'].name =='INFO', enqueue=True)

    cleaning_data = pd.read_excel("../../data/OfflineData/2024-08-27-2024-09-23_model_cleaning_data.xlsx", 
                                  parse_dates=['sampling_time']).iloc[0].to_dict()
    
    out_state_list = [out_path/f'state{i}.xml' for i in range(4)]

    st = TestMlss(model, state_list, out_state_list, out_path, freq=120, is_do = False)
    mlss_all_list = [[] for _ in range(4)]
    tn_all_list = []
    x_list = []
    for x in np.linspace(left, right, step):
        x_list.append(x)
        if test_type == "flow6_qpump":
            mlss_list, tn_list = st.run(cleaning_data, flow6_qpump=x, step=sim_step)
        else:
            mlss_list, tn_list = st.run(cleaning_data, inf_tn=x, step=sim_step)
           
        tn_all_list.append(tn_list)
        for i in range(4):
            
            mlss_all_list[i].append(mlss_list[i])
        
        
    

    
    for i in range(4):
        data = pd.concat(mlss_all_list[i], axis=1)  
        df = pd.DataFrame(data.values, columns=[f'{prefix}={x}' for x in x_list])
        print(df)
        df.to_excel(out_path / f'mlss_result{i}.xlsx')
        for j in range(len(mlss_all_list[i])):
            plt.plot(range(len(mlss_all_list[i][j])),mlss_all_list[i][j],'-', alpha=0.8, linewidth=1, label=f'{prefix}={x_list[j]:.2f}')
            # plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
        plt.legend(loc="best") # upper right
        plt.xlabel('iter')
        plt.ylabel('mlss')
        plt.title(f"MLSS-{prefix}")
        plt.savefig(out_path/f'mlss{i}.png')
        plt.clf()
    data = pd.concat(tn_all_list, axis=1)      
    df = pd.DataFrame(data, columns=[f'{prefix}={x}' for x in x_list])
    df.to_excel(out_path / f'tn_result.xlsx')
    for j in range(len(tn_all_list)):
        plt.plot(range(len(tn_all_list[j])),tn_all_list[j],'-', alpha=0.8, linewidth=1, label=f'{prefix}={x_list[j]:.2f}')
        # plt.axhline(y=goal_mlss, color='r', linestyle='-', alpha=1, linewidth=1, label=f'goal_mlss {goal_mlss}')
    plt.legend(loc="best") # upper right
    plt.title(f"effTN-{prefix}")
    plt.xlabel('iter')
    plt.ylabel('tn')
    plt.savefig(out_path/f'tn.png')
    plt.clf()


if __name__ == "__main__":
    start = datetime.datetime.now()
    test_flow6_qpump("inf_tn", 0.01, 20, 3, 2)
    end = datetime.datetime.now()
    logger.info(f"################## cost time: {end-start} ##################")
