# -*- coding: UTF-8 -*-
"""
单次的并行模拟模块,模拟时长两个小时,单点数据
2024/10/19 ysq改了之后的
"""
import os
from loguru import logger
import sys
sys.path.append('..')
import dynamita.scheduler as ds
import dynamita.tool as dtool
import time
import multiprocessing as mp


# 采用进程隔离后暂不需要使用锁
# muti_simulate_locker = threading.Lock() # 模拟是同一个进程内两个线程（基准和离线模拟）跑sumo，出现问题。优化的逻辑是开了多个进程跑sumo 暂未发现问题。

class AAO_Singlemulation:
    """
    example:
    >>> taskname = "aao_do_auto_optimize"
    >>> model = [f'xxx/sumoproject4.3.{i+1}.dll' for i in range(4)]
    >>> state = [f'xxx/state{i+1}_xxx.xml' for i in range(4)]
    >>> state_flag = True       # 是否覆盖(保存)状态文件
    >>> freq = 120
    >>> input_args = pd.DataFrame()的一行
    >>> model_param = {}
    >>> is_do = False
    >>> sumo = AAO_Singlemulation(taskname, model, state, state_flag, freq, input_args, model_param, is_do)
    >>> sumo.run_sim()
    """
    def __init__(self, taskname, model, state_flag, freq,model_param,is_do = False):
        self.model = model
        self.model_param = model_param
        self.frequency = freq
        self.flag = state_flag
        self.taskname = taskname
        self.is_do = is_do
    
    def data_Callback(self, job, data):
        """
        function:
            数据回调函数
        """
        jobData = ds.sumo.getJobData(job)
        jobData["data"] = data  

    def msg_Callback(self, state_name_list:list, i=-1):
        def msg_callback_save_for_opt(job, msg):
            """消息回调函数"""
            # 打印任务号和消息(调试时使用)
            # print(f'#{job} {msg}')
            # 离线模式模拟完直接结束任务
            # logger = setup_logger()
            # logger.debug(f'#{job} {msg}')
            if ds.sumo.isSimFinishedMsg(msg) and not self.flag:
                ds.sumo.finish(job)
            elif ds.sumo.isSimFinishedMsg(msg) and self.flag:
                if i==-1:
                    jobnumber = ((job + 3) % 4) + 1
                    state_name = state_name_list[jobnumber-1]
                else:
                    state_name = state_name_list[i]
                ds.sumo.sendCommand(job, f'save {state_name}')

            if msg.startswith("530045"):
                ds.sumo.finish(job)
        return msg_callback_save_for_opt

    # 2024.10.28, 增加跑单个分支
    def run_sim(self,state_file, influent_data,result_dic=None, cmdline=None, water_line_no=-1): 
        """
        :param result_dic: Manager() 共享参数
        :cmdline: 控制量 默认为None 控制量固定即纯模拟 优化时可以赋值从而做到变化的控制量
        :return: 混合输出
        :water_line_no: 只跑一条分支, 此参数代表分支编号, 0开始(cmd_line还是4条的); -1表示所有分支都跑
        """
        line = 1 if water_line_no>=0 else 4
        if cmdline is None:
            cmdline = [[''],[''],[''],['']]
        else:
            assert len(cmdline) == 4
        ds.sumo.setParallelJobs(line)
        # 绑定消息回调函数
        ds.sumo.message_callback = self.msg_Callback(state_file, water_line_no)
        # 绑定数据回调函数
        ds.sumo.datacomm_callback = self.data_Callback 

        sumo_suffix = ""
        db_para_suffix = ""
        if not self.is_do:
            sumo_suffix = 'Qair_NTP'
            db_para_suffix = 'qair_ntp'
        else: # self.taskname == 'aao_do_auto_optimize':
            sumo_suffix = 'SO2'
            db_para_suffix = 'so2'
        
        commands=[
            [
                f"load {state_file[0]}",  # 载入初始状态
                # # 设置模型参数
                # f'set Sumo__Plant__param__Sumo2__YOHO_SB_anox {self.model_param["yoho_sb_anox"]}',    # 缺氧条件下易生物降解基质上OHOs的产率
                # f'set Sumo__Plant__param__Sumo2__YOHO_SB_ox {self.model_param["yoho_sb_ox"]}',      # 耗氧条件下易生物降解基质上OHOs的产率
                # # 该阶段自校准所要优化的参数
                # f'set Sumo__Plant__param__Sumo2__bAOB {self.model_param["baob"]}',              # AOBs的衰变率
                # f'set Sumo__Plant__param__Sumo2__KNHx_AOB_AS {self.model_param["knhx_aob_as"]}',       # AOBs中NHx的半饱和度
                # f'set Sumo__Plant__param__Sumo2__KO2_AOB_AS  {self.model_param["ko2_aob_as"]}',       # AOBs中O2的半饱和度
                # f'set Sumo__Plant__PAC1__param__G {self.model_param["pac1"]}',                  # PAC1的G值
                # f'set Sumo__Plant__PAC2__param__G {self.model_param["pac2"]}',                  # PAC2的G值
                # f'set Sumo__Plant__PAC3__param__G {self.model_param["pac3"]}',                  # PAC3的G值
                # f'set Sumo__Plant__Clarifier2__param__fXTSS_sludge_base {self.model_param["clarifier2_fxtss_sludge_base"]}',        # 深度处理沉淀池无聚合物去除率
                # f'set Sumo__Plant__Clarifier2__param__fXTSS_sludge_polymer {self.model_param["clarifier2_fxtss_sludge_polymer"]}',     # 深度处理沉淀池有聚合物去除率
                # f"set Sumo__Plant__aao_cstr3_1_1__param__alpha {self.model_param['aao_cstr3_1_1_alpha_cpa']}",  # 自动校准1-1#AAO好氧廊道1氧传递系数 
                # f"set Sumo__Plant__aao_cstr4_1_1__param__alpha {self.model_param['aao_cstr4_1_1_alpha_cpa']}",  # 自动校准1-1#AAO好氧廊道2氧传递系数 
                # f"set Sumo__Plant__aao_cstr5_1_1__param__alpha {self.model_param['aao_cstr5_1_1_alpha_cpa']}",  # 自动校准1-1#AAO好氧廊道3氧传递系数 
                # f"set Sumo__Plant__aao_cstr6_1_1__param__alpha {self.model_param['aao_cstr6_1_1_alpha_cpa']}",  # 自动校准1-1#AAO好氧廊道4氧传递系数 
                # f"set Sumo__Plant__aao_cstr7_1_1__param__alpha {self.model_param['aao_cstr7_1_1_alpha_cpa']}",  # 自动校准1-1#AAO好氧廊道5氧传递系数 
                # f"set Sumo__Plant__aao_cstr8_1_1__param__alpha {self.model_param['aao_cstr8_1_1_alpha_cpa']}",  # 自动校准1-1#AAO脱气池氧传递系数 
                # 设置进水参数
                f"set Sumo__Plant__aao_influent_1_1__param__Q {influent_data['aao_influent_1_1_q']}",  # 基准模拟1#AAO生化池进水流量
                f"set Sumo__Plant__aao_influent_1_1__param__TCOD {influent_data['aao_influent_1_1_tcod']}",  # 基准模拟1#AAO进水化学需氧量
                f"set Sumo__Plant__aao_influent_1_1__param__TP {influent_data['aao_influent_1_1_tp']}",  # 基准模拟1#AAO进水总磷
                f"set Sumo__Plant__aao_influent_1_1__param__TKN {influent_data['aao_influent_1_1_tn']}",  # 基准模拟1#AAO进水总氮
                f"set Sumo__Plant__aao_influent_1_1__param__frSNHx_TKN {influent_data['aao_influent_1_1_frsnhx_tkn']}",  # 基准模拟1#AAO进水氨氮在总凯氏氮中的比例
                f"set Sumo__Plant__aao_influent_1_1__param__T {influent_data['aao_influent_1_1_t']}",  # 基准模拟1#AAO进水温度
                
                
                # 设置运行参数
                # 选择:厌氧:缺氧=1:3:6 选择池站总进水比例=1/10  
                # f"set Sumo__Plant__aao_flowdivider1_1_1__param__fr1_Q {influent_data['opt1_aao_sel_inf_rat_1'] / (influent_data['opt1_aao_sel_inf_rat_1'] + influent_data['opt1_aao_ana_inf_rat_1'] + influent_data['opt1_aao_ano_inf_rat_1'])}",
                # # 缺氧:厌氧=6/9
                # f"set Sumo__Plant__aao_flowdivider2_1_1__param__fr1_Q {influent_data['opt1_aao_ano_inf_rat_1'] / (influent_data['opt1_aao_ana_inf_rat_1'] + influent_data['opt1_aao_ano_inf_rat_1'])}",
                # f"set Sumo__Plant__aao_carbon_1_1__param__Q {influent_data['aao_carbon_1_1_q']}",  # 基准模拟1-1#AAO生化池碳源投加量
                # f"set Sumo__Plant__aao_pac_1_1__param__Q {influent_data['aao_pac_1_1_q']}",  # 基准模拟1#AAO聚合氯化铝投加量
                # f"set Sumo__Plant__aao_pac_3__param__Q {influent_data['aao_pac_3_q']}",  # 基准模拟2号加药间聚合氯化铝投加量
                f"set Sumo__Plant__aao_flowdivider3_1_1_influx__param__Qpumped_target {influent_data['aao_flowdivider3_1_1_influx_q']}",  # 基准模拟1-1#AAO生化池内回流量
                f"set Sumo__Plant__aao_clarifier_1_1__param__Qsludge_target {influent_data['aao_clarifier_1_1_sludge_target_q']}",  # 基准模拟1号二沉池底流流量        
                f"set Sumo__Plant__aao_flowdivider4_1_1_sludge__param__Qpumped_target {influent_data['aao_flowdivider4_1_1_sludge_q']}",  # 基准模拟1号二沉池剩余污泥量
                f"set Sumo__Plant__aao_cstr3_1_1__param__{sumo_suffix} {influent_data[f'aao_cstr3_1_1_{db_para_suffix}']}",  # 基准模拟1-1#AAO生化池好氧段1曝气量
                f"set Sumo__Plant__aao_cstr4_1_1__param__{sumo_suffix} {influent_data[f'aao_cstr4_1_1_{db_para_suffix}']}",  # 基准模拟1-1#AAO生化池好氧段2曝气量
                f"set Sumo__Plant__aao_cstr5_1_1__param__{sumo_suffix} {influent_data[f'aao_cstr5_1_1_{db_para_suffix}']}",  # 基准模拟1-1#AAO生化池好氧段3曝气量
                f"set Sumo__Plant__aao_cstr6_1_1__param__{sumo_suffix} {influent_data[f'aao_cstr6_1_1_{db_para_suffix}']}",  # 基准模拟1-1#AAO生化池好氧段4曝气量
                f"set Sumo__Plant__aao_cstr7_1_1__param__{sumo_suffix} {influent_data[f'aao_cstr7_1_1_{db_para_suffix}']}",  # 基准模拟1-1#AAO生化池好氧段5曝气量


                f"set Sumo__StopTime {self.frequency*dtool.minute}",  # 设置模拟时长
                f"set Sumo__DataComm {self.frequency*dtool.minute}",  # 设置模拟通讯间隔
                'maptoic',
                'mode dynamic',  # 设置为动态模拟
                'start',
                *cmdline[0]
            ],[
                f"load {state_file[1]}",  # 载入初始状态
                # # 设置模型参数
                # f'set Sumo__Plant__param__Sumo2__YOHO_SB_anox {self.model_param["yoho_sb_anox"]}',    # 缺氧条件下易生物降解基质上OHOs的产率
                # f'set Sumo__Plant__param__Sumo2__YOHO_SB_ox {self.model_param["yoho_sb_ox"]}',      # 耗氧条件下易生物降解基质上OHOs的产率
                # # 该阶段自校准所要优化的参数
                # f'set Sumo__Plant__param__Sumo2__bAOB {self.model_param["baob"]}',              # AOBs的衰变率
                # f'set Sumo__Plant__param__Sumo2__KNHx_AOB_AS {self.model_param["knhx_aob_as"]}',       # AOBs中NHx的半饱和度
                # f'set Sumo__Plant__param__Sumo2__KO2_AOB_AS  {self.model_param["ko2_aob_as"]}',       # AOBs中O2的半饱和度
                # f'set Sumo__Plant__PAC1__param__G {self.model_param["pac1"]}',                  # PAC1的G值
                # f'set Sumo__Plant__PAC2__param__G {self.model_param["pac2"]}',                  # PAC2的G值
                # f'set Sumo__Plant__PAC3__param__G {self.model_param["pac3"]}',                  # PAC3的G值
                # f'set Sumo__Plant__Clarifier2__param__fXTSS_sludge_base {self.model_param["clarifier2_fxtss_sludge_base"]}',        # 深度处理沉淀池无聚合物去除率
                # f'set Sumo__Plant__Clarifier2__param__fXTSS_sludge_polymer {self.model_param["clarifier2_fxtss_sludge_polymer"]}',     # 深度处理沉淀池有聚合物去除率       
                # 设置进水参数
                f"set Sumo__Plant__aao_influent_1_2__param__Q {influent_data['aao_influent_1_2_q']}",  # 基准模拟1#AAO生化池进水流量
                f"set Sumo__Plant__aao_influent_1_2__param__TCOD {influent_data['aao_influent_1_2_tcod']}",  # 基准模拟1#AAO进水化学需氧量
                f"set Sumo__Plant__aao_influent_1_2__param__TP {influent_data['aao_influent_1_2_tp']}",  # 基准模拟1#AAO进水总磷
                f"set Sumo__Plant__aao_influent_1_2__param__TKN {influent_data['aao_influent_1_2_tn']}",  # 基准模拟1#AAO进水总氮
                f"set Sumo__Plant__aao_influent_1_2__param__frSNHx_TKN {influent_data['aao_influent_1_2_frsnhx_tkn']}",  # 基准模拟1#AAO进水氨氮在总凯氏氮中的比例
                f"set Sumo__Plant__aao_influent_1_2__param__T {influent_data['aao_influent_1_2_t']}",  # 基准模拟1#AAO进水温度
                
                
                # 设置运行参数
                # f"set Sumo__Plant__aao_flowdivider1_1_2__param__fr1_Q {influent_data['opt1_aao_sel_inf_rat_2'] / (influent_data['opt1_aao_sel_inf_rat_2'] + influent_data['opt1_aao_ana_inf_rat_2'] + influent_data['opt1_aao_ano_inf_rat_2'])}",
                # f"set Sumo__Plant__aao_flowdivider2_1_2__param__fr1_Q {influent_data['opt1_aao_ano_inf_rat_2'] / (influent_data['opt1_aao_ana_inf_rat_2'] + influent_data['opt1_aao_ano_inf_rat_2'])}",
                # f"set Sumo__Plant__aao_carbon_1_2__param__Q {influent_data['aao_carbon_1_2_q']}",  # 基准模拟1-2#AAO生化池碳源投加量
                # f"set Sumo__Plant__aao_pac_1_2__param__Q {influent_data['aao_pac_1_2_q']}",  # 基准模拟1#AAO聚合氯化铝投加量
                # f"set Sumo__Plant__aao_pac_4__param__Q {influent_data['aao_pac_4_q']}",  # 基准模拟2号加药间聚合氯化铝投加量
                f"set Sumo__Plant__aao_flowdivider3_1_2_influx__param__Qpumped_target {influent_data['aao_flowdivider3_1_2_influx_q']}",  # 基准模拟1-2#AAO生化池内回流量
                f"set Sumo__Plant__aao_clarifier_1_2__param__Qsludge_target {influent_data['aao_clarifier_1_2_sludge_target_q']}",  # 基准模拟1号二沉池底流流量        
                f"set Sumo__Plant__aao_flowdivider4_1_2_sludge__param__Qpumped_target {influent_data['aao_flowdivider4_1_2_sludge_q']}",  # 基准模拟1号二沉池剩余污泥量
                f"set Sumo__Plant__aao_cstr3_1_2__param__{sumo_suffix} {influent_data[f'aao_cstr3_1_2_{db_para_suffix}']}",  # 基准模拟1-2#AAO生化池好氧段1曝气量
                f"set Sumo__Plant__aao_cstr4_1_2__param__{sumo_suffix} {influent_data[f'aao_cstr4_1_2_{db_para_suffix}']}",  # 基准模拟1-2#AAO生化池好氧段2曝气量
                f"set Sumo__Plant__aao_cstr5_1_2__param__{sumo_suffix} {influent_data[f'aao_cstr5_1_2_{db_para_suffix}']}",  # 基准模拟1-2#AAO生化池好氧段3曝气量
                f"set Sumo__Plant__aao_cstr6_1_2__param__{sumo_suffix} {influent_data[f'aao_cstr6_1_2_{db_para_suffix}']}",  # 基准模拟1-2#AAO生化池好氧段4曝气量
                f"set Sumo__Plant__aao_cstr7_1_2__param__{sumo_suffix} {influent_data[f'aao_cstr7_1_2_{db_para_suffix}']}",  # 基准模拟1-2#AAO生化池好氧段5曝气量

                f"set Sumo__StopTime {self.frequency*dtool.minute}",  # 设置模拟时长
                f"set Sumo__DataComm {self.frequency*dtool.minute}",  # 设置模拟通讯间隔
                'maptoic',
                'mode dynamic',  # 设置为动态模拟
                'start',
                *cmdline[1] #解引用出新的指令 新的指令会覆盖前面的指令
            ],[
                f"load {state_file[2]}",  # 载入初始状态
                # # 设置模型参数
                # f'set Sumo__Plant__param__Sumo2__YOHO_SB_anox {self.model_param["yoho_sb_anox"]}',    # 缺氧条件下易生物降解基质上OHOs的产率
                # f'set Sumo__Plant__param__Sumo2__YOHO_SB_ox {self.model_param["yoho_sb_ox"]}',      # 耗氧条件下易生物降解基质上OHOs的产率
                # # 该阶段自校准所要优化的参数
                # f'set Sumo__Plant__param__Sumo2__bAOB {self.model_param["baob"]}',              # AOBs的衰变率
                # f'set Sumo__Plant__param__Sumo2__KNHx_AOB_AS {self.model_param["knhx_aob_as"]}',       # AOBs中NHx的半饱和度
                # f'set Sumo__Plant__param__Sumo2__KO2_AOB_AS  {self.model_param["ko2_aob_as"]}',       # AOBs中O2的半饱和度
                # f'set Sumo__Plant__PAC1__param__G {self.model_param["pac1"]}',                  # PAC1的G值
                # f'set Sumo__Plant__PAC2__param__G {self.model_param["pac2"]}',                  # PAC2的G值
                # f'set Sumo__Plant__PAC3__param__G {self.model_param["pac3"]}',                  # PAC3的G值
                # f'set Sumo__Plant__Clarifier2__param__fXTSS_sludge_base {self.model_param["clarifier2_fxtss_sludge_base"]}',        # 深度处理沉淀池无聚合物去除率
                # f'set Sumo__Plant__Clarifier2__param__fXTSS_sludge_polymer {self.model_param["clarifier2_fxtss_sludge_polymer"]}',     # 深度处理沉淀池有聚合物去除率                
                
                # 设置进水参数
                f"set Sumo__Plant__aao_influent_2_1__param__Q {influent_data['aao_influent_2_1_q']}",  # 基准模拟2#AAO生化池进水流量
                f"set Sumo__Plant__aao_influent_2_1__param__TCOD {influent_data['aao_influent_2_1_tcod']}",  # 基准模拟2#AAO进水化学需氧量
                f"set Sumo__Plant__aao_influent_2_1__param__TP {influent_data['aao_influent_2_1_tp']}",  # 基准模拟2#AAO进水总磷
                f"set Sumo__Plant__aao_influent_2_1__param__TKN {influent_data['aao_influent_2_1_tn']}",  # 基准模拟2#AAO进水总氮
                f"set Sumo__Plant__aao_influent_2_1__param__frSNHx_TKN {influent_data['aao_influent_2_1_frsnhx_tkn']}",  # 基准模拟2#AAO进水氨氮在总凯氏氮中的比例    
                f"set Sumo__Plant__aao_influent_2_1__param__T {influent_data['aao_influent_2_1_t']}",  # 基准模拟2#AAO进水温度
                
                # 设置运行参数
                
                # f"set Sumo__Plant__aao_flowdivider1_2_1__param__fr1_Q {influent_data['opt1_aao_sel_inf_rat_3'] / (influent_data['opt1_aao_sel_inf_rat_3'] + influent_data['opt1_aao_ana_inf_rat_3'] + influent_data['opt1_aao_ano_inf_rat_3'])}",
                # f"set Sumo__Plant__aao_flowdivider2_2_1__param__fr1_Q {influent_data['opt1_aao_ano_inf_rat_3'] / (influent_data['opt1_aao_ana_inf_rat_3'] + influent_data['opt1_aao_ano_inf_rat_3'])}",
                # f"set Sumo__Plant__aao_carbon_2_1__param__Q {influent_data['aao_carbon_2_1_q']}",  # 基准模拟2-1#AAO生化池碳源投加量
                # f"set Sumo__Plant__aao_pac_2_1__param__Q {influent_data['aao_pac_2_1_q']}",  # 基准模拟2号加药间聚合氯化铝投加量
                # f"set Sumo__Plant__aao_pac_5__param__Q {influent_data['aao_pac_5_q']}",  # 基准模拟2#AAO聚合氯化铝投加量
                f"set Sumo__Plant__aao_flowdivider3_2_1_influx__param__Qpumped_target {influent_data['aao_flowdivider3_2_1_influx_q']}",  # 基准模拟2-1#AAO生化池内回流量     
                f"set Sumo__Plant__aao_clarifier_2_1__param__Qsludge_target {influent_data['aao_clarifier_2_1_sludge_target_q']}",  # 基准模拟2号二沉池底流流量
                f"set Sumo__Plant__aao_flowdivider4_2_1_sludge__param__Qpumped_target {influent_data['aao_flowdivider4_2_1_sludge_q']}",  # 基准模拟2号二沉池剩余污泥量

                f"set Sumo__Plant__aao_cstr3_2_1__param__{sumo_suffix} {influent_data[f'aao_cstr3_2_1_{db_para_suffix}']}",  # 基准模拟2-1#AAO生化池好氧段1曝气量
                f"set Sumo__Plant__aao_cstr4_2_1__param__{sumo_suffix} {influent_data[f'aao_cstr4_2_1_{db_para_suffix}']}",  # 基准模拟2-1#AAO生化池好氧段2曝气量
                f"set Sumo__Plant__aao_cstr5_2_1__param__{sumo_suffix} {influent_data[f'aao_cstr5_2_1_{db_para_suffix}']}",  # 基准模拟2-1#AAO生化池好氧段3曝气量
                f"set Sumo__Plant__aao_cstr6_2_1__param__{sumo_suffix} {influent_data[f'aao_cstr6_2_1_{db_para_suffix}']}",  # 基准模拟2-1#AAO生化池好氧段4曝气量
                f"set Sumo__Plant__aao_cstr7_2_1__param__{sumo_suffix} {influent_data[f'aao_cstr7_2_1_{db_para_suffix}']}",  # 基准模拟2-1#AAO生化池好氧段5曝气量

                f"set Sumo__StopTime {self.frequency*dtool.minute}",  # 设置模拟时长
                f"set Sumo__DataComm {self.frequency*dtool.minute}",  # 设置模拟通讯间隔
                'maptoic',
                'mode dynamic',  # 设置为动态模拟
                'start',
                *cmdline[2]
            ],[
                f"load {state_file[3]}",  # 载入初始状态
                # # 设置模型参数
                # f'set Sumo__Plant__param__Sumo2__YOHO_SB_anox {self.model_param["yoho_sb_anox"]}',    # 缺氧条件下易生物降解基质上OHOs的产率
                # f'set Sumo__Plant__param__Sumo2__YOHO_SB_ox {self.model_param["yoho_sb_ox"]}',      # 耗氧条件下易生物降解基质上OHOs的产率
                # # 该阶段自校准所要优化的参数
                # f'set Sumo__Plant__param__Sumo2__bAOB {self.model_param["baob"]}',              # AOBs的衰变率
                # f'set Sumo__Plant__param__Sumo2__KNHx_AOB_AS {self.model_param["knhx_aob_as"]}',       # AOBs中NHx的半饱和度
                # f'set Sumo__Plant__param__Sumo2__KO2_AOB_AS  {self.model_param["ko2_aob_as"]}',       # AOBs中O2的半饱和度
                # f'set Sumo__Plant__PAC1__param__G {self.model_param["pac1"]}',                  # PAC1的G值
                # f'set Sumo__Plant__PAC2__param__G {self.model_param["pac2"]}',                  # PAC2的G值
                # f'set Sumo__Plant__PAC3__param__G {self.model_param["pac3"]}',                  # PAC3的G值
                # f'set Sumo__Plant__Clarifier2__param__fXTSS_sludge_base {self.model_param["clarifier2_fxtss_sludge_base"]}',        # 深度处理沉淀池无聚合物去除率
                # f'set Sumo__Plant__Clarifier2__param__fXTSS_sludge_polymer {self.model_param["clarifier2_fxtss_sludge_polymer"]}',     # 深度处理沉淀池有聚合物去除率                 
                                
                # 设置进水参数
                f"set Sumo__Plant__aao_influent_2_2__param__Q {influent_data['aao_influent_2_2_q']}",  # 基准模拟2#AAO生化池进水流量
                f"set Sumo__Plant__aao_influent_2_2__param__TCOD {influent_data['aao_influent_2_2_tcod']}",  # 基准模拟2#AAO进水化学需氧量
                f"set Sumo__Plant__aao_influent_2_2__param__TP {influent_data['aao_influent_2_2_tp']}",  # 基准模拟2#AAO进水总磷
                f"set Sumo__Plant__aao_influent_2_2__param__TKN {influent_data['aao_influent_2_2_tn']}",  # 基准模拟2#AAO进水总氮
                f"set Sumo__Plant__aao_influent_2_2__param__frSNHx_TKN {influent_data['aao_influent_2_2_frsnhx_tkn']}",  # 基准模拟2#AAO进水氨氮在总凯氏氮中的比例    
                f"set Sumo__Plant__aao_influent_2_2__param__T {influent_data['aao_influent_2_2_t']}",  # 基准模拟2#AAO进水温度
                
                # 设置运行参数
                # f"set Sumo__Plant__aao_flowdivider1_2_2__param__fr1_Q {influent_data['opt1_aao_sel_inf_rat_4'] / (influent_data['opt1_aao_sel_inf_rat_4'] + influent_data['opt1_aao_ana_inf_rat_4'] + influent_data['opt1_aao_ano_inf_rat_4'])}",
                # f"set Sumo__Plant__aao_flowdivider2_2_2__param__fr1_Q {influent_data['opt1_aao_ano_inf_rat_4'] / (influent_data['opt1_aao_ana_inf_rat_4'] + influent_data['opt1_aao_ano_inf_rat_4'])}",
                # f"set Sumo__Plant__aao_carbon_2_2__param__Q {influent_data['aao_carbon_2_2_q']}",  # 基准模拟2-2#AAO生化池碳源投加量
                # f"set Sumo__Plant__aao_pac_2_2__param__Q {influent_data['aao_pac_2_2_q']}",  # 基准模拟2号加药间聚合氯化铝投加量
                # f"set Sumo__Plant__aao_pac_6__param__Q {influent_data['aao_pac_6_q']}",  # 基准模拟2#AAO聚合氯化铝投加量
                f"set Sumo__Plant__aao_flowdivider3_2_2_influx__param__Qpumped_target {influent_data['aao_flowdivider3_2_2_influx_q']}",  # 基准模拟2-2#AAO生化池内回流量    
                f"set Sumo__Plant__aao_clarifier_2_2__param__Qsludge_target {influent_data['aao_clarifier_2_2_sludge_target_q']}",  # 基准模拟2号二沉池底流流量
                f"set Sumo__Plant__aao_flowdivider4_2_2_sludge__param__Qpumped_target {influent_data['aao_flowdivider4_2_2_sludge_q']}",  # 基准模拟2号二沉池剩余污泥量
                f"set Sumo__Plant__aao_cstr3_2_2__param__{sumo_suffix} {influent_data[f'aao_cstr3_2_2_{db_para_suffix}']}",  # 基准模拟2-2#AAO生化池好氧段1曝气量
                f"set Sumo__Plant__aao_cstr4_2_2__param__{sumo_suffix} {influent_data[f'aao_cstr4_2_2_{db_para_suffix}']}",  # 基准模拟2-2#AAO生化池好氧段2曝气量
                f"set Sumo__Plant__aao_cstr5_2_2__param__{sumo_suffix} {influent_data[f'aao_cstr5_2_2_{db_para_suffix}']}",  # 基准模拟2-2#AAO生化池好氧段3曝气量
                f"set Sumo__Plant__aao_cstr6_2_2__param__{sumo_suffix} {influent_data[f'aao_cstr6_2_2_{db_para_suffix}']}",  # 基准模拟2-2#AAO生化池好氧段4曝气量
                f"set Sumo__Plant__aao_cstr7_2_2__param__{sumo_suffix} {influent_data[f'aao_cstr7_2_2_{db_para_suffix}']}",  # 基准模拟2-2#AAO生化池好氧段5曝气量
                f"set Sumo__StopTime {self.frequency*dtool.minute}",  # 设置模拟时长
                f"set Sumo__DataComm {self.frequency*dtool.minute}",  # 设置模拟通讯间隔
                'maptoic',
                'mode dynamic',  # 设置为动态模拟
                'start',
                *cmdline[3]
            ]
        ]

        variables = [
            [
                #时间戳
                "Sumo__Time",
                #1-1进水
                "Sumo__Plant__aao_influent_1_1__Q",
                "Sumo__Plant__aao_influent_1_1__TCOD ",      
                "Sumo__Plant__aao_influent_1_1__TBOD_5",
                "Sumo__Plant__aao_influent_1_1__XTSS",
                "Sumo__Plant__aao_influent_1_1__TP ",        
                "Sumo__Plant__aao_influent_1_1__TKN",        
                "Sumo__Plant__aao_influent_1_1__frSNHx_TKN ",
                "Sumo__Plant__aao_influent_1_1__SNOx",
                "Sumo__Plant__aao_influent_1_1__frSPO4_TP",  
                "Sumo__Plant__aao_influent_1_1__T ",
                #进水比例
                "Sumo__Plant__aao_flowdivider1_1_1__fr1_Q",
                "Sumo__Plant__aao_flowdivider2_1_1__fr1_Q",
                #1-1缺氧池数据
                "Sumo__Plant__aao_pfr_1_1_inf__TBOD_5",
                "Sumo__Plant__aao_pfr_1_1_inf__SNHx",
                "Sumo__Plant__aao_pfr_1_1_inf__SNOx",
                "Sumo__Plant__aao_pfr_1_1_inf__SPO4",
                
                "Sumo__Plant__aao_pfr_1_1_eff__SNHx",
                "Sumo__Plant__aao_pfr_1_1_eff__SNOx",
                "Sumo__Plant__aao_pfr_1_1_eff__SPO4",
                #1-1好氧池数据
                "Sumo__Plant__aao_mlss_1_1__Q",
                "Sumo__Plant__aao_mlss_1_1__TCOD",
                "Sumo__Plant__aao_mlss_1_1__TBOD_5",
                "Sumo__Plant__aao_mlss_1_1__XTSS",
                "Sumo__Plant__aao_mlss_1_1__TP",
                "Sumo__Plant__aao_mlss_1_1__TN",
                "Sumo__Plant__aao_mlss_1_1__SNHx",
                "Sumo__Plant__aao_mlss_1_1__SNOx",
                # "Sumo__Plant__aao_mlss_1_1__pH",
                "Sumo__Plant__aao_mlss_1_1__SPO4",
                #1-1二沉池进水数据
                "Sumo__Plant__aao_clarifier_1_1_inf__Q",
                "Sumo__Plant__aao_clarifier_1_1_inf__TCOD",
                "Sumo__Plant__aao_clarifier_1_1_inf__TBOD_5",
                "Sumo__Plant__aao_clarifier_1_1_inf__XTSS",
                "Sumo__Plant__aao_clarifier_1_1_inf__TP",
                "Sumo__Plant__aao_clarifier_1_1_inf__TN",
                "Sumo__Plant__aao_clarifier_1_1_inf__SNHx",
                "Sumo__Plant__aao_clarifier_1_1_inf__SNOx",
                # "Sumo__Plant__aao_clarifier_1_1_inf__pH",
                "Sumo__Plant__aao_clarifier_1_1_inf__SPO4",
                #1-1外回流
                "Sumo__Plant__aao_ras_1_1__Q",
                "Sumo__Plant__aao_ras_1_1__TCOD",
                "Sumo__Plant__aao_ras_1_1__TBOD_5",
                "Sumo__Plant__aao_ras_1_1__XTSS",
                "Sumo__Plant__aao_ras_1_1__TP",
                "Sumo__Plant__aao_ras_1_1__TN",
                "Sumo__Plant__aao_ras_1_1__SNHx",
                "Sumo__Plant__aao_ras_1_1__SNOx",
                # "Sumo__Plant__aao_ras_1_1__pH",
                "Sumo__Plant__aao_ras_1_1__SPO4",
                #1-1高效沉淀池数据
                "Sumo__Plant__aao_clarifier_3_inf__Q",                            
                "Sumo__Plant__aao_clarifier_3_inf__TCOD",
                "Sumo__Plant__aao_clarifier_3_inf__TBOD_5",
                "Sumo__Plant__aao_clarifier_3_inf__XTSS",
                "Sumo__Plant__aao_clarifier_3_inf__TP",
                "Sumo__Plant__aao_clarifier_3_inf__TN",
                "Sumo__Plant__aao_clarifier_3_inf__SNHx",
                "Sumo__Plant__aao_clarifier_3_inf__SNOx",
                # "Sumo__Plant__aao_clarifier_inf__pH",
                "Sumo__Plant__aao_clarifier_3_inf__SPO4",
                #1-1出水
                "Sumo__Plant__aao_effluent_1_1__Q",
                "Sumo__Plant__aao_effluent_1_1__TCOD",
                "Sumo__Plant__aao_effluent_1_1__TBOD_5",
                "Sumo__Plant__aao_effluent_1_1__XTSS",
                "Sumo__Plant__aao_effluent_1_1__TP",
                "Sumo__Plant__aao_effluent_1_1__TN",
                "Sumo__Plant__aao_effluent_1_1__SNHx",
                "Sumo__Plant__aao_effluent_1_1__SNOx",
                # "Sumo__Plant__aao_effluent_1_1__pH",
                "Sumo__Plant__aao_effluent_1_1__SPO4",
                

                # 运行参数
                
                #1-1碳源
                "Sumo__Plant__aao_carbon_1_1__Q",
                #1-1二层池段的PAC
                "Sumo__Plant__aao_pac_1_1__Q",
                #1-1高效部分的PAC
                "Sumo__Plant__aao_pac_3__Q",
                #1-1内回流
                "Sumo__Plant__aao_flowdivider3_1_1_influx__Qpumped_target",
                #1-1二层池底流
                "Sumo__Plant__aao_clarifier_1_1__Qsludge_target",
                #1-1排泥
                "Sumo__Plant__aao_flowdivider4_1_1_sludge__Qpumped_target",
                #1-1好氧段曝气
                "Sumo__Plant__aao_cstr3_1_1__Qair_NTP",
                "Sumo__Plant__aao_cstr4_1_1__Qair_NTP",
                "Sumo__Plant__aao_cstr5_1_1__Qair_NTP",
                "Sumo__Plant__aao_cstr6_1_1__Qair_NTP",
                "Sumo__Plant__aao_cstr7_1_1__Qair_NTP",
                #1-1好氧段溶解
                "Sumo__Plant__aao_cstr3_1_1__SO2",
                "Sumo__Plant__aao_cstr4_1_1__SO2",
                "Sumo__Plant__aao_cstr5_1_1__SO2",
                "Sumo__Plant__aao_cstr6_1_1__SO2",
                "Sumo__Plant__aao_cstr7_1_1__SO2",
                "Sumo__Plant__aao_cstr8_1_1__SO2",
                #1-1悬浮固体浓度
                "Sumo__Plant__aao_cstr7_1_1__XTSS",
                #1-1泥龄
                "Sumo__Plant__SRT1",
                #1-1比例
                "Sumo__Plant__Ratio1",
                "Sumo__Plant__Ratio2",
                #1-1二层池出水
                "Sumo__Plant__aao_clarifier_1_1_eff__Q",
                "Sumo__Plant__aao_clarifier_1_1_eff__TCOD",
                "Sumo__Plant__aao_clarifier_1_1_eff__TBOD_5",
                "Sumo__Plant__aao_clarifier_1_1_eff__XTSS",
                "Sumo__Plant__aao_clarifier_1_1_eff__TP",
                "Sumo__Plant__aao_clarifier_1_1_eff__TN",
                "Sumo__Plant__aao_clarifier_1_1_eff__SNHx",
                "Sumo__Plant__aao_clarifier_1_1_eff__SNOx",
                #"Sumo__Plant__aao_clarifier_1_1_eff__pH",
                "Sumo__Plant__aao_clarifier_1_1_eff__SPO4",    
            ],[
                #1-2进水
                "Sumo__Plant__aao_influent_1_2__Q",
                "Sumo__Plant__aao_influent_1_2__TCOD ",      
                "Sumo__Plant__aao_influent_1_2__TBOD_5",
                "Sumo__Plant__aao_influent_1_2__XTSS",
                "Sumo__Plant__aao_influent_1_2__TP ",        
                "Sumo__Plant__aao_influent_1_2__TKN",        
                "Sumo__Plant__aao_influent_1_2__frSNHx_TKN ",
                "Sumo__Plant__aao_influent_1_2__SNOx",
                "Sumo__Plant__aao_influent_1_2__frSPO4_TP",  
                "Sumo__Plant__aao_influent_1_2__T ",
                #进水比例
                "Sumo__Plant__aao_flowdivider1_1_2__fr1_Q",
                "Sumo__Plant__aao_flowdivider2_1_2__fr1_Q",
                #1-2缺氧池数据
                "Sumo__Plant__aao_pfr_1_2_inf__TBOD_5",
                "Sumo__Plant__aao_pfr_1_2_inf__SNHx",
                "Sumo__Plant__aao_pfr_1_2_inf__SNOx",
                "Sumo__Plant__aao_pfr_1_2_inf__SPO4",
            
                "Sumo__Plant__aao_pfr_1_2_eff__SNHx",
                "Sumo__Plant__aao_pfr_1_2_eff__SNOx",
                "Sumo__Plant__aao_pfr_1_2_eff__SPO4",
                #1-2好氧池数据
                "Sumo__Plant__aao_mlss_1_2__Q",
                "Sumo__Plant__aao_mlss_1_2__TCOD",
                "Sumo__Plant__aao_mlss_1_2__TBOD_5",
                "Sumo__Plant__aao_mlss_1_2__XTSS",
                "Sumo__Plant__aao_mlss_1_2__TP",
                "Sumo__Plant__aao_mlss_1_2__TN",
                "Sumo__Plant__aao_mlss_1_2__SNHx",
                "Sumo__Plant__aao_mlss_1_2__SNOx",
                # "Sumo__Plant__aao_mlss_1_2__pH",
                "Sumo__Plant__aao_mlss_1_2__SPO4",
                #1-2二沉池进水数据
                "Sumo__Plant__aao_clarifier_1_2_inf__Q",
                "Sumo__Plant__aao_clarifier_1_2_inf__TCOD",
                "Sumo__Plant__aao_clarifier_1_2_inf__TBOD_5",
                "Sumo__Plant__aao_clarifier_1_2_inf__XTSS",
                "Sumo__Plant__aao_clarifier_1_2_inf__TP",
                "Sumo__Plant__aao_clarifier_1_2_inf__TN",
                "Sumo__Plant__aao_clarifier_1_2_inf__SNHx",
                "Sumo__Plant__aao_clarifier_1_2_inf__SNOx",
                # "Sumo__Plant__aao_clarifier_1_2_inf__pH",
                "Sumo__Plant__aao_clarifier_1_2_inf__SPO4",
                #1-2外回流
                "Sumo__Plant__aao_ras_1_2__Q",
                "Sumo__Plant__aao_ras_1_2__TCOD",
                "Sumo__Plant__aao_ras_1_2__TBOD_5",
                "Sumo__Plant__aao_ras_1_2__XTSS",
                "Sumo__Plant__aao_ras_1_2__TP",
                "Sumo__Plant__aao_ras_1_2__TN",
                "Sumo__Plant__aao_ras_1_2__SNHx",
                "Sumo__Plant__aao_ras_1_2__SNOx",
                # "Sumo__Plant__aao_ras_1_2__pH",
                "Sumo__Plant__aao_ras_1_2__SPO4",
                #1-2高效沉淀池数据
                "Sumo__Plant__aao_clarifier_4_inf__Q",                            
                "Sumo__Plant__aao_clarifier_4_inf__TCOD",
                "Sumo__Plant__aao_clarifier_4_inf__TBOD_5",
                "Sumo__Plant__aao_clarifier_4_inf__XTSS",
                "Sumo__Plant__aao_clarifier_4_inf__TP",
                "Sumo__Plant__aao_clarifier_4_inf__TN",
                "Sumo__Plant__aao_clarifier_4_inf__SNHx",
                "Sumo__Plant__aao_clarifier_4_inf__SNOx",
                # "Sumo__Plant__aao_clarifier_inf__pH",
                "Sumo__Plant__aao_clarifier_4_inf__SPO4",
                #1-2出水
                "Sumo__Plant__aao_effluent_1_2__Q",
                "Sumo__Plant__aao_effluent_1_2__TCOD",
                "Sumo__Plant__aao_effluent_1_2__TBOD_5",
                "Sumo__Plant__aao_effluent_1_2__XTSS",
                "Sumo__Plant__aao_effluent_1_2__TP",
                "Sumo__Plant__aao_effluent_1_2__TN",
                "Sumo__Plant__aao_effluent_1_2__SNHx",
                "Sumo__Plant__aao_effluent_1_2__SNOx",
                # "Sumo__Plant__aao_effluent_1_2__pH",
                "Sumo__Plant__aao_effluent_1_2__SPO4",
                

                # 运行参数
                
                #1-2碳源
                "Sumo__Plant__aao_carbon_1_2__Q",
                #1-2二层池段的PAC
                "Sumo__Plant__aao_pac_1_2__Q",
                #1-2高效部分的PAC
                "Sumo__Plant__aao_pac_4__Q",
                #1-2内回流
                "Sumo__Plant__aao_flowdivider3_1_2_influx__Qpumped_target",
                #1-2二层池底流
                "Sumo__Plant__aao_clarifier_1_2__Qsludge_target",
                #1-2排泥
                "Sumo__Plant__aao_flowdivider4_1_2_sludge__Qpumped_target",
                #1-2好氧段曝气
                "Sumo__Plant__aao_cstr3_1_2__Qair_NTP",
                "Sumo__Plant__aao_cstr4_1_2__Qair_NTP",
                "Sumo__Plant__aao_cstr5_1_2__Qair_NTP",
                "Sumo__Plant__aao_cstr6_1_2__Qair_NTP",
                "Sumo__Plant__aao_cstr7_1_2__Qair_NTP",
                #1-2好氧段溶解
                "Sumo__Plant__aao_cstr3_1_2__SO2",
                "Sumo__Plant__aao_cstr4_1_2__SO2",
                "Sumo__Plant__aao_cstr5_1_2__SO2",
                "Sumo__Plant__aao_cstr6_1_2__SO2",
                "Sumo__Plant__aao_cstr7_1_2__SO2",
                "Sumo__Plant__aao_cstr8_1_2__SO2",
                #1-2悬浮固体浓度
                "Sumo__Plant__aao_cstr7_1_2__XTSS",
                #1-2泥龄
                "Sumo__Plant__SRT1",
                #1-2比例
                "Sumo__Plant__Ratio3",
                "Sumo__Plant__Ratio4",
                #1-2二层池出水
                "Sumo__Plant__aao_clarifier_1_2_eff__Q",
                "Sumo__Plant__aao_clarifier_1_2_eff__TCOD",
                "Sumo__Plant__aao_clarifier_1_2_eff__TBOD_5",
                "Sumo__Plant__aao_clarifier_1_2_eff__XTSS",
                "Sumo__Plant__aao_clarifier_1_2_eff__TP",
                "Sumo__Plant__aao_clarifier_1_2_eff__TN",
                "Sumo__Plant__aao_clarifier_1_2_eff__SNHx",
                "Sumo__Plant__aao_clarifier_1_2_eff__SNOx",
                # "Sumo__Plant__aao_clarifier_1_2_eff__pH",
                "Sumo__Plant__aao_clarifier_1_2_eff__SPO4",  
            ],[
                #2-1进水
                "Sumo__Plant__aao_influent_2_1__Q",
                "Sumo__Plant__aao_influent_2_1__TCOD ",      
                "Sumo__Plant__aao_influent_2_1__TBOD_5",
                "Sumo__Plant__aao_influent_2_1__XTSS",
                "Sumo__Plant__aao_influent_2_1__TP ",        
                "Sumo__Plant__aao_influent_2_1__TKN",        
                "Sumo__Plant__aao_influent_2_1__frSNHx_TKN ",
                "Sumo__Plant__aao_influent_2_1__SNOx",
                "Sumo__Plant__aao_influent_2_1__frSPO4_TP",  
                "Sumo__Plant__aao_influent_2_1__T ",
                #进水比例
                "Sumo__Plant__aao_flowdivider1_2_1__fr1_Q",
                "Sumo__Plant__aao_flowdivider2_2_1__fr1_Q",
                #2-1缺氧池数据
                "Sumo__Plant__aao_pfr_2_1_inf__TBOD_5",
                "Sumo__Plant__aao_pfr_2_1_inf__SNHx",
                "Sumo__Plant__aao_pfr_2_1_inf__SNOx",
                "Sumo__Plant__aao_pfr_2_1_inf__SPO4",
                
                "Sumo__Plant__aao_pfr_2_1_eff__SNHx",
                "Sumo__Plant__aao_pfr_2_1_eff__SNOx",
                "Sumo__Plant__aao_pfr_2_1_eff__SPO4",
                #2-1好氧池数据
                "Sumo__Plant__aao_mlss_2_1__Q",
                "Sumo__Plant__aao_mlss_2_1__TCOD",
                "Sumo__Plant__aao_mlss_2_1__TBOD_5",
                "Sumo__Plant__aao_mlss_2_1__XTSS",
                "Sumo__Plant__aao_mlss_2_1__TP",
                "Sumo__Plant__aao_mlss_2_1__TN",
                "Sumo__Plant__aao_mlss_2_1__SNHx",
                "Sumo__Plant__aao_mlss_2_1__SNOx",
                # "Sumo__Plant__aao_mlss_2_1__pH",
                "Sumo__Plant__aao_mlss_2_1__SPO4",
                #2-1二沉池进水数据
                "Sumo__Plant__aao_clarifier_2_1_inf__Q",
                "Sumo__Plant__aao_clarifier_2_1_inf__TCOD",
                "Sumo__Plant__aao_clarifier_2_1_inf__TBOD_5",
                "Sumo__Plant__aao_clarifier_2_1_inf__XTSS",
                "Sumo__Plant__aao_clarifier_2_1_inf__TP",
                "Sumo__Plant__aao_clarifier_2_1_inf__TN",
                "Sumo__Plant__aao_clarifier_2_1_inf__SNHx",
                "Sumo__Plant__aao_clarifier_2_1_inf__SNOx",
                # "Sumo__Plant__aao_clarifier_2_1_inf__pH",
                "Sumo__Plant__aao_clarifier_2_1_inf__SPO4",
                #2-1外回流
                "Sumo__Plant__aao_ras_2_1__Q",
                "Sumo__Plant__aao_ras_2_1__TCOD",
                "Sumo__Plant__aao_ras_2_1__TBOD_5",
                "Sumo__Plant__aao_ras_2_1__XTSS",
                "Sumo__Plant__aao_ras_2_1__TP",
                "Sumo__Plant__aao_ras_2_1__TN",
                "Sumo__Plant__aao_ras_2_1__SNHx",
                "Sumo__Plant__aao_ras_2_1__SNOx",
                # "Sumo__Plant__aao_ras_2_1__pH",
                "Sumo__Plant__aao_ras_2_1__SPO4",
                #2-1高效沉淀池数据
                "Sumo__Plant__aao_clarifier_5_inf__Q",                            
                "Sumo__Plant__aao_clarifier_5_inf__TCOD",
                "Sumo__Plant__aao_clarifier_5_inf__TBOD_5",
                "Sumo__Plant__aao_clarifier_5_inf__XTSS",
                "Sumo__Plant__aao_clarifier_5_inf__TP",
                "Sumo__Plant__aao_clarifier_5_inf__TN",
                "Sumo__Plant__aao_clarifier_5_inf__SNHx",
                "Sumo__Plant__aao_clarifier_5_inf__SNOx",
                # "Sumo__Plant__aao_clarifier_inf__pH",
                "Sumo__Plant__aao_clarifier_5_inf__SPO4",
                #2-1出水
                "Sumo__Plant__aao_effluent_2_1__Q",
                "Sumo__Plant__aao_effluent_2_1__TCOD",
                "Sumo__Plant__aao_effluent_2_1__TBOD_5",
                "Sumo__Plant__aao_effluent_2_1__XTSS",
                "Sumo__Plant__aao_effluent_2_1__TP",
                "Sumo__Plant__aao_effluent_2_1__TN",
                "Sumo__Plant__aao_effluent_2_1__SNHx",
                "Sumo__Plant__aao_effluent_2_1__SNOx",
                # "Sumo__Plant__aao_effluent_2_1__pH",
                "Sumo__Plant__aao_effluent_2_1__SPO4",
                

                # 运行参数
                
                #2-1碳源
                "Sumo__Plant__aao_carbon_2_1__Q",
                #2-1二层池段的PAC
                "Sumo__Plant__aao_pac_2_1__Q",
                #2-1高效部分的PAC
                "Sumo__Plant__aao_pac_5__Q",
                #2-1内回流
                "Sumo__Plant__aao_flowdivider3_2_1_influx__Qpumped_target",
                #2-1二层池底流
                "Sumo__Plant__aao_clarifier_2_1__Qsludge_target",
                #2-1排泥
                "Sumo__Plant__aao_flowdivider4_2_1_sludge__Qpumped_target",
                #2-1好氧段曝气
                "Sumo__Plant__aao_cstr3_2_1__Qair_NTP",
                "Sumo__Plant__aao_cstr4_2_1__Qair_NTP",
                "Sumo__Plant__aao_cstr5_2_1__Qair_NTP",
                "Sumo__Plant__aao_cstr6_2_1__Qair_NTP",
                "Sumo__Plant__aao_cstr7_2_1__Qair_NTP",
                #2-1好氧段溶解
                "Sumo__Plant__aao_cstr3_2_1__SO2",
                "Sumo__Plant__aao_cstr4_2_1__SO2",
                "Sumo__Plant__aao_cstr5_2_1__SO2",
                "Sumo__Plant__aao_cstr6_2_1__SO2",
                "Sumo__Plant__aao_cstr7_2_1__SO2",
                "Sumo__Plant__aao_cstr8_2_1__SO2",
                #2-1悬浮固体浓度
                "Sumo__Plant__aao_cstr7_2_1__XTSS",
                #2-1泥龄
                "Sumo__Plant__SRT2",
                #2-1比例
                "Sumo__Plant__Ratio5",
                "Sumo__Plant__Ratio6",
                #2-1二层池出水
                "Sumo__Plant__aao_clarifier_2_1_eff__Q",
                "Sumo__Plant__aao_clarifier_2_1_eff__TCOD",
                "Sumo__Plant__aao_clarifier_2_1_eff__TBOD_5",
                "Sumo__Plant__aao_clarifier_2_1_eff__XTSS",
                "Sumo__Plant__aao_clarifier_2_1_eff__TP",
                "Sumo__Plant__aao_clarifier_2_1_eff__TN",
                "Sumo__Plant__aao_clarifier_2_1_eff__SNHx",
                "Sumo__Plant__aao_clarifier_2_1_eff__SNOx",
                # "Sumo__Plant__aao_clarifier_2_1_eff__pH",
                "Sumo__Plant__aao_clarifier_2_1_eff__SPO4",  
            ],[
                #2-2进水
                "Sumo__Plant__aao_influent_2_2__Q",
                "Sumo__Plant__aao_influent_2_2__TCOD ",      
                "Sumo__Plant__aao_influent_2_2__TBOD_5",
                "Sumo__Plant__aao_influent_2_2__XTSS",
                "Sumo__Plant__aao_influent_2_2__TP ",        
                "Sumo__Plant__aao_influent_2_2__TKN",        
                "Sumo__Plant__aao_influent_2_2__frSNHx_TKN ",
                "Sumo__Plant__aao_influent_2_2__SNOx",
                "Sumo__Plant__aao_influent_2_2__frSPO4_TP",  
                "Sumo__Plant__aao_influent_2_2__T ",
                #进水比例
                "Sumo__Plant__aao_flowdivider1_2_2__fr1_Q",
                "Sumo__Plant__aao_flowdivider2_2_2__fr1_Q",
                #2-2缺氧池数据
                "Sumo__Plant__aao_pfr_2_2_inf__TBOD_5",
                "Sumo__Plant__aao_pfr_2_2_inf__SNHx",
                "Sumo__Plant__aao_pfr_2_2_inf__SNOx",
                "Sumo__Plant__aao_pfr_2_2_inf__SPO4",
                
                "Sumo__Plant__aao_pfr_2_2_eff__SNHx",
                "Sumo__Plant__aao_pfr_2_2_eff__SNOx",
                "Sumo__Plant__aao_pfr_2_2_eff__SPO4",
                #2-2好氧池数据
                "Sumo__Plant__aao_mlss_2_2__Q",
                "Sumo__Plant__aao_mlss_2_2__TCOD",
                "Sumo__Plant__aao_mlss_2_2__TBOD_5",
                "Sumo__Plant__aao_mlss_2_2__XTSS",
                "Sumo__Plant__aao_mlss_2_2__TP",
                "Sumo__Plant__aao_mlss_2_2__TN",
                "Sumo__Plant__aao_mlss_2_2__SNHx",
                "Sumo__Plant__aao_mlss_2_2__SNOx",
                # "Sumo__Plant__aao_mlss_2_2__pH",
                "Sumo__Plant__aao_mlss_2_2__SPO4",
                #2-2二沉池进水数据
                "Sumo__Plant__aao_clarifier_2_2_inf__Q",
                "Sumo__Plant__aao_clarifier_2_2_inf__TCOD",
                "Sumo__Plant__aao_clarifier_2_2_inf__TBOD_5",
                "Sumo__Plant__aao_clarifier_2_2_inf__XTSS",
                "Sumo__Plant__aao_clarifier_2_2_inf__TP",
                "Sumo__Plant__aao_clarifier_2_2_inf__TN",
                "Sumo__Plant__aao_clarifier_2_2_inf__SNHx",
                "Sumo__Plant__aao_clarifier_2_2_inf__SNOx",
                # "Sumo__Plant__aao_clarifier_2_2_inf__pH",
                "Sumo__Plant__aao_clarifier_2_2_inf__SPO4",
                #2-2外回流
                "Sumo__Plant__aao_ras_2_2__Q",
                "Sumo__Plant__aao_ras_2_2__TCOD",
                "Sumo__Plant__aao_ras_2_2__TBOD_5",
                "Sumo__Plant__aao_ras_2_2__XTSS",
                "Sumo__Plant__aao_ras_2_2__TP",
                "Sumo__Plant__aao_ras_2_2__TN",
                "Sumo__Plant__aao_ras_2_2__SNHx",
                "Sumo__Plant__aao_ras_2_2__SNOx",
                # "Sumo__Plant__aao_ras_2_2__pH",
                "Sumo__Plant__aao_ras_2_2__SPO4",
                #2-2高效沉淀池数据
                "Sumo__Plant__aao_clarifier_6_inf__Q",                            
                "Sumo__Plant__aao_clarifier_6_inf__TCOD",
                "Sumo__Plant__aao_clarifier_6_inf__TBOD_5",
                "Sumo__Plant__aao_clarifier_6_inf__XTSS",
                "Sumo__Plant__aao_clarifier_6_inf__TP",
                "Sumo__Plant__aao_clarifier_6_inf__TN",
                "Sumo__Plant__aao_clarifier_6_inf__SNHx",
                "Sumo__Plant__aao_clarifier_6_inf__SNOx",
                # "Sumo__Plant__aao_clarifier_inf__pH",
                "Sumo__Plant__aao_clarifier_6_inf__SPO4",
                #2-2出水
                "Sumo__Plant__aao_effluent_2_2__Q",
                "Sumo__Plant__aao_effluent_2_2__TCOD",
                "Sumo__Plant__aao_effluent_2_2__TBOD_5",
                "Sumo__Plant__aao_effluent_2_2__XTSS",
                "Sumo__Plant__aao_effluent_2_2__TP",
                "Sumo__Plant__aao_effluent_2_2__TN",
                "Sumo__Plant__aao_effluent_2_2__SNHx",
                "Sumo__Plant__aao_effluent_2_2__SNOx",
                # "Sumo__Plant__aao_effluent_2_2__pH",
                "Sumo__Plant__aao_effluent_2_2__SPO4",
                

                # 运行参数
                
                #2-2碳源
                "Sumo__Plant__aao_carbon_2_2__Q",
                #2-2二层池段的PAC
                "Sumo__Plant__aao_pac_2_2__Q",
                #2-2高效部分的PAC
                "Sumo__Plant__aao_pac_6__Q",
                #2-2内回流
                "Sumo__Plant__aao_flowdivider3_2_2_influx__Qpumped_target",
                #2-2二层池底流
                "Sumo__Plant__aao_clarifier_2_2__Qsludge_target",
                #2-2排泥
                "Sumo__Plant__aao_flowdivider4_2_2_sludge__Qpumped_target",
                #2-2好氧段曝气
                "Sumo__Plant__aao_cstr3_2_2__Qair_NTP",
                "Sumo__Plant__aao_cstr4_2_2__Qair_NTP",
                "Sumo__Plant__aao_cstr5_2_2__Qair_NTP",
                "Sumo__Plant__aao_cstr6_2_2__Qair_NTP",
                "Sumo__Plant__aao_cstr7_2_2__Qair_NTP",
                #2-2好氧段溶解
                "Sumo__Plant__aao_cstr3_2_2__SO2",
                "Sumo__Plant__aao_cstr4_2_2__SO2",
                "Sumo__Plant__aao_cstr5_2_2__SO2",
                "Sumo__Plant__aao_cstr6_2_2__SO2",
                "Sumo__Plant__aao_cstr7_2_2__SO2",
                "Sumo__Plant__aao_cstr8_2_2__SO2",
                #2-2悬浮固体浓度
                "Sumo__Plant__aao_cstr7_2_2__XTSS",
                #2-2泥龄
                "Sumo__Plant__SRT2",
                #2-2比例
                "Sumo__Plant__Ratio7",
                "Sumo__Plant__Ratio8",
                #2-2二层池出水
                "Sumo__Plant__aao_clarifier_2_2_eff__Q",
                "Sumo__Plant__aao_clarifier_2_2_eff__TCOD",
                "Sumo__Plant__aao_clarifier_2_2_eff__TBOD_5",
                "Sumo__Plant__aao_clarifier_2_2_eff__XTSS",
                "Sumo__Plant__aao_clarifier_2_2_eff__TP",
                "Sumo__Plant__aao_clarifier_2_2_eff__TN",
                "Sumo__Plant__aao_clarifier_2_2_eff__SNHx",
                "Sumo__Plant__aao_clarifier_2_2_eff__SNOx",
                # "Sumo__Plant__aao_clarifier_2_2_eff__pH",
                "Sumo__Plant__aao_clarifier_2_2_eff__SPO4",  
            ]
        ]

        if water_line_no==-1:
            jobs = [ds.sumo.schedule(
                model=self.model[i],  # 载入模型文件
                commands=commands[i],  # 启动模拟
                variables=variables[i],
                jobData={
                "data": {}, ds.sumo.persistent: True
                }
            ) for i in range(4)]
        else:
            jobs = ds.sumo.schedule(
                model=self.model[water_line_no],  # 载入模型文件
                commands=commands[water_line_no],  # 启动模拟
                variables=variables[water_line_no],
                jobData={
                "data": {}, ds.sumo.persistent: True
                }
            )
        
        
        # 等待模拟结束
        aao_optimize_stop_flag_path = os.path.join(os.getcwd(), f"DigitalTwinAPI\\aao_optimize_stop_flag") 
        aao_basic_schedule_stop_flag_path = os.path.join(os.getcwd(), f"DigitalTwinAPI\\aao_basic_schedule_stop_flag")
        aao_manual_simulate_stop_flag_path = os.path.join(os.getcwd(), f"DigitalTwinAPI\\aao_manual_simulate_stop_flag")

        running_ok_flag = True
        while ds.sumo.scheduledJobs:
            time.sleep(0.1)
            if (self.taskname == "aao_auto_optimize" and os.path.exists(aao_optimize_stop_flag_path)) \
                or (self.taskname == "aao_basic_schedule" and os.path.exists(aao_basic_schedule_stop_flag_path)) \
                or (self.taskname == "aao_manual_simulate" and os.path.exists(aao_manual_simulate_stop_flag_path)): # 这里需要调整为更加具体任务做匹配
                if water_line_no==-1:
                    for i in range(4):
                        ds.sumo.sendCommand(jobs[i], "stop")
                else:
                    ds.sumo.sendCommand(jobs, "stop")
                    
                ds.sumo.cleanup()
                running_ok_flag = False 
                break
        if not running_ok_flag:
            if result_dic is None:
                raise KeyboardInterrupt
            result_dic["sumo_run_result"] = "stopped" #增加一个sumo 运行结果字段，用于标识在进程中执行有没有外部触发的停止
            return
        if water_line_no==-1:
            final_output = self.total_result(ds.sumo.jobData[jobs[0]]["data"], ds.sumo.jobData[jobs[1]]["data"], ds.sumo.jobData[jobs[2]]["data"],
                                   ds.sumo.jobData[jobs[3]]["data"])
        else:
            final_output = ds.sumo.jobData[jobs]["data"]
            final_output = self.rename_keys(final_output, f"Sumo__Plant__SRT{int((water_line_no//2)+1)}", f"Sumo__Plant__SRT{water_line_no+1}")

        ds.sumo.cleanup()  # 清除sumo的任务规划器
        if result_dic == None:  
                return final_output
        else:
            for key,value in final_output.items():
                result_dic[key] = value
            result_dic["sumo_run_result"] = "success"

    def new_single_process_rum_sim(self,state_file, influent_data, cmdline=None): 
        '''
            cmdline默认为空 即只进行输入固定值的模拟
            如果是优化调用的, 因为参数在不断变化, 为了使得单次模拟模块还能应用, 采用解引用的方法
        '''
        logger.info("new_single_process_rum_sim start") 
        result_dic = mp.Manager().dict()
        process = mp.Process(target=self.run_sim, args=(state_file, influent_data,result_dic, cmdline))
        process.start()
        process.join()
        logger.info("new_single_process_rum_sim process.join()") 
        final_output = result_dic.copy()
        return final_output

    def rename_keys(self, old_dict, key_old_name, key_new_name):
        '''
        把旧字典中某个键名字改成另外一个新名字, 并返回这个重新构造的新字典
        :param old_dict: 原始字典
        :param key_old_name: 需要改动的旧键名
        :param key_new_name: 想要构造的新键名
        :return: 一个新字典, 其中的键名进行了重命名
        '''
        
        new_dict = {}
        for old_key, value in old_dict.items():
            if old_key == key_old_name:
                new_dict[key_new_name] = value
            else:
                # 其余保留
                new_dict[old_key] = value
        return new_dict
        
    # 融合各路输出, 不需要计算 注意:由于四条路的SRT分别命名为SRT1、SRT1、SRT2、SRT2 所以必须手动区分开
    def total_result(self, aao1_output, aao2_output, aao3_output, aao4_output):
       
        # 首先需要重命名 SRT
        new_aao2_output = self.rename_keys(aao2_output, "Sumo__Plant__SRT1", "Sumo__Plant__SRT2")
        new_aao3_output = self.rename_keys(aao3_output, "Sumo__Plant__SRT2", "Sumo__Plant__SRT3")
        new_aao4_output = self.rename_keys(aao4_output, "Sumo__Plant__SRT2", "Sumo__Plant__SRT4")
        # 使用字典解包的方式合并字典(如果有相同的键，则后面的字典中的值会覆盖前面的值)
        total_data = {**aao1_output, **new_aao2_output, **new_aao3_output, **new_aao4_output}      
        return total_data

    def cal_effluent(self, state_file, influent_data, cmdline=None):
        final_output = self.run_sim(state_file, influent_data, result_dic=None, cmdline=cmdline)
        eff_var = [
            "Sumo__Plant__aao_effluent_{}_TCOD",
            "Sumo__Plant__aao_effluent_{}_TBOD_5",
            "Sumo__Plant__aao_effluent_{}_XTSS",
            "Sumo__Plant__aao_effluent_{}_TP",
            "Sumo__Plant__aao_effluent_{}_TN",
            "Sumo__Plant__aao_effluent_{}_SNHx",
            "Sumo__Plant__aao_effluent_{}_SNOx",
            # "Sumo__Plant__aao_effluent_{idx}_pH",
            "Sumo__Plant__aao_effluent_{}_SPO4",
        ]

        idx = ['1_1_', '1_2_', '2_1_', '2_2_']
        eff_Q_list = [final_output[f'Sumo__Plant__aao_effluent_{i}_Q'] for i in idx]
        all_Q = sum(eff_Q_list)
        final_output['Sumo__Plant__aao_effluent__Q'] = all_Q
        for var in eff_var:
            final_output[var.format('')] = sum([eff_Q_list[i]*final_output[var.format(idx[i])] for i in range(4)])/all_Q
        return final_output
