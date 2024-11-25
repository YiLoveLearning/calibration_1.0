# 查看xml中对应出水的状态信息，用于评估初始状态校准的准确性
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import datetime
######################## 参数配置 ########################
if __name__ == "__main__":
    version_type = ["my", "xueqian", "24_7", "24_10", "mbr"]    # 学谦的sumo版本为22.0.0, 模型命名也不同
    version = version_type[3]
    if version=="my":
        xml_format = "../../data/OfflineStates/state{}_init.xml"    # 校准后的初始状态文件
        # xml_format = "../../data/OfflineStates/22.1.0dll_23/state{}_init.xml"    # 校准后的初始状态文件
        effluent_data = ["../../data/OfflineData/2022-07-07-2024-01-05_effluent_data.xlsx", 
                     "../../data/OfflineData/2022-07-07-2024-01-05_operating_data.xlsx"]
        ct = datetime.datetime(2022, 9, 6, 1)
        # ct = datetime.datetime(2023, 4, 11, 1)
    elif version=="xueqian":
        xml_format = "../../output/fwq230420state/state{}_2022-09-06-01-00.xml"    # 校准后的初始状态文件
        # xml_format = "../../output/fwq230420state/state{}.xml"    # 校准后的初始状态文件
        effluent_data = ["../../data/OfflineData/2022-07-07-2024-01-05_effluent_data.xlsx", 
                     "../../data/OfflineData/2022-07-07-2024-01-05_operating_data.xlsx"]
        ct = datetime.datetime(2022, 9, 6, 1)
        # ct = datetime.datetime(2023, 4, 11, 1)
    elif version=="24_7":
        # xml_format = "../../data/OfflineStates/22.1.0dll_24_7/V4.3-{}.xml"    # 校准后的初始状态文件
        xml_format = "../../data/OfflineStates/22.1.0dll_24_7/state{}_init1.xml"    # 校准后的初始状态文件
        # xml_format = "../../data/OfflineStates/22.1.0dll_24_7/state{}.xml"    # 校准后的初始状态文件
        # xml_format = "../../data/OfflineStates/22.1.0dll_24_7/state{}_2024-04-08-01-00.xml"    # 校准后的初始状态文件
        effluent_data = ["../../data/OfflineData/2024-04-08-2024-06-28_effluent_data.xlsx", 
                     "../../data/OfflineData/2024-04-08-2024-06-28_operating_data.xlsx"]
        ct = datetime.datetime(2024, 4, 8, 1)
    elif version=="24_10":
        # xml_format = 'D:/ysq/drr/init_state/output/0.01_0.05state{}_init.xml'    # 校准后的初始状态文件
        xml_format = 'D:/ysq/drr/init_state/output/0.1_0.05state{}_init.xml'    # 校准后的初始状态文件
        effluent_data = ["../../data/OfflineData/2024-08-27-2024-09-23_effluent_data.xlsx", 
                     "../../data/OfflineData/2024-08-27-2024-09-23_cleaning_data.xlsx"]
        ct = datetime.datetime(2024, 8, 27, 1)
    elif version=="mbr":
        xml_format = "D:/cmm/DT_Self_Calibration/data/22.1.0dll_24_8_mbr_tmp/after_7/state{}_init.xml"    # 校准后的初始状态文件
        effluent_data = ["D:/cmm/DT_Self_Calibration/data/OfflineData/2024-04-08-2024-06-28_mbr_effluent_data.xlsx", 
                     "D:/cmm/DT_Self_Calibration/data/OfflineData/2024-03-09-2024-06-28_mbr_operating_data.xlsx"]
        ct = datetime.datetime(2024, 4, 8, 1)
#########################################################


def get_xml_value(xml_file, xml_name):
    """根据一个xml_name获取xml_file中的一个值(应该是str)"""
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 找到具有指定 name 属性的 <real> 元素
    real_element = root.find(f".//real[@name='{xml_name}']")
    if real_element is not None:
        # 找到 <value> 元素并获取其文本内容
        value_element = real_element.find('value')
        if value_element is not None:
            return float(value_element.text)
        else:
            raise ValueError(f"The <value> element is not found inside the <real> element with name '{xml_name}'.")
    else:
        raise ValueError(f"The <real> element with name '{xml_name}' is not found in the XML file.")

def get_xml_value_dict(xml_file, xml_name_list, rename_list):
    """根据xml_name_list获取xml_file中一系列值, 并用rename_list命名, 返回dict"""
    res = {}
    try:
        for i, name in enumerate(xml_name_list):
            real_value = get_xml_value(xml_file, name)
            res.update({rename_list[i]: real_value})
            print(f"The value of the '{name}' element is: {real_value}")
    except Exception as e:
        print(f"Error: {e}")
    return res

def get_true_value_dict(xlsx_file, ct, real_name_list, rename_list):
    if isinstance(xlsx_file, str):
        # 输入参数是单个 Excel 文件路径,直接读取
        df = pd.read_excel(xlsx_file, parse_dates=['sampling_time'])
    elif isinstance(xlsx_file, list):
        # 输入参数是 Excel 文件路径列表,读取并合并
        df = pd.read_excel(xlsx_file[0], parse_dates=['sampling_time'])
        for file_path in xlsx_file[1:]:
            dfs = pd.read_excel(file_path, parse_dates=['sampling_time'])
            df = pd.merge(df, dfs, on='sampling_time', how='outer')
    df = df[df['sampling_time']==ct]
    res = {}
    try:
        for i, name in enumerate(real_name_list):
            real_value = df[name].iloc[0]
            res.update({rename_list[i]: real_value})
            print(f"The value of the '{name}' element is: {real_value}")
    except Exception as e:
        print(f"Error: {e}")
    return res

# 主程序
if __name__ == "__main__":
    output_path = (Path('../../output')/'utils' / Path(__file__).stem)    # 可复现, 只用跑一次完整的结果
    output_path.mkdir(parents=True, exist_ok=True)
    xml_file_list = [xml_format.format(i) for i in range(1, 5)]
    rename_list = ['SNHx', 'TP', 'XTSS', 'TCOD', 'TN', 'MLSS']
    if version=="my":
        name_list = [['Sumo__Plant__Effluent2__SNHx',  # 出水NH4
                    'Sumo__Plant__Effluent2__TP',  # 出水TP
                    'Sumo__Plant__Effluent2__XTSS',  # 出水SS
                    'Sumo__Plant__Effluent2__TCOD',
                    'Sumo__Plant__Effluent2__TN',
                    'Sumo__Plant__CSTR1_7__XTSS',
        ],[
                    'Sumo__Plant__Effluent__SNHx',  # 出水NH4
                    'Sumo__Plant__Effluent__TP',  # 出水TP
                    'Sumo__Plant__Effluent__XTSS',  # 出水SS
                    'Sumo__Plant__Effluent__TCOD',
                    'Sumo__Plant__Effluent__TN',
                    'Sumo__Plant__CSTR2_7__XTSS'
        ],[
                    'Sumo__Plant__Effluent__SNHx',  # 出水NH4
                    'Sumo__Plant__Effluent__TP',  # 出水TP
                    'Sumo__Plant__Effluent__XTSS',  # 出水SS
                    'Sumo__Plant__Effluent__TCOD',
                    'Sumo__Plant__Effluent__TN',
                    'Sumo__Plant__CSTR3_7__XTSS'
        ],[
                    'Sumo__Plant__Effluent__SNHx',  # 出水NH4
                    'Sumo__Plant__Effluent__TP',  # 出水TP
                    'Sumo__Plant__Effluent__XTSS',  # 出水SS
                    'Sumo__Plant__Effluent__TCOD',
                    'Sumo__Plant__Effluent__TN',
                    'Sumo__Plant__CSTR4_7__XTSS'
        ]]
    elif version=="xueqian" or version=="24_7":
        name_list = [['Sumo__Plant__aao_effluent_1__SNHx',  # 出水NH4
                    'Sumo__Plant__aao_effluent_1__TP',  # 出水TP
                    'Sumo__Plant__aao_effluent_1__XTSS',  # 出水SS
                    'Sumo__Plant__aao_effluent_1__TCOD',
                    'Sumo__Plant__aao_effluent_1__TN',
                    'Sumo__Plant__aao_cstr7_1_1__XTSS', # MLSS(仪表装在这, MLSS约等于TSS)
        ],[
                    'Sumo__Plant__aao_effluent_2__SNHx',  # 出水NH4
                    'Sumo__Plant__aao_effluent_2__TP',  # 出水TP
                    'Sumo__Plant__aao_effluent_2__XTSS',  # 出水SS
                    'Sumo__Plant__aao_effluent_2__TCOD',
                    'Sumo__Plant__aao_effluent_2__TN',
                    'Sumo__Plant__aao_cstr7_1_2__XTSS', # MLSS
        ],[
                    'Sumo__Plant__aao_effluent_3__SNHx',  # 出水NH4
                    'Sumo__Plant__aao_effluent_3__TP',  # 出水TP
                    'Sumo__Plant__aao_effluent_3__XTSS',  # 出水SS
                    'Sumo__Plant__aao_effluent_3__TCOD',
                    'Sumo__Plant__aao_effluent_3__TN',
                    'Sumo__Plant__aao_cstr7_2_1__XTSS', # MLSS
        ],[
                    'Sumo__Plant__aao_effluent_4__SNHx',  # 出水NH4
                    'Sumo__Plant__aao_effluent_4__TP',  # 出水TP
                    'Sumo__Plant__aao_effluent_4__XTSS',  # 出水SS
                    'Sumo__Plant__aao_effluent_4__TCOD',
                    'Sumo__Plant__aao_effluent_4__TN',
                    'Sumo__Plant__aao_cstr7_2_2__XTSS', # MLSS
        ]]
    elif version=="24_10":
        name_list = [[
                    'Sumo__Plant__aao_effluent_1_1__SNHx',  # 出水NH4
                    'Sumo__Plant__aao_effluent_1_1__TP',  # 出水TP
                    'Sumo__Plant__aao_effluent_1_1__XTSS',  # 出水SS
                    'Sumo__Plant__aao_effluent_1_1__TCOD',
                    'Sumo__Plant__aao_effluent_1_1__TN',
                    'Sumo__Plant__aao_cstr7_1_1__XTSS', # MLSS(仪表装在这, MLSS约等于TSS)
        ],[
                    'Sumo__Plant__aao_effluent_1_2__SNHx',  # 出水NH4
                    'Sumo__Plant__aao_effluent_1_2__TP',  # 出水TP
                    'Sumo__Plant__aao_effluent_1_2__XTSS',  # 出水SS
                    'Sumo__Plant__aao_effluent_1_2__TCOD',
                    'Sumo__Plant__aao_effluent_1_2__TN',
                    'Sumo__Plant__aao_cstr7_1_2__XTSS', # MLSS
        ],[
                    'Sumo__Plant__aao_effluent_2_1__SNHx',  # 出水NH4
                    'Sumo__Plant__aao_effluent_2_1__TP',  # 出水TP
                    'Sumo__Plant__aao_effluent_2_1__XTSS',  # 出水SS
                    'Sumo__Plant__aao_effluent_2_1__TCOD',
                    'Sumo__Plant__aao_effluent_2_1__TN',
                    'Sumo__Plant__aao_cstr7_2_1__XTSS', # MLSS
        ],[
                    'Sumo__Plant__aao_effluent_2_2__SNHx',  # 出水NH4
                    'Sumo__Plant__aao_effluent_2_2__TP',  # 出水TP
                    'Sumo__Plant__aao_effluent_2_2__XTSS',  # 出水SS
                    'Sumo__Plant__aao_effluent_2_2__TCOD',
                    'Sumo__Plant__aao_effluent_2_2__TN',
                    'Sumo__Plant__aao_cstr7_2_2__XTSS', # MLSS
        ]]
    elif version=="mbr":
        name_list = [[
                    'Sumo__Plant__mbr_effluent_1_1__SNHx',  # 出水NH4
                    'Sumo__Plant__mbr_effluent_1_1__TP',  # 出水TP
                    'Sumo__Plant__mbr_effluent_1_1__XTSS',  # 出水SS
                    'Sumo__Plant__mbr_effluent_1_1__TCOD',
                    'Sumo__Plant__mbr_effluent_1_1__TN',
                    'Sumo__Plant__mbr_cstr2_8_1_1__XTSS', # MLSS(仪表装在这, MLSS约等于TSS)
        ],[
                    'Sumo__Plant__mbr_effluent_1_2__SNHx',  # 出水NH4
                    'Sumo__Plant__mbr_effluent_1_2__TP',  # 出水TP
                    'Sumo__Plant__mbr_effluent_1_2__XTSS',  # 出水SS
                    'Sumo__Plant__mbr_effluent_1_2__TCOD',
                    'Sumo__Plant__mbr_effluent_1_2__TN',
                    'Sumo__Plant__mbr_cstr2_8_1_2__XTSS', # MLSS
        ],[
                    'Sumo__Plant__mbr_effluent_2_1__SNHx',  # 出水NH4
                    'Sumo__Plant__mbr_effluent_2_1__TP',  # 出水TP
                    'Sumo__Plant__mbr_effluent_2_1__XTSS',  # 出水SS
                    'Sumo__Plant__mbr_effluent_2_1__TCOD',
                    'Sumo__Plant__mbr_effluent_2_1__TN',
                    'Sumo__Plant__mbr_cstr2_8_2_1__XTSS', # MLSS
        ],[
                    'Sumo__Plant__mbr_effluent_2_2__SNHx',  # 出水NH4
                    'Sumo__Plant__mbr_effluent_2_2__TP',  # 出水TP
                    'Sumo__Plant__mbr_effluent_2_2__XTSS',  # 出水SS
                    'Sumo__Plant__mbr_effluent_2_2__TCOD',
                    'Sumo__Plant__mbr_effluent_2_2__TN',
                    'Sumo__Plant__mbr_cstr2_8_2_2__XTSS', # MLSS
        ]]
        
    real_name_list1 = [
        'effluent_secpump_snhx',
        'effluent_dis1_tp', 
        'effluent_dis1_ss',
        'effluent_tol_cod',
        'effluent_dis1_tn',
    ]
    real_name_list = [[
            *real_name_list1,
            "aao1_1_mlss",
            # "mbr1_1_mlss",
    ],[
            *real_name_list1,
            "aao1_2_mlss",
            # "mbr1_2_mlss",
    ],[
            *real_name_list1,
            "aao2_1_mlss",
            # "mbr2_1_mlss",
    ],[
            *real_name_list1,
            "aao2_2_mlss",
            # "mbr2_2_mlss",
    ]]
    res = []
    for i in range(4):
        res.append(get_xml_value_dict(xml_file_list[i], name_list[i], rename_list))
    res = pd.DataFrame(res)
    real_res = []
    for i in range(4):
        real_res.append(get_true_value_dict(effluent_data, ct, real_name_list[i], rename_list))
    real_res = pd.DataFrame(real_res)
    df_comparison = res.add_suffix('_init').join(real_res.add_suffix('_real')).sort_index(axis=1)

    column_means = df_comparison.mean()
    # 将每列的平均值添加到 DataFrame 的最后一行
    df_comparison.loc[len(df_comparison)] = column_means
    # 保存到 Excel 文件
    df_comparison.to_excel(output_path/'combined_results.xlsx', index=False)
    #######################

