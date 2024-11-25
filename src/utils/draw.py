import pandas as pd
import copy
import matplotlib.pyplot as plt
from pathlib import Path

def draw_same_column(df_list, df_label, out_name, out_path, pic_name, 
                     x_name=None, one_file=False, marker_list=None, figsize=(12, 4)):
    """把多个df同名的列画到一张折线图上(可选全部图拼接成一张图或者多个图)
    
    Args:
        df_list (list) - DataFrame 列表
        df_label (list) - 每个df给个标签(因为列名是一样的无法区分)
        out_name (list) - 待绘制的列名(df可能有很多, 但不是所有都想画)  # [ ] TODO: None为全画
        out_path (str) - 图表保存路径
        pic_name (str) - 折线图名称(一张图)或折线图名称前缀(多张图)
        x_name (str) - x轴列名
        one_file (bool) - 是否保存到一个文件中
        marker_list (list) - 点的样式列表, 按顺序和df_list对应, 未指定的用默认样式
            可取样式: ['o', '.', 'v', '^', '<', '>', 's', 'p', '*', 
                      'h', 'H', '+', 'x', 'D', 'd', '|', '_', Unicode]
        
    Examples:
        >>> df1 = pd.read_excel('result1.xlsx')
        >>> df2 = pd.read_excel('result2.xlsx')
        >>> df3 = pd.read_excel('result3.xlsx')
        >>> df_list = [df1, df2, df3]
        >>> df_label = ['df1', 'df2', 'df3']
        >>> out_name = ['col1', 'col2']
        >>> draw_same_column(df_list, df_label, out_name, '../../output', 'result')
    """
    out_path = Path(out_path)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    # 设置默认样式(折线图)
    markers = {label:'-' for label in df_label}
    if marker_list is not None:
        for i, marker in enumerate(marker_list):
            markers[df_label[i]] = marker
    df_list_copy = copy.deepcopy(df_list)
    for i, df in enumerate(df_list_copy):
        df["file"] = df_label[i]
    # 合并DataFrame
    merged_df = pd.concat(df_list_copy)

    if one_file:
        fig, axes = plt.subplots(nrows=len(out_name), ncols=1, figsize=figsize)
        for i, name in enumerate(out_name): # 遍历每个列名
            for file, group in merged_df.groupby("file"):   # 遍历每个df
                if x_name is not None and group.index.name != x_name:
                    axes[i].plot(pd.to_datetime(group[x_name]), group[name], markers[file], label=file)
                else:
                    axes[i].plot(group[name], markers[file], label=file)

            axes[i].legend()
            axes[i].set_title(name)
            # axes[i].axvline(x=pd.to_datetime(group["sampling_time"].iloc[7*12 - 1]), color='r', linestyle='--')

        # 调整子图之间的间距
        plt.tight_layout()
        # 显示图表
        plt.savefig(Path(out_path) / f'{pic_name}.png')
    else:
        for i, name in enumerate(out_name):
            plt.figure(figsize=figsize)
            plt.subplots_adjust(left=0.04, right=0.97, top=0.9, bottom=0.1)
            for file, group in merged_df.groupby("file"):
                if x_name is not None and group.index.name != x_name:
                    plt.plot(pd.to_datetime(group["sampling_time"]), group[name], markers[file], label=file)
                else:
                    plt.plot(group[name], markers[file], label=file)

            plt.legend()
            plt.title(name)
            # plt.axvline(x=pd.to_datetime(group["sampling_time"].iloc[7*12 - 1]), color='r', linestyle='--')
            plt.savefig(Path(out_path) / f'{pic_name}_{name}.png')
            plt.clf()

def draw_scatter_plot(df_list, df_label, out_name, out_path, pic_name, x_name=None, one_file=False):
    draw_same_column(df_list, df_label, out_name, out_path, pic_name, x_name, one_file, marker_list=['o'])

# # 根据draw_same_column更改
# # 把多个df同名的列画到一张图上，其中真实数据用点，模拟结果用折线
# # 输入list的第一个元素为真实数据
# def draw_scatter_plot(df_list, df_label, out_name, out_path, pic_name, x_name=None, one_file=False):
#     true_name = df_label[0]
#     df_list_copy = copy.deepcopy(df_list)
#     for i, df in enumerate(df_list_copy):
#         df["file"] = df_label[i]
#     # 合并DataFrame
#     merged_df = pd.concat(df_list_copy)

#     if one_file:
#         fig, axes = plt.subplots(nrows=len(out_name), ncols=1, figsize=(10, 30))
#         for i, name in enumerate(out_name):
#             for file, group in merged_df.groupby("file"):
#                 if x_name is not None:
#                     axes[i].plot(pd.to_datetime(group[x_name]), group[name], label=file)
#                 else:
#                     axes[i].plot(group[name], label=file)
#             axes[i].legend()
#             axes[i].set_title(name)
#             # axes[i].axvline(x=pd.to_datetime(group["sampling_time"].iloc[7*12 - 1]), color='r', linestyle='--')

#         # 调整子图之间的间距
#         plt.tight_layout()
#         # 显示图表
#         # plt.savefig(Path(out_path) / f'{pic_name}.png')
#     else:
#         for i, name in enumerate(out_name):
#             for file, group in merged_df.groupby("file"):
#                 if x_name is not None:
#                         plt.plot(pd.to_datetime(group["sampling_time"]), group[name], label=file)
#                 else:
#                     if file == true_name:
#                         plt.scatter(group.index, group[name], label=file,s=15,c='green')
#                     else:
#                         plt.plot(group.index, group[name], label=file)

#             plt.legend()
#             plt.title(name)
#             # plt.axvline(x=pd.to_datetime(group["sampling_time"].iloc[7*12 - 1]), color='r', linestyle='--')
#             plt.savefig(Path(out_path) / f'{pic_name}_{name}.png')
#             plt.clf()

def draw_df(df, out_path, figsize=(10, 6), prefix='', suffix=''):
    """绘制 DataFrame 的折线图。以index为横坐标, 列名为纵坐标, 几列就绘制几张。
    
    Args:
        df (pandas.DataFrame) - 待绘制的 DataFrame
        figsize (tuple) - 图表大小
        out_path (str) - 图表保存路径
    
    Examples:
    >>> df = pd.DataFrame({'A': [10, 4, 8, 10], 'B': [4, 0, np.NaN, 5], 'C': [5, 2, 0, 1]})
    >>> draw_df(df, figsize=(10, 6), out_path='../../output')
    """
    plt.figure(figsize=figsize)
    for column_name in df.columns:
        df[column_name].plot()
        plt.title(f"{column_name} Column")
        plt.xlabel(df.index.name)
        plt.ylabel(column_name)
        plt.savefig(Path(out_path)/f"{prefix}{column_name}{suffix}.png")  # 保存图表到当前目录
        plt.clf()

def draw_loss(data, x=None, figsize=(10, 6), 
              title="Loss Iteration Curve", 
              xlabel="Iteration", ylabel="Loss", out_path=None):
    """绘制损失函数的迭代曲线。

    Args:
        data (list) - 损失数据
        x (list) - 迭代次数
        figsize (tuple) - 图表大小
        title (str) - 图表标题
        xlabel (str) - x轴标题
        ylabel (str) - y轴标题
        out_path (str) - 图表保存路径

    Examples:
    >>> data = [10, 4, 8, 10]
    >>> draw_loss(data out_path='../../output')
    """
    plt.figure(figsize=figsize)
    if x is not None:
        plt.plot(x, data)
    else:
        plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if out_path is not None:
        plt.savefig(out_path)
    else:
        print('No output path')

def draw_cmp_eff(df_list, df_label, out_name, out_path, pic_name, x_name, one_file=False, version='24_7'):
    """画图比较跑出来的出水结果等

    Args:
        df_list (list) - DataFrame 列表, 其中第一个为真实数据文件(列名指定), 其他为模拟结果(列名同sumo中命名)
        df_label (list) - 每个df给个标签(因为列名是一样的无法区分), 与df_list一一对应
        out_name (list) - 待绘制的列名(df可能有很多, 但不是所有都想画)  # [ ] TODO: None为全画
        out_path (str) - 图表保存路径
        pic_name (str) - 折线图名称(一张图)或折线图名称前缀(多张图)
        x_name (str) - x轴列名
        one_file (bool) - 是否保存到一个文件中
    """
    if version=='24_7':
        column_names_map = {
            'Me_VSS' : 'Sumo__Plant__aao_cstr7_1_1__XVSS',
            'Me_VSS' : 'Sumo__Plant__aao_cstr7_1_2__XVSS',
            'Me_VSS' : 'Sumo__Plant__aao_cstr7_2_1__XVSS',
            'Me_VSS' : 'Sumo__Plant__aao_cstr7_2_2__XVSS',

            'aao1_1_mlss' : 'Sumo__Plant__aao_cstr7_1_1__XTSS',
            'aao1_2_mlss' : 'Sumo__Plant__aao_cstr7_1_2__XTSS',
            'aao2_1_mlss' : 'Sumo__Plant__aao_cstr7_2_1__XTSS',
            'aao2_2_mlss' : 'Sumo__Plant__aao_cstr7_2_2__XTSS',

            'effluent_tol_cod': 'Sumo__Plant__aao_effluent__TCOD',
            'effluent_dis1_ss': 'Sumo__Plant__aao_effluent__XTSS',

            'aao1_1_aerobic_terminal_do': 'Sumo__Plant__aao_cstr7_1_1__SO2',
            'aao1_2_aerobic_terminal_do': 'Sumo__Plant__aao_cstr7_1_2__SO2',
            'aao2_1_aerobic_terminal_do': 'Sumo__Plant__aao_cstr7_2_1__SO2',
            'aao2_2_aerobic_terminal_do': 'Sumo__Plant__aao_cstr7_2_2__SO2',

            'effluent_secpump_snhx': 'Sumo__Plant__aao_effluent__SNHx',
            'effluent_dis1_tn': 'Sumo__Plant__aao_effluent__TN',
            'effluent_dis1_tp': 'Sumo__Plant__aao_effluent__TP',
        }
    elif version=='24_8':
        column_names_map = {
            'Me_VSS' : 'Sumo__Plant__CSTR1_7__XVSS',
            'Me_VSS' : 'Sumo__Plant__CSTR2_7__XVSS',
            'Me_VSS' : 'Sumo__Plant__CSTR3_7__XVSS',
            'Me_VSS' : 'Sumo__Plant__CSTR4_7__XVSS',

            'aao1_1_mlss' : 'Sumo__Plant__CSTR1_7__XTSS',
            'aao1_2_mlss' : 'Sumo__Plant__CSTR2_7__XTSS',
            'aao2_1_mlss' : 'Sumo__Plant__CSTR3_7__XTSS',
            'aao2_2_mlss' : 'Sumo__Plant__CSTR4_7__XTSS',

            'effluent_tol_cod': 'Sumo__Plant__Effluent__TCOD',
            'effluent_dis1_ss': 'Sumo__Plant__Effluent__XTSS',

            'aao1_1_aerobic_terminal_do': 'Sumo__Plant__CSTR1_8__SO2',
            'aao1_2_aerobic_terminal_do': 'Sumo__Plant__CSTR2_8__SO2',
            'aao2_1_aerobic_terminal_do': 'Sumo__Plant__CSTR3_8__SO2',
            'aao2_2_aerobic_terminal_do': 'Sumo__Plant__CSTR4_8__SO2',

            'effluent_secpump_snhx': 'Sumo__Plant__Effluent__SNHx',
            'effluent_dis1_tn': 'Sumo__Plant__Effluent__TN',
            'effluent_dis1_tp': 'Sumo__Plant__Effluent__TP',
        }
    elif version=='24_8_mbr':
        column_names_map = {
            'mbr1_1_mlss' : 'Sumo__Plant__mbr_cstr2_8_1_1__XTSS',
            'mbr1_2_mlss' : 'Sumo__Plant__mbr_cstr2_8_1_2__XTSS',
            'mbr2_1_mlss' : 'Sumo__Plant__mbr_cstr2_8_2_1__XTSS',
            'mbr2_2_mlss' : 'Sumo__Plant__mbr_cstr2_8_2_2__XTSS',

            'effluent_tol_cod': 'Sumo__Plant__mbr_effluent__TCOD',
            'effluent_dis1_ss': 'Sumo__Plant__mbr_effluent__XTSS',

            'mbr1_1_aerobic_terminal_do': 'Sumo__Plant__mbr_cstr2_8_1_1__SO2',
            'mbr1_2_aerobic_terminal_do': 'Sumo__Plant__mbr_cstr2_8_1_2__SO2',
            'mbr2_1_aerobic_terminal_do': 'Sumo__Plant__mbr_cstr2_8_2_1__SO2',
            'mbr2_2_aerobic_terminal_do': 'Sumo__Plant__mbr_cstr2_8_2_2__SO2',

            'effluent_secpump_snhx': 'Sumo__Plant__mbr_effluent__SNHx',
            'effluent_dis1_tn': 'Sumo__Plant__mbr_effluent__TN',
            'effluent_dis1_tp': 'Sumo__Plant__mbr_effluent__TP',
        }
    
    for key, value in df_list[0].items():
        new_key = column_names_map.get(key, key)
        df_list[0][new_key] = value
        if key != new_key:  # 可选
            del df_list[0][key]
    draw_same_column(df_list, df_label, out_name, out_path, pic_name, 
                     x_name, one_file, marker_list=['o'])

if __name__ == '__main__':
    # path = 'D:/ysq/DT_Self_Calibration/output/pre_study/main/pre_result/2024-04-13T14-38-24/'
    df1 = pd.read_excel(r"D:\ysq\DT_Self_Calibration\data\OfflineData\2024-04-08-2024-06-28_effluent_data.xlsx")
    df2 = pd.read_excel(r"D:\ysq\DT_Self_Calibration\tmp\miaomiao_test\output\human\2024-08-21T12-22-29\result_0.xlsx")
    df3 = pd.read_excel(r"D:\ysq\DT_Self_Calibration\data\OfflineData\human_eff.xlsx")
    # df3 = pd.read_excel(path+"result3.xlsx")
    # df4 = pd.read_excel(path+"result4.xlsx")

    # df1 = pd.read_excel('../../tmp/miaomiao_test/output/human/2024-07-09T19-44-24/result_0.xlsx')
    # df2 = pd.read_excel('../../tmp/miaomiao_test/output/human/2024-07-09T19-44-24/result_1.xlsx')
    # df3 = pd.read_excel('../../tmp/miaomiao_test/output/human/2024-07-11T19-28-53/result_mlss_callback_0.xlsx')
    df_list = [df1, df2]
    df_label = ['my', 'human']

    out_name = [
        'Sumo__Plant__aao_effluent__SNHx',
        'Sumo__Plant__aao_effluent__TP',
        'Sumo__Plant__aao_effluent__XTSS',
        'Sumo__Plant__aao_effluent__TN',
        'Sumo__Plant__aao_cstr7_1_1__SO2',
        'Sumo__Plant__aao_cstr7_1_2__SO2',
        'Sumo__Plant__aao_cstr7_2_1__SO2',
        'Sumo__Plant__aao_cstr7_2_2__SO2',
    ]

    draw_scatter_plot(df_list, df_label, out_name, '../../output/test/', 'test')