import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from my_math import choose_sample_time

# garbage_Q_day.xlsx和北湖运行数据的相关性分析

# 计算数据的相关系数矩阵
def corr_matrix(df):
    return df.corr()

# 绘制相关性热力图
def corr_heatmap(corr_matrix, name, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    # 创建自定义颜色条
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1)
    plt.title(f"{name} Correlation Matrix")
    plt.savefig(f"{name}_heatmap.png", bbox_inches='tight', pad_inches=0)

# 绘制相关性散点图矩阵
def corr_scatter_plot(df, name, figsize=(12, 10)):
    plt.figure(figsize=figsize)
    sns.pairplot(df)
    plt.title(f"{name} Correlation Scatter Plot Matrix")
    plt.savefig(f"{name}_scatter_plot.png")

def concat_df(file_list):
    col_names = set()
    df_list = []
    # 把所有表格列名拼上文件名名字后拼接到一起
    for file in file_list:
        df = pd.read_excel(file, header=0)
        df = df.iloc[1:]
        df = df.drop('date', axis=1)
        df = df.astype(float)
        col_names.update(df.columns)
        df.columns = [f"{col}_{file.split('.')[0]}" for col in df.columns]
        df_list.append(df)
    all_df = pd.concat(df_list, axis=1)
    return all_df, col_names

if __name__ == '__main__':
    df1_name = '../../data/OfflineData/2024-04-08-2024-06-28_effluent_data.xlsx'
    df2_name = '../../data/OfflineData/2024-04-08-2024-06-28_operating_data.xlsx'
    df1 = pd.read_excel(df1_name, parse_dates=['sampling_time'])
    df2 = pd.read_excel(df2_name, parse_dates=['sampling_time'])
    df2 = df2.drop(columns=['sampling_time'])
    df = pd.concat([df1, df2], axis=1)
    df = df[['sampling_time', 'aao1_1_aerobic_front_do', 'aao1_1_aerobic_terminal_do',
            'aao1_1_deg_do', 'aao1_2_aerobic_front_do', 'aao1_2_aerobic_terminal_do',
            'aao1_2_deg_do', 'aao2_1_aerobic_front_do', 'aao2_1_aerobic_terminal_do',
            'aao2_1_deg_do', 'aao2_2_aerobic_front_do', 'aao2_2_aerobic_terminal_do',
            'effluent_secpump_snhx'
            ]]
    df['effluent_secpump_snhx'] = 1/df['effluent_secpump_snhx']
    start_time = datetime.datetime(2024, 4, 8, 1)
    end_time = datetime.datetime(2024, 6, 8, 1)
    time_series = pd.date_range(start=start_time, end=end_time, freq='2H')
    df = choose_sample_time(df, time_series=None, time_col='sampling_time', freq='2H')
    # # 垃圾渗滤液的数据整理(天总量)
    # garbage = df.reindex(date_range)   # 按天采样并排序，缺失值用NaN填充

    # # 北湖运行数据的数据整理(天均值)
    # inf_df = pd.read_excel(inf_file, parse_dates=['date'])
    # # inf_df['date'] = pd.to_datetime(inf_df['date'])
    # inf_df.dropna(inplace=True)
    # inf_df = inf_df.groupby(inf_df['date'].dt.date).mean(numeric_only=True)
    # inf_df = inf_df.reindex(date_range)   # 按天采样并排序，缺失值用NaN填充
    # # 合并两个表，并且把date列去掉
    # all_df = pd.concat([garbage, inf_df], axis=1)
    corr = corr_matrix(df)
    corr_heatmap(corr, 'day')
    corr_scatter_plot(df, 'day')
