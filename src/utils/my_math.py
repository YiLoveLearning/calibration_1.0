import numpy as np
import pandas as pd
import copy

def freqstr2minutes(freq):
    """将pd频率字符串解析为分钟数。
    
    Args:
        freq (str) - Pandas 中支持的频率字符串
            'D'、'W'、'M'、'Q'、'Y'分别表示日、周、月、季度、年
            'H'、'T'、'S'、'L'、'U'、'N'分别表示小时、分钟、秒、毫秒、微秒、纳秒
    
    Returns:
        float - 对应的分钟数
    """
    # 将频率字符串转换为 Timedelta 对象
    delta = pd.Timedelta(freq)
    # 转换为分钟数
    minutes = delta.total_seconds() / 60
    return minutes

def minutes2freqstr(minutes):
    """将分钟数转换为 Pandas 支持的频率字符串。
    
    Args:
        minutes (int) - 以分钟为单位的频率。
    
    Returns:
        str - 对应的频率字符串。
    """
    # 创建 Timedelta 对象
    delta = pd.Timedelta(minutes=minutes)
    
    # 根据 Timedelta 对象提取频率字符串
    if delta.days > 0:
        return f"{delta.days}D"
    elif delta.seconds // 3600 > 0:
        return f"{delta.seconds // 3600}H"
    elif delta.seconds // 60 > 0:
        return f"{delta.seconds // 60}T"
    else:
        return f"{delta.seconds}S"

def SMA(df, window_size):
    """计算数据序列的简单移动平均值。
    返回序列与原始数据序列等长: 得到的移动平均序列放在结果中间, 两边用缩短的滑动窗口计算的平均值填充。
    
    Args:
        df (pandas.DataFrame) - 待计算移动平均的数据
        window_size (int) - 移动窗口的大小
    
    Returns:
        pandas.DataFrame - 加权移动平均值数据
    
    Examples:
    >>> df = pd.DataFrame([1,2,3,4,5])
    >>> sm_df = SMA(df, 2)
    """
    # 计算中间部分的加权平均值
    sm_df = df.rolling(window=window_size, center=True).mean()

    # 计算两边部分的加权平均值
    for i in range(window_size//2):
        sm_df.iloc[i] = df[:1+i].mean()
    for i in range((window_size-1)//2):
        sm_df.iloc[-1-i] = df[-1-i:].mean()
    return sm_df

def mask_nan(df):
    """标记nan值为True
    
    Args:
        df (pandas.DataFrame) - 待处理的数据
    
    Returns:
        pandas.DataFrame - 标记nan值的数据
    
    Examples:
    >>> data = pd.DataFrame([1,2,3,4,5])
    >>> mask_nan(data)
    """
    return df.isnull()

def mask_val(df, val):
    """标记指定值为True
    
    Args:
        df (pandas.DataFrame) - 待处理的数据
        val (float) - 需要标记的值
    
    Returns:
        pandas.DataFrame - 标记指定值的数据
    
    Examples:
    >>> data = pd.DataFrame([1,2,3,4,5])
    >>> mask_val(data, 3)
    """
    return df==val

def mask_zero(df):
    """标记零值为True"""
    return df==0

def mask_exceed(df, threshold):
    """标记爆表值为True
    
    Args:
        df (pandas.DataFrame) - 待处理的数据
        threshold (float) - max的个数或者占比超出多少则认为是爆表值
    
    Returns:
        pandas.DataFrame - 标记爆表值的数据
    
    Examples:
    >>> data = pd.DataFrame({'A': [10, 0, 8, 10], 'B': [5, 5, 0, np.NaN], 'C': [0, 2, 0, 0]})
    >>> mask_exceed(data, 1)
    """
    if threshold<1:
        assert threshold>0, 'threshold should be positive'
        threshold = len(df)*threshold
    # 统计每列最大值个数
    max_count = df.apply(lambda x: (x==x.max()).sum() if x.dtype=='float' else 10)
    # 如果该列最大值个数超过阈值, 则认为是爆表值, 该列该值设为True; 其他列均设为False, 整个的形状不变
    return df.apply(lambda x: x==x.max() if max_count[x.name]>threshold 
                    else pd.Series(False, index=x.index, name=x.name))

def union_masks(*masks):
    """将任意个 DataFrame 类型的 mask 进行并操作,得到一个新的 mask。
    
    Args:
        *masks (pandas.DataFrame) - 需要合并的 mask,可以传入任意个
    
    Returns:
        pandas.DataFrame - 所有输入 mask 的并操作结果

    Examples:
    >>> m1 = pd.DataFrame({'A': [True, False, True], 'B': [False, False, True]})
    >>> m2 = pd.DataFrame({'A': [False, True, False], 'B': [True, True, False]})
    >>> mask = union_masks(m1, m2)
    """
    result = None
    for mask in masks:
        if result is None:
            result = mask
        else:
            result = result | mask
    return result

def interp_mask(df, mask):
    """将 DataFrame 中的mask用线性插值的方式填充。
    
    Args:
        df (pandas.DataFrame) - 待处理的 DataFrame
        mask (pandas.DataFrame) - 需要填充的 mask
    
    Returns:
        pandas.DataFrame - 填充后的 DataFrame
    
    Examples:
    >>> df = pd.DataFrame({'A': [10, 4, 8, 10], 'B': [4, 0, np.NaN, 5], 'C': [5, 2, 0, 1]})
    >>> m1 = mask_nan(df)
    >>> m2 = mask_zero(df)
    >>> m3 = mask_exceed(df, 1)
    >>> mask = union_masks(m1, m2, m3)
    >>> df_filled = interp_mask(df, mask)
    """
    # 使用线性插值的方式填充mask
    df_filled = df[~mask].interpolate(method='linear', axis=0, limit_direction='both')
    return df_filled

def choose_sample_time(data, time_series=None, time_col='sampling_time', freq='2H'):
    """选择采样时间数据, 并按照时间序列排序, 缺失数据填nan。

    Args:
        data (pandas.DataFrame) - 原始数据, 带有时间列
        time_series (pandas.Series) - 采样时间序列, 若为None则选取所有时间数据
        time_col (str) - 时间列名
        freq (str) - 时间频率

    Returns:
        pandas.DataFrame - 填补排序后的数据

    Examples:
    >>> data = pd.read_excel('../../data/OfflineData/230411-240105_clean.xlsx')
    >>> time_col = 'simulate_time'
    >>> data = choose_sample_time(data, time_series=None, time_col=time_col, freq='2H')
    """
    if data.index.name is None:
        data[time_col] = pd.to_datetime(data[time_col])
        # 设置时间列为索引
        data.set_index(time_col, inplace=True)
    # 把'#VALUE!'等特殊值转换为NaN
    for col in data.select_dtypes(include='object').columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # 缺失数据填nan
    data = data.asfreq(freq).fillna(np.nan)
    # 选取采样时间数据
    if time_series is not None: # 采样时间则返回拷贝, 防止原数据信息损失
        data = copy.deepcopy(data.loc[time_series])
    else:   # 选取所有时间数据, 排序
        data = data.sort_index()
    data.index.name=time_col
    return data