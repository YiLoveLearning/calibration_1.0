import ast
import numpy as np
import copy
import pandas as pd

def str2tuple(tuple_str):
    """Convert a string to a tuple.
    Args:
        tuple_str (str): The string representation of a tuple.
        
    Returns:
        tuple: The tuple.
    
    Examples:
    >>> tuple_str = "(1, 2, 3)"
    >>> tuple_value = str2tuple(tuple_str)
    """
    try:
        tuple_value = ast.literal_eval(tuple_str)
        if isinstance(tuple_value, tuple):
            return tuple_value
        else:
            raise ValueError("Invalid tuple literal")
    except ValueError:
        raise ValueError("Invalid tuple literal")
    except SyntaxError:
        raise ValueError("Invalid syntax in tuple literal")

def dict2cmd_line(d, name_map):
    """Convert a dict to a cmd_line(list[list[str]]).
    Args:
        d: {'总name1':值1, '总name2':值2}
        name_map: {'总name1':[分支1name, 分支2name, ...], '总name2':[...], ...}
            或 {'总name1':[[分支1name, 同值的另一变量], [分支2name, 同值的], ...], '总name2':[...], ...}
        
    Returns:
    
    Examples:
    >>> d = {'a':123, 'b':345}
    >>> name_map = {'a':['a1', 'a2'], 'b':['b1', 'b2'], 'c':['c1', 'c2']}
    >>> cmd_line = dict2cmd_line(d, name_map)
    """
    WATER_LINE = len(list(name_map.values())[0])
    cmd_line = [[] for _ in range(WATER_LINE)]
    # cmd_line = [[]]*WATER_LINE    # 被钉上耻辱架的bug: 和上面那行不同-是复制了相同的WATER_LINE个[]对象, 而不是WATER_LINE个独立的list
    for name in d:
        if isinstance(name_map[name][0], str):
            for line in range(WATER_LINE):
                cmd_line[line].append(f'set {name_map[name][line]} {d[name]}')
        else:   # list(np.array之类的有些麻烦)
            assert isinstance(name_map[name][0], list), "bad dict2cmd_line name_map type"
            for line in range(WATER_LINE):
                for i in name_map[name][line]:
                    cmd_line[line].append(f'set {i} {d[name]}')
    # cmd_line = [[f'set {name_map[name][line]} {d[name]}' 
    #                     for name in d]
    #                             for line in range(WATER_LINE)]
    return cmd_line

# Helper function to wrap a function with arguments
def wrapper(args):
    func = args[0]
    func_args = args[1:]
    return func(*func_args)

def get_mean(data, time_series, time_col='cleaning_time', freq ='2H'):
    '''
    取指定采样时间数据的均值
    '''
    if data.index is None:
        data[time_col] = pd.to_datetime(data[time_col])
        # 设置时间列为索引
        data.set_index(time_col, inplace=True)
    # 缺失数据填nan
    data = data.asfreq(freq).fillna(np.nan)
    # 选取采样时间数据
    if time_series is not None: # 采样时间则返回拷贝, 防止原数据信息损失
        data = copy.deepcopy(data.loc[time_series])
    else:   # 选取所有时间数据, 排序
        data = data.sort_index()
    mean = data.mean(axis=0)
    return mean