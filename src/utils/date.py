from pathlib import Path
from datetime import datetime
from bisect import bisect_left

def find_closest_date_file(target_date: datetime, folder: str, 
                           file_name_pattern: str = 'state{}_%Y-%m-%d-%H-00.xml',
                           left_or_right: str = 'left') -> str:
    """在指定文件夹中查找最接近目标日期的文件，并返回文件名。

    Args:
        target_date (datetime): 目标日期和时间。
        folder (str): 文件夹路径。
        file_name_pattern (str): 文件名格式字符串，默认为 'state{}_%Y-%m-%d-%H-00.xml'。
        left_or_right (str): 如果找到多个匹配的文件，选择最接近目标日期的文件。
                'left' 表示选择较早的文件，'right' 表示选择较晚的文件。默认为 'left'。right会取到, left不会取到
    
    Returns:
        str|None: 最接近目标日期的文件名。如果没有找到匹配的文件，返回 None。

    Examples:
    >>> target_date = datetime(2024, 4, 28, 1, 0)
    >>> folder_path = '../../data/OfflineStates/22.1.0dll_24_8'
    >>> result = find_closest_date_file(target_date, folder_path, 'state{}_%Y-%m-%d-%H-00.xml', 'left')
    """
    dates = []
    file_names = file_name_pattern.format(1)    # 将{}->1(format变成具体文件名)

    # 遍历文件夹中的文件
    for file_path in Path(folder).iterdir():
        if file_path.is_file():
            try:
                file_date = datetime.strptime(file_path.name, file_names)
                dates.append(file_date)
            except ValueError:
                # 如果文件名不匹配预期格式，跳过该文件
                continue

    # 如果没有找到匹配的文件，返回 None
    if not dates:
        return None

    # 对日期列表进行排序
    sorted_indices = sorted(range(len(dates)), key=lambda k: dates[k])
    sorted_dates = [dates[i] for i in sorted_indices]

    # 使用二分查找找到最接近的索引
    index = bisect_left(sorted_dates, target_date)
    
    if left_or_right == 'left':
        index -= 1
    elif left_or_right != 'right':
        raise ValueError("left_or_right must be either 'left' or 'right'")

    # 如果索引超出范围，返回 None
    if index < 0 or index >= len(dates):
        return None

    # 返回找到的文件名
    return dates[index].strftime(file_name_pattern), dates[index]

if __name__ == "__main__":
    last_state_str, last_date = find_closest_date_file(now, state_folder, state_pattern)
    if 数据库last_date到now的数据存在缺失:
        原来调用初始状态校准的部分放在这
    else:
        从last_date开始仿真到now并保存状态
