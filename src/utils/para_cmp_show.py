# 画feedback_res_model或nn_model生成的para_cmp.xlsx, 即0_gt, 0_pd, ..., 
# _gt为横坐标, _pd为纵坐标的散点图

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def para_cmp_scatter(file_path, output_dir, name_list=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(file_path)

    # 找到所有匹配的列对
    pairs = [(col, col.replace('_gt', '_pd')) for col in df.columns if '_gt' in col]

    if name_list is not None:
        assert(len(name_list)==len(pairs))

    # 为每一对列绘制散点图
    for i, (_gt, _pd) in enumerate(pairs):
        plt.figure()
        plt.scatter(df[_gt], df[_pd], alpha=0.5)
        plt.title(f'Scatter plot for {_gt} vs {_pd}({name_list[i]})')
        plt.xlabel(_gt)
        plt.ylabel(_pd)
        plt.grid(True)
        plt.savefig(output_dir / f'{name_list[i]}.png')

if __name__ == "__main__":
    root = Path('../../output/res_nn')
    output_dir = root / Path(__file__).stem
    file_path = root / 'para_cmp.xlsx'

    sorted_name_list = ['alpha1', 'alpha2', 'alpha3', 'alpha4', 'fXTSS_sludge_base', 'PAC1_G', 
        'PAC_G', 'KNHx_AOB_AS', 'KO2_AOB_AS', 'YOHO_SB_anox', 'YOHO_SB_ox', 'bAOB', 'muAOB']
    nn_name_list = ['YOHO_SB_anox', 'YOHO_SB_ox', 'muAOB', 'bAOB', 'KNHx_AOB_AS', 'KO2_AOB_AS',
        'PAC1_G', 'PAC_G', 'fXTSS_sludge_base', 'alpha1', 'alpha2', 'alpha3', 'alpha4']
    para_cmp_scatter(file_path, output_dir, nn_name_list)