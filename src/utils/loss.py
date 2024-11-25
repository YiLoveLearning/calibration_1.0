import numpy as np
import pandas as pd

# def mae(y_true: np.ndarray, y_pred: np.ndarray):
#     y_true = np.where(y_true!=0, y_true, np.nan)
#     y_true = np.where(y_true!=1e-6, y_true, np.nan)
#     mae = np.nanmean(np.abs(y_true - y_pred))
#     return mae
# def mape(y_true, y_pred):
#     y_true = np.where(y_true!=0, y_true, np.nan)
#     y_true = np.where(y_true!=1e-6, y_true, np.nan)
#     mape = np.nanmean(np.abs((y_true - y_pred) / (y_true+1e-6)))
#     return mape
# def rmse(y_true, y_pred):
#     y_true = np.where(y_true!=0, y_true, np.nan)
#     y_true = np.where(y_true!=1e-6, y_true, np.nan)
#     mse = np.nanmean((y_true - y_pred) ** 2)
#     rmse = mse ** 0.5
#     return rmse
# def pso_loss(y_true, y_pred):
#     std = np.nanstd(y_true)
#     loss = np.nansum(np.abs(y_pred - y_true) / (std+1e-6))
#     return loss

def mae(y_true: pd.DataFrame, y_pred: pd.DataFrame, clean=True):
    if clean:
        y_true = y_true.replace(0, np.nan)        # 0值一般为异常值
        y_true = y_true.replace(1e-6, np.nan)     # 1e-6在数据库一般为？值
    if isinstance(y_true, pd.DataFrame):
        y_true_ = y_true.select_dtypes(include='number')
        y_pred_ = y_pred.select_dtypes(include='number')
    else:
        y_true_ = y_true.copy()
        y_pred_ = y_pred.copy()
    mae_loss = (y_true_ - y_pred_).abs().dropna().mean()
    return mae_loss

def mape(y_true: pd.DataFrame, y_pred: pd.DataFrame, clean=True):
    if clean:
        y_true = y_true.replace(0, np.nan)
        y_true = y_true.replace(1e-6, np.nan)
    if isinstance(y_true, pd.DataFrame):
        y_true_ = y_true.select_dtypes(include='number')
        y_pred_ = y_pred.select_dtypes(include='number')
    else:
        y_true_ = y_true.copy()
        y_pred_ = y_pred.copy()
    mape_loss = ((y_true_ - y_pred_) / (y_true_+1e-6)).abs().dropna().mean()
    return mape_loss

def rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame, clean=True):
    if clean:
        y_true = y_true.replace(0, np.nan)
        y_true = y_true.replace(1e-6, np.nan)
    if isinstance(y_true, pd.DataFrame):
        y_true_ = y_true.select_dtypes(include='number')
        y_pred_ = y_pred.select_dtypes(include='number')
    else:
        y_true_ = y_true.copy()
        y_pred_ = y_pred.copy()
    mse_loss = ((y_true_ - y_pred_) ** 2).dropna().mean()
    rmse_loss = mse_loss ** 0.5
    return rmse_loss

# 不适用于真实值用平均值代替的情况, 因为真实值的std作为分母
def pso_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame, clean=True):
    if clean:
        y_true = y_true.replace(0, np.nan)
        y_true = y_true.replace(1e-6, np.nan)
    if isinstance(y_true, pd.DataFrame):
        y_true_ = y_true.select_dtypes(include='number')
        y_pred_ = y_pred.select_dtypes(include='number')
    else:
        y_true_ = y_true.copy()
        y_pred_ = y_pred.copy()
    std = y_true_.dropna().std()
    loss = ((y_pred_ - y_true_) / (std+1e-6)).abs().dropna().mean()
    return loss

# 输入为未归一化的数据(参数数据, 出水数据没有意义)
def nn_loss(y_true: pd.DataFrame, y_pred: pd.DataFrame, w=True):
    import torch
    from torch.nn import functional as F
    import sys
    sys.path.append('..')
    from nn.nn_key import output_mean, output_std, output_key
    from nn.loss_weight import loss_weight
    mean = torch.tensor(output_mean)
    std = torch.tensor(output_std)
    y_true_ = torch.tensor(y_true.get(output_key).values)
    y_pred_ = torch.tensor(y_pred.get(output_key).values)
    y_true_ = (y_true_ - mean) / std
    y_pred_ = (y_pred_ - mean) / std
    weight = torch.tensor(pd.DataFrame([loss_weight]).get(output_key).values)
    sqrt_weight = (weight**0.5).repeat(y_true_.shape[0], 1).to(y_true_.device)
    if w:
        y_true_ = y_true_*sqrt_weight
        y_pred_ = y_pred_*sqrt_weight
    loss = F.mse_loss(y_pred_, y_true_)
    return loss
