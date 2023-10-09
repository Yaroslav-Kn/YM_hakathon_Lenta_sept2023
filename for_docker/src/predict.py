import os

import pickle
from catboost import CatBoostRegressor
from datetime import datetime, timedelta
import pandas as pd


def get_dict_models (path: str) -> dict:
    """Функция для получения словаря с моделями
    :param path: Путь до папки с моделями
    :return model: модель для прогноза
    """    
    with open(path, 'rb') as file:
        model = pickle.load(file)       
    
    return model


def prediction_submitions(df_pred, model, sumissions_df=None):
    '''функция для прогноза
    :param df_pred: датасет для прогноза
    :param model: модель для прогноза
    :param sumissions_df: датасет с конкретным запросом (если None, то возвращается полный прогноз по всем сочетаниям товар / магазин)
    :return new_df: Возвращает датасет с прогнозом
    '''
    if sumissions_df is not None and 'target' in sumissions_df.columns:
        sumissions_df = sumissions_df.drop('target', axis = 1)
        
    new_df = df_pred[['pr_sku_id', 'st_id']].reset_index(drop=True)
    new_df['date'] = pd.to_datetime(df_pred.index)
    X_test = df_pred.drop('target', axis=1)    
    y_pred = model.predict(X_test)    
    new_df['target'] = y_pred
    
    if sumissions_df is not None:
        sumissions_df['date'] = pd.to_datetime(sumissions_df['date'])
        new_df = new_df.merge(sumissions_df, on=['date', 'pr_sku_id', 'st_id'], how='right')
        new_df = new_df.fillna(0)
    return new_df
