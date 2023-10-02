from fastapi import FastAPI
from .predict import make_predict, get_dict_models
from .preprocessing import get_df_for_pred, get_df_for_predict, get_cat, preprocessing_st_df
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

cwd = os.getcwd()

PATH_SALES_DF_TRAIN = cwd + '/src/data/raw/sales_df_train.csv'
PATH_PR_DF = cwd + '/src/data/raw/pr_df.csv'
PATH_ST_DF = cwd + '/src/data/raw/st_df.csv'

PATH_TO_HOLIDAYS = cwd + '/src/data/holidays.csv'

PATH_TO_MODELS= cwd + '/src/models'

app = FastAPI(title='Product demand forecasting')


@app.get("/health")
def health():
    """Функция для проверки работы сервера
    :return: Возвращает статус ok, если сервер работает
    """
    return {"status": "ok"}


@app.get('/get_predict')
def get_predict():
    """Функция для получения предсказания спроса
    :return: Возвращает json с прогнозами всех товаров на 14 дней вперёд
    """
    sales_df_train = pd.read_csv(PATH_SALES_DF_TRAIN)
    pr_df = pd.read_csv(PATH_PR_DF)
    st_df = pd.read_csv(PATH_ST_DF)
    list_holidays = (pd.read_csv(PATH_TO_HOLIDAYS)['holidays'].values)
    dict_mod = get_dict_models(PATH_TO_MODELS)   

    df_best_dict = {'n_day_lag_list': [1, 2, 3, 4, 5, 6, 7],
                 'n_week_for_lag': 2,
                 'strareg_agg': 'mean',
                 'drop_outliers': True,
                 'threshold': 1.4,
                 'var_for_null': 0.01,
                 'var_for_na': 0.01}

    pr_df = get_cat(pr_df)
    st_df = preprocessing_st_df(st_df, sales_df_train)
    df_pred = get_df_for_pred(sales_df_train, pr_df, st_df, list_holidays, df_best_dict)
    df_for_submissions = get_df_for_predict(sales_df_train)
    
    predict = make_predict(df_pred, 
                          df_for_submissions, 
                          st_df, 
                          pr_df, 
                          dict_mod, 
                          not_promo=True, 
                          only_st_sku=True) 
    return predict.to_json(orient="records")
