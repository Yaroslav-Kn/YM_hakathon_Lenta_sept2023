from fastapi import FastAPI
from .predict import make_predict, get_dict_models
from .preprocessing import get_df_for_pred, get_df_for_subbmisions, get_cat, preprocessing_st_df
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

cwd = os.getcwd()

PATH_SALES_DF_TRAIN = cwd + '/src/data/raw/sales_df_train.csv'
PATH_PR_DF = cwd + '/src/data/raw/pr_df.csv'
PATH_ST_DF = cwd + '/src/data/raw/st_df.csv'
PATH_TO_SUMISSION_DF = cwd + '/src/data/sales_submission.csv'
PATH_TO_HOLIDAYS = cwd + '/src/data/holidays.csv'
PATH_TO_MODELS= cwd + '/src/models'

app = FastAPI(title='Product demand forecasting')


def update_df_sales(path: str, json_row: list):
    """Функция для вставки новых записей о совершённых продажах 
    :param path: путь к датасету по продажам
    :param json_row: список json с новыми записями
    :return: 
    """  
    try: 
        df_json = pd.read_json(json_row)
    except:
        json_row = eval(json_row)
        df_json = pd.read_json(json_row)

    df = pd.read_csv(path)
    df_json['date'] = df_json['date'].astype('str')
    new_df = pd.concat([df, df_json], axis=0)
    new_df = new_df.sort_values(['date', 'st_id', 'pr_sku_id']).drop_duplicates()
    new_df.to_csv(path, index=False)


def update_predictions(path_sales, path_pr, path_st, path_holidays, path_models, path_submissions):
    """Функция для совершения прогноза 14 дней вперёд и записи в файл с предсказаниями 
    :param path_sales: путь к датасету по продажам
    :param path_pr: путь к датасету по товарной иерархии
    :param path_st: путь к датасету с данными по магазинам
    :param path_holidays: путь к датасету с праздниками на 2022-2023 год
    :param path_models: путь к папке с моделями
    :param path_submissions: путь к файлу с прогнозами
    :return: 
    """  
    sales_df_train = pd.read_csv(path_sales)
    pr_df = pd.read_csv(path_pr)
    st_df = pd.read_csv(path_st)
    list_holidays = (pd.read_csv(path_holidays)['holidays'].values)
    dict_mod = get_dict_models(path_models)   
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
    df_for_submissions = get_df_for_subbmisions(sales_df_train)    
    predict = make_predict(df_pred, 
                          df_for_submissions, 
                          st_df, 
                          pr_df, 
                          dict_mod, 
                          not_promo=True, 
                          only_st_sku=True) 
    predict.to_csv(path_submissions, index=False)


@app.get("/health")
def health():
    """Функция для проверки работы сервера
    :return: Возвращает статус ok, если сервер работает
    """   
    return {"status": "ok"}  


@app.post("/update_sales")
def update_sales(json_row):
    """Функция для вставки новых записей о совершённых продажах и запуска прогноза на 14 дней вперёд 
    :return: Возвращает статус update sales and submissions completed, если обновление проведено успешно
    """    
    update_df_sales(PATH_SALES_DF_TRAIN, json_row)
    update_predictions(PATH_SALES_DF_TRAIN, 
                       PATH_PR_DF, 
                       PATH_ST_DF, 
                       PATH_TO_HOLIDAYS, 
                       PATH_TO_MODELS, 
                       PATH_TO_SUMISSION_DF)
    return {"status": "update sales and submissions completed"} 


@app.get('/get_predict')
def get_predict():
    """Функция для получения предсказания спроса
    :return: Возвращает json с прогнозами всех товаров на 14 дней вперёд
    """
    predict = pd.read_csv(PATH_TO_SUMISSION_DF) 
    return predict.to_json(orient="records")
