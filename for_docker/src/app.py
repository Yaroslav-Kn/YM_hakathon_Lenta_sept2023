from fastapi import FastAPI
from predict import make_predict, get_dict_models
from preprocessing import get_df_for_pred, get_df_for_subbmisions, get_cat, preprocessing_st_df
import pandas as pd
import json
import requests
import yaml

import os

# задаём пути и ссылки
cwd = os.getcwd()
CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

PATH_SALES_DF_TRAIN = cwd + config['PATH_SALES_DF_TRAIN']
PATH_PR_DF = cwd + config['PATH_PR_DF']
PATH_ST_DF = cwd + config['PATH_ST_DF']
PATH_TO_SUMISSION_DF = cwd + config['PATH_TO_SUMISSION_DF']
PATH_TO_HOLIDAYS = cwd + config['PATH_TO_HOLIDAYS']
PATH_TO_MODELS= cwd + config['PATH_TO_MODELS']

URL_FOR_READY = config['URL_FOR_READY']

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


def send_predict_ready() -> None:
    ''' функция для отправки сообщения, что предикт готов'''
    url: str = f'{URL_FOR_READY}/api/v1/ready'
    requests.post(url, json.dumps(True))


@app.get("/api/v1/health")
def health():
    """Функция для проверки работы сервера
    :return: Возвращает статус ok, если сервер работает
    """   
    return {"status": "ok"}  


@app.get("/api/v1/get_last_date")
def get_last_date():
    """Функция для получения последней даты в датасете
    :return: Возвращает статус ok, если сервер работает
    """   
    sales_df_train = pd.read_csv(PATH_SALES_DF_TRAIN)
    return {"last_date": sales_df_train['date'].max()}  


@app.post("/api/v1/update_sales")
def update_sales(json_row):
    """Функция для вставки новых записей о совершённых продажах и запуска прогноза на 14 дней вперёд 
    :return: Возвращает статус update sales and submissions completed, если обновление проведено успешно и сообщение на бэк по указаному api
    """    
    update_df_sales(PATH_SALES_DF_TRAIN, json_row)
    update_predictions(PATH_SALES_DF_TRAIN, 
                       PATH_PR_DF, 
                       PATH_ST_DF, 
                       PATH_TO_HOLIDAYS, 
                       PATH_TO_MODELS, 
                       PATH_TO_SUMISSION_DF)
    send_predict_ready()
    return {"status": "update sales and submissions completed"} 


@app.get('/api/v1/get_predict')
def get_predict():
    """Функция для получения предсказания спроса
    :return: Возвращает json с прогнозами всех товаров на 14 дней вперёд
    """
    predict = pd.read_csv(PATH_TO_SUMISSION_DF) 
    return predict.to_json(orient="records")
