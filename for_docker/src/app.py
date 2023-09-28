from fastapi import FastAPI
from predict import make_predict, get_dict_models
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

cwd = os.getcwd()

PATH_TO_MODELS= cwd + '/models'
PATH_TO_PREPROC_DF_FOR_PRED = cwd + '/data/preprocessing/preproc_df_test_no.csv'

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
    :param st_id: id магазина
    :param sku_id: id товара
    :return: Возвращает json с датами и спросом для соответствующего товара и магазина
    """
    dict_mod = get_dict_models(PATH_TO_MODELS)
    predict = make_predict(PATH_TO_PREPROC_DF_FOR_PRED, dict_mod)
    return predict.to_json(orient="records")