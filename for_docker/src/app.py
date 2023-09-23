from fastapi import FastAPI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

cwd = os.getcwd()
PATH_TO_LAST_DAY= cwd + '/last_date.txt'

app = FastAPI(title='Product demand forecasting')


def get_last_data(path_to_last_day: str) -> datetime:
    """Функция для получения последней даты в обучающей выборке
    :param path_to_train: Путь до тренировочного датасета
    :return:
    """    
    with open(path_to_last_day) as file:
        last_date = file.read()
    return datetime.strptime(last_date, '%Y-%m-%d')


@app.get("/health")
def health():
    """Функция для проверки работы сервера
    :return: Возвращает статус ok, если сервер работает
    """
    return {"status": "ok"}


@app.get('/get_predict')
def get_predict(st_id: str, sku_id: str):
    """Функция для получения предсказания спроса
    :param st_id: id магазина
    :param sku_id: id товара
    :return: Возвращает json с датами и спросом для соответствующего товара и магазина
    """
    last_day = get_last_data(PATH_TO_LAST_DAY)
    list_date = [last_day + timedelta(days=x) for x in range(14)]
    list_sales_in_units = np.random.randint(0, 20, 14)
    df = pd.DataFrame({ 'date': list_date,
                        'sales_in_units': list_sales_in_units})
    df['date'] = df['date'].astype('str')
    return df.to_json(orient="records")