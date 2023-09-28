import os

import pickle
from catboost import CatBoostRegressor
from datetime import datetime, timedelta
import pandas as pd


def get_dict_models (path: str) -> dict:
    """Функция для получения словаря с моделями
    :param path: Путь до папки с моделями
    :return dict_mod: Словарь с моделями
    """    
    dict_mod = {}
    for f in os.listdir(path):
        with open(f'{path}/{f}', 'rb') as file:
            model = pickle.load(file)        
        name_mod = f.split('.')[0]
        dict_mod[name_mod] = model
    return dict_mod


def make_predict(path_to_pred_df: str, dict_mod: dict) -> pd.DataFrame:
    """Функция для получения предсказания модели
    :param path_to_pred_df: Путь до предобработанного датасета для прогноза
    :param dict_mod: Словарь с моделями
    :return predict_df: Датафрейм с прогнозом на 14 дней
    """   
    # читаем данные, находим последнюю дату в датасете (по нему будем строить прогноз)
    df = pd.read_csv(path_to_pred_df, index_col=0)
    df.index = pd.to_datetime(df.index)
    last_date = df.index.max()
    list_models = list(df.loc[last_date, 'group_shop_cat'].unique())

    # получаем список дат (потребуется для вывода прогноза)
    list_columns = []
    for i in range(14):
        list_columns.append(datetime.date(last_date + timedelta(days=i)))

    list_predict = []
    # поочерёдно выбираем типы моделей
    for name_model in list_models: 
        # фильтруем данные для этих моделей
        df_for_pred = df.loc[(df.index == last_date) & 
                                  (df['group_shop_cat'] == name_model)]    
        # если есть подходящие данные, то начинаем прогноз
        if df_for_pred.shape[0] > 0:
            if 'target_0' in df_for_pred.columns:
                X_test = df_for_pred.drop('target_0', axis=1)

            #Запоминаем толбцы товара и магазины (их нужно будет удалить на предикте)
            sku_df = df_for_pred.loc[last_date, 'pr_sku_id'].reset_index(drop=True)
            st_df = df_for_pred.loc[last_date, 'st_id'].reset_index(drop=True)
            X_test = X_test.drop(['group_shop_cat', 'pr_sku_id', 'st_id', 'pr_group_id'],axis=1)


            #делаем прогноз
            model = dict_mod[name_model]            
            y_pred = model.predict(X_test)

            # Преобразуем прогноз в требуемый формат
            predictions_data = pd.DataFrame(data=y_pred, columns=list_columns)
            predictions_data['pr_sku_id'] = sku_df
            predictions_data['st_id'] = st_df
            predictions_data = predictions_data.melt(['st_id', 'pr_sku_id'], 
                                                     var_name='date', 
                                                     value_name='target')
            list_predict.append(predictions_data)

    predict_df = pd.concat(list_predict,axis=0)
    predict_df['date'] = predict_df['date'].astype('str')
    return predict_df
