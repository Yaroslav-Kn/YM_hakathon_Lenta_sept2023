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


def make_predict(df_test: pd.DataFrame, 
                 df_for_pred: pd.DataFrame, 
                 st_df: pd.DataFrame, 
                 pr_df: pd.DataFrame, 
                 dict_mod: dict, 
                 not_promo: bool = False, 
                 return_y_true: bool = False, 
                 only_st_sku: bool = False) -> pd.DataFrame:
    '''функция разделяет датасет на соответствующие категории, выбирает соответствующую модель и прогнозирует данные,
    имеет возможность указания прогноза с промо и без, возвращать ли y_true, если файл зарпоса сабмита имел соответствующий столбец
    или возвращать только код магазина, sku товара, дату и прогноз
    :param df_test: датасет с данными для прогноза
    :param df_for_pred: датасет информацией по каким товарам и магазинам нужен прогноз (по умолчанию передаются все существующие сочетания товаров и магазинов)
    :param st_df: датасет с информацией о товарах
    :param pr_df: датасет с информацией о магазинах
    :param dict_mod: словарь с моделями
    :param not_promo: флаг указывающий, нужно ли возвращать товары по промо
    :param return_y_true: флаг указывающий, нужно ли возвращать информацию о реальной стоимости товара (если данная ифнормация передавалась в df_for_pred)
    :param only_st_sku: флаг указывающий, что нужно вернуть только информацию о магазине и товаре (помимо даты и прогноза)
    :return pr_data: Возвращает датасет с прогнозом
    '''
    # получим последнюю дату в датасете
    last_date = df_test.index.max()
    list_models = df_test.loc[last_date, 'group_shop_cat'].unique()

    #создадим списко для названия столбцов предикта в соответствии с датами
    list_columns = []
    for i in range(14):
        list_columns.append(last_date + timedelta(days=i))

    # создадим список для предсказаний и пройдёмся по всем моделям
    list_predict = []
    for name_model in list_models: 
        df_test_for_pred = df_test[df_test['group_shop_cat'] == name_model].copy(deep=True)   
        if df_test_for_pred.shape[0] > 0:
            #Сохраним столбцы, чтобы в дальнейшем их можно было присоединить к датасету
            group_shop_cat = df_test_for_pred['group_shop_cat'].reset_index(drop=True)
            pr_sales_type = df_test_for_pred['pr_sales_type_id'].reset_index(drop=True)
            pr_pr_uom_id = df_test_for_pred['pr_uom_id'].reset_index(drop=True)
            
            # сделаем прогноз
            X_test = df_test_for_pred.drop(['group_shop_cat'],axis=1)
            model = dict_mod[name_model]            
            y_pred = model.predict(X_test)

            # получим широкий датасет, где для каждой строки будет столбец с прогнозом на конкретную дату на 14 дней вперёд
            predictions_data = pd.DataFrame(data=y_pred, columns=list_columns)
            predictions_data['group_shop_cat'] = group_shop_cat
            predictions_data['pr_sales_type_id'] = pr_sales_type
            predictions_data['pr_uom_id'] = pr_pr_uom_id

            # развернём получившийся датасет в узкую таблицу (поскольку именно в таком формате она нужна по заданию)
            predictions_data = predictions_data.melt(['group_shop_cat', 'pr_sales_type_id', 'pr_uom_id'], 
                                                     var_name='date', 
                                                     value_name='target')
            list_predict.append(predictions_data)

    # соединим прогнозы каждой из модели
    pred = pd.concat(list_predict)
    # получим пересечение всех товаров со всеми магазинами, добавим столбец с таким же названием какой был у моделей
    # и соединим их с нашим рогнозом
    st_pr_df = st_df.merge(pr_df, how='cross')
    st_pr_df['group_shop_cat'] = st_pr_df['group_shop'] + '_' + st_pr_df['group_cat']
    st_pr_df = st_pr_df.drop(['group_cat', 'group_shop'], axis=1)
    pr_data = pred.merge(st_pr_df, on='group_shop_cat', how='left')
    df_for_pred = df_for_pred.merge(pr_df[['pr_sku_id', 'pr_uom_id']], on = 'pr_sku_id', how='left')

    # приведём столбец с датой и прогнозного датасета и датасета с запросом к одному типу, чтобы не было проблем при их соединении
    pr_data['date'] = pd.to_datetime(pr_data['date'])
    df_for_pred['date'] = pd.to_datetime(df_for_pred['date'])

    # объединим прогнозный датасет и датасет с запросом на предсказание с учётом того нужно ли возвращать все столбцы
    if only_st_sku:
        pr_data = pr_data.merge(df_for_pred, on=['st_id', 'pr_sku_id', 'date'], how='right')                
    else:
        pr_data = pr_data.merge(df_for_pred, on=['st_id', 'pr_sku_id', 'date', 'pr_sales_type_id', 'pr_sku_id'], how='right')
    pr_data = pr_data.fillna(0)

    # обработаем возврат информации по промо
    if not_promo:
        pr_data = pr_data[pr_data['pr_sales_type_id'] == 0]
        return_list = ['st_id', 'pr_sku_id', 'date', 'target']
        if return_y_true and 'y_true' in pr_data.columns:
            return_list.append('y_true')
        pr_data = pr_data[return_list]
    # сгруппируем данные по столбцам и сагрегируем таргет как среднее
    # одинаковые записи могу твозник при пересечении разных множеств (например изменение типа продажи товара по штукам или на вес и т.д.) 
    pr_columns = list(pr_data.columns)
    pr_columns.remove('target')
    if not only_st_sku:
        pr_columns.remove('y_true')
        pr_data = pr_data.groupby(pr_columns)[['target', 'y_true']].agg('mean').reset_index(drop=False)
    else:
        pr_data = pr_data.groupby(pr_columns)[['target']].agg('mean').reset_index(drop=False)
    pr_data['date'] = pr_data['date'].astype('str')

    return pr_data
