import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from datetime import datetime, timedelta

import pickle

import os

import warnings
warnings.filterwarnings('ignore')


### Далее функции для обработки датасета по товарам
def get_cat(df: pd.DataFrame) -> pd.DataFrame:
    '''функция для преобразования сочетаний товар-магазин в группа магазина-категория товаров
    :param df: сырой датасет с информацией о товарах
    :return df: Возвращает датасет со столбцом категорий в соответствии с выбранной разбивкой 
    '''
    df.loc[df['pr_cat_id']=='c559da2ba967eb820766939a658022c8', 'group_cat'] = 'cat_1'
    df.loc[df['pr_subcat_id']=='60787c41b04097dfea76addfccd12243', 'group_cat'] = 'cat_2'
    df.loc[df['pr_subcat_id']=='ca34f669ae367c87f0e75dcae0f61ee5', 'group_cat'] = 'cat_3'
    df.loc[df['pr_cat_id'].isin(['e58cc5ca94270acaceed13bc82dfedf7', 
                                          'fb2fcd534b0ff3bbed73cc51df620323']), 'group_cat'] = 'cat_4'
    df.loc[df['pr_cat_id'].isin(['3de2334a314a7a72721f1f74a6cb4cee', 
                                          'f3173935ed8ac4bf073c1bcd63171f8a',
                                          'b59c67bf196a4758191e42f76670ceba']), 'group_cat'] = 'cat_5'
    df.loc[df['pr_cat_id'].isin(['28fc2782ea7ef51c1104ccf7b9bea13d', 
                                          '9701a1c165dd9420816bfec5edd6c2b1', 
                                          '5caf41d62364d5b41a893adc1a9dd5d4', 
                                          '186a157b2992e7daed3677ce8e9fe40f', 
                                          '2df45244f09369e16ea3f9117ca45157', 
                                          '6d9c547cf146054a5a720606a7694467', 
                                          '535ab76633d94208236a2e829ea6d888', 
                                          'a6ea8471c120fe8cc35a2954c9b9c595']), 'group_cat'] = 'cat_6'
    df.loc[df['pr_cat_id']=='f9ab16852d455ce9203da64f4fc7f92d', 'group_cat'] = 'cat_7'
    df.loc[df['pr_cat_id'].isin(['b7087c1f4f89e63af8d46f3b20271153', 
                                          'f93882cbd8fc7fb794c1011d63be6fb6']), 'group_cat'] = 'cat_8'
    df.loc[df['pr_cat_id']=='faafda66202d234463057972460c04f5', 'group_cat'] = 'cat_9'
    df.loc[df['pr_cat_id']=='fd5c905bcd8c3348ad1b35d7231ee2b1', 'group_cat'] = 'cat_10'
    df.loc[df['pr_cat_id']=='c9f95a0a5af052bffce5c89917335f67', 'group_cat'] = 'cat_11'
    df['group_cat'] = df['group_cat'].fillna('cat_12')
    df['pr_uom_id'] = df['pr_uom_id']==1
    df = df.drop(['pr_cat_id', 'pr_subcat_id'], axis=1) #df['pr_uom_id']
    return df


### Далее функции для обработки датасета по магазинам
def get_ohe(df: pd.DataFrame, column: list) -> pd.DataFrame:   
    '''функция для OHE, позволяет получить датасет с названими столбцов
    :param df: датасет для кодировки OHE
    :param column: столбцы для кодировки
    :return df: Возвращает датасет со столбцами закодированными OHE с корректными названиями
    ''' 
    df_ohe = pd.get_dummies(df[column])
    new_columns = [f'{column}_{c}' for c in df_ohe.columns]
    df_ohe.columns = new_columns
    df = df.drop(column, axis=1)    
    df = pd.concat([df, df_ohe], axis=1)
    return df

def do_ohe_and_clear_unective(df: pd.DataFrame) -> pd.DataFrame: 
    '''функция вызывает OHE  к нужным столбцам магазинов и оставляет только активные магазины
    :param df: сырой датасет с информацией по магазинам
    :return df: Возвращает датасет обработанный OHE датасет по магазинам
    ''' 
    list_columns_for_ohe = ['st_city_id', 'st_division_code', 'st_type_format_id', 'st_type_loc_id', 'st_type_size_id']
    for column in list_columns_for_ohe:
        df = get_ohe(df, column)
    df = df[df['st_is_active']!=0]
    df = df.drop('st_is_active', axis=1)
    return df


def segmentetion_st(df: pd.DataFrame, sales_df_train: pd.DataFrame) -> pd.DataFrame:
    '''функция проводит сегментацию магазинов в зависимости от количества средних продаж
    :param df: датасет с информацией по магазинам
    :param sales_df_train: датасет с информацией о продажах
    :return df: Возвращает датасет со столбцами сегментации
    ''' 
    df_st_mean = (sales_df_train.groupby('st_id')['pr_sales_in_units'].agg('mean')
                    .reset_index(drop=False)
                    .sort_values(by='pr_sales_in_units'))

    df = df.merge(df_st_mean, on ='st_id')

    df['mean_seale_1'] = df['pr_sales_in_units'] < 2.5
    df['mean_seale_2'] = (df['pr_sales_in_units'] >= 2.5) & (df['pr_sales_in_units'] < 4)
    df['mean_seale_3'] = (df['pr_sales_in_units'] >= 4) & (df['pr_sales_in_units'] < 5)
    df['mean_seale_4'] = (df['pr_sales_in_units'] >= 5)
    df = df.drop('pr_sales_in_units', axis=1)
    return df


def get_df_ts_store(df: pd.DataFrame, store_columns: str) -> pd.DataFrame:
    '''функция проводит агрегацию по соответствующему столбцу и делает индекс в дату
    :param df: датасет с информацией по магазинам
    :param store_columns: столбец для агрегации
    :return df_st_id: Возвращает агрегированный датасет
    ''' 
    df_st_id = df.groupby(['date',store_columns])['pr_sales_in_units'].agg('sum').reset_index(drop=False)
    df_st_id.index = df_st_id['date']
    df_st_id = df_st_id.drop('date', axis=1)
    return df_st_id


def preprocessing_st_df(df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    '''функция вызывает ряд преобразований написанных выше над датасетом по магазинам    
    :param df: входящий датасет по магазинам
    :param sales_df: входящий датасет по продажам
    :return df: Возвращает обработанный датасет
    ''' 
    df =  do_ohe_and_clear_unective(df)
    df =  segmentetion_st(df, sales_df)
    df = get_group_st(df, sales_df)
    df = df.drop('st_city_id_1587965fb4d4b5afe8428a4a024feb0d', axis=1) #удаляем столбец с константой
    return df


def get_rolling_mean(df: pd.DataFrame, group_column: str, column: str) -> pd.DataFrame:
    '''функция нахоит скользящее среднее за 30 дней
    :param df: входящий датасет по магазинам
    :param group_column: столбец для группировки
    :param column: столбец для расчёта скользящего среднего
    :return df: Возвращает обработанный датасет
    '''
    list_group_column = df[group_column].unique()
    for gr_col in list_group_column:
        new_name = f'rolling_mean_{column}'
        df.loc[df[group_column]==gr_col, new_name] = (df[df[group_column]==gr_col][column]
                                                                    .shift()
                                                                    .rolling(30)
                                                                    .mean())
    df = df.drop(column, axis=1)
    return df


def get_ratio_summer_winter(df: pd.DataFrame) -> pd.DataFrame:
    '''функция находит отношение продаж за летний и зимний период
    :param df: входящий датасет
    :return new_df: Возвращает датасет из id магазина и значения отношения продаж
    '''
    df_july = df.loc['2023-07-01'][['st_id', 'rolling_mean_pr_sales_in_units']]
    df_jan = df.loc['2023-01-01'][['st_id', 'rolling_mean_pr_sales_in_units']]
    new_df = df_july.merge(df_jan, on='st_id', how='left')
    new_df.loc[new_df['rolling_mean_pr_sales_in_units_y']==0, 'rolling_mean_pr_sales_in_units_y'] =0.01 #защита от деления на 0
    new_df['ratio_summer_winter'] = (new_df['rolling_mean_pr_sales_in_units_x']
                                     / new_df['rolling_mean_pr_sales_in_units_y'])
    return new_df[['st_id', 'ratio_summer_winter']]


def get_group_st(df: pd.DataFrame, sales_df_train: pd.DataFrame) -> pd.DataFrame:
    '''функция рассчитывает метрики для разделения магазинов по разным группам на основе продаж, и присоединяет к датасету по магазинам
    :param df: входящий датасет по магазинам
    :param sales_df_train: датасет по продажам
    :return df: Возвращает датасет по магазинам со столбцом группы
    '''
    df_st_id = get_df_ts_store(sales_df_train, 'st_id')
    df_st_id = get_rolling_mean(df_st_id, 'st_id', 'pr_sales_in_units')

    df_st_id_1 = df_st_id.groupby('st_id')['rolling_mean_pr_sales_in_units'].agg('max').reset_index(drop=False)
    df_st_id_2 = get_ratio_summer_winter(df_st_id)
    df_st_id = df_st_id_1.merge(df_st_id_2, on='st_id', how='left')

    df_st_id.loc[df_st_id['rolling_mean_pr_sales_in_units']<500,'group_shop'] = 'group_1'
    df_st_id.loc[df_st_id['ratio_summer_winter']>1,'group_shop'] = 'group_2'
    df_st_id['group_shop'] = df_st_id['group_shop'].fillna('group_3')
    df_st_id = df_st_id.drop(['rolling_mean_pr_sales_in_units', 'ratio_summer_winter'], axis=1)

    df = df.merge(df_st_id, on ='st_id')
    return df


### Далее функции для обработки датасета по продажам
def get_clear_df(df: pd.DataFrame, var_for_na: float = 0.01, var_for_null: float = 0.01) -> pd.DataFrame:
    '''функция очищает датасет от отрицательных значений, восстанавливает все даты для всех товаров, заполняет получившиеся пропуски и 0
    :param df: входящий датасет
    :param var_for_na: значение для заполнения пропусков
    :param var_for_null: значения для замены нулей
    :return df: обработанный датасет
    '''
    dict_df = {}     
    df_date = pd.DataFrame(data=df['date'].copy(deep=True).unique(), columns=['date'])
    df_date = df_date.sort_values('date').reset_index(drop=True)
    df = df.copy(deep=True)
    df['all_st_pr'] =  df['st_id'] + '_' + df['pr_sku_id']
    for x in df['all_st_pr'].unique():
        new_df = df.loc[df['all_st_pr']==x].copy(deep=True)
        
        # оставим в датасете только значения с целевым признаком больше или равных 0
        new_df = new_df[new_df['pr_sales_in_units']>=0].reset_index(drop=True)
        if new_df.shape[0] > 0:
            #сделаем словарь для заполнения пропусков
            list_col = list(new_df.columns)
            list_col.remove('pr_sales_in_units')
            dict_for_fillna = {k: new_df.loc[0, k] for k in list_col}

            new_df = new_df.merge(df_date, on='date', how='outer')
            
            for k, v in dict_for_fillna.items():
                new_df[k] = new_df[k].fillna(v)

            new_df['pr_sales_in_units'] = new_df['pr_sales_in_units'].fillna(var_for_na)
            new_df[new_df['pr_sales_in_units']==0] = var_for_null
            new_df['date'] = pd.to_datetime(new_df['date'])
            new_df = new_df.sort_values('date')
            new_df = new_df.drop('all_st_pr', axis=1)
            dict_df[x] = new_df
    return pd.concat(dict_df.values(),axis=0).sort_values(['date','st_id', 'pr_sku_id'])


def get_group_and_agg(df: pd.DataFrame, strareg_agg: str='mean') -> pd.DataFrame:
    '''Функция для аггрегации данных, поскольку будем модель предсказвает группы категория товара - группа магазинов, а не конкретный магазин
    :param df: входящий датасет
    :param strareg_agg: функция аггрегации при групперовке
    :return df: обработанный датасет
    '''
   
    group_column = ['st_id', 'group_cat', 'pr_sales_type_id', 'pr_uom_id']
        
    df = (df.groupby(['date', *group_column])['pr_sales_in_units']
                       .agg(strareg_agg)
                       .reset_index(drop=False)
                       .sort_values(['date', *group_column]))  
    df['group_column'] = df['st_id'].astype('str') + '_' + df['group_cat'].astype('str') + '_' + df['pr_sales_type_id'].astype('str')
    return df 


def get_date_and_weekday(df: pd.DataFrame) -> pd.DataFrame:
    '''функция для приведения даты в нужный формат, получения дня недели, отметки, что это выходной день, и флаги для каждого дня недели
    :param df: входящий датасет
    :return df: обработанный датасет
    '''
    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']
    df['weekday'] = df['date'].dt.weekday 
    df = df.drop('date', axis=1)
    df['weekend'] = (df['weekday'] == 5) | (df['weekday'] == 6)
    for i in range(7):
        df[f'weekday_{i}'] = df['weekday']==i         
    return df


def get_dict(df: pd.DataFrame, group_column: str) -> dict:
    '''функция для преобразования датасета в словарь датасетов (позволяет ускорить работу в дальнейшем, так как 
    признаки будут высчитываться в рамках одних и тех же групировок (по факту партицирование над пандами))
    :param df: входящий датасет
    :param group_column: название столбца для группировки
    :return dict_df: словарь датасетов
    '''
    dict_df = {}     
    for x in df[group_column].unique():
        new_df = df.loc[df[group_column]==x].copy(deep=True).reset_index(drop=True)
        dict_df[x] = new_df
    return dict_df


def processing_outliers(df: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    '''функция для удаления выбросов по отношению количеству продаж к скользящему среднему за последние 30 дней
    :param df: входящий датасет
    :param column: столбец для рассчёта значений
    :param threshold: порог, относительно которого идёт отбрасывание выбросов
    :return df: обработанный датасет
    '''
    df['rolling_mean_30'] = (df[column]
                           .shift()
                           .rolling(30)
                           .mean())
    df.loc[df['rolling_mean_30']==0, 'rolling_mean_30'] = 0.001 #защита от деления на 0
    df['ratio'] = df[column].shift() / df['rolling_mean_30']
    df.loc[df['ratio'] > threshold, column] = df.loc[df['ratio'] > threshold, 'rolling_mean_30'] * threshold
    df = df.drop(['ratio', 'rolling_mean_30'], axis=1)
    return df


def get_mean_in_day(df: pd.DataFrame, weekday_column: str, column: str, n_week) -> pd.DataFrame:
    '''функция для нахождения среднего в конкретный день недели за последние n_week недель
    :param df: входящий датасет
    :param weekday_column: столбец с номером дня недели
    :param column: столбец для рассчёта значений
    :param n_week: количество недель, по которым рассчитывается среднее
    :return df: обработанный датасет
    '''
    list_weekday_column = df[weekday_column].unique()
    for day in list_weekday_column: 
        new_name_mean = f'mean_in_weekday_{n_week}_week'  
        df.loc[(df[weekday_column]==day), new_name_mean] = (df[df[weekday_column]==day][column]
                                                           .shift()
                                                           .rolling(n_week)
                                                           .mean())
    df = df.drop(weekday_column, axis=1)     
    return df


def get_rolling(df: pd.DataFrame, column: str, n_day_list: list) -> pd.DataFrame:
    '''функция нахождение скользящих стаитистик (среднего, макимального, миниального) по окнам заданным в n_day_list
    :param df: входящий датасет
    :param column: столбец для рассчёта значений
    :param n_day_list: список с количеством дней по которому рассчитываются статистики
    :return df: обработанный датасет
    ''' 
    for n_day in n_day_list:
        new_name_mean = f'rolling_mean_{n_day}'
        new_name_max = f'rolling_max_{n_day}'
        new_name_min = f'rolling_min_{n_day}'        
        new_name_max_min = f'rolling_max_min_{n_day}'
        new_name_ratio = f'rolling_ratio_{n_day}'     
        df[new_name_mean] = (df[column]
                               .shift()
                               .rolling(n_day)
                               .mean())
        df[new_name_max] = (df[column]
                               .shift()
                               .rolling(n_day)
                               .max())
        df[new_name_min] = (df[column]
                               .shift()
                               .rolling(n_day)
                               .min())
        df[new_name_max_min] = (df[new_name_max] + df[new_name_min]) / 2
        
        df.loc[df[new_name_mean]==0, new_name_mean] = 0.001 #защита от деления на 0
        df[new_name_ratio] = df[new_name_max_min] / df[new_name_mean]
        df
    return df


def get_lag(df: pd.DataFrame, column: str, n_day_list: list) -> pd.DataFrame:
    '''функция для получения лагов по списку n_day_list
    :param df: входящий датасет
    :param column: столбец для рассчёта значений
    :param n_day_list: список с днями по которым нужно получить лаги
    :return df: обработанный датасет
    ''' 
    for n_day in n_day_list:
        new_name = f'lag_{n_day}'
        df[new_name] = (df[column].shift(n_day)) 
        
    df['mean_week_lag'] = df[['lag_5', 'lag_6', 'lag_7']].mean(axis=1)
    return df


def get_features_ny_e_h(df: pd.DataFrame, list_holidays: list) -> pd.DataFrame:
    '''функция для обработки нового года и пасхи и праздников
    :param df: входящий датасет
    :param list_holidays: список с праздниками с учётом переносов
    :return df: обработанный датасет
    ''' 
    #Добавим флаг нового года и пасхи
    df['new_year'] = df.index=='2023-01-01'
    df['easter'] = df.index=='2023-04-16'
    #
    df['week_after_new_year'] = (df.index > '2023-01-01') & (df.index <= '2023-01-08')
    df['week_after_easter'] = (df.index > '2023-04-16') & (df.index <= '2023-01-23')
    # Добавим флаг после нового года и пасхи
    df['week_befor_new_year'] = (df.index > '2022-12-24') & (df.index < '2023-01-01')
    df['week_befor_easter'] = (df.index > '2023-04-09') & (df.index <= '2023-04-16')

    #Обработаем список праздников
    df['holiday'] = df.index.isin(list_holidays)
    return df


def get_target_diapazon(df: pd.DataFrame, n_day: int, is_train: bool=True) -> pd.DataFrame:    
    '''функция для получения столбцв с таргетами
    :param df: входящий датасет
    :param n_day: горизонт прогноза в днях
    :param is_train: флаг, показывающий обрабатывается тренировочный набор или нет
    :return df: обработанный датасет
    ''' 
    df = df.copy(deep = True)
    df = df.rename(columns = {'pr_sales_in_units': 'target_0'})
    if is_train:
        for n_day in range(1, n_day):           
            new_name = f'target_{n_day}'
            df[new_name] = df['target_0'].shift(-n_day)
    return df


def get_features_for_ts(df: pd.DataFrame, 
                        is_train: bool = True, 
                        n_day_target: int = 14,
                        column: str = 'pr_sales_in_units',
                        strareg_agg: str='mean',
                        var_for_na: float = 0.01, 
                        var_for_null: float = 0,
                        weekday_column: str = 'weekday',
                        n_week_for_lag: int = 4,
                        n_day_rolling_list: list = [7, 14, 30],
                        n_day_lag_list: list = list(range(1,15)),
                        list_holidays : list= [],
                        drop_outliers: bool = False, 
                        threshold: float = 1.7) -> pd.DataFrame:
    '''функция для получения фичей времянного ряда при помощи написанных выше функций
    :param df: входящий датасет
    :param is_train: флаг, показывающий обрабатывается тренировочный набор или нет
    :param n_day_target: горизонт прогноза в днях
    :param column: столбец для рассчёта значений
    :param strareg_agg: функция аггрегации при групперовке
    :param var_for_na: значение для заполнения пропусков
    :param var_for_null: значения для замены нулей
    :param weekday_column: столбец с номером дня недели
    :param n_week_for_lag: список с днями по которым нужно получить лаги
    :param n_day_rolling_list: список с днями по которым нужно получить лаги   
    :param n_day_lag_list: список с количеством дней по которому рассчитываются статистики
    :param list_holidays: список с праздниками с учётом переносов
    :param drop_outliers: флаг, показывающий производить ли очистку выбросов
    :param threshold: порог, относительно которого идёт отбрасывание выбросов
    :return df: обработанный датасет
    ''' 
    df = df.copy(deep = True)
    df = get_clear_df(df, var_for_na, var_for_null)
    df = get_group_and_agg(df, strareg_agg) 
    new_dict = {}
    dict_df = get_dict(df, 'group_column')
    for x in dict_df:
        df = dict_df[x].copy(deep = True)         
        if drop_outliers:
            df = processing_outliers(df, column, threshold)
        df = get_date_and_weekday(df)  
        df = get_mean_in_day(df, weekday_column, column, n_week_for_lag)
        df = get_rolling(df, column, n_day_rolling_list)
        df = get_lag(df, column, n_day_lag_list)
        df = get_features_ny_e_h(df, list_holidays)
        df = get_target_diapazon(df, n_day_target, is_train=is_train)        
        new_dict[x] = df
        
    return pd.concat(new_dict.values(),axis=0).sort_values(['date','group_column'])


def combine_shops_sales(st_df, df_ts):
    '''функция для объединения датасета продаж с датасетом по магазинам
    :param df_ts: обработанный веремнной ряд по продажам
    :param st_df: обработанный датасет с информацией по магазинам
    :param is_train: флаг, показывающий обрабатывается тренировочный набор или нет 
    :return df: обработанный датасет
    ''' 
    #объединим получившиеся датасеты, перезададим индексы и удалим пропуски в отсутсвующих торговых центрах
    df = df_ts.merge(st_df, on ='st_id', how='left')
    df.index = df_ts.index
    df = df.dropna(subset='group_shop')
    # Создадим столбец с уникальным сочитанием группы магазина и группы категории товра 
    # удалим ненужные столбцы и пропуски в данных
    df['group_shop_cat'] = df['group_shop'] + '_' + df['group_cat']
    df = df.drop(['st_id', 'group_cat', 'group_column', 'group_shop'], axis=1)
    df = df.dropna()
    return df


def preproceccing_df(df: pd.DataFrame, 
                     pr_df: pd.DataFrame, 
                     st_df: pd.DataFrame, 
                     is_train: bool = True, 
                     n_day_target: int = 14,
                     column: str = 'pr_sales_in_units',
                     strareg_agg: str='mean',
                     var_for_na: float = 0.01, 
                     var_for_null: float  = 0,
                     weekday_column: str = 'weekday',
                     n_week_for_lag : int= 4,
                     n_day_rolling_list: list = [7, 14, 30],
                     n_day_lag_list: list = list(range(1,15)),
                     list_holidays: list = [],
                     drop_outliers: bool = False, 
                     threshold: float = 1.7) -> pd.DataFrame:
    '''Полная функция предообработки данных
    :param df: входящий датасет по продажам
    :param st_df: обработанный датасет с информацией по магазинам
    :param st_df: обработанный датасет с информацией по товарам
    :param is_train: флаг, показывающий обрабатывается тренировочный набор или нет
    :param n_day_target: горизонт прогноза в днях
    :param column: столбец для рассчёта значений
    :param strareg_agg: функция аггрегации при групперовке
    :param var_for_na: значение для заполнения пропусков
    :param var_for_null: значения для замены нулей
    :param weekday_column: столбец с номером дня недели
    :param n_week_for_lag: список с днями по которым нужно получить лаги
    :param n_day_rolling_list: список с днями по которым нужно получить лаги   
    :param n_day_lag_list: список с количеством дней по которому рассчитываются статистики
    :param list_holidays: список с праздниками с учётом переносов
    :param drop_outliers: флаг, показывающий производить ли очистку выбросов
    :param threshold: порог, относительно которого идёт отбрасывание выбросов
    :return df: обработанный датасет
    '''    
    df = df[['st_id', 'pr_sales_type_id', 'pr_sku_id', 'date', 'pr_sales_in_units']]
    df = df.merge(pr_df, on ='pr_sku_id')
    df = get_features_for_ts(df, 
                             is_train = is_train, 
                             n_day_target = n_day_target,
                             column = column,
                             strareg_agg = strareg_agg,
                             var_for_na = var_for_na, 
                             var_for_null = var_for_null,
                             weekday_column = weekday_column,
                             n_week_for_lag = n_week_for_lag,
                             n_day_rolling_list = n_day_rolling_list,
                             n_day_lag_list = n_day_lag_list,
                             list_holidays = list_holidays,
                             drop_outliers = drop_outliers, 
                             threshold = threshold)    
    df = combine_shops_sales(st_df, df)
    return df


def get_df_for_pred(sales_df_train: pd.DataFrame, 
                    pr_df: pd.DataFrame, 
                    st_df: pd.DataFrame, 
                    list_holidays: list, 
                    df_best_dict: dict) -> pd.DataFrame:
    '''функция, позволяет получить готовый датасет для прогноза
    :param sales_df_train: необработанный датасет по продажам
    :param st_df: необработанный датасет с информацией по магазинам
    :param st_df: необработанный датасет с информацией по товарам   
    :param list_holidays: список с праздниками с учётом переносов
    :param df_best_dict: словарь со спискам параметров для обработки датасетов
    :return df: обработанный датасет
    '''    
    #преобразуем данные с магазинами и товарами

    sales_df_train['date'] = pd.to_datetime(sales_df_train['date'])
    sales_df_train = sales_df_train.sort_values('date')
    df_for_pred = sales_df_train.copy(deep=True)
    
    #добавление в датасет следующей даты(для прогноза начиная с завтрашнего дня)
    last_date = sales_df_train['date'].max() 
    df_next_day = df_for_pred.loc[df_for_pred['date']==last_date]
    df_next_day['pr_sales_in_units'] = 0
    df_next_day['date'] = last_date + timedelta(days=1)
    df_next_day = df_next_day.reset_index(drop=True)
    df_for_pred = df_for_pred.reset_index(drop=True)
    df_for_pred.loc[len(df_for_pred.index)] = df_next_day.loc[0]
    
     #проводим препроцессинг только для данных, за 30 дней назад так как это максимальные скользящие средние, которые мы используем
    df_pred = preproceccing_df(df_for_pred[df_for_pred['date'] > last_date - timedelta(days=30)].copy(deep=True), 
                               pr_df, 
                               st_df,                              
                               list_holidays = list_holidays,
                               is_train=False,
                               **df_best_dict).loc[last_date + timedelta(days=1)]    
    if 'target_0' in df_pred.columns:
        df_pred = df_pred.drop('target_0', axis=1)
    return df_pred


def get_df_for_subbmisions(sales_df_train: pd.DataFrame) -> pd.DataFrame:  
    '''функция для получения датасета для предсказаний со всеми встречающимися товарами и магазинами
    :param sales_df_train: необработанный датасет по продажам
    :return df: обработанный датасет с уникальными значениями товар- магазин и датами на 14 дней вперёд
    '''   
    df = sales_df_train[['st_id', 'pr_sku_id']].reset_index(drop=True)
    df = df.drop_duplicates()
    last_date = pd.to_datetime(sales_df_train['date']).max()
    list_df = []    
    for i in range(1, 14):
        actual_date = last_date + timedelta(days=i)
        new_df = df.copy(deep=True)
        new_df['date'] = actual_date
        new_df.index = pd.to_datetime(new_df['date'])
        list_df.append(new_df)
    return pd.concat(list_df, axis=0)
