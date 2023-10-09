import pandas as pd
from datetime import datetime, timedelta
import numpy as np

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
    df = df.drop(['pr_cat_id', 'pr_subcat_id', 'pr_group_id'], axis=1)    
    return df


def get_mean_sale_group(df: pd.DataFrame, sales_df_train: pd.DataFrame) -> pd.DataFrame:
    '''функция проводит сегментацию магазинов в зависимости от количества средних продаж
    :param df: датасет с информацией по магазинам
    :param sales_df_train: датасет с информацией о продажах
    :return df: Возвращает датасет со столбцом сегментации магазинов
    ''' 
    df_st_mean = (sales_df_train.groupby('st_id')['pr_sales_in_units'].agg('mean')
                      .reset_index(drop=False)
                      .sort_values(by='pr_sales_in_units'))
    df = df.merge(df_st_mean, on ='st_id')
    
    df['mean_seale'] = np.nan
    df.loc[df['pr_sales_in_units'] < 2.5, 'mean_seale'] = 'mean_seale_1'
    df.loc[(df['pr_sales_in_units'] >= 2.5) & (df['pr_sales_in_units'] < 4), 'mean_seale'] = 'mean_seale_2'
    df.loc[(df['pr_sales_in_units'] >= 4) & (df['pr_sales_in_units'] < 5), 'mean_seale'] = 'mean_seale_3'
    df.loc[(df['pr_sales_in_units'] >= 5), 'mean_seale'] = 'mean_seale_4'
  
    df = df.drop('pr_sales_in_units', axis=1)
    return df


def get_ratio_promo(df: pd.DataFrame, sales_df_train: pd.DataFrame) -> pd.DataFrame:
    '''функция рассчитывает отношение товаров по промо к товарам без промо в рамках конкретного магазина
    :param df: датасет с информацией по магазинам
    :param sales_df_train: датасет с информацией о продажах
    :return df: Возвращает датасет со столбцом ratio_promo
    ''' 
    df_st_ratio_promo = (sales_df_train.groupby(['st_id', 'pr_sales_type_id'])['pr_sales_in_units']
                                .agg('sum')
                                .reset_index(drop=False))
    df_st_ratio_promo = (df_st_ratio_promo.pivot(columns = 'pr_sales_type_id', index = 'st_id', values = 'pr_sales_in_units')
                            .reset_index(drop=False))

    df_st_ratio_promo['ratio_promo'] = df_st_ratio_promo[1] / df_st_ratio_promo[0]

    df = df.merge(df_st_ratio_promo[['st_id', 'ratio_promo']], on ='st_id')
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
                                                                    .shift(13).shift()
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


def get_st_df(df: pd.DataFrame, sales_df_train: pd.DataFrame) -> pd.DataFrame:
    '''функция для получения датасета по магазинам
    :param df: датасет с информацией по магазинам
    :param sales_df_train: датасет с продажами
    :return df: Возвращает агрегированный датасет по магазинам
    '''
    # Оставим только активные магазины и удалим столбец st_is_active
    df = df.copy(deep=True)
    df = df[df['st_is_active']!=0]
    df = df.drop('st_is_active', axis=1)  
    
    df = get_mean_sale_group(df, sales_df_train)
    df = get_ratio_promo(df, sales_df_train)
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


def get_clear_df(df: pd.DataFrame, df_date: pd.DataFrame, limit_nan: int = 10) -> pd.DataFrame:
    '''функция очищает датасет от отрицательных значений, восстанавливает все даты для всех товаров, заполняет получившиеся пропуски и 0
    :param df: входящий датасет
    :param df_date: датасет с датами
    :param limit_nan: лимит заполнения пропусков
    :return df: обработанный датасет
    '''
    new_df = df.copy(deep=True)
    # оставим в датасете только значения с целевым признаком больше или равных 0
    new_df = new_df[new_df['pr_sales_in_units']>=0].reset_index(drop=True)
    if new_df.shape[0] > 0:
        #сделаем словарь для заполнения пропусков    
        list_col = list(new_df.columns)
        list_col.remove('pr_sales_in_units')
        dict_for_fillna = {k: new_df.loc[0, k] for k in list_col}

        # присоединим датасет с датами
        new_df = new_df.merge(df_date, on='date', how='right')
        
        new_df['pr_sales_in_units'] = new_df['pr_sales_in_units'].fillna(0)
        new_df = new_df.sort_values('date').reset_index(drop=True)
        
        # заполним пропуски, отступаем первые 60 дней (большая их часть будет отброшена за счёт пропусков в скользящих средних)
        # заполняем только первые 10 значений, чтобы модель не стремилась предсказывать везде 0
        for k, v in dict_for_fillna.items():
            new_df.loc[60:,k] = new_df.loc[60:,k].fillna(v, limit=limit_nan)
                   
    else:
        new_df = None
    
    return new_df


def get_date_and_weekday(df: pd.DataFrame) -> pd.DataFrame:
    '''функция для приведения даты в нужный формат, получения дня недели, отметки, что это выходной день, и флаги для каждого дня недели
    :param df: входящий датасет
    :return df: обработанный датасет
    '''
    #приведём дату в нужный формат, укажем новый индекс и день недели
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday 
    df['weekend'] = (df['weekday'] == 5) | (df['weekday'] == 6)
    # преобразуем день недели через тригонометрическую функцию    
    df['weekday_cos'] =  np.cos((2 * np.pi) / 7 * (df['weekday'] + 1))
    # аналогично поступим с неделями
    df['week'] = df['date'].dt.isocalendar().week
    df['week'] =  np.cos((2 * np.pi) / 53 * (df['week'] + 1))
    
    df.index = df['date']
    df = df.drop('date', axis=1)
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
                                                           .shift(13).shift()
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
                               .shift(13).shift()
                               .rolling(n_day)
                               .mean())
        df[new_name_max] = (df[column]
                               .shift(13).shift()
                               .rolling(n_day)
                               .max())
        df[new_name_min] = (df[column]
                               .shift(13).shift()
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
        df[new_name] = (df[column].shift(13).shift(n_day)) 
                
    df['mean_week_lag'] = df[['lag_5', 'lag_6', 'lag_7']].mean(axis=1)
    df['ratio_lag_1_to_mean_week_lag'] = df['lag_1'] / df['mean_week_lag']
    df['ratio_lag_1_to_mean_week_lag'] = df['ratio_lag_1_to_mean_week_lag'].fillna(0)
    return df


def get_list_before_2days_holidays(list_holidays: list, n_day_before: int = 2) -> list:
    '''функция для получения списка дат 2 дней перед праздниками (с объединением продолжительных праздников)
    :param list_holidays: список с праздниками с учётом переносов
    :param n_day_before: количество дней перед праздниками
    :return before_2days_holidays: список дат 2 дней перед праздникам
    ''' 
    before_2days_holidays = [list_holidays[0]]
    for i in range(1, len(list_holidays)):
        last_date = datetime.strptime(list_holidays[i - 1], '%Y-%m-%d') 
        date = datetime.strptime(list_holidays[i], '%Y-%m-%d')
        if last_date != date - timedelta(days = 1):
            new_list_day_befor = []
            for i in range(1, n_day_before + 1):
                new_date = date - timedelta(days = i)
                new_date = datetime.strftime(new_date, '%Y-%m-%d')
                new_list_day_befor.append(new_date)
            before_2days_holidays += new_list_day_befor
    return before_2days_holidays


def get_features_ny_e_h(df: pd.DataFrame, list_holidays: list, before_2days_holidays_list: list) -> pd.DataFrame:
    '''функция для обработки нового года и пасхи и праздников
    :param df: входящий датасет
    :param list_holidays: список с праздниками с учётом переносов
    :param before_2days_holidays_list: список дат 2 дней перед праздникам
    :return df: обработанный датасет
    ''' 
    #Добавим флаг нового года и пасхи
    df['new_year'] = df.index=='2023-01-01'
    df['easter'] = df.index=='2023-04-16'
    #
    df['week_after_new_year'] = (df.index > '2023-01-01') & (df.index <= '2023-01-08')
    df['week_after_easter'] = (df.index > '2023-04-16') & (df.index <= '2023-01-23')
    # Добавим флаг после нового года и пасхи
    df['week_before_new_year'] = (df.index > '2022-12-24') & (df.index < '2023-01-01')
    df['week_before_easter'] = (df.index > '2023-04-09') & (df.index <= '2023-04-16')

    #Обработаем список праздников
    df['holiday'] = df.index.isin(list_holidays)
    # получим данные о днях, которые являются двумя днями, предшествующими праздникам
    df['before_2days_holidays'] = df.index.isin(before_2days_holidays_list)
    return df


def rename_target(df: pd.DataFrame, column:str) -> pd.DataFrame:
    '''функция для переименования целевого признака
    :param df: входящий датасет
    :param column: старое название целевого столбца
    :return df: обработанный датасет
    ''' 
    df = df.rename(columns = {column: 'target'})    
    return df


def get_dict(df: pd.DataFrame, 
             df_date: pd.DataFrame,
             column: str = 'pr_sales_in_units',
             limit_nan: int = 10, 
             weekday_column: str = 'weekday',
             n_week_for_lag: int = 4,
             n_day_rolling_list: list = [7, 14, 30],
             n_day_lag_list = list(range(1,15)),
             list_holidays: list = [], 
             n_day_before: int = 2) -> dict:
    '''функция для преобразования датасета в словарь датасетов с обработкой каждого из датасетов(позволяет ускорить работу в дальнейшем, так как 
    признаки будут высчитываться в рамках одних и тех же групировок (по факту партицирование над пандами))
    :param df: входящий датасет
    :param df_date: датасет с датами
    :param column: столбец для рассчёта значений
    :param limit_nan: лимит заполнения пропусков
    :param weekday_column: столбец с номером дня недели
    :param n_week_for_lag: список с днями по которым нужно получить лаги
    :param n_day_rolling_list: список с днями по которым нужно получить лаги   
    :param n_day_lag_list: список с количеством дней по которому рассчитываются статистики
    :param list_holidays: список с праздниками с учётом переносов   
    :param n_day_before: количество дней перед праздниками
    :return dict_df: словарь датасетов
    '''
    #согласно тз оставим только товары без промо
    df = df[df['pr_sales_type_id']==0]
    df = df.drop('pr_sales_type_id', axis=1)
    df['st_sku'] =  df['st_id'] + '_' + df['pr_sku_id']    
    dict_df = {}    
    before_2days_holidays_list = get_list_before_2days_holidays(list_holidays, n_day_before)
    for x in df['st_sku'].unique():
        new_df = df.loc[df['st_sku']==x].copy(deep=True).reset_index(drop=True)
        new_df = get_clear_df(new_df, df_date, limit_nan)
        if not new_df is None:         
            new_df = get_date_and_weekday(new_df)  
            new_df = get_mean_in_day(new_df, weekday_column, column, n_week_for_lag)
            new_df = get_rolling(new_df, column, n_day_rolling_list)
            new_df = get_lag(new_df, column, n_day_lag_list)
            new_df = get_features_ny_e_h(new_df, list_holidays, before_2days_holidays_list)
            new_df = rename_target(new_df, column)            
            new_df = new_df.drop('st_sku', axis=1)      
            
            dict_df[x] = new_df
    
    return dict_df


def get_features_for_ts(df: pd.DataFrame, 
                        column: str = 'pr_sales_in_units',
                        limit_nan: int = 10, 
                        weekday_column: str = 'weekday',
                        n_week_for_lag: int = 4,
                        n_day_rolling_list: list = [7, 14, 30],
                        n_day_lag_list = list(range(1,15)),
                        list_holidays: list = [], 
                        n_day_before: int = 2) -> pd.DataFrame:
    '''функция для получения фичей времянного ряда при помощи написанных выше функций
    :param df: входящий датасет
    :param column: столбец для рассчёта значений
    :param limit_nan: лимит заполнения пропусков
    :param weekday_column: столбец с номером дня недели
    :param n_week_for_lag: список с днями по которым нужно получить лаги
    :param n_day_rolling_list: список с днями по которым нужно получить лаги   
    :param n_day_lag_list: список с количеством дней по которому рассчитываются статистики
    :param list_holidays: список с праздниками с учётом переносов   
    :param n_day_before: количество дней перед праздниками
    :return df: обработанный датасет
    ''' 
    df = df.copy(deep = True)       
    df_date = pd.DataFrame(data=df['date'].copy(deep=True).unique(), columns=['date'])
    dict_df = get_dict(df,
                       df_date,
                       column = column,
                       limit_nan = limit_nan,
                       weekday_column = weekday_column,
                       n_week_for_lag = n_week_for_lag,
                       n_day_rolling_list = n_day_rolling_list,
                       n_day_lag_list = n_day_lag_list,
                       list_holidays = list_holidays,
                       n_day_before = n_day_before)
       
    return pd.concat(dict_df.values(),axis=0).sort_values(['date','st_id', 'pr_sku_id'])


def combine_shops_sales(st_df: pd.DataFrame, df_ts: pd.DataFrame, pr_df: pd.DataFrame) -> pd.DataFrame:
    '''функция для объединения датафреймов
    :param st_df: обработанный датасет по магазинам
    :param df_ts: обработанный датасет по прадажам
    :param pr_df: обработанный датасет по товарам
    :return df: объединённый датасет
    '''   
    #объединим получившиеся датасеты, перезададим индексы и удалим пропуски в отсутсвующих торговых центрах
    df_ts['date'] = df_ts.index
    df = df_ts.merge(st_df, on ='st_id', how='left')
    df = df.merge(pr_df, on ='pr_sku_id', how='left')
    df.index = df['date']
    df = df.drop('date', axis=1)
    # Создадим столбец с уникальным сочитанием группы магазина и группы категории товра 
    # удалим ненужные столбцы и пропуски в данных
    df['group_shop_cat'] = df['group_shop'] + '_' + df['group_cat']
    df = df.dropna()
    df = df.drop(['group_shop', 'group_cat'], axis=1)
    
      # преобразуем формат столбцов 
    df['st_type_format_id'] = df['st_type_format_id'].astype('int')
    df['st_type_loc_id'] = df['st_type_loc_id'].astype('int')    
    df['st_type_size_id'] = df['st_type_size_id'].astype('int')
    return df


def get_test_df(df: pd.DataFrame, first_date:str = None) -> pd.DataFrame:
    '''функция для получения тестового датасета 
    :param df: входящий датасет
    :param first_date: первая дата от которой формируется двухнедельная заготовка для прогноза (если None, то берётся последня дата в датасете)
    :return df: датасет для прогноза
    '''    
    # Если дата среза неизвестна, тогда берём последнюю дату в датасете
    if first_date is None:
        first_date = df['date'].max()
        
    unique_st_pr = df[df['pr_sales_type_id']==0].copy(deep=True)
    unique_st_pr = unique_st_pr[['st_id', 'pr_sku_id', 'pr_sales_type_id']].drop_duplicates()
    
    # получим заготовку для предсказаний (будем её делать с завтрашнего дня, поэтому диапазон от 1 до 15)
    df_list = [] 
    for i in range(1, 15):
        date = datetime.strptime(first_date, '%Y-%m-%d') + timedelta(days = i)
        date = datetime.strftime(date, '%Y-%m-%d')
        new_df = unique_st_pr.copy(deep=True)
        new_df['pr_sales_in_units'] = 0
        new_df['date'] = date
        df_list.append(new_df)
    new_df = pd.concat(df_list, axis=0)
    
    #добавим информацию по предыдущему периоду     
    old_df = df[['st_id', 'pr_sales_type_id', 'pr_sku_id', 'date', 'pr_sales_in_units']]
    
    return pd.concat([old_df, new_df], axis=0).reset_index(drop=True)


def preproceccing_df(df: pd.DataFrame, 
                     pr_df: pd.DataFrame, 
                     st_df: pd.DataFrame, 
                     first_date: bool = None,
                     is_train: bool = True, 
                     column: str = 'pr_sales_in_units',
                     limit_nan: int = 10, 
                     weekday_column: str = 'weekday',
                     n_week_for_lag: int = 4,
                     n_day_rolling_list: list = [7, 14, 30],
                     n_day_lag_list = list(range(1,15)),
                     list_holidays: list = [], 
                     n_day_before: int = 2) -> pd.DataFrame:
    '''Полная функция предообработки данных
    :param df: входящий датасет по продажам
    :param st_df: обработанный датасет с информацией по магазинам
    :param st_df: обработанный датасет с информацией по товарам
    :param first_date: первая дата от которой формируется двухнедельная заготовка для прогноза или по которой обрезается тренировочный датасет
    :param is_train: флаг, показывающий обрабатывается тренировочный набор или нет
    :param column: столбец для рассчёта значений
    :param limit_nan: лимит заполнения пропусков
    :param weekday_column: столбец с номером дня недели
    :param n_week_for_lag: список с днями по которым нужно получить лаги
    :param n_day_rolling_list: список с днями по которым нужно получить лаги   
    :param n_day_lag_list: список с количеством дней по которому рассчитываются статистики
    :param list_holidays: список с праздниками с учётом переносов   
    :param n_day_before: количество дней перед праздниками
    :return df: обработанный датасет
    '''    
    pr_df = get_cat(pr_df)
    st_df = get_st_df(st_df, df)  
    
    if first_date is not None:
        #если указана первая дата перед прогнозом, то обрезаем датасет
        df = df.copy(deep=True)
        df.index = pd.to_datetime(df['date'])        
        df = df.sort_index()
        df = df.loc[:first_date]
            
    if first_date is None:
        # eсли дата среза неизвестна, тогда берём последнюю дату в датасете
        first_date = df['date'].max()  
        
    if not is_train:
        #если не тренировочный, то получаем данные через функцию
        df = get_test_df(df, first_date = first_date)
    
    df = df[['st_id', 'pr_sales_type_id', 'pr_sku_id', 'date', 'pr_sales_in_units']]
    df = get_features_for_ts(df, 
                             column = column,
                             limit_nan = limit_nan,
                             weekday_column = weekday_column,
                             n_week_for_lag = n_week_for_lag,
                             n_day_rolling_list = n_day_rolling_list,
                             n_day_lag_list = n_day_lag_list,
                             list_holidays = list_holidays,
                             n_day_before = n_day_before)
    if not is_train:
        first_date = datetime.strptime(first_date, '%Y-%m-%d') + timedelta(days = 1)
        first_date = datetime.strftime(first_date, '%Y-%m-%d')
        df = df.loc[first_date:]
    df = combine_shops_sales(st_df, df, pr_df)
    return df
