from typing import Dict
import numpy as np
import pandas as pd
import app.src.preprocess_input as prep
from app.constant import PREDICT_COLUMN_TYPE, PREDICT_COLUMN

def set_dtypes(data_input):
    '''
    Check data input datatypes consistency with predefined DTYPES
    Set data datatypes as DTYPE

    Parameters
    ----------
    data_input: pd.DataFrame
        DaraFrame for modeling

    Returns
    -------
    data: pd.DataFrame
        Checked dataset for columns consistency
    '''
    data = data_input.astype(PREDICT_COLUMN_TYPE)
    return data

def clean_column_name(df):
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
    }, inplace=True,
    )
    return df

def mathematical_transforms(df):
    df["_OQuGLA"] = df.OverallQual * df.GrLivArea
    return df

def adding_feature(x, state):
    df = x.copy()
    if (state=="predict"):
        df_transform = mathematical_transforms(df)
    else:
        df_clean = clean_column_name(df)
        df_transform = mathematical_transforms(df_clean)
    return df_transform


def construct_df(data_to_predict: dict) -> pd.DataFrame:
    df_to_predict = pd.DataFrame(data=data_to_predict)
    df_to_predict = set_dtypes(df_to_predict)
    # COLUMN = set(params['NUM_COLUMN']+params['CAT_COLUMN'])
    COLUMN = set(PREDICT_COLUMN)
    column_in_data = set(df_to_predict.columns)
    remain_columns = list(COLUMN-column_in_data)
    df_to_predict[remain_columns] = np.NaN
    return df_to_predict

def feature_engineering_predict(data_to_predict) -> pd.DataFrame:
    state = 'transform'
    data_to_predict = data_to_predict.copy()

    house_numerical = data_to_predict[PREDICT_COLUMN]
    df_add_feature = adding_feature(house_numerical, state="predict")
    df_numerical_imputed = prep.numerical_imputer(df_add_feature, state=state)
    x_predict = prep.normalization(df_numerical_imputed, state=state)
    return x_predict