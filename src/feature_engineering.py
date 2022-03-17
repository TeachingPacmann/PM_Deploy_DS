import pandas as pd
import numpy as np 
import joblib

def clean(df):
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
    }, inplace=True,
    )
    return df
    
def mathematical_transforms(df):
    df["_OQuGLA"] = df.OverallQual * df.GrLivArea
    return df

def main(x):
    df = x.copy
    df_clean = clean(df)
    df_transform_math = mathematical_transforms(df_clean)
    df_transform = group_transforms(df_transform_math)

    joblib.dump(df_transform, f"output/feature.pkl")
    return df_transform
