import pandas as pd
import numpy as np 
import joblib

def clean(df):
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
    # Some values of GarageYrBlt are corrupt, so we'll replace them
    # with the year the house was built
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    # Names beginning with numbers are awkward to work with
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "Threeseasonporch",
    }, inplace=True,
    )
    return df
    
def mathematical_transforms(df):
    df["LivLotRatio"] = df.GrLivArea / df.LotArea
    df["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    df["_OQuGLA"] = df.OverallQual * df.GrLivArea
    return df

def group_transforms(df):
    df["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
    return df

def main(x):
    df = x.copy
    df_clean = clean(df)
    df_transform_math = mathematical_transforms(df_impute)
    df_transform = group_transforms(df_transform_math)

    
    joblib.dump(df_transform, f"output/feature.pkl")
    return df_transform
