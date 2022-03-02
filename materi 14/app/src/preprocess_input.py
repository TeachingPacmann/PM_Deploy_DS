import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import yaml


def numerical_imputer(numerical,
                    state = 'transform'):

    index = numerical.index
    cols = numerical.columns

    if state == 'fit':
        imputer = SimpleImputer(
            missing_values=np.nan,
            strategy="mean")

        imputer.fit(numerical)
        joblib.dump(imputer,
                    "app/models/estimator/numerical_imputer.pkl")
    elif state == 'transform':
        imputer = joblib.load("app/models/estimator/numerical_imputer.pkl")

    imputed = imputer.transform(numerical)
    imputed = pd.DataFrame(imputed)
    imputed.index = index
    imputed.columns = cols
    return imputed


def normalization(x_all,
                  state = 'fit'):
    index = x_all.index
    cols = x_all.columns


    if state == 'fit':
        normalizer = StandardScaler()
        normalizer.fit(x_all)
        joblib.dump(normalizer,
                    "app/models/estimator/normalizer.pkl")

    elif state == 'transform':
        normalizer = joblib.load("app/models/estimator/normalizer.pkl")

    normalized = normalizer.transform(x_all)
    normalized = pd.DataFrame(normalized)
    normalized.index = index
    normalized.columns = cols
    return normalized