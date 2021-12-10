import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
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
                    "output/numerical_imputer.pkl")
    elif state == 'transform':
        imputer = joblib.load("output/numerical_imputer.pkl")
        
    imputed = imputer.transform(numerical)
    imputed = pd.DataFrame(imputed)
    imputed.index = index
    imputed.columns = cols
    return imputed


def categorical_imputer(df_categorical):
    df = df_categorical.copy()
    df.fillna(value = 'KOSONG', inplace=True)
    return df


def one_hot_encoder(x_cat,
                    state='fit'):
    df = x_cat.copy()
    index = x_cat.index
    col = x_cat.columns
    
    if state == 'fit':
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(x_cat)
        joblib.dump(encoder,
                    "output/onehotencoder.pkl")
        
    elif state == 'transform':
        encoder = joblib.load("output/onehotencoder.pkl")
    
    encoded = encoder.transform(x_cat)
    feat_names = encoder.get_feature_names_out(col)
    encoded = pd.DataFrame(encoded)
    encoded.index = index
    encoded.columns = feat_names
    return encoded


def normalization(x_all,
                  state = 'fit'):
    index = x_all.index
    cols = x_all.columns
    

    if state == 'fit':
        normalizer = StandardScaler()
        normalizer.fit(x_all)
        joblib.dump(normalizer,
                    "output/normalizer.pkl")

    elif state == 'transform':
        normalizer = joblib.load("output/normalizer.pkl")
        
    normalized = normalizer.transform(x_all)
    normalized = pd.DataFrame(normalized)
    normalized.index = index
    normalized.columns = cols
    return normalized

def run(params, xpath, ypath, dump_path, state='fit'):
    house_variables = joblib.load(xpath)
    house_target = joblib.load(ypath)
    
    house_numerical = house_variables[params['NUM_COLUMN']]
    house_categorical = house_variables[params['CAT_COLUMN']]
    
    df_numerical_imputed = numerical_imputer(house_numerical, state=state)
    df_categorical_imputed = categorical_imputer(house_categorical)
    
    df_categorical_encoded = one_hot_encoder(df_categorical_imputed, state=state)
    
    df_joined = pd.concat([df_numerical_imputed, df_categorical_encoded], axis=1)
    
    df_normalized = normalization(df_joined, state=state)
    
    joblib.dump(df_normalized, dump_path)


#if __name__ == "__main__":
#    f = open("src/params/preprocess_params.yaml", "r")
#    params = yaml.load(f, Loader=yaml.SafeLoader)
#    f.close()
    
#    temp = ['TRAIN','VALID','TEST']
    
#    for subgroup in temp:
#        xpath = params[f'X_PATH_{subgroup}']
#        ypath = params[f'Y_PATH_{subgroup}']
#        dump_path = params[f'DUMP_{subgroup}']
        
#        if subgroup == 'TRAIN':
#            state = 'fit'
#        else:
#            state = 'transform'
        
#        run(params, xpath, ypath, dump_path, state)