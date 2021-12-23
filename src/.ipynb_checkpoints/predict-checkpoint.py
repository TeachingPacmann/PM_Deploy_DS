import numpy as np
import pandas as pd
import model_lib
import time
import joblib
import yaml
import preprocess_data as prep

def set_dtypes(data_input, params):
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
    data = data_input.astype(params["PREDICT_COLUMN_TYPE"])
    return data

def construct_df(params, data_to_predict, file=None):
    if file == 'csv':
        df_to_predict = pd.read_csv(data_to_predict, sep = ';')
    elif file == 'excel':
        df_to_predict = pd.read_excel(data_to_predict)
    else:
        df_to_predict = pd.DataFrame(data=data_to_predict)
        df_to_predict = set_dtypes(df_to_predict, params)
        # COLUMN = set(params['NUM_COLUMN']+params['CAT_COLUMN'])
        COLUMN = set(params['PREDICT_COLUMN'])
        column_in_data = set(df_to_predict.columns)
        remain_columns = list(COLUMN-column_in_data)
        df_to_predict[remain_columns] = np.NaN
    return df_to_predict

def feature_engineering_predict(data_to_predict):
    state = 'transform'
    dump_path = params[f'DUMP_PREDICT']
    data_to_predict = data_to_predict.copy()
    # house_numerical = data_to_predict[params['NUM_COLUMN']]
    # house_categorical = data_to_predict[params['CAT_COLUMN']]
    
    house_numerical = data_to_predict[params['PREDICT_COLUMN']]
    df_numerical_imputed = prep.numerical_imputer(house_numerical, state=state)
    # df_categorical_imputed = prep.categorical_imputer(house_categorical)
    # df_categorical_encoded = prep.one_hot_encoder(df_categorical_imputed, state=state)
    # df_joined = pd.concat([df_numerical_imputed, df_categorical_encoded], axis=1)
    x_predict = prep.normalization(df_numerical_imputed, state=state) # df_joined
    joblib.dump(x_predict, dump_path)
    return x_predict

if __name__ == "__main__":
    
    # Open yaml
    f = open("src/params/params.yaml", "r")
    params = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()
    
    # load model param and best model
    model_name = joblib.load(params['MODEL_NAME'])
    main_model = joblib.load(params['BEST_MODEL'])
    
    print(f"Working on predict data with {model_name} model\n")
    
    # construct dictionary as log file
    predict_dict = {'model': [main_model],
                  'model_name': [model_name],
                  'predicted': []}
    
    # input data to predict
    
    # 1. through file
    # data_predict = 'data/2data.csv'
    # file = 'csv'
    
    # 2. through input
    n_data = int(input(f"Input data (enter int value): "))
    data_predict = {}
    for i in range(n_data):
        for i in params["PREDICT_COLUMN"]:
            if i in data_predict:
                data_predict[i].append(input(f"Input {i}: "))
            else:
                data_predict[i] = [input(f"Input {i}: ")]
    
    # make input data to df
    x_input = construct_df(params, data_predict)
    
    # feature engineering on df
    print(f"Running on feature engineering...\n")
    x_predict = feature_engineering_predict(x_input)
    
    # make prediction
    y_predicted = main_model.predict(x_predict)
    
    # dump log prediction result
    predict_dict['predicted'].append(y_predicted)
    joblib.dump(predict_dict, 'output/predict/predict_log.pkl')
    
    print(f"Model: {predict_dict['model_name']},\n Predicted: {predict_dict['predicted']}\n")
    
    for i in range(len(x_predict)):
        print(f"{i+1}. Data with rates (1-10) the overall condition of the house {x_input['OverallCond'][i]}, First Floor {x_input['1stFlrSF'][i]} square feet, were predict to have sale price {y_predicted[i]}\n")
    