import numpy as np
import pandas as pd
import model_lib
import time
import joblib
import yaml
import preprocess_data as prep

def construct_df(data_to_predict, file='csv'):
    if file == 'csv':
        df_to_predict = pd.read_csv(data_to_predict, sep = ';')
    elif file == 'excel':
        df_to_predict = pd.read_excel(data_to_predict)
    return df_to_predict

def feature_engineering_predict(data_to_predict):
    state = 'transform'
    dump_path = params[f'DUMP_PREDICT']
    data_to_predict = data_to_predict.copy()
    house_numerical = data_to_predict[params['NUM_COLUMN']]
    house_categorical = data_to_predict[params['CAT_COLUMN']]
    df_numerical_imputed = prep.numerical_imputer(house_numerical, state=state)
    df_categorical_imputed = prep.categorical_imputer(house_categorical)
    df_categorical_encoded = prep.one_hot_encoder(df_categorical_imputed, state=state)
    df_joined = pd.concat([df_numerical_imputed, df_categorical_encoded], axis=1)
    x_predict = prep.normalization(df_joined, state=state)
    # joblib.dump(x_predict, dump_path)
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
    data_predict = 'data/2data.csv'
    file = 'csv'
    
    # make input data to dfh
    x_input = construct_df(data_predict, file)
    
    # feature engineering on df
    print(f"Running on feature engineering...\n")
    x_predict = feature_engineering_predict(x_input)
    
    # make prediction
    y_predicted = main_model.predict(x_predict)
    
    # dump log prediction result
    predict_dict['predicted'].append(y_predicted)
    # joblib.dump(predicted, 'output/predict_log.pkl')
    
    print(f"Model: {predict_dict['model_name']},\n Predicted: {predict_dict['predicted']}\n")
    
    for i in range(len(x_predict)):
        print(f"{i+1}. Data with rates (1-10) the overall condition of the house {x_input['OverallCond'][i]}, year built in {x_input['YearBuilt'][i]}, First Floor {x_input['1stFlrSF'][i]} square feet, were predict to have sale price {y_predicted[i]}\n")
    