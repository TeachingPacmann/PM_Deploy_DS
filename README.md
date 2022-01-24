# PM_Deploy_DS
# **Prediksi Harga Rumah**

## **Latar Belakang**
Untuk membeli rumah, seseorang biasanya melalui banyak pertimbangan, dimulai dari kebutuhan pribadi dan keadaan rumah. Namun di titik akhir biasanya pembeli rumah lebih memperhatikan harga rumah dibandingkan fasilitas yang ada didalam rumah. Oleh karena itu penting mengetahui atau memprediksi harga akhir rumah untuk menjadi preferensi seorang pembeli rumah untuk menentukan keputusan.


## **Objectives**
- Membuat pipeline untuk membangun model prediksi rumah 


## **Methods**
![](-)

## **Splitting Data**

## **Wrangling and Feature Engineering**

```python
def run(params, xpath, ypath, dump_path, state='fit'):
    '''
    Main function of wrangling and feature engineering.
    This function will applied in data training, testing and validation.
    
    Parameters
    ----------
    params: .yaml file
        File containing necessary variables as constant variable such as location file and features name 
        - PREDICT_COLUMN(str) : list of features to be used   
    xpath: string
        Location of features pickle file

    ypath: string
        Location of target pickle file

    dump_path: string
        Location to save the result of preprocessing

    state: string
        Data state for leakage handling. fit for training data, transform for validation and testing data

    '''

    # Load variables and target pickle file
    house_variables = joblib.load(xpath)
    house_target = joblib.load(ypath)

    # Due to resource issue, we just use six features with highest correlation to target class
    house_numerical = house_variables[params['PREDICT_COLUMN']] 
    
    # Handling missing value
    df_numerical_imputed = numerical_imputer(house_numerical, state=state)
    
    # Add a representative feature
    df_add_feature = add_feature(df_numerical_imputed)
    
    # Normalization
    df_normalized = normalization(df_add_feature, state=state) # df_joined
    
    # Save the result of preprocessed feature
    joblib.dump(df_normalized, dump_path)

```


## **Modelling**

```python
def main(params):
    '''
    Main function of modelling
    
    Parameters
    ----------
    params: .yaml file contain (dict) of general parameters for the read_data and model_lib function
        - DUMP_TRAIN (str)  : location of preprocessed training data pickle
        - Y_PATH_TRAIN (str): location of target column pickle for training data
        - DUMP_VALID (str)  : location of preprocessed validation data pickle
        - Y_PATH_VALID (str): location of target column  pickle validation data

        - target(str) : y column to be used   
        - scoring(str) : sklearn cross-val scoring scheme
        - n_iter_search : RandomizedSearchCV number of iteration

    '''
    lasso = model_lib.model_lasso
    rf = model_lib.model_rf
    lsvr = model_lib.model_svr

    # Initiate dictionary train_log_dict to be saved later in pickle containing model information in training phase
    train_log_dict = {'model': [lasso, rf, lsvr],
                      'model_name': [],
                      'model_fit': [],
                      'model_report': [],
                      'model_score': [],
                      'fit_time': []}

    # Read data after preprocessing
    x_train, y_train, x_valid, y_valid  = model_lib.read_data(params)

    # Iterate list model 
    for model in train_log_dict['model']:
        # initiate the model
        param_model, base_model = model()
        # logging model name
        train_log_dict['model_name'].append(base_model.__class__.__name__)
        print(
           f'Fitting {base_model.__class__.__name__}')

        # Training
        t0 = time.time()

        # Searching best parameter using Random Search CV
        fitted_model,best_estimator = model_lib.fit(
            x_train, y_train, base_model, param_model, params)
        elapsed_time = time.time() - t0
        print(f'elapsed time: {elapsed_time} s \n')
        train_log_dict['fit_time'].append(elapsed_time)
        train_log_dict['model_fit'].append(best_estimator.__class__.__name__)
        
        # Fitting model with best params to data training
        best_estimator.fit(x_train, y_train)
        train_log_dict['model_report'].append(best_estimator)

        
        # Validate model to validation data
        score = model_lib.validation_score( x_valid, y_valid, best_estimator)
        train_log_dict['model_score'].append(
            score)

    # Select which model in model list has best score evaluation (minimum rmse) in validation data
    best_model, best_estimator, best_report = model_lib.select_model(
        train_log_dict)
    print(
        f"Model: {best_model}, Score: {best_report}, Parameter: {best_estimator}")

    # Dump model name
    joblib.dump(best_model, f'output/model/train/base_model.pkl')
    # Dump best model estimator with best param
    joblib.dump(best_estimator, 'output/model/train/best_estimator.pkl')
    # Dump training log
    joblib.dump(train_log_dict, 'output/model/train/train_log.pkl')
```


## **Testing**


## **Prediction**

```python
if __name__ == "__main__":
    '''
    Main function of prediction.
    1. Load the model with best parameters in pickle
    2. Get input data (manual user input)
    3. Construct DataFrame of input data (including defense in data type)
    4. Preprocessed and Feature engineering (load estimator in pickle file)
    5. Make prediction
    '''
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

    # through input
    n_data = int(input(f"Input data (enter int value): "))
    data_predict = {}
    for i in range(n_data):
        for i in params["PREDICT_COLUMN"]:
            if i in data_predict:
                data_predict[i].append(input(f"Input {i}: "))
            else:
                data_predict[i] = [input(f"Input {i}: ")]
    
    # Make input data to DataFrame
    x_input = construct_df(params, data_predict)
    
    # Feature engineering on input DataFrame
    print(f"Running on feature engineering...\n")
    x_predict = feature_engineering_predict(x_input)
    
    # Make prediction
    y_predicted = main_model.predict(x_predict)
    
    # Dump log prediction result
    predict_dict['predicted'].append(y_predicted)
    joblib.dump(predict_dict, 'output/predict/predict_log.pkl')
    
    # Show the result of price prediction
    print(f"Model: {predict_dict['model_name']},\n Predicted: {predict_dict['predicted']}\n")
    
    for i in range(len(x_predict)):
        print(f"{i+1}. Data with rates (1-10) the overall condition of the house {x_input['OverallCond'][i]}, First Floor {x_input['1stFlrSF'][i]} square feet, were predict to have sale price {y_predicted[i]}\n")

```

## **Sumber Data**
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
