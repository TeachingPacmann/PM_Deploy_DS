import numpy as np
import pandas as pd
import model_lib
import time
import joblib
import yaml


def main(params):

    lasso = model_lib.model_lasso
    rf = model_lib.model_rf
    lsvr = model_lib.model_svr

    train_log_dict = {'model': [lasso, rf, lsvr],
                      'model_name': [],
                      'model_fit': [],
                      'model_report': [],
                      'model_score': [],
                      'fit_time': []}

    x_train, y_train, x_valid, y_valid  = model_lib.read_data(params)

    for model in train_log_dict['model']:
        param_model, base_model = model()
        train_log_dict['model_name'].append(base_model.__class__.__name__)
        print(
           f'Fitting {base_model.__class__.__name__}')

        # Train
        t0 = time.time()
        fitted_model,best_estimator = model_lib.fit(
            x_train, y_train, base_model, param_model, params)
        elapsed_time = time.time() - t0
        print(f'elapsed time: {elapsed_time} s \n')
        train_log_dict['fit_time'].append(elapsed_time)
        train_log_dict['model_fit'].append(best_estimator.__class__.__name__)
        
        best_estimator.fit(x_train, y_train)
        train_log_dict['model_report'].append(best_estimator)

        
        # Validate
        score = model_lib.validation_score( x_valid, y_valid, best_estimator)
        #train_log_dict['model_score'].append(
        #    report['f1-score']['macro avg'])
        train_log_dict['model_score'].append(
            score)


    best_model, best_estimator, best_report = model_lib.select_model(
        train_log_dict)
    print(
        f"Model: {best_model}, Score: {best_report}, Parameter: {best_estimator}")
    joblib.dump(best_model, f'output/model/train/base_model.pkl')
    joblib.dump(best_estimator, 'output/model/train/best_estimator.pkl')
    joblib.dump(train_log_dict, 'output/model/train/train_log.pkl')
    