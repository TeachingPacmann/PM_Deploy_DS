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
            f'Fitting {base_model.__class__.__name__}, with weight: {weights}')

        # Train
        t0 = time.time()
        fitted_model = model_lib.fit(
            x_train, y_train, base_model, param_model, params)
        elapsed_time = time.time() - t0
        print(f'elapsed time: {elapsed_time} s \n')
        train_log_dict['fit_time'].append(elapsed_time)
        train_log_dict['model_fit'].append(fitted_model)

        # Validate
        score = model_lib.validation_score( x_valid, y_valid, fitted_model)
        train_log_dict['model_score'].append(
            report['f1-score']['macro avg'])

    best_model, best_report, best_threshold, name = model_lib.select_model(
        train_log_dict)
    print(
        f"Model: {name}, Score: {best_report['f1-score']['macro avg']}")
    joblib.dump(best_model, 'output/isrelated_model.pkl')
    joblib.dump(best_threshold, 'output/isrelated_threshold.pkl')
    joblib.dump(train_log_dict, 'output/isrelated_train_log.pkl')

    print(f'\n {best_report}')