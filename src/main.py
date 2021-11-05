import numpy as np
import pandas as pd
import joblib
import yaml
from read_data import read_data, split_input_output, split_data
from preprocess_data import run

f = open("src/params/preprocess_params.yaml", "r")
params = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

data_house = read_data(params['DATA_PATH'])
output_df, input_df = split_input_output(
                            data_house,
                            params['TARGET_COLUMN'])

X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(input_df,
                                                                output_df,
                                                                True,
                                                                params['TEST_SIZE'])

temp = ['TRAIN','VALID','TEST']

for subgroup in temp:
    xpath = params[f'X_PATH_{subgroup}']
    ypath = params[f'Y_PATH_{subgroup}']
    dump_path = params[f'DUMP_{subgroup}']

    if subgroup == 'TRAIN':
        state = 'fit'
    else:
        state = 'transform'

    run(params, xpath, ypath, dump_path, state)