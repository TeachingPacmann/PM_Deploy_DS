import numpy as np
import pandas as pd
import joblib
from app.constant import MODEL_NAME, BEST_MODEL

def modelling(x_predict: pd.DataFrame):
    model_name = joblib.load(MODEL_NAME)
    main_model = joblib.load(BEST_MODEL)

    y_predicted = main_model.predict(x_predict)
    y_predicted = y_predicted[0]
    return y_predicted