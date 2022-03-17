# DUMP_TRAIN: "models/preprocessed_x_train.pkl"
# DUMP_VALID: "models/preprocessed_x_valid.pkl"
# DUMP_TEST: "models/preprocessed_x_test.pkl"
# DUMP_PREDICT: "models/preprocessed_x_predict.pkl"
MODEL_NAME = "app/models/base_model.pkl"
BEST_MODEL = "app/models/best_estimator.pkl"
# TRAIN_LOG: "models/train_log.pkl"

PREDICT_COLUMN_TYPE = { "OverallCond": "int",
    "GrLivArea": "int",
    "TotalBsmtSF": "int",
    "1stFlrSF": "int",
    "GarageCars": "int",
    "GarageArea": "int" }

PREDICT_COLUMN = ["OverallCond", "GrLivArea", "TotalBsmtSF",
                  "1stFlrSF", "GarageCars", "GarageArea"]