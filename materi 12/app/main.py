# import uvicorn
from cmath import log
import pickle
from app.src.feature_engineering_input import construct_df, feature_engineering_predict
from app.src.modelling import modelling
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

import logging

app = FastAPI()

# setup loggers
logging.config.fileConfig('app/logging.conf', disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__)

class Item(BaseModel):
    OverallCond: int
    GrLivArea: int
    TotalBsmtSF: int
    fstFlrSF: int
    GarageCars: int
    GarageArea: int

@app.get("/")
async def read_root():
    return {"Hello": "World"}

# first endpoint
@app.get("/log_now")
def log_now():
    logger.info("logging from the root logger")

    return {"result": "OK"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

@app.get("/status")
async def status_web():
    return {"STATUS": "OKE"}

@app.post("/predict-house/v1/")
def house_pricing(item: Item):
    data_predict = {}
    for i, value in enumerate(item):
        if i == 3:
            data_predict["1stFlrSF"] = [value[1]]

        data_predict[value[0]] = [value[1]]

    x_input = construct_df(data_predict)
    logger.info("finished construct df")

    x_predict = feature_engineering_predict(x_input)
    logger.info("finish feature engineering")
    # make prediction
    y_predicted = modelling(x_predict)
    logger.info("finish modelling")
    logger.info(y_predicted)

    return {"result": y_predicted}