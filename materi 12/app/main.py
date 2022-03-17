# import uvicorn
from cmath import log
import pickle
from app.src.feature_engineering_input import construct_df, feature_engineering_predict
from app.src.modelling import modelling
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

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

    x_predict = feature_engineering_predict(x_input)
    # make prediction
    y_predicted = modelling(x_predict)

    return {"result": y_predicted}