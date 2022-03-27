# import uvicorn
from app.src.feature_engineering_input import construct_df, feature_engineering_predict
from app.src.modelling import modelling
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    OverallQual: int
    GrLivArea: int
    TotalBsmtSF: int
    FirstFlrSF: int
    GarageCars: int
    GarageArea: int

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/status")
async def status_web():
    return {"STATUS": "OKE"}

@app.post("/predict-house/v1/")
def house_pricing(item: Item):
    data_predict = {}
    for i, value in enumerate(item):
        data_predict[value[0]] = [value[1]]

    x_input = construct_df(data_predict)

    x_predict = feature_engineering_predict(x_input)
    # make prediction
    y_predicted = modelling(x_predict)

    return {"result": y_predicted}