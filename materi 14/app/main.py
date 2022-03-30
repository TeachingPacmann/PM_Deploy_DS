import logging.config

from app.src.feature_engineering_input import construct_df, feature_engineering_predict
from app.src.modelling import modelling
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from app.exception import BaseException, validation_exception_handler, starlette_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


app = FastAPI(
    exception_handlers={
        RequestValidationError: validation_exception_handler,
        StarletteHTTPException: starlette_exception_handler,
    },
)

# setup loggers
logging.config.fileConfig('app/logging.conf', disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__)

class Item(BaseModel):
    OverallQual: int
    GrLivArea: int
    TotalBsmtSF: int
    FirstFlrSF: int
    GarageCars: int
    GarageArea: int

fake_secret_token = "pacmannpmdata"


@app.get("/")
async def read_root():
    return {"msg": "Hello World"}

# first endpoint
@app.get("/log_now")
def log_now():
    logger.info("logging from the root logger")

    return {"result": "OK"}

@app.post("/predict-house/v1/")
def house_pricing(item: Item, x_token: str = Header(...)):
    if x_token != fake_secret_token:
        raise HTTPException(status_code=400, detail="Invalid X-Token header")

    try:
        data_predict = {}
        for i, value in enumerate(item):
            data_predict[value[0]] = [value[1]]

        x_input = construct_df(data_predict)
        logger.info("finished construct df")

        x_predict = feature_engineering_predict(x_input)
        logger.info("finished feature engineering")
        # make prediction
        y_predicted = modelling(x_predict)
        logger.info("finished modelling")

        result = {"result": y_predicted}

        return JSONResponse(
            status_code=200,
            content=result
        )

    except Exception as e:
        message = "There is error in our config!"
        logger.error(e)
        raise BaseException(message=message)

