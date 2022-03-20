# import uvicorn
from app.src.feature_engineering_input import construct_df, feature_engineering_predict
from app.src.modelling import modelling
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse, PlainTextResponse
from app.exception import BaseException
from fastapi.exceptions import RequestValidationError
from typing import Any, Dict
import logging

app = FastAPI()

# setup loggers
logging.config.fileConfig("app/logging.conf", disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(__name__)


class Item(BaseModel):
    OverallCond: int
    GrLivArea: int
    TotalBsmtSF: int
    fstFlrSF: int
    GarageCars: int
    GarageArea: int


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    try:
        response: Dict[str, Any] = {}
        response["message"] = ", ".join(
            [f"{x['loc'][-1]} - {x['ctx']['msg']} - {x['type']}" for x in exc.errors()]
        )
        response["status"] = "UNPROCESSABLE_ENTITY"

        return JSONResponse(
            status_code=422,
            content=response,
        )

    except Exception as exc:
        raise BaseException(
            message=f"There's an error in the validation handler - {str(exc)}"
        )


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# first endpoint
@app.get("/log_now")
def log_now():
    logger.info("logging from the root logger")

    return {"result": "OK"}


@app.get("/status")
async def status_web():
    return {"STATUS": "OKE"}


@app.post("/predict-house/v1/")
def house_pricing(item: Item):
    try:
        data_predict = {}
        for i, value in enumerate(item):
            if i == 3:
                data_predict["1stFlrSF"] = [value[1]]

            data_predict[value[0]] = [value[1]]

        x_input = construct_df(data_predict)
        logger.info("finished construct df")

        x_predict = feature_engineering_predict(x_input)
        logger.info("finished feature engineering")
        # make prediction
        y_predicted = modelling(x_predict)
        logger.info("finished modelling")

        result = {"result": y_predicted}

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        message = "There is error in our config!"
        logger.error(e)
        raise BaseException(message=message)
