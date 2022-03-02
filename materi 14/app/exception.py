from enum import Enum
from typing import Any, Dict
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY


class ErrorMessage(str, Enum):
    BAD_REQUEST = "BAD_REQUEST"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    UNPROCESSABLE_ENTITY = "UNPROCESSABLE_ENTITY"

async def CustomExceptionHandler(request: Request, exception: BaseException):
    return JSONResponse (status_code = 500, content = {"message": "Something happened in our config!"})

# async def validation_exception_handler(
#     request: Request, exc: RequestValidationError
# ):
#     # return PlainTextResponse(str(exc), status_code=400)
#     try:
#         response: Dict[str, Any] = {}
#         response["message"] = ", ".join(
#             [f"{x['loc'][-1]} - {x['msg']} - {x['type']}" for x in exc.errors()]
#         )
#         response["status"] = ErrorMessage.UNPROCESSABLE_ENTITY.value

#         return JSONResponse(
#             status_code=HTTP_422_UNPROCESSABLE_ENTITY,
#             content=response,
#         )

#     except Exception as exc:
#         raise BaseException(
#             message=f"There's an error in the validation handler - {str(exc)}"
#         )


async def starlette_exception_handler(request, exc: StarletteHTTPException):
    response = {
        "message": str(exc.detail),
        "status": str(exc.detail),
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=response,
    )

class BaseException(Exception):
    def __init__(self, message: str):
        self.message = message
