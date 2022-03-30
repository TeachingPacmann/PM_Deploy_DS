# from enum import Enum
from typing import Any, Dict
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

async def starlette_exception_handler(request, exc: StarletteHTTPException):
    response = {
        "message": str(exc.detail),
        "status": str(exc.detail),
    }
    return JSONResponse(
        status_code=exc.status_code,
        content=response,
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        response: Dict[str, Any] = {}
        response["message"] = ", ".join(
             [f"{x['loc'][-1]} - {x['msg']} - {x['type']}" for x in exc.errors()]
        )
        response["status"] = "UNPROCESSABLE_ENTITY"

        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            content=response,
        )

    except Exception as exc:
        raise BaseException(
            message=f"There's an error in the validation handler - {str(exc)}"
        )

class BaseException(Exception):
    message = "INTERNAL SERVER ERROR"
    status_code = 500

    def __init__(self, message: str):
        self.message = message

    def base_return(self) -> Dict:
        return {
            "message": self.message,
            "status": "INTERNAL SERVER ERROR"
        }
