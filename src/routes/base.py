from fastapi import APIRouter, Depends
from ..helpers import get_settings , Settings

base_router = APIRouter(
    prefix= "/v1",
    tags= ["api_v1"]
)

@base_router.get("/")
async def index(app_settings: Settings = Depends(get_settings)):

    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION
    
    return {
        "app_name": app_name,
        "app_version": app_version,
    }
