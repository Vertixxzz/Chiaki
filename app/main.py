from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.config import PROJECT_DIR
from app.routes import router

app = FastAPI(title='Forte Vision')
app.mount('/static', StaticFiles(directory=str(PROJECT_DIR / 'static')), name='static')
app.include_router(router)
