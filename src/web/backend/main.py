from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api_authors import router as authors_router
from .api_clusters import router as clusters_router
from .api_recommendations import router as recommendations_router

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

app = FastAPI(title="AI Coauthor")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(clusters_router)
app.include_router(authors_router)
app.include_router(recommendations_router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
