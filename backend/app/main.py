from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.children import router as children_router
from app.api.visits import router as visits_router
from app.api.videos import router as videos_router
from app.api.questionnaires import router as questionnaires_router
from app.api.reports import router as reports_router


app = FastAPI(title="Neurimo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(children_router)
app.include_router(visits_router)
app.include_router(videos_router)
app.include_router(questionnaires_router)
app.include_router(reports_router)
app.mount("/static", StaticFiles(directory="data"), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}

