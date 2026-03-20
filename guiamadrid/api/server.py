"""FastAPI server for Guía del Ocio Madrid."""

from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from guiamadrid.db.database import (
    get_cinemas,
    get_movies_for_date,
    get_showtimes_for_date,
    init_db,
)

app = FastAPI(
    title="Guía del Ocio Madrid",
    description="API de cartelera de cines en Madrid",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def root():
    return {"service": "Guía del Ocio Madrid", "version": "0.1.0"}


@app.get("/api/showtimes")
def showtimes(
    fecha: str = Query(
        default=None,
        description="Fecha en formato YYYY-MM-DD (default: hoy)",
    ),
):
    """Get all showtimes for a given date."""
    target = fecha or str(date.today())
    try:
        date.fromisoformat(target)
    except ValueError:
        raise HTTPException(400, "Fecha inválida. Usa formato YYYY-MM-DD.")
    results = get_showtimes_for_date(target)
    return {"date": target, "count": len(results), "showtimes": results}


@app.get("/api/movies")
def movies(
    fecha: str = Query(
        default=None,
        description="Fecha en formato YYYY-MM-DD (default: hoy)",
    ),
):
    """Get distinct movies showing on a date."""
    target = fecha or str(date.today())
    try:
        date.fromisoformat(target)
    except ValueError:
        raise HTTPException(400, "Fecha inválida. Usa formato YYYY-MM-DD.")
    results = get_movies_for_date(target)
    return {"date": target, "count": len(results), "movies": results}


@app.get("/api/cinemas")
def cinemas():
    """Get all cinemas in the database."""
    results = get_cinemas()
    return {"count": len(results), "cinemas": results}


@app.get("/api/showtimes/{cinema_id}")
def showtimes_by_cinema(
    cinema_id: str,
    fecha: str = Query(default=None),
):
    """Get showtimes for a specific cinema."""
    target = fecha or str(date.today())
    all_showtimes = get_showtimes_for_date(target)
    filtered = [s for s in all_showtimes if s["cinema_id"] == cinema_id]
    return {"date": target, "cinema_id": cinema_id, "count": len(filtered), "showtimes": filtered}


def run():
    """Run the server (for CLI usage)."""
    import uvicorn
    from guiamadrid.config import API_HOST, API_PORT
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    run()
