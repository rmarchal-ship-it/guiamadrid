"""Full integration test: scrape (mock) → DB → API → email digest.

Run with: python tests/test_full_pipeline.py
"""

import os
import sys
from pathlib import Path

# Use a temp DB for testing
os.environ["GUIAMADRID_TEST"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

# Override DB path before importing
import guiamadrid.config as config

TEST_DB = Path("/tmp/guiamadrid_test.db")
if TEST_DB.exists():
    TEST_DB.unlink()
config.DB_PATH = TEST_DB
config.DATABASE_URL = f"sqlite:///{TEST_DB}"

# Re-initialize engine with test DB
import guiamadrid.db.database as db_mod
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db_mod.engine = create_engine(config.DATABASE_URL, echo=False)
db_mod.SessionLocal = sessionmaker(bind=db_mod.engine)

from guiamadrid.db.database import (
    get_cinemas,
    get_movies_for_date,
    get_showtimes_for_date,
    init_db,
    store_scrape_result,
)
from guiamadrid.notifications.email_sender import build_digest_html, build_digest_text
from guiamadrid.scrapers.base import ScrapeResult, Showtime


def test_pipeline():
    print("=== FULL PIPELINE TEST ===\n")

    # 1. Init DB
    init_db()
    print("[1/6] DB initialized")

    # 2. Create realistic scrape result
    showtimes = [
        Showtime("Yelmo Ideal", "E0621", "Capitán América: Brave New World", "301843",
                 "14:30", "2026-03-20", "Castellano", "2D", "Julius Onah",
                 "https://example.com/cap4.jpg", "Sam Wilson asume el manto...",
                 3.2, "Acción, Aventura", 118),
        Showtime("Yelmo Ideal", "E0621", "Capitán América: Brave New World", "301843",
                 "17:15", "2026-03-20", "VOSE", "2D", "Julius Onah",
                 "", "", 3.2, "Acción, Aventura", 118),
        Showtime("Yelmo Ideal", "E0621", "Dune: Parte Dos", "295937",
                 "15:00", "2026-03-20", "VOSE", "IMAX", "Denis Villeneuve",
                 "", "", 4.5, "Ciencia ficción", 166),
        Showtime("Yelmo Ideal", "E0621", "Anora", "312456",
                 "17:30", "2026-03-20", "VOSE", "2D", "Sean Baker",
                 "", "", 4.1, "Drama, Comedia", 139),
        Showtime("Cinesa Proyecciones", "E0402", "Capitán América: Brave New World", "301843",
                 "20:00", "2026-03-20", "Castellano", "2D", "Julius Onah",
                 "", "", 3.2, "Acción, Aventura", 118),
        Showtime("Cinesa Proyecciones", "E0402", "Flow", "320001",
                 "12:00", "2026-03-20", "VOSE", "2D", "Gints Zilbalodis",
                 "", "", 4.3, "Animación", 85),
        Showtime("Cines Princesa (Renoir)", "E0364", "Anora", "312456",
                 "19:00", "2026-03-20", "VOSE", "2D", "Sean Baker",
                 "", "", 4.1, "Drama, Comedia", 139),
        Showtime("Cine Doré (Filmoteca)", "G02GQ", "El séptimo sello", "000001",
                 "18:00", "2026-03-20", "VOSE", "2D", "Ingmar Bergman",
                 "", "", 4.8, "Drama", 96),
    ]

    result = ScrapeResult(
        showtimes=showtimes, cinemas_count=4, movies_count=5
    )
    print(f"[2/6] Scrape result: {len(showtimes)} showtimes, 4 cinemas, 5 movies")

    # 3. Store in DB
    inserted = store_scrape_result(result)
    print(f"[3/6] Stored: {inserted} new showtimes")
    assert inserted == 8, f"Expected 8, got {inserted}"

    # Verify idempotency
    inserted2 = store_scrape_result(result)
    assert inserted2 == 0, f"Idempotency failed: {inserted2} re-inserted"
    print(f"       Idempotency OK (0 re-inserted)")

    # 4. Query back
    db_showtimes = get_showtimes_for_date("2026-03-20")
    db_movies = get_movies_for_date("2026-03-20")
    db_cinemas = get_cinemas()
    print(f"[4/6] DB queries: {len(db_showtimes)} showtimes, {len(db_movies)} movies, {len(db_cinemas)} cinemas")
    assert len(db_showtimes) == 8
    assert len(db_movies) == 5
    assert len(db_cinemas) == 4

    # 5. Test API
    from fastapi.testclient import TestClient
    from guiamadrid.api.server import app

    client = TestClient(app)

    r = client.get("/api/showtimes?fecha=2026-03-20")
    assert r.status_code == 200
    assert r.json()["count"] == 8

    r = client.get("/api/movies?fecha=2026-03-20")
    assert r.status_code == 200
    assert r.json()["count"] == 5

    r = client.get("/api/cinemas")
    assert r.status_code == 200
    assert r.json()["count"] == 4

    r = client.get("/api/showtimes/E0621?fecha=2026-03-20")
    assert r.status_code == 200
    assert r.json()["count"] == 4  # 4 showtimes at Yelmo Ideal

    r = client.get("/api/showtimes?fecha=invalid")
    assert r.status_code == 400

    print(f"[5/6] API endpoints: all 5 tests passed")

    # 6. Email digest
    html = build_digest_html("2026-03-20")
    text = build_digest_text("2026-03-20")
    assert "5 películas" in text
    assert "8 sesiones" in text
    assert "Capitán América" in text
    assert "Dune" in text
    assert "Anora" in text
    assert "Flow" in text
    assert "El séptimo sello" in text
    assert "Yelmo Ideal" in text
    assert "Cinesa Proyecciones" in text
    assert "<html>" in html
    print(f"[6/6] Email digest: HTML ({len(html)} chars) + text ({len(text)} chars) OK")

    print(f"\n{'='*50}")
    print("ALL TESTS PASSED")
    print(f"{'='*50}")
    print(f"\nText digest preview:\n")
    print(text)

    # Cleanup
    TEST_DB.unlink(missing_ok=True)


if __name__ == "__main__":
    test_pipeline()
