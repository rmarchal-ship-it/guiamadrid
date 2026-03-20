"""CLI entry point for Guía del Ocio Madrid.

Usage:
    python -m guiamadrid scrape          # Scrape today's showtimes → DB
    python -m guiamadrid scrape 2026-03-25  # Scrape specific date
    python -m guiamadrid serve           # Start FastAPI server
    python -m guiamadrid digest          # Send email digest for today
    python -m guiamadrid stats           # Show DB stats
"""

import sys
from datetime import date


def cmd_scrape(target_date: str | None = None):
    from guiamadrid.db.database import store_scrape_result
    from guiamadrid.scrapers.cine.sensacine import SensaCineScraper

    d = date.fromisoformat(target_date) if target_date else None
    print(f"Scraping SensaCine for {target_date or 'today'}...")

    with SensaCineScraper() as scraper:
        result = scraper.scrape(d)

    print(f"  {len(result.showtimes)} showtimes, {result.cinemas_count} cinemas, {result.movies_count} movies")
    if result.errors:
        print(f"  {len(result.errors)} errors:")
        for e in result.errors[:5]:
            print(f"    - {e}")

    inserted = store_scrape_result(result)
    print(f"  {inserted} new showtimes stored in DB")


def cmd_serve():
    from guiamadrid.api.server import run
    run()


def cmd_digest(target_date: str | None = None):
    from guiamadrid.notifications.email_sender import send_digest
    send_digest(target_date)


def cmd_stats():
    from guiamadrid.db.database import SessionLocal, init_db
    from guiamadrid.db.models import Cinema, Movie, ScrapeLog, Showtime

    init_db()
    session = SessionLocal()
    try:
        print("=== Guía Madrid DB Stats ===")
        print(f"  Cinemas:   {session.query(Cinema).count()}")
        print(f"  Movies:    {session.query(Movie).count()}")
        print(f"  Showtimes: {session.query(Showtime).count()}")
        print(f"  Scrapes:   {session.query(ScrapeLog).count()}")

        last = session.query(ScrapeLog).order_by(ScrapeLog.id.desc()).first()
        if last:
            print(f"  Last scrape: {last.date_scraped} ({last.showtimes_count} showtimes)")
    finally:
        session.close()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    command = args[0]
    extra = args[1] if len(args) > 1 else None

    if command == "scrape":
        cmd_scrape(extra)
    elif command == "serve":
        cmd_serve()
    elif command == "digest":
        cmd_digest(extra)
    elif command == "stats":
        cmd_stats()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
