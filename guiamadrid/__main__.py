"""CLI entry point for Guía del Ocio Madrid.

Usage:
    python -m guiamadrid scrape              # Scrape today's showtimes → DB
    python -m guiamadrid scrape 2026-03-25   # Scrape specific date
    python -m guiamadrid scrape-concerts     # Scrape concerts (Ticketmaster + datos.madrid.es)
    python -m guiamadrid scrape-concerts 2026-03-25
    python -m guiamadrid serve               # Start FastAPI server
    python -m guiamadrid digest              # Send email digest for today
    python -m guiamadrid stats               # Show DB stats
    python -m guiamadrid trailers            # Find YouTube trailers for current movies
    python -m guiamadrid posters             # Fetch TMDB poster URLs (needs TMDB_API_KEY)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path


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


def cmd_scrape_concerts(target_date: str | None = None):
    from guiamadrid.db.database import store_concert_scrape_result

    d = date.fromisoformat(target_date) if target_date else None
    total_inserted = 0

    # Ticketmaster
    try:
        from guiamadrid.scrapers.conciertos.ticketmaster import TicketmasterScraper
        print(f"Scraping Ticketmaster for {target_date or 'today'}...")
        with TicketmasterScraper() as scraper:
            result = scraper.scrape(d)
        print(f"  {len(result.events)} events, {result.venues_count} venues")
        if result.errors:
            print(f"  {len(result.errors)} errors")
            for e in result.errors[:3]:
                print(f"    - {e}")
        if result.events:
            inserted = store_concert_scrape_result(result, source="ticketmaster")
            total_inserted += inserted
            print(f"  {inserted} new concerts stored")
    except ValueError as e:
        print(f"  Ticketmaster skipped: {e}")

    # Datos Madrid
    try:
        from guiamadrid.scrapers.conciertos.datos_madrid import DatosMadridScraper
        print(f"Scraping datos.madrid.es for {target_date or 'today'}...")
        with DatosMadridScraper() as scraper:
            result = scraper.scrape(d)
        print(f"  {len(result.events)} events, {result.venues_count} venues")
        if result.errors:
            print(f"  {len(result.errors)} errors")
        if result.events:
            inserted = store_concert_scrape_result(result, source="datos_madrid")
            total_inserted += inserted
            print(f"  {inserted} new concerts stored")
    except Exception as e:
        print(f"  datos.madrid.es error: {e}")

    # DICE.fm
    try:
        from guiamadrid.scrapers.conciertos.dice import DiceScraper
        print(f"Scraping DICE.fm for {target_date or 'today'}...")
        with DiceScraper() as scraper:
            result = scraper.scrape(d)
        print(f"  {len(result.events)} events, {result.venues_count} venues")
        if result.errors:
            print(f"  {len(result.errors)} errors")
            for e in result.errors[:3]:
                print(f"    - {e}")
        if result.events:
            inserted = store_concert_scrape_result(result, source="dice")
            total_inserted += inserted
            print(f"  {inserted} new concerts stored")
    except Exception as e:
        print(f"  DICE.fm error: {e}")

    print(f"Total: {total_inserted} new concerts stored in DB")


def cmd_serve():
    from guiamadrid.api.server import run
    run()


def cmd_digest(target_date: str | None = None):
    from guiamadrid.notifications.email_sender import send_digest
    send_digest(target_date)


def cmd_trailers():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "find_trailers", Path(__file__).parent.parent / "find_trailers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.exit(mod.main())


def cmd_posters():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_posters", Path(__file__).parent.parent / "fetch_posters.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.exit(mod.main())


def cmd_stats():
    from guiamadrid.db.database import SessionLocal, init_db
    from guiamadrid.db.models import Cinema, Concert, Movie, ScrapeLog, Showtime, Venue

    init_db()
    session = SessionLocal()
    try:
        print("=== Guía Madrid DB Stats ===")
        print("── Cine ──")
        print(f"  Cinemas:   {session.query(Cinema).count()}")
        print(f"  Movies:    {session.query(Movie).count()}")
        print(f"  Showtimes: {session.query(Showtime).count()}")
        print("── Conciertos ──")
        print(f"  Venues:    {session.query(Venue).count()}")
        print(f"  Concerts:  {session.query(Concert).count()}")
        print("── General ──")
        print(f"  Scrapes:   {session.query(ScrapeLog).count()}")

        last = session.query(ScrapeLog).order_by(ScrapeLog.id.desc()).first()
        if last:
            print(f"  Last scrape: {last.date_scraped} ({last.source}, {last.showtimes_count} items)")
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
    elif command == "scrape-concerts":
        cmd_scrape_concerts(extra)
    elif command == "serve":
        cmd_serve()
    elif command == "digest":
        cmd_digest(extra)
    elif command == "stats":
        cmd_stats()
    elif command == "trailers":
        cmd_trailers()
    elif command == "posters":
        cmd_posters()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
