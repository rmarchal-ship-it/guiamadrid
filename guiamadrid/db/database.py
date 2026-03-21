"""Database initialization and session management."""

from __future__ import annotations

import json
from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from guiamadrid.config import DATA_DIR, DATABASE_URL
from guiamadrid.db.models import Base, Cinema, Movie, ScrapeLog, Showtime
from guiamadrid.scrapers.base import ScrapeResult
from guiamadrid.scrapers.base import Showtime as ShowtimeDTO

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()


def store_scrape_result(result: ScrapeResult, source: str = "sensacine") -> int:
    """Store a ScrapeResult into the database. Returns number of new showtimes inserted."""
    init_db()
    session = SessionLocal()
    inserted = 0

    try:
        # Cache for cinema/movie lookups
        cinema_cache: dict[str, int] = {}
        movie_cache: dict[str, int] = {}

        for st in result.showtimes:
            cinema_db_id = _get_or_create_cinema(session, st, cinema_cache)
            movie_db_id = _get_or_create_movie(session, st, movie_cache)

            # Insert showtime if not duplicate
            existing = (
                session.query(Showtime)
                .filter_by(
                    cinema_id=cinema_db_id,
                    movie_id=movie_db_id,
                    date=st.date,
                    time=st.showtime,
                    language=st.language,
                    format=st.format,
                )
                .first()
            )
            if not existing:
                session.add(
                    Showtime(
                        cinema_id=cinema_db_id,
                        movie_id=movie_db_id,
                        date=st.date,
                        time=st.showtime,
                        language=st.language,
                        format=st.format,
                    )
                )
                inserted += 1

        # Log the scrape
        session.add(
            ScrapeLog(
                source=source,
                date_scraped=result.showtimes[0].date if result.showtimes else str(date.today()),
                showtimes_count=len(result.showtimes),
                cinemas_count=result.cinemas_count,
                movies_count=result.movies_count,
                errors=json.dumps(result.errors) if result.errors else "",
            )
        )

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return inserted


def _get_or_create_cinema(
    session: Session, st: ShowtimeDTO, cache: dict[str, int]
) -> int:
    """Get or create a Cinema record, return its DB id."""
    if st.cinema_id in cache:
        return cache[st.cinema_id]

    cinema = (
        session.query(Cinema).filter_by(external_id=st.cinema_id).first()
    )
    if not cinema:
        cinema = Cinema(
            external_id=st.cinema_id,
            name=st.cinema_name,
            source="sensacine",
        )
        session.add(cinema)
        session.flush()

    cache[st.cinema_id] = cinema.id
    return cinema.id


def _get_or_create_movie(
    session: Session, st: ShowtimeDTO, cache: dict[str, int]
) -> int:
    """Get or create a Movie record, return its DB id."""
    # Use title as key (external_id may be empty)
    key = st.movie_title
    if key in cache:
        return cache[key]

    movie = session.query(Movie).filter_by(title=st.movie_title).first()
    if not movie:
        movie = Movie(
            external_id=st.movie_id or "",
            title=st.movie_title,
            director=st.director,
            genre=st.genre,
            duration_min=st.duration_min,
            synopsis=st.synopsis,
            poster_url=st.poster_url,
            rating=st.rating,
            source="sensacine",
        )
        session.add(movie)
        session.flush()

    cache[key] = movie.id
    return movie.id


def get_showtimes_for_date(target_date: str) -> list[dict]:
    """Get all showtimes for a date, joined with cinema and movie info."""
    session = SessionLocal()
    try:
        rows = (
            session.query(Showtime, Cinema, Movie)
            .join(Cinema, Showtime.cinema_id == Cinema.id)
            .join(Movie, Showtime.movie_id == Movie.id)
            .filter(Showtime.date == target_date)
            .order_by(Cinema.name, Movie.title, Showtime.time)
            .all()
        )
        return [
            {
                "id": st.id,
                "cinema": cinema.name,
                "cinema_id": cinema.external_id,
                "movie": movie.title,
                "director": movie.director,
                "genre": movie.genre,
                "duration_min": movie.duration_min,
                "poster_url": movie.poster_url,
                "rating": movie.rating,
                "time": st.time,
                "date": st.date,
                "language": st.language,
                "format": st.format,
            }
            for st, cinema, movie in rows
        ]
    finally:
        session.close()


def get_movies_for_date(target_date: str) -> list[dict]:
    """Get distinct movies showing on a date."""
    session = SessionLocal()
    try:
        movies = (
            session.query(Movie)
            .join(Showtime, Showtime.movie_id == Movie.id)
            .filter(Showtime.date == target_date)
            .distinct()
            .order_by(Movie.title)
            .all()
        )
        return [
            {
                "id": m.id,
                "title": m.title,
                "director": m.director,
                "genre": m.genre,
                "duration_min": m.duration_min,
                "poster_url": m.poster_url,
                "rating": m.rating,
                "synopsis": m.synopsis,
            }
            for m in movies
        ]
    finally:
        session.close()


def get_available_dates() -> list[str]:
    """Get all dates that have showtimes, sorted descending."""
    session = SessionLocal()
    try:
        rows = (
            session.query(Showtime.date)
            .distinct()
            .order_by(Showtime.date.desc())
            .all()
        )
        return [r[0] for r in rows]
    finally:
        session.close()


def get_cinemas() -> list[dict]:
    """Get all cinemas."""
    session = SessionLocal()
    try:
        cinemas = session.query(Cinema).order_by(Cinema.name).all()
        return [
            {
                "id": c.id,
                "external_id": c.external_id,
                "name": c.name,
                "address": c.address,
            }
            for c in cinemas
        ]
    finally:
        session.close()
