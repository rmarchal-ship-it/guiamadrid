"""SQLAlchemy models for Guía del Ocio Madrid."""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Cinema(Base):
    __tablename__ = "cinemas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(20), unique=True, nullable=False)  # "E0621"
    name = Column(String(200), nullable=False)
    source = Column(String(50), default="sensacine")  # scraper source
    address = Column(String(500), default="")
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    showtimes = relationship("Showtime", back_populates="cinema")


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(50), nullable=True)
    title = Column(String(500), nullable=False)
    director = Column(String(300), default="")
    genre = Column(String(200), default="")
    duration_min = Column(Integer, nullable=True)
    synopsis = Column(Text, default="")
    poster_url = Column(String(1000), default="")
    rating = Column(Float, nullable=True)
    source = Column(String(50), default="sensacine")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    showtimes = relationship("Showtime", back_populates="movie")


class Showtime(Base):
    __tablename__ = "showtimes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cinema_id = Column(Integer, ForeignKey("cinemas.id"), nullable=False)
    movie_id = Column(Integer, ForeignKey("movies.id"), nullable=False)
    date = Column(String(10), nullable=False)  # "2026-03-20"
    time = Column(String(5), nullable=False)  # "14:30"
    language = Column(String(50), default="")
    format = Column(String(20), default="2D")
    created_at = Column(DateTime, default=datetime.utcnow)

    cinema = relationship("Cinema", back_populates="showtimes")
    movie = relationship("Movie", back_populates="showtimes")

    __table_args__ = (
        UniqueConstraint(
            "cinema_id", "movie_id", "date", "time", "language", "format",
            name="uq_showtime",
        ),
    )


class ScrapeLog(Base):
    __tablename__ = "scrape_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)
    date_scraped = Column(String(10), nullable=False)
    showtimes_count = Column(Integer, default=0)
    cinemas_count = Column(Integer, default=0)
    movies_count = Column(Integer, default=0)
    errors = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
