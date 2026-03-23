"""Base scraper with shared HTTP logic and rate limiting."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date

import cloudscraper

from guiamadrid.config import REQUEST_DELAY, REQUEST_TIMEOUT, USER_AGENT


@dataclass
class Showtime:
    """A single movie showtime at a cinema."""
    cinema_name: str
    cinema_id: str
    movie_title: str
    movie_id: str | None = None
    showtime: str = ""  # "14:30"
    date: str = ""  # "2026-03-20"
    language: str = ""  # "VOSE", "Castellano", etc.
    format: str = ""  # "2D", "3D", "IMAX"
    director: str = ""
    poster_url: str = ""
    synopsis: str = ""
    rating: float | None = None
    genre: str = ""
    duration_min: int | None = None


@dataclass
class ScrapeResult:
    """Result of a scrape run."""
    showtimes: list[Showtime] = field(default_factory=list)
    cinemas_count: int = 0
    movies_count: int = 0
    errors: list[str] = field(default_factory=list)


class BaseScraper(ABC):
    """Base class for all scrapers."""

    def __init__(self):
        self._client = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "linux", "desktop": True},
        )
        self._client.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "es-ES,es;q=0.9",
        })
        self._timeout = REQUEST_TIMEOUT
        self._last_request_time = 0.0

    def _get(self, url: str):
        """Make a rate-limited GET request."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()
        return self._client.get(url, timeout=self._timeout)

    def _get_json(self, url: str) -> dict:
        """GET request expecting JSON response."""
        resp = self._get(url)
        resp.raise_for_status()
        return resp.json()

    @abstractmethod
    def scrape(self, target_date: date | None = None) -> ScrapeResult:
        """Run the scraper for a given date (default: today)."""
        ...

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
