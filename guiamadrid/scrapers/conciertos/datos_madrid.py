"""Scraper for datos.madrid.es open cultural events API.

Uses the public JSON endpoint (no API key required):
    https://datos.madrid.es/egob/catalogo/206974-0-agenda-eventos-culturales-100.json

Filters for music-related events in Madrid.
"""

from __future__ import annotations

from datetime import date

import requests

from guiamadrid.config import DATOS_MADRID_EVENTS_URL
from guiamadrid.scrapers.base import ConcertEvent, ConcertScrapeResult

_MUSIC_KEYWORDS = {
    "concierto", "música", "musica", "jazz", "flamenco", "ópera", "opera",
    "recital", "coro", "sinfónic", "sinfonic", "orquesta", "acústic",
    "cantautor", "festival", "dj", "electrónic",
}


class DatosMadridScraper:
    """Scrapes music events from Madrid's open data portal."""

    def scrape(self, target_date: date | None = None) -> ConcertScrapeResult:
        target_date = target_date or date.today()
        target_str = target_date.strftime("%Y-%m-%d")

        events: list[ConcertEvent] = []
        venues_seen: set[str] = set()
        errors: list[str] = []

        try:
            data = self._fetch()
        except Exception as e:
            return ConcertScrapeResult(errors=[f"Fetch failed: {e}"])

        graph = data.get("@graph", [])

        for item in graph:
            try:
                event = self._parse_event(item, target_str)
                if event:
                    events.append(event)
                    if event.venue_id:
                        venues_seen.add(event.venue_id)
            except Exception as e:
                errors.append(f"Parse error: {e}")

        return ConcertScrapeResult(
            events=events,
            venues_count=len(venues_seen),
            errors=errors,
        )

    def _fetch(self) -> dict:
        resp = requests.get(DATOS_MADRID_EVENTS_URL, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _is_music_event(item: dict) -> bool:
        title = (item.get("title") or "").lower()
        desc = (item.get("description") or "").lower()
        event_type = (item.get("/tipo") or "").lower()

        text = f"{title} {desc} {event_type}"
        return any(kw in text for kw in _MUSIC_KEYWORDS)

    @classmethod
    def _parse_event(cls, item: dict, target_str: str) -> ConcertEvent | None:
        if not cls._is_music_event(item):
            return None

        title = item.get("title", "")
        if not title:
            return None

        # Date filtering: event must overlap with target date
        date_start = (item.get("dtstart") or "")[:10]
        date_end = (item.get("dtend") or "")[:10]
        if date_start and date_start > target_str:
            return None
        if date_end and date_end < target_str:
            return None

        # Time
        time_str = (item.get("time") or "")[:5]

        # Price
        price = item.get("price") or ""
        is_free = item.get("free") == 1
        if is_free:
            price = "Gratis"

        # Venue
        location = item.get("event-location") or item.get("location") or {}
        if isinstance(location, str):
            venue_name = location
            venue_address = ""
            lat, lon = None, None
        else:
            venue_name = location.get("facility-name") or location.get("area", {}).get("locality", "") or ""
            venue_address = location.get("address", {}).get("street-address", "") if isinstance(location.get("address"), dict) else ""
            lat = location.get("latitude")
            lon = location.get("longitude")
            if lat:
                try:
                    lat = float(lat)
                except (ValueError, TypeError):
                    lat = None
            if lon:
                try:
                    lon = float(lon)
                except (ValueError, TypeError):
                    lon = None

        venue_id = f"dm_{venue_name}" if venue_name else ""

        # External link
        link = item.get("link") or ""

        return ConcertEvent(
            event_name=title,
            artist=title,  # datos.madrid.es doesn't separate artist from event name
            venue_name=venue_name,
            venue_id=venue_id,
            venue_address=venue_address,
            venue_latitude=lat,
            venue_longitude=lon,
            date=target_str,
            time=time_str,
            genre="",
            price_range=str(price),
            ticket_url=link,
            image_url="",
            source="datos_madrid",
            external_id=str(item.get("id", item.get("@id", ""))),
        )

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
