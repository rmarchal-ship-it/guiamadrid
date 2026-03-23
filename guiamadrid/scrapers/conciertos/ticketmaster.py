"""Ticketmaster Discovery API scraper for Madrid concerts.

Uses the public Discovery API v2:
    https://app.ticketmaster.com/discovery/v2/events.json
    ?countryCode=ES&city=Madrid&classificationName=music
    &startDateTime=...&endDateTime=...&size=200&apikey={KEY}

Requires TICKETMASTER_API_KEY environment variable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import requests

from guiamadrid.config import TICKETMASTER_API_KEY, TICKETMASTER_BASE_URL
from guiamadrid.scrapers.base import ConcertEvent, ConcertScrapeResult


class TicketmasterScraper:
    """Scrapes concert events from Ticketmaster Discovery API."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or TICKETMASTER_API_KEY
        if not self._api_key:
            raise ValueError(
                "TICKETMASTER_API_KEY not set. "
                "Register at https://developer.ticketmaster.com/"
            )

    def scrape(self, target_date: date | None = None) -> ConcertScrapeResult:
        target_date = target_date or date.today()
        end_date = target_date + timedelta(days=7)

        events: list[ConcertEvent] = []
        venues_seen: set[str] = set()
        errors: list[str] = []
        page = 0

        while True:
            try:
                data = self._fetch_page(target_date, end_date, page)
            except Exception as e:
                errors.append(f"Page {page}: {e}")
                break

            embedded = data.get("_embedded")
            if not embedded or "events" not in embedded:
                break

            for event_data in embedded["events"]:
                try:
                    event = self._parse_event(event_data)
                    if event:
                        events.append(event)
                        if event.venue_id:
                            venues_seen.add(event.venue_id)
                except Exception as e:
                    errors.append(f"Event parse error: {e}")

            # Pagination
            page_info = data.get("page", {})
            total_pages = page_info.get("totalPages", 1)
            page += 1
            if page >= total_pages:
                break

        return ConcertScrapeResult(
            events=events,
            venues_count=len(venues_seen),
            errors=errors,
        )

    def _fetch_page(self, start: date, end: date, page: int) -> dict:
        params = {
            "apikey": self._api_key,
            "countryCode": "ES",
            "city": "Madrid",
            "classificationName": "music",
            "startDateTime": f"{start}T00:00:00Z",
            "endDateTime": f"{end}T23:59:59Z",
            "size": 200,
            "page": page,
            "sort": "date,asc",
        }
        resp = requests.get(
            f"{TICKETMASTER_BASE_URL}/events.json",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_event(data: dict) -> ConcertEvent | None:
        name = data.get("name", "")
        if not name:
            return None

        # Artist(s)
        attractions = data.get("_embedded", {}).get("attractions", [])
        artist = ", ".join(a.get("name", "") for a in attractions) or name

        # Venue
        venues = data.get("_embedded", {}).get("venues", [])
        venue = venues[0] if venues else {}
        venue_name = venue.get("name", "")
        venue_id = venue.get("id", "")
        venue_address = ""
        addr = venue.get("address", {})
        if addr:
            venue_address = addr.get("line1", "")
        location = venue.get("location", {})
        lat = None
        lon = None
        if location:
            try:
                lat = float(location.get("latitude", 0))
                lon = float(location.get("longitude", 0))
            except (ValueError, TypeError):
                pass

        # Date & time
        dates = data.get("dates", {}).get("start", {})
        event_date = dates.get("localDate", "")
        event_time = dates.get("localTime", "")[:5] if dates.get("localTime") else ""

        # Genre
        classifications = data.get("classifications", [])
        genre_parts = []
        if classifications:
            c = classifications[0]
            for key in ("genre", "subGenre"):
                g = c.get(key, {})
                if g and g.get("name") and g["name"] != "Undefined":
                    genre_parts.append(g["name"])
        genre = ", ".join(genre_parts)

        # Price
        price_range = ""
        prices = data.get("priceRanges", [])
        if prices:
            p = prices[0]
            currency = p.get("currency", "EUR")
            symbol = "€" if currency == "EUR" else currency
            low = p.get("min")
            high = p.get("max")
            if low and high:
                price_range = f"{low}{symbol} - {high}{symbol}"
            elif low:
                price_range = f"{low}{symbol}"

        # Image
        images = data.get("images", [])
        image_url = ""
        if images:
            # Prefer higher resolution
            for img in sorted(images, key=lambda i: i.get("width", 0), reverse=True):
                image_url = img.get("url", "")
                break

        # Ticket URL
        ticket_url = data.get("url", "")

        return ConcertEvent(
            event_name=name,
            artist=artist,
            venue_name=venue_name,
            venue_id=venue_id,
            venue_address=venue_address,
            venue_latitude=lat,
            venue_longitude=lon,
            date=event_date,
            time=event_time,
            genre=genre,
            price_range=price_range,
            ticket_url=ticket_url,
            image_url=image_url,
            source="ticketmaster",
            external_id=data.get("id", ""),
        )

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
