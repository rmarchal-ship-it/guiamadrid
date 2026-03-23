"""DICE.fm scraper for Madrid concert/music events.

DICE.fm is a JavaScript SPA. This scraper uses cloudscraper to bypass
Cloudflare and extracts JSON-LD structured data (schema.org/Event)
embedded in venue and event pages.

Madrid city ID: 5d8cef380de1e404dc962211
Browse URL: https://dice.fm/browse/madrid-5d8cef380de1e404dc962211/music/gig?lng=es
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime

from guiamadrid.scrapers.base import BaseScraper, ConcertEvent, ConcertScrapeResult

_MADRID_CITY_ID = "5d8cef380de1e404dc962211"
_BROWSE_URL = f"https://dice.fm/browse/madrid-{_MADRID_CITY_ID}/music/{{category}}?lng=es"

# Event categories to scrape
_CATEGORIES = ["gig", "party", "dj"]

# Regex to extract JSON-LD blocks from HTML
_JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL,
)

# Regex to extract __NEXT_DATA__ (Next.js hydration payload)
_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
    re.DOTALL,
)


class DiceScraper(BaseScraper):
    """Scrapes concert events from DICE.fm Madrid pages."""

    def scrape(self, target_date: date | None = None) -> ConcertScrapeResult:
        target_date = target_date or date.today()
        target_str = target_date.strftime("%Y-%m-%d")

        events: list[ConcertEvent] = []
        venues_seen: set[str] = set()
        errors: list[str] = []
        seen_ids: set[str] = set()

        for category in _CATEGORIES:
            url = _BROWSE_URL.format(category=category)
            try:
                html = self._fetch_html(url)
                page_events = self._extract_events(html, target_str)
                for ev in page_events:
                    key = ev.external_id or f"{ev.event_name}_{ev.date}"
                    if key not in seen_ids:
                        seen_ids.add(key)
                        events.append(ev)
                        if ev.venue_id:
                            venues_seen.add(ev.venue_id)
            except Exception as e:
                errors.append(f"DICE {category}: {e}")

        return ConcertScrapeResult(
            events=events,
            venues_count=len(venues_seen),
            errors=errors,
        )

    def _fetch_html(self, url: str) -> str:
        resp = self._get(url)
        resp.raise_for_status()
        return resp.text

    def _extract_events(self, html: str, target_str: str) -> list[ConcertEvent]:
        """Try multiple extraction methods: JSON-LD, __NEXT_DATA__, or both."""
        events: list[ConcertEvent] = []

        # Method 1: JSON-LD
        events.extend(self._parse_jsonld(html, target_str))

        # Method 2: __NEXT_DATA__ (Next.js)
        if not events:
            events.extend(self._parse_next_data(html, target_str))

        return events

    def _parse_jsonld(self, html: str, target_str: str) -> list[ConcertEvent]:
        """Extract events from JSON-LD schema.org blocks."""
        events: list[ConcertEvent] = []

        for match in _JSONLD_RE.finditer(html):
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

            # Handle single objects and arrays
            items = data if isinstance(data, list) else [data]

            for item in items:
                event = self._jsonld_to_event(item, target_str)
                if event:
                    events.append(event)

        return events

    @staticmethod
    def _jsonld_to_event(item: dict, target_str: str) -> ConcertEvent | None:
        """Convert a JSON-LD Event object to ConcertEvent."""
        item_type = item.get("@type", "")
        if item_type not in ("Event", "MusicEvent"):
            return None

        name = item.get("name", "")
        if not name:
            return None

        # Date
        start_date = item.get("startDate", "")
        event_date = start_date[:10] if start_date else ""
        event_time = ""
        if start_date and "T" in start_date:
            try:
                dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                event_time = dt.strftime("%H:%M")
                event_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        # Venue
        location = item.get("location", {})
        venue_name = ""
        venue_address = ""
        venue_id = ""
        if isinstance(location, dict):
            venue_name = location.get("name", "")
            addr = location.get("address", {})
            if isinstance(addr, dict):
                venue_address = addr.get("streetAddress", "")
            elif isinstance(addr, str):
                venue_address = addr
            venue_id = f"dice_{venue_name}" if venue_name else ""

        # Artist/performer
        performers = item.get("performer", [])
        if isinstance(performers, dict):
            performers = [performers]
        artist_names = []
        for p in performers:
            if isinstance(p, dict):
                artist_names.append(p.get("name", ""))
        artist = ", ".join(filter(None, artist_names)) or name

        # Image
        image = item.get("image", "")
        if isinstance(image, list):
            image = image[0] if image else ""
        if isinstance(image, dict):
            image = image.get("url", "")

        # Price
        price_range = ""
        offers = item.get("offers", {})
        if isinstance(offers, dict):
            low = offers.get("lowPrice") or offers.get("price")
            high = offers.get("highPrice")
            currency = offers.get("priceCurrency", "EUR")
            symbol = "€" if currency == "EUR" else currency
            if low and high and low != high:
                price_range = f"{low}{symbol} - {high}{symbol}"
            elif low:
                price_range = f"{low}{symbol}"

        # URL
        ticket_url = item.get("url", "")

        return ConcertEvent(
            event_name=name,
            artist=artist,
            venue_name=venue_name,
            venue_id=venue_id,
            venue_address=venue_address,
            date=event_date,
            time=event_time,
            genre="",
            price_range=price_range,
            ticket_url=ticket_url,
            image_url=image,
            source="dice",
            external_id=ticket_url or name,
        )

    def _parse_next_data(self, html: str, target_str: str) -> list[ConcertEvent]:
        """Extract events from Next.js __NEXT_DATA__ hydration payload."""
        events: list[ConcertEvent] = []

        match = _NEXT_DATA_RE.search(html)
        if not match:
            return events

        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return events

        # Navigate the Next.js data structure to find events
        # Structure varies, so we search recursively for event-like objects
        self._find_events_in_data(data, events, target_str)

        return events

    def _find_events_in_data(
        self, data, events: list[ConcertEvent], target_str: str, depth: int = 0
    ):
        """Recursively search for event objects in nested data."""
        if depth > 8:
            return

        if isinstance(data, dict):
            # Look for event-like objects
            if "name" in data and ("date" in data or "startDate" in data or "event_date" in data):
                event = self._dict_to_event(data)
                if event:
                    events.append(event)
                    return

            for value in data.values():
                self._find_events_in_data(value, events, target_str, depth + 1)

        elif isinstance(data, list):
            for item in data:
                self._find_events_in_data(item, events, target_str, depth + 1)

    @staticmethod
    def _dict_to_event(data: dict) -> ConcertEvent | None:
        """Try to convert a generic dict to a ConcertEvent."""
        name = data.get("name") or data.get("title") or ""
        if not name:
            return None

        # Date
        event_date = (
            data.get("date", "")
            or data.get("startDate", "")
            or data.get("event_date", "")
        )[:10]
        if not event_date:
            return None

        # Time
        start = data.get("startDate") or data.get("date") or ""
        event_time = ""
        if "T" in start:
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                event_time = dt.strftime("%H:%M")
            except ValueError:
                pass

        # Venue
        venue = data.get("venue", {}) or data.get("location", {})
        if isinstance(venue, str):
            venue_name = venue
            venue_address = ""
        elif isinstance(venue, dict):
            venue_name = venue.get("name", "")
            venue_address = venue.get("address", "")
            if isinstance(venue_address, dict):
                venue_address = venue_address.get("streetAddress", "")
        else:
            venue_name = ""
            venue_address = ""

        # Artist
        artists = data.get("artists", []) or data.get("lineup", [])
        if isinstance(artists, list):
            artist_names = []
            for a in artists:
                if isinstance(a, str):
                    artist_names.append(a)
                elif isinstance(a, dict):
                    artist_names.append(a.get("name", ""))
            artist = ", ".join(filter(None, artist_names)) or name
        else:
            artist = name

        image = data.get("image", "") or data.get("image_url", "") or data.get("images", [""])[0] if isinstance(data.get("images"), list) else ""
        ticket_url = data.get("url", "") or data.get("ticket_url", "")

        return ConcertEvent(
            event_name=name,
            artist=artist,
            venue_name=venue_name,
            venue_id=f"dice_{venue_name}" if venue_name else "",
            venue_address=venue_address if isinstance(venue_address, str) else "",
            date=event_date,
            time=event_time,
            genre=data.get("genre", ""),
            price_range="",
            ticket_url=ticket_url,
            image_url=image if isinstance(image, str) else "",
            source="dice",
            external_id=data.get("id", "") or ticket_url or name,
        )
