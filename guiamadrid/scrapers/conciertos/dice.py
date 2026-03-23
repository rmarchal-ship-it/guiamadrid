"""DICE.fm scraper for Madrid concert/music events.

DICE.fm is a JavaScript SPA — browse/listing pages don't contain event data
in the server-rendered HTML. However, individual **venue pages** include
JSON-LD (schema.org/Event) structured data that can be extracted.

Strategy: scrape each known Madrid venue's DICE page for JSON-LD events.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime

from guiamadrid.scrapers.base import BaseScraper, ConcertEvent, ConcertScrapeResult

# Known Madrid venue slugs on DICE.fm
# Format: (venue_name, dice_slug)
_DICE_VENUES = [
    ("Sala Clamores", "sala-clamores-rgdy"),
    ("Siroco", "siroco-987o"),
    ("Sala Villanos", "sala-villanos-27d9v"),
    ("Café Berlín", "caf-berln-dqx3"),
    ("Tempo Club", "tempo-club-2bvw5"),
    ("Sala La Riviera", "sala-la-riviera-3o3yp"),
    ("Galileo Galilei", "sala-galileo-galilei-7r5v"),
    ("Sala Rockville", "rockville-madrid-5q8o"),
    ("Babylon Club", "babylon-2y77p"),
    ("Café Central", "caf-central-madrid-dddae"),
    ("Sala BUT", "but-bk95"),
    ("Teatro Barceló", "teatro-barcel-7b9n"),
    ("La Sala", "la-sala-1q3e"),
    ("Moby Dick Club", "moby-dick-club-2orqg"),
    ("WiZink Center", "wizink-center-2bxlr"),
]

_VENUE_URL = "https://dice.fm/venue/{slug}?lng=es"

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
    """Scrapes concert events from DICE.fm Madrid venue pages."""

    _TIMEOUT = 8  # shorter timeout per venue page

    def scrape(self, target_date: date | None = None) -> ConcertScrapeResult:
        target_date = target_date or date.today()
        target_str = target_date.strftime("%Y-%m-%d")

        events: list[ConcertEvent] = []
        venues_seen: set[str] = set()
        errors: list[str] = []
        seen_ids: set[str] = set()

        for venue_name, slug in _DICE_VENUES:
            url = _VENUE_URL.format(slug=slug)
            try:
                html = self._fetch_html(url)
                page_events = self._extract_events(html, target_str)
                for ev in page_events:
                    # Override venue name with our canonical name
                    ev = ConcertEvent(
                        event_name=ev.event_name,
                        artist=ev.artist,
                        venue_name=venue_name,
                        venue_id=ev.venue_id or f"dice_{venue_name}",
                        venue_address=ev.venue_address,
                        date=ev.date,
                        time=ev.time,
                        genre=ev.genre,
                        price_range=ev.price_range,
                        ticket_url=ev.ticket_url,
                        image_url=ev.image_url,
                        source="dice",
                        external_id=ev.external_id,
                    )
                    key = ev.external_id or f"{ev.event_name}_{ev.date}"
                    if key not in seen_ids:
                        seen_ids.add(key)
                        events.append(ev)
                        venues_seen.add(venue_name)
            except Exception as e:
                errors.append(f"DICE {venue_name}: {e}")

        return ConcertScrapeResult(
            events=events,
            venues_count=len(venues_seen),
            errors=errors,
        )

    def _fetch_html(self, url: str) -> str:
        resp = self._client.get(url, timeout=self._TIMEOUT)
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
            image_url=image if isinstance(image, str) else "",
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

        self._find_events_in_data(data, events, target_str)
        return events

    def _find_events_in_data(
        self, data, events: list[ConcertEvent], target_str: str, depth: int = 0
    ):
        """Recursively search for event objects in nested data."""
        if depth > 8:
            return

        if isinstance(data, dict):
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

        event_date = (
            data.get("date", "")
            or data.get("startDate", "")
            or data.get("event_date", "")
        )[:10]
        if not event_date:
            return None

        start = data.get("startDate") or data.get("date") or ""
        event_time = ""
        if "T" in start:
            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                event_time = dt.strftime("%H:%M")
            except ValueError:
                pass

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
