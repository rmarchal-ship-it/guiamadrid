"""Songkick scraper for Madrid concerts.

Scrapes concert listings from Songkick's Madrid metro area page.
No API key required — uses the public HTML listings with JSON-LD microdata.

Songkick metro ID for Madrid: 28755-spain-madrid
URL: https://www.songkick.com/metro-areas/28755-spain-madrid/calendar
"""

from __future__ import annotations

import json
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from guiamadrid.scrapers.base import ConcertEvent, ConcertScrapeResult

METRO_ID = "28755-spain-madrid"
BASE_URL = "https://www.songkick.com"
CALENDAR_URL = f"{BASE_URL}/metro-areas/{METRO_ID}/calendar"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}
REQUEST_DELAY = 1.5
MAX_PAGES = 5  # ~50 events/page → up to 250 events


class SongkickScraper:
    """Scrapes upcoming concerts in Madrid from Songkick."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def scrape(self, target_date: date | None = None) -> ConcertScrapeResult:
        target_date = target_date or date.today()
        events: list[ConcertEvent] = []
        errors: list[str] = []
        venues_seen: set[str] = set()

        for page in range(1, MAX_PAGES + 1):
            try:
                page_events, has_next = self._scrape_page(page, target_date, errors)
                events.extend(page_events)
                for e in page_events:
                    if e.venue_name:
                        venues_seen.add(e.venue_name)
                if not has_next:
                    break
                time.sleep(REQUEST_DELAY)
            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                break

        return ConcertScrapeResult(
            events=self._dedupe(events),
            venues_count=len(venues_seen),
            errors=errors,
        )

    def _scrape_page(
        self, page: int, target_date: date, errors: list[str]
    ) -> tuple[list[ConcertEvent], bool]:
        params = {"page": page}
        try:
            resp = requests.get(
                CALENDAR_URL, params=params, headers=HEADERS, timeout=15
            )
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"HTTP error: {exc}") from exc

        soup = BeautifulSoup(resp.text, "html.parser")
        events: list[ConcertEvent] = []

        for li in soup.select("li.event-listings-element"):
            try:
                event = self._parse_li(li, target_date)
                if event:
                    events.append(event)
            except Exception as exc:
                errors.append(f"Parse error: {exc}")

        has_next = bool(soup.select_one("a.next_page"))
        return events, has_next

    def _parse_li(self, li, target_date: date) -> ConcertEvent | None:
        time_tag = li.select_one("time[datetime]")
        event_date = ""
        event_time = ""
        if time_tag:
            dt_str = time_tag.get("datetime", "")
            if "T" in dt_str:
                event_date = dt_str[:10]
                event_time = dt_str[11:16]
            elif re.match(r"\d{4}-\d{2}-\d{2}", dt_str):
                event_date = dt_str[:10]

        if event_date and event_date < target_date.isoformat():
            return None

        ld_tag = li.select_one("script[type='application/ld+json']")
        ld = None
        if ld_tag:
            try:
                data = json.loads(ld_tag.string)
                ld = data[0] if isinstance(data, list) else data
            except Exception:
                pass

        if ld:
            event_name = ld.get("name", "")
            image_url = ld.get("image", "")
            ticket_url = ld.get("url", "").split("?")[0]
            location = ld.get("location", {})
            venue = location.get("name", "")
            city = location.get("address", {}).get("addressLocality", "")
            if city and city.lower() not in ("madrid",):
                return None
        else:
            artist_tag = li.select_one("p.artists a strong") or li.select_one("p.artists a")
            if not artist_tag:
                return None
            event_name = artist_tag.get_text(strip=True)
            venue_tag = li.select_one("p.location a.venue-link")
            venue = venue_tag.get_text(strip=True) if venue_tag else ""
            city_tag = li.select_one("p.location .city-name")
            city = city_tag.get_text(strip=True) if city_tag else ""
            if city and "madrid" not in city.lower():
                return None
            link_tag = li.select_one("a.event-link[href*='/concerts/']")
            ticket_url = (BASE_URL + link_tag["href"]) if link_tag else ""
            img_tag = li.select_one("img.artist-profile-image")
            image_url = ""
            if img_tag:
                src = img_tag.get("data-src") or img_tag.get("src", "")
                if "default" not in src:
                    image_url = "https:" + src if src.startswith("//") else src

        artist_tag = li.select_one("p.artists a strong")
        artist = artist_tag.get_text(strip=True) if artist_tag else event_name
        if image_url and image_url.startswith("//"):
            image_url = "https:" + image_url
        image_url = re.sub(r"/large_avatar$", "/huge_avatar", image_url)

        return ConcertEvent(
            event_name=event_name,
            artist=artist,
            venue_name=venue,
            date=event_date,
            time=event_time,
            ticket_url=ticket_url,
            image_url=image_url,
            source="songkick",
        )

    def _dedupe(self, events: list[ConcertEvent]) -> list[ConcertEvent]:
        seen: set[tuple] = set()
        unique: list[ConcertEvent] = []
        for e in events:
            key = (e.event_name.lower(), e.date, e.venue_name.lower())
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique
