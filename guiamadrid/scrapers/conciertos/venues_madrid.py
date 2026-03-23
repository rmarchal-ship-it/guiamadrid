"""Scraper for specific Madrid music venue websites.

Scrapes event listings directly from 26 Madrid venue websites.
Uses multiple strategies per venue:
  1. WordPress REST API (The Events Calendar plugin)
  2. WordPress REST API (posts/custom post types)
  3. HTML parsing with BeautifulSoup

Venues that only have Instagram/Facebook are skipped (no scrapable web).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import NamedTuple

from bs4 import BeautifulSoup

from guiamadrid.scrapers.base import BaseScraper, ConcertEvent, ConcertScrapeResult


# ---------------------------------------------------------------------------
# Venue configuration
# ---------------------------------------------------------------------------

class Venue(NamedTuple):
    name: str
    url: str  # base URL (no trailing slash)
    agenda_paths: list[str]  # paths to try for event listings


VENUES: list[Venue] = [
    # --- Common ownership group (Café Berlín, Clamores, Villanos, Tempo) ---
    # WordPress sites with similar structure
    Venue("Café Berlín", "https://berlincafe.es", ["/programas/", "/programa/"]),
    Venue("Sala Clamores", "https://www.salaclamores.es", ["/calendario", "/agenda/", "/"]),
    Venue("Sala Villanos", "https://salavillanos.es", ["/agenda/", "/"]),
    Venue("Tempo Club", "https://tempoclub.es", ["/conciertos/", "/en/concerts/", "/"]),
    # --- Major venues ---
    Venue("Sala La Riviera", "https://salariviera.com", ["/events/", "/conciertossalariviera/", "/"]),
    Venue("Siroco", "https://siroco.es", ["/agenda/", "/conciertos/", "/"]),
    Venue("Galileo Galilei", "https://salagalileo.es", ["/programacion/", "/"]),
    Venue("Sala Rockville", "https://rockville.es", ["/programacion/", "/"]),
    # --- Jazz / intimate venues ---
    Venue("Café Central", "https://cafecentralmadrid.com", ["/programacion/", "/"]),
    Venue("Recoletos Jazz", "https://www.recoletosjazz.com", ["/Tickets-Shows/", "/"]),
    Venue("The Jungle Jazz Club", "https://www.thejunglejazzclub.com", ["/agenda/", "/programacion/", "/"]),
    Venue("Café El Despertar", "https://www.cafeeldespertar.com", ["/agenda/", "/programacion/", "/"]),
    Venue("Babylon Club", "https://www.babylonmadrid.com", ["/agenda", "/"]),
    # --- Smaller venues ---
    Venue("Sala Honky Tonk", "https://www.clubhonky.com", ["/agenda/", "/programacion/", "/"]),
    Venue("ContraClub", "https://www.contraclub.es", ["/agenda/", "/programacion/", "/"]),
    Venue("Sala La Caverna", "https://www.salalacaverna.com", ["/agenda/", "/"]),
    Venue("Sala Vesta", "https://www.salavesta.com", ["/agenda/", "/programacion/", "/"]),
    Venue("Madreams Music", "https://madreamsmusic.es", ["/agenda/", "/"]),
    Venue("El Sótano", "https://www.salaelsotano.com", ["/agenda/", "/programacion/", "/"]),
    Venue("Intruso Bar", "https://www.intrusobar.com", ["/agenda/", "/programacion/", "/"]),
    Venue("Blackbird", "https://www.blackbirdrockbar.com", ["/agenda/", "/programacion/", "/"]),
    Venue("La Fontana de Oro", "https://www.fontanadeoro.com", ["/agenda/", "/programacion/", "/"]),
]

# Venues with only Instagram/Facebook (not scrapable via web)
SOCIAL_ONLY_VENUES = [
    ("Búho Real", "https://www.instagram.com/salabuhorealbar"),
    ("La Coquette Blues Bar", "https://www.facebook.com/La-Coquette-Blues-Bar-335324683222436/"),
    ("Jazzville", "https://www.instagram.com/salajazzville"),
    ("Casa Brava", "https://www.instagram.com/casa.brava.madrid/"),
]

# ---------------------------------------------------------------------------
# Spanish date/month parsing
# ---------------------------------------------------------------------------

_MONTHS_ES = {
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "jun": 6,
    "julio": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "sep": 9, "sept": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11,
    "diciembre": 12, "dic": 12,
}

_DATE_PATTERNS = [
    # "23 de marzo de 2026", "23 marzo 2026"
    re.compile(
        r"(\d{1,2})\s+(?:de\s+)?("
        + "|".join(_MONTHS_ES.keys())
        + r")(?:\s+(?:de\s+)?(\d{4}))?",
        re.IGNORECASE,
    ),
    # "2026-03-23"
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    # "23/03/2026" or "23-03-2026"
    re.compile(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})"),
    # "23/03" (no year, assume current year)
    re.compile(r"(\d{1,2})[/\-](\d{1,2})(?!\d)"),
]

_TIME_PATTERN = re.compile(r"(\d{1,2})[:\.](\d{2})\s*h?(?:oras)?", re.IGNORECASE)
_PRICE_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*€|€\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*euros",
    re.IGNORECASE,
)


def _parse_spanish_date(text: str, ref_year: int) -> str | None:
    """Try to extract a YYYY-MM-DD date from Spanish text."""
    text_lower = text.lower()

    # Pattern 1: "23 de marzo de 2026"
    m = _DATE_PATTERNS[0].search(text_lower)
    if m:
        day = int(m.group(1))
        month = _MONTHS_ES.get(m.group(2).lower())
        year = int(m.group(3)) if m.group(3) else ref_year
        if month and 1 <= day <= 31:
            return f"{year}-{month:02d}-{day:02d}"

    # Pattern 2: "2026-03-23"
    m = _DATE_PATTERNS[1].search(text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Pattern 3: "23/03/2026"
    m = _DATE_PATTERNS[2].search(text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year}-{month:02d}-{day:02d}"

    # Pattern 4: "23/03" (no year)
    m = _DATE_PATTERNS[3].search(text)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{ref_year}-{month:02d}-{day:02d}"

    return None


def _parse_time(text: str) -> str:
    """Extract HH:MM time from text."""
    m = _TIME_PATTERN.search(text)
    if m:
        h, mins = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mins <= 59:
            return f"{h:02d}:{mins:02d}"
    return ""


def _parse_price(text: str) -> str:
    """Extract price info from text."""
    prices = []
    for m in _PRICE_PATTERN.finditer(text):
        val = m.group(1) or m.group(2) or m.group(3)
        if val:
            prices.append(float(val.replace(",", ".")))
    if not prices:
        return ""
    if len(prices) == 1:
        return f"{prices[0]:.0f}€"
    return f"{min(prices):.0f}€ - {max(prices):.0f}€"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class VenuesMadridScraper(BaseScraper):
    """Scrapes concert events from individual Madrid venue websites."""

    # Shorter timeout for API probing (WP REST API), full timeout for HTML pages
    _API_TIMEOUT = 5
    _HTML_TIMEOUT = 10

    def __init__(self):
        super().__init__()
        # Accept HTML for venue sites
        self._client.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        # Shorter delay between requests (0.3s instead of 1s from base)
        self._delay = 0.3

    def scrape(self, target_date: date | None = None) -> ConcertScrapeResult:
        target_date = target_date or date.today()
        target_str = target_date.strftime("%Y-%m-%d")
        ref_year = target_date.year

        all_events: list[ConcertEvent] = []
        venues_with_events: set[str] = set()
        errors: list[str] = []

        for venue in VENUES:
            try:
                events = self._scrape_venue(venue, target_str, ref_year)
                if events:
                    all_events.extend(events)
                    venues_with_events.add(venue.name)
            except Exception as e:
                errors.append(f"{venue.name}: {e}")

        return ConcertScrapeResult(
            events=all_events,
            venues_count=len(venues_with_events),
            errors=errors,
        )

    def _scrape_venue(
        self, venue: Venue, target_str: str, ref_year: int
    ) -> list[ConcertEvent]:
        """Try multiple strategies to scrape a venue."""
        events: list[ConcertEvent] = []

        # Strategy 1: WordPress Events Calendar API
        wp_events = self._try_wp_events_api(venue, target_str)
        if wp_events:
            return wp_events

        # Strategy 2: WordPress REST API (posts)
        wp_posts = self._try_wp_posts_api(venue, target_str, ref_year)
        if wp_posts:
            return wp_posts

        # Strategy 3: HTML parsing of agenda pages
        for path in venue.agenda_paths:
            url = venue.url + path
            try:
                resp = self._client.get(url, timeout=self._HTML_TIMEOUT)
                if resp.status_code != 200:
                    continue
                html = resp.text
                parsed = self._parse_html_events(html, venue, target_str, ref_year)
                if parsed:
                    events.extend(parsed)
                    break  # found events, stop trying other paths
            except Exception:
                continue

        # Strategy 4: JSON-LD structured data (already in HTML)
        # (handled within _parse_html_events)

        return events

    def _try_wp_events_api(
        self, venue: Venue, target_str: str
    ) -> list[ConcertEvent] | None:
        """Try The Events Calendar WordPress plugin API."""
        url = f"{venue.url}/wp-json/tribe/events/v1/events"
        try:
            resp = self._client.get(url, timeout=self._API_TIMEOUT)
            if resp.status_code != 200:
                return None
            data = resp.json()
            events = []
            for ev in data.get("events", []):
                event_date = (ev.get("start_date", "") or "")[:10]
                if not event_date:
                    continue
                event_time = ""
                start = ev.get("start_date", "")
                if "T" in start or " " in start:
                    try:
                        dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                        event_time = dt.strftime("%H:%M")
                    except ValueError:
                        pass

                # Price
                price_range = ""
                cost = ev.get("cost", "")
                if cost:
                    price_range = _parse_price(str(cost)) or str(cost)

                events.append(ConcertEvent(
                    event_name=ev.get("title", ""),
                    artist=ev.get("title", ""),
                    venue_name=venue.name,
                    venue_id=f"venue_{venue.name.lower().replace(' ', '_')}",
                    venue_address=self._extract_wp_venue_address(ev),
                    date=event_date,
                    time=event_time,
                    genre="",
                    price_range=price_range,
                    ticket_url=ev.get("url", ""),
                    image_url=ev.get("image", {}).get("url", "") if isinstance(ev.get("image"), dict) else "",
                    source="venue_web",
                    external_id=f"venue_{venue.name}_{ev.get('id', '')}",
                ))
            return events if events else None
        except Exception:
            return None

    @staticmethod
    def _extract_wp_venue_address(event: dict) -> str:
        venue_data = event.get("venue", {})
        if isinstance(venue_data, dict):
            parts = [
                venue_data.get("address", ""),
                venue_data.get("city", ""),
            ]
            return ", ".join(p for p in parts if p)
        return ""

    def _try_wp_posts_api(
        self, venue: Venue, target_str: str, ref_year: int
    ) -> list[ConcertEvent] | None:
        """Try WordPress REST API for posts."""
        url = f"{venue.url}/wp-json/wp/v2/posts?per_page=20"
        try:
            resp = self._client.get(url, timeout=self._API_TIMEOUT)
            if resp.status_code != 200:
                return None
            posts = resp.json()
            if not isinstance(posts, list):
                return None
            events = []
            for post in posts:
                title = ""
                title_data = post.get("title", {})
                if isinstance(title_data, dict):
                    title = title_data.get("rendered", "")
                elif isinstance(title_data, str):
                    title = title_data
                # Strip HTML from title
                title = re.sub(r"<[^>]+>", "", title).strip()
                if not title:
                    continue

                # Try to find date in content
                content = ""
                content_data = post.get("content", {})
                if isinstance(content_data, dict):
                    content = content_data.get("rendered", "")
                elif isinstance(content_data, str):
                    content = content_data
                content_text = re.sub(r"<[^>]+>", " ", content)

                event_date = _parse_spanish_date(content_text, ref_year)
                if not event_date:
                    event_date = _parse_spanish_date(title, ref_year)
                if not event_date:
                    continue

                event_time = _parse_time(content_text) or _parse_time(title)
                price_range = _parse_price(content_text)

                # Image
                image_url = ""
                featured = post.get("_embedded", {}).get("wp:featuredmedia", [])
                if featured and isinstance(featured, list):
                    image_url = featured[0].get("source_url", "")
                if not image_url:
                    img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content)
                    if img_match:
                        image_url = img_match.group(1)

                events.append(ConcertEvent(
                    event_name=title,
                    artist=title,
                    venue_name=venue.name,
                    venue_id=f"venue_{venue.name.lower().replace(' ', '_')}",
                    date=event_date,
                    time=event_time,
                    genre="",
                    price_range=price_range,
                    ticket_url=post.get("link", ""),
                    image_url=image_url,
                    source="venue_web",
                    external_id=f"venue_{venue.name}_{post.get('id', '')}",
                ))
            return events if events else None
        except Exception:
            return None

    def _parse_html_events(
        self, html: str, venue: Venue, target_str: str, ref_year: int
    ) -> list[ConcertEvent]:
        """Parse events from raw HTML using BeautifulSoup."""
        events: list[ConcertEvent] = []

        # First try JSON-LD
        events.extend(self._extract_jsonld_events(html, venue))
        if events:
            return events

        soup = BeautifulSoup(html, "html.parser")

        # Strategy A: Look for common event container patterns
        event_selectors = [
            "article",
            ".event", ".evento", ".concierto", ".concert",
            ".programacion-item", ".programa-item",
            ".tribe-events-calendar-list__event",
            '[class*="event"]', '[class*="concert"]', '[class*="programa"]',
            ".entry", ".post",
            "li.wp-block-post",
        ]

        containers = []
        for selector in event_selectors:
            found = soup.select(selector)
            if found and len(found) >= 2:  # at least 2 items suggests event listing
                containers = found
                break

        if containers:
            for container in containers:
                event = self._parse_event_container(container, venue, ref_year)
                if event:
                    events.append(event)
        else:
            # Strategy B: Look for date patterns in the whole page
            # and extract surrounding text as events
            events.extend(self._extract_events_by_dates(soup, venue, ref_year))

        return events

    def _extract_jsonld_events(
        self, html: str, venue: Venue
    ) -> list[ConcertEvent]:
        """Extract events from JSON-LD schema.org blocks."""
        events: list[ConcertEvent] = []
        for match in re.finditer(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html,
            re.DOTALL,
        ):
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("@type", "")
                if item_type not in ("Event", "MusicEvent"):
                    continue
                name = item.get("name", "")
                if not name:
                    continue

                start_date = item.get("startDate", "")
                event_date = start_date[:10] if start_date else ""
                event_time = ""
                if "T" in start_date:
                    try:
                        dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                        event_time = dt.strftime("%H:%M")
                        event_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pass

                # Performer
                performers = item.get("performer", [])
                if isinstance(performers, dict):
                    performers = [performers]
                artist_names = []
                if isinstance(performers, list):
                    for p in performers:
                        if isinstance(p, dict):
                            artist_names.append(p.get("name", ""))
                artist = ", ".join(filter(None, artist_names)) or name

                # Price
                price_range = ""
                offers = item.get("offers", {})
                if isinstance(offers, dict):
                    low = offers.get("lowPrice") or offers.get("price")
                    high = offers.get("highPrice")
                    if low and high and str(low) != str(high):
                        price_range = f"{low}€ - {high}€"
                    elif low:
                        price_range = f"{low}€"

                # Image
                image = item.get("image", "")
                if isinstance(image, list):
                    image = image[0] if image else ""
                if isinstance(image, dict):
                    image = image.get("url", "")

                events.append(ConcertEvent(
                    event_name=name,
                    artist=artist,
                    venue_name=venue.name,
                    venue_id=f"venue_{venue.name.lower().replace(' ', '_')}",
                    date=event_date,
                    time=event_time,
                    genre="",
                    price_range=price_range,
                    ticket_url=item.get("url", ""),
                    image_url=image if isinstance(image, str) else "",
                    source="venue_web",
                    external_id=f"venue_{venue.name}_{name}_{event_date}",
                ))

        return events

    def _parse_event_container(
        self, container, venue: Venue, ref_year: int
    ) -> ConcertEvent | None:
        """Parse a single event from an HTML container element."""
        text = container.get_text(separator=" ", strip=True)
        if len(text) < 5:
            return None

        # Title: first heading, or link text, or strong text
        title = ""
        for tag in ["h1", "h2", "h3", "h4", "a", "strong", ".title", ".event-title"]:
            el = container.select_one(tag)
            if el:
                title = el.get_text(strip=True)
                if len(title) > 3:
                    break
        if not title:
            # Use first significant text chunk
            title = text[:80].split("·")[0].split("|")[0].strip()

        if not title or len(title) < 3:
            return None

        # Date
        event_date = _parse_spanish_date(text, ref_year)
        if not event_date:
            # Check datetime attributes
            time_el = container.find("time")
            if time_el and time_el.get("datetime"):
                event_date = time_el["datetime"][:10]

        # Time
        event_time = _parse_time(text)

        # Price
        price_range = _parse_price(text)

        # Image
        image_url = ""
        img = container.find("img")
        if img:
            image_url = img.get("src", "") or img.get("data-src", "") or img.get("data-lazy-src", "")

        # Link
        ticket_url = ""
        link = container.find("a", href=True)
        if link:
            href = link["href"]
            if href.startswith("http"):
                ticket_url = href
            elif href.startswith("/"):
                ticket_url = venue.url + href

        return ConcertEvent(
            event_name=title,
            artist=title,
            venue_name=venue.name,
            venue_id=f"venue_{venue.name.lower().replace(' ', '_')}",
            date=event_date or "",
            time=event_time,
            genre="",
            price_range=price_range,
            ticket_url=ticket_url,
            image_url=image_url,
            source="venue_web",
            external_id=f"venue_{venue.name}_{title}_{event_date or 'nodate'}",
        )

    def _extract_events_by_dates(
        self, soup: BeautifulSoup, venue: Venue, ref_year: int
    ) -> list[ConcertEvent]:
        """Last resort: scan the page for date patterns and extract nearby text."""
        events: list[ConcertEvent] = []
        body = soup.find("body")
        if not body:
            return events

        # Get all text blocks
        for element in body.find_all(["div", "section", "li", "p", "article"]):
            text = element.get_text(separator=" ", strip=True)
            if len(text) < 10 or len(text) > 500:
                continue

            event_date = _parse_spanish_date(text, ref_year)
            if not event_date:
                continue

            # Find a title in this block
            title = ""
            for tag in ["h1", "h2", "h3", "h4", "a", "strong"]:
                el = element.find(tag)
                if el:
                    t = el.get_text(strip=True)
                    if len(t) > 3:
                        title = t
                        break
            if not title:
                # Skip dates without a clear title
                continue

            event_time = _parse_time(text)
            price_range = _parse_price(text)

            img = element.find("img")
            image_url = ""
            if img:
                image_url = img.get("src", "") or img.get("data-src", "")

            link = element.find("a", href=True)
            ticket_url = ""
            if link:
                href = link["href"]
                if href.startswith("http"):
                    ticket_url = href
                elif href.startswith("/"):
                    ticket_url = venue.url + href

            events.append(ConcertEvent(
                event_name=title,
                artist=title,
                venue_name=venue.name,
                venue_id=f"venue_{venue.name.lower().replace(' ', '_')}",
                date=event_date,
                time=event_time,
                genre="",
                price_range=price_range,
                ticket_url=ticket_url,
                image_url=image_url,
                source="venue_web",
                external_id=f"venue_{venue.name}_{title}_{event_date}",
            ))

        return events
