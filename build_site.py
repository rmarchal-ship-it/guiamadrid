#!/usr/bin/env python3
"""
build_site.py — Full pipeline to build cartelera_standalone.html.

Scrapes SensaCine for today's showtimes across all 62 Madrid cinemas,
finds YouTube trailers, fetches poster URLs, and updates the standalone
HTML file with fresh embedded data.

No API keys needed. Runnable as: python build_site.py
"""

from __future__ import annotations

import html as html_module
import json
import re
import subprocess
import sys
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path

import cloudscraper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
HTML_FILE = PROJECT_ROOT / "cartelera_standalone.html"

# ---------------------------------------------------------------------------
# Config (inline to avoid import issues in CI)
# ---------------------------------------------------------------------------
SENSACINE_BASE_URL = "https://www.sensacine.com"
SHOWTIMES_URL = SENSACINE_BASE_URL + "/_/showtimes/theater-{theater_id}/d-{date}/p-{page}"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
}

# Import cinema IDs from config
try:
    from guiamadrid.config import SENSACINE_THEATER_IDS
except ImportError:
    # Fallback: parse config.py directly
    _config_text = (PROJECT_ROOT / "guiamadrid" / "config.py").read_text(encoding="utf-8")
    SENSACINE_THEATER_IDS = {}
    for m in re.finditer(r'"([EG]\w+)":\s*"([^"]+)"', _config_text):
        SENSACINE_THEATER_IDS[m.group(1)] = m.group(2)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Scrape SensaCine
# ═══════════════════════════════════════════════════════════════════════════

# Shared session — cloudscraper handles Cloudflare challenges automatically
_session = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "linux", "desktop": True},
)
_session.headers.update(HEADERS)


def _fetch_json(url: str, referer: str = "") -> dict:
    """Fetch a URL and return parsed JSON, with retries."""
    extra: dict[str, str] = {}
    if referer:
        extra["Referer"] = referer
    for attempt in range(MAX_RETRIES):
        try:
            resp = _session.get(url, headers=extra, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                if hasattr(e, "response") and e.response is not None:
                    r = e.response
                    print(f"\n    HTTP {r.status_code} | body[:200]: {r.text[:200]}")
                raise
            time.sleep(2 ** attempt)
    return {}


def _fetch_text(url: str) -> str:
    """Fetch a URL and return text."""
    resp = _session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _parse_time(starts_at: str) -> str:
    """Parse 'startsAt' field to 'HH:MM' string."""
    if not starts_at:
        return ""
    if "T" in starts_at:
        try:
            dt = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
            return dt.strftime("%H:%M")
        except ValueError:
            pass
    if ":" in starts_at and len(starts_at) <= 8:
        return starts_at[:5]
    try:
        return datetime.fromtimestamp(int(starts_at)).strftime("%H:%M")
    except (ValueError, OSError):
        pass
    return ""


def _diffusion_to_language(diffusion: str) -> str:
    d = diffusion.upper()
    if "ORIGINAL" in d or "VOSE" in d or "VOS" in d:
        return "VOSE"
    if "DUBBED" in d or "LOCAL" in d:
        return "Castellano"
    if "VO" in d:
        return "VO"
    return diffusion


def _version_key_to_format(version_key: str) -> str:
    key = version_key.upper()
    if "IMAX" in key:
        return "IMAX"
    if "3D" in key:
        return "3D"
    if "4DX" in key:
        return "4DX"
    if "ATMOS" in key:
        return "Atmos"
    return "2D"


def _extract_movie_info(movie: dict) -> dict:
    """Extract movie metadata from the SensaCine movie object."""
    title = movie.get("title") or movie.get("originalTitle") or "Unknown"

    # Director
    director = ""
    for credit in movie.get("credits", []):
        pos = credit.get("position", {})
        if pos.get("name") == "DIRECTOR":
            person = credit.get("person", {})
            name = f"{person.get('firstName', '')} {person.get('lastName', '')}".strip()
            if name:
                director = f"{director} | {name}" if director else name

    # Poster URL
    poster = ""
    poster_obj = movie.get("poster")
    if isinstance(poster_obj, dict):
        poster = poster_obj.get("url", "")
    elif isinstance(poster_obj, str):
        poster = poster_obj

    # Synopsis (strip HTML tags and decode entities)
    synopsis_raw = movie.get("synopsisFull") or movie.get("synopsis", "")
    synopsis = re.sub(r"<[^>]+>", "", synopsis_raw)
    synopsis = html_module.unescape(synopsis).strip()

    # Rating
    rating = None
    user_rating = (
        movie.get("statistics", {}).get("userRating")
        if isinstance(movie.get("statistics"), dict)
        else None
    )
    if user_rating is None:
        user_rating = movie.get("userRating")
    if user_rating is not None:
        try:
            rating = float(user_rating)
        except (ValueError, TypeError):
            pass

    # Genres
    genres_raw = movie.get("genres", [])
    if isinstance(genres_raw, list):
        genre_names = []
        for g in genres_raw:
            if isinstance(g, dict):
                genre_names.append(g.get("translate") or g.get("name", ""))
            elif isinstance(g, str):
                genre_names.append(g)
        genre_str = ", ".join(filter(None, genre_names))
    else:
        genre_str = str(genres_raw)

    # Runtime (seconds -> minutes)
    runtime = movie.get("runtime")
    duration = None
    if isinstance(runtime, (int, float)) and runtime > 0:
        duration = int(runtime) // 60 if runtime > 300 else int(runtime)

    return {
        "id": str(movie.get("internalId", movie.get("id", ""))),
        "title": title,
        "director": director,
        "poster": poster,
        "synopsis": synopsis,
        "rating": rating,
        "genre": genre_str,
        "duration": duration,
    }


def scrape_all_cinemas(target_date: date) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """
    Scrape all cinemas from SensaCine for the given date.

    Returns:
        (movies, showtimes, cinemas, errors)
    """
    date_str = target_date.strftime("%Y-%m-%d")
    movies_by_title: dict[str, dict] = {}  # title -> movie dict
    all_showtimes: list[dict] = []
    cinemas_seen: dict[str, dict] = {}  # external_id -> cinema dict
    errors: list[str] = []
    showtime_id = 0

    total = len(SENSACINE_THEATER_IDS)
    for idx, (theater_id, cinema_name) in enumerate(SENSACINE_THEATER_IDS.items(), 1):
        print(f"  [{idx}/{total}] {cinema_name} ({theater_id})...", end=" ", flush=True)
        try:
            page = 1
            total_pages = 1
            theater_showtimes = 0

            referer = f"{SENSACINE_BASE_URL}/cines/cine/{theater_id}/"
            while page <= total_pages:
                url = SHOWTIMES_URL.format(theater_id=theater_id, date=date_str, page=page)
                data = _fetch_json(url, referer=referer)

                pagination = data.get("pagination", {})
                total_pages = int(pagination.get("totalPages", 1))

                results = data.get("results", [])
                if not isinstance(results, list):
                    break

                for entry in results:
                    movie_data = entry.get("movie")
                    if movie_data is None:
                        continue

                    info = _extract_movie_info(movie_data)

                    # Collect movie (deduplicate by title)
                    if info["title"] not in movies_by_title:
                        movies_by_title[info["title"]] = {
                            "title": info["title"],
                            "director": info["director"],
                            "genre": info["genre"],
                            "duration_min": info["duration"],
                            "poster_url": info["poster"],
                            "rating": info["rating"],
                            "synopsis": info["synopsis"],
                            "sensacine_id": info["id"],
                        }
                    elif not movies_by_title[info["title"]].get("poster_url") and info["poster"]:
                        movies_by_title[info["title"]]["poster_url"] = info["poster"]

                    # Collect cinema
                    if theater_id not in cinemas_seen:
                        cinemas_seen[theater_id] = {
                            "external_id": theater_id,
                            "name": cinema_name,
                            "address": "",
                        }

                    # Collect showtimes
                    showtimes_dict = entry.get("showtimes", {})
                    if not isinstance(showtimes_dict, dict):
                        continue

                    seen_ids: set[int] = set()
                    for version_key, version_sessions in showtimes_dict.items():
                        if not isinstance(version_sessions, list):
                            continue
                        for session in version_sessions:
                            internal_id = session.get("internalId")
                            if internal_id is not None and internal_id in seen_ids:
                                continue
                            if internal_id is not None:
                                seen_ids.add(internal_id)

                            time_str = _parse_time(session.get("startsAt", ""))
                            if not time_str:
                                continue

                            diffusion = session.get("diffusionVersion", version_key)
                            language = _diffusion_to_language(diffusion)
                            screen_fmt = _version_key_to_format(version_key)

                            showtime_id += 1
                            all_showtimes.append({
                                "id": showtime_id,
                                "cinema": cinema_name,
                                "cinema_id": theater_id,
                                "movie": info["title"],
                                "director": info["director"],
                                "genre": info["genre"],
                                "duration_min": info["duration"],
                                "poster_url": info["poster"],
                                "rating": info["rating"],
                                "time": time_str,
                                "date": date_str,
                                "language": language,
                                "format": screen_fmt,
                            })
                            theater_showtimes += 1

                page += 1
                if page <= total_pages:
                    time.sleep(REQUEST_DELAY)

            print(f"{theater_showtimes} showtimes")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append(f"{cinema_name} ({theater_id}): {e}")

        # If the first 5 cinemas all failed, abort early (likely IP-blocked)
        if idx == 5 and len(errors) == 5:
            print("\n  First 5 cinemas all failed — SensaCine likely blocking this IP.")
            print("  Aborting early to save time.")
            break

        time.sleep(REQUEST_DELAY)

    # Build final movies list with sequential IDs
    movies_list = []
    for i, (title, mv) in enumerate(sorted(movies_by_title.items()), 1):
        movies_list.append({
            "id": i,
            "title": mv["title"],
            "director": mv["director"],
            "genre": mv["genre"],
            "duration_min": mv["duration_min"],
            "poster_url": mv["poster_url"],
            "rating": mv["rating"],
            "synopsis": mv["synopsis"],
            "sensacine_id": mv["sensacine_id"],
            "tmdb_id": None,  # will be filled later
        })

    # Build cinemas list with sequential IDs
    cinemas_list = []
    for i, (ext_id, cinema) in enumerate(sorted(cinemas_seen.items(), key=lambda x: x[1]["name"]), 1):
        cinemas_list.append({
            "id": i,
            "external_id": cinema["external_id"],
            "name": cinema["name"],
            "address": cinema["address"],
        })

    return movies_list, all_showtimes, cinemas_list, errors


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Fetch posters for movies missing them
# ═══════════════════════════════════════════════════════════════════════════

def fetch_poster_tmdb_scrape(tmdb_id: int) -> str:
    """Scrape poster path from TMDB movie page (no API key needed)."""
    url = f"https://www.themoviedb.org/movie/{tmdb_id}"
    try:
        html = _fetch_text(url)
        og = re.search(
            r'<meta\s[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
            html,
        )
        if og:
            img_url = og.group(1)
            img_url = re.sub(r"/t/p/w\d+/", "/t/p/w500/", img_url)
            return img_url
        m = re.search(r"(https://image\.tmdb\.org/t/p/\w+/\w+\.jpg)", html)
        if m:
            return re.sub(r"/t/p/w\d+/", "/t/p/w500/", m.group(1))
    except Exception as e:
        print(f"    TMDB scrape {tmdb_id}: {e}")
    return ""


def fetch_ratings(movies: list[dict]) -> None:
    """Fetch user ratings from individual SensaCine movie pages."""
    missing = [m for m in movies if not m.get("rating") and m.get("sensacine_id")]
    if not missing:
        print("  All movies already have ratings.")
        return

    print(f"  Fetching ratings for {len(missing)} movies...")
    found = 0
    for mv in missing:
        sid = mv["sensacine_id"]
        url = f"{SENSACINE_BASE_URL}/_/entities/movie/{sid}"
        try:
            data = _fetch_json(url)
            # Try multiple rating fields
            rating = None
            stats = data.get("statistics") or {}
            if isinstance(stats, dict):
                for key in ("userRating", "pressRating"):
                    val = stats.get(key)
                    if val is not None:
                        try:
                            rating = float(val)
                            break
                        except (ValueError, TypeError):
                            pass
            if rating is None:
                for key in ("userRating", "pressRating"):
                    val = data.get(key)
                    if val is not None:
                        try:
                            rating = float(val)
                            break
                        except (ValueError, TypeError):
                            pass
            if rating and rating > 0:
                mv["rating"] = rating
                found += 1
                print(f"    {mv['title']}: {rating}")
        except Exception as e:
            print(f"    {mv['title']}: error ({e})")
        time.sleep(REQUEST_DELAY)

    print(f"  Found {found}/{len(missing)} ratings.")


def fill_missing_posters(movies: list[dict], tmdb_ids: dict[str, int]) -> None:
    """Try to fill poster_url for movies that don't have one, using TMDB scraping."""
    missing = [m for m in movies if not m.get("poster_url")]
    if not missing:
        print("  All movies have posters from SensaCine.")
        return

    print(f"  {len(missing)} movies missing posters, trying TMDB scrape...")
    for mv in missing:
        tmdb_id = tmdb_ids.get(mv["title"])
        if not tmdb_id:
            continue
        url = fetch_poster_tmdb_scrape(tmdb_id)
        if url:
            mv["poster_url"] = url
            print(f"    Found poster for: {mv['title']}")
        time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Find YouTube trailers
# ═══════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Remove accents & lowercase for fuzzy matching."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _search_youtube(query: str, max_results: int = 5) -> list[dict]:
    """Search YouTube via yt-dlp and return video metadata."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                f"ytsearch{max_results}:{query}",
                "--dump-json",
                "--no-download",
                "--no-playlist",
                "--flat-playlist",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                videos.append({
                    "id": data.get("id", ""),
                    "title": data.get("title", ""),
                    "channel": data.get("channel", data.get("uploader", "")),
                    "description": data.get("description", ""),
                })
            except json.JSONDecodeError:
                continue
        return videos
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    yt-dlp error: {e}")
        return []


def _verify_video(video: dict, title: str, verify_keywords: list[str]) -> bool:
    """Check if a YouTube result actually matches the movie."""
    vtitle = _normalize(video.get("title") or "")
    vchannel = _normalize(video.get("channel") or "")
    vdesc = _normalize(video.get("description") or "")
    combined = f"{vtitle} {vchannel} {vdesc}"

    keyword_match = any(_normalize(kw) in combined for kw in verify_keywords)
    if not keyword_match:
        return False

    # Reject reactions, reviews, etc.
    reject_patterns = [
        r"\breaccion\b", r"\breaction\b", r"\branking\b", r"\btop \d+\b",
        r"\bexplicaci[oó]n\b", r"\bresumen\b", r"\bcrítica\b", r"\breview\b",
    ]
    for pat in reject_patterns:
        if re.search(pat, vtitle):
            return False

    return True


def find_trailers(movies: list[dict], existing_trailers: dict[str, str]) -> dict[str, str]:
    """Find YouTube trailers for movies that don't have one yet."""
    trailer_map = dict(existing_trailers)  # start with existing
    stopwords = {"del", "de", "la", "el", "los", "las", "un", "una", "en", "con", "por", "para", "que", "y"}

    # Check if yt-dlp is available
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  yt-dlp not found, skipping trailer search.")
        return trailer_map

    new_count = 0
    for mv in movies:
        title = mv["title"]
        if title in trailer_map:
            continue

        clean_title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
        director = mv.get("director", "")

        # Build search query
        parts = [clean_title]
        if director:
            parts.append(director)
        parts.append("trailer")
        if not re.search(r"[aeiouioun]", clean_title.lower()):
            parts.append("espanol")
        query = " ".join(parts)

        # Build verify keywords
        words = [w for w in re.findall(r"\w+", clean_title.lower()) if len(w) >= 3 and w not in stopwords]
        verify = sorted(words, key=len, reverse=True)[:3] if words else [_normalize(clean_title)]
        verify.append(_normalize(clean_title))

        print(f"    Searching trailer: {title}...", end=" ", flush=True)
        videos = _search_youtube(query)
        found = False
        for video in videos:
            if _verify_video(video, title, verify):
                trailer_map[title] = video["id"]
                print(f"found: {video['id']}")
                new_count += 1
                found = True
                break

        if not found:
            # Retry with simpler query
            alt_query = f"{clean_title} pelicula trailer oficial"
            videos2 = _search_youtube(alt_query, max_results=3)
            for video in videos2:
                if _verify_video(video, title, verify):
                    trailer_map[title] = video["id"]
                    print(f"found (retry): {video['id']}")
                    new_count += 1
                    found = True
                    break

        if not found:
            print("not found")

    print(f"  {new_count} new trailers found, {len(trailer_map)} total.")
    return trailer_map


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Update the HTML file
# ═══════════════════════════════════════════════════════════════════════════

def _load_existing_tmdb_ids() -> dict[str, int]:
    """Load existing TMDB_IDS from the HTML file."""
    html = HTML_FILE.read_text(encoding="utf-8")
    match = re.search(r"const TMDB_IDS\s*=\s*\{([^}]*)\}", html)
    if not match:
        return {}
    ids = {}
    for line in match.group(1).split("\n"):
        m = re.match(r'\s*"([^"]+)":\s*(\d+)', line)
        if m:
            ids[m.group(1)] = int(m.group(2))
    return ids


def _load_existing_trailers() -> dict[str, str]:
    """Load existing YOUTUBE_TRAILERS from the HTML file."""
    html = HTML_FILE.read_text(encoding="utf-8")
    match = re.search(r"const YOUTUBE_TRAILERS\s*=\s*\{([^}]*)\}", html)
    if not match:
        return {}
    existing = {}
    for line in match.group(1).split("\n"):
        m = re.match(r'\s*"([^"]+)":\s*"([^"]+)"', line)
        if m:
            existing[m.group(1)] = m.group(2)
    return existing


def _build_js_object(data: dict, value_type: str = "string") -> str:
    """Build a JS object literal from a dict."""
    lines = []
    for key in sorted(data.keys()):
        escaped_key = key.replace('"', '\\"')
        val = data[key]
        if value_type == "int":
            lines.append(f'  "{escaped_key}": {val}')
        else:
            escaped_val = str(val).replace('"', '\\"')
            lines.append(f'  "{escaped_key}": "{escaped_val}"')
    return "{\n" + ",\n".join(lines) + "\n}"


def update_html(
    movies: list[dict],
    showtimes: list[dict],
    cinemas: list[dict],
    tmdb_ids: dict[str, int],
    trailers: dict[str, str],
) -> None:
    """Replace the embedded data constants in the HTML file."""
    html = HTML_FILE.read_text(encoding="utf-8")

    # --- EMBEDDED_MOVIES ---
    movies_json = json.dumps(movies, ensure_ascii=False, separators=(", ", ": "))
    html = re.sub(
        r"const EMBEDDED_MOVIES\s*=\s*\[.*?\];\s*\n",
        f"const EMBEDDED_MOVIES = {movies_json};\n",
        html,
    )

    # --- EMBEDDED_SHOWTIMES ---
    showtimes_json = json.dumps(showtimes, ensure_ascii=False, separators=(", ", ": "))
    html = re.sub(
        r"const EMBEDDED_SHOWTIMES\s*=\s*\[.*?\];\s*\n",
        f"const EMBEDDED_SHOWTIMES = {showtimes_json};\n",
        html,
    )

    # --- EMBEDDED_CINEMAS ---
    cinemas_json = json.dumps(cinemas, ensure_ascii=False, separators=(", ", ": "))
    html = re.sub(
        r"const EMBEDDED_CINEMAS\s*=\s*\[.*?\];\s*\n",
        f"const EMBEDDED_CINEMAS = {cinemas_json};\n",
        html,
    )

    # --- TMDB_IDS ---
    tmdb_obj = _build_js_object(tmdb_ids, value_type="int")
    html = re.sub(
        r"const TMDB_IDS\s*=\s*\{[^}]*\};",
        f"const TMDB_IDS = {tmdb_obj};",
        html,
    )

    # --- YOUTUBE_TRAILERS ---
    trailers_obj = _build_js_object(trailers, value_type="string")
    html = re.sub(
        r"const YOUTUBE_TRAILERS\s*=\s*\{[^}]*\};",
        f"const YOUTUBE_TRAILERS = {trailers_obj};",
        html,
    )

    # --- Update datePicker default value ---
    if showtimes:
        # Use the date from the first showtime
        data_date = showtimes[0].get("date", "")
        if data_date:
            html = re.sub(
                r'(<input type="date" id="datePicker" value=")[^"]*(")',
                rf'\g<1>{data_date}\2',
                html,
            )

    HTML_FILE.write_text(html, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    today = date.today()
    print("=" * 65)
    print(f"  Cartelera Madrid — Build Pipeline")
    print(f"  Date: {today.strftime('%Y-%m-%d')}")
    print(f"  Cinemas: {len(SENSACINE_THEATER_IDS)}")
    print("=" * 65)

    if not HTML_FILE.exists():
        print(f"ERROR: HTML file not found: {HTML_FILE}")
        return 1

    # Quick connectivity test
    print("\nTesting SensaCine connectivity...")
    test_id = list(SENSACINE_THEATER_IDS.keys())[0]
    test_url = SHOWTIMES_URL.format(theater_id=test_id, date=today.strftime("%Y-%m-%d"), page=1)
    try:
        resp = _session.get(test_url, timeout=REQUEST_TIMEOUT)
        ct = resp.headers.get("content-type", "N/A")
        print(f"  Status: {resp.status_code} | Content-Type: {ct}")
        print(f"  Response body[:500]: {resp.text[:500]}")
        if resp.status_code != 200 or "json" not in ct.lower():
            print("\n  SensaCine is NOT returning JSON. Likely blocking or captcha.")
            print("  The pipeline cannot update data in this environment.")
            return 1
        # Verify it actually parses as JSON
        try:
            data = resp.json()
            results = data.get("results", [])
            print(f"  Test OK: got {len(results)} movies from test cinema")
        except Exception:
            print("  Response is not valid JSON despite Content-Type header.")
            return 1
    except Exception as e:
        print(f"  Connection failed: {e}")
        return 1

    # Load existing data from HTML (for merging)
    existing_tmdb_ids = _load_existing_tmdb_ids()
    existing_trailers = _load_existing_trailers()

    # --- Step 1: Scrape ---
    print(f"\n[1/4] Scraping SensaCine for {today}...")
    movies, showtimes, cinemas, errors = scrape_all_cinemas(today)
    print(f"\n  Result: {len(movies)} movies, {len(showtimes)} showtimes, {len(cinemas)} cinemas")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    if not showtimes:
        print("\nERROR: No showtimes scraped. Aborting to preserve existing data.")
        # Write debug info
        debug = {"movies": len(movies), "showtimes": len(showtimes), "cinemas": len(cinemas), "errors": errors[:20]}
        (PROJECT_ROOT / "sensacine_test.json").write_text(json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1

    # --- Step 2: Fetch ratings ---
    print("\n[2/5] Fetching ratings from SensaCine...")
    fetch_ratings(movies)

    # --- Step 3: Merge TMDB IDs ---
    print("\n[3/5] Merging TMDB IDs...")
    # Keep existing TMDB IDs for movies that are still showing
    current_titles = {m["title"] for m in movies}
    tmdb_ids: dict[str, int] = {}
    for title, tid in existing_tmdb_ids.items():
        if title in current_titles:
            tmdb_ids[title] = tid
    # Set tmdb_id field on movie objects
    for mv in movies:
        tid = tmdb_ids.get(mv["title"])
        if tid:
            mv["tmdb_id"] = tid
    print(f"  {len(tmdb_ids)} TMDB IDs carried over for current movies.")

    # --- Step 4: Fill missing posters ---
    print("\n[4/5] Checking posters...")
    fill_missing_posters(movies, tmdb_ids)

    # --- Step 5: Find trailers ---
    print("\n[5/5] Finding YouTube trailers...")
    # Only keep existing trailers for movies still showing
    relevant_trailers = {t: v for t, v in existing_trailers.items() if t in current_titles}
    trailers = find_trailers(movies, relevant_trailers)

    # --- Write HTML ---
    print("\nWriting updated HTML...")
    update_html(movies, showtimes, cinemas, tmdb_ids, trailers)

    print(f"\nDone! Updated {HTML_FILE.name}")
    print(f"  Movies: {len(movies)}")
    print(f"  Showtimes: {len(showtimes)}")
    print(f"  Cinemas: {len(cinemas)}")
    print(f"  Trailers: {len(trailers)}")
    print(f"  TMDB IDs: {len(tmdb_ids)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        # Write traceback to file for debugging in CI
        debug_file = PROJECT_ROOT / "sensacine_test.json"
        debug_file.write_text(json.dumps({"crash": True, "traceback": tb}, indent=2), encoding="utf-8")
        sys.exit(1)
