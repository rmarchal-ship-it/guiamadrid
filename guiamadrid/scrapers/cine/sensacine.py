"""SensaCine scraper — uses internal JSON API for showtimes.

SensaCine (sensacine.com) and AlloCiné (allocine.fr) share the same
backend API. The endpoint is:

    https://www.sensacine.com/_/showtimes/theater-{ID}/d-{YYYY-MM-DD}/p-{page}

Response JSON structure (confirmed via allocine-seances package):
    {
      "pagination": {"page": 1, "totalPages": 2},
      "results": [
        {
          "movie": {
            "internalId": 301843,
            "title": "...",
            "originalTitle": "...",
            "credits": [{"position": {"name": "DIRECTOR"}, "person": {"firstName": "...", "lastName": "..."}}],
            "genres": [{"translate": "Acción"}],
            "runtime": 7080,          // seconds
            "poster": {"url": "..."},
            "synopsisFull": "...",
            "releases": [...],
            "languages": [...],
            "flags": {"hasDvdRelease": false},
            "customFlags": {"isPremiere": false, "weeklyOuting": false}
          },
          "showtimes": {
            "dubbed": [
              {"internalId": 123, "startsAt": "2026-03-20T14:30:00", "diffusionVersion": "DUBBED"},
              ...
            ],
            "original": [
              {"internalId": 456, "startsAt": "2026-03-20T16:00:00", "diffusionVersion": "ORIGINAL"},
              ...
            ]
          }
        }
      ]
    }
"""

from datetime import date, datetime

from guiamadrid.config import SENSACINE_BASE_URL, SENSACINE_THEATER_IDS
from guiamadrid.scrapers.base import BaseScraper, ScrapeResult, Showtime

# Showtimes URL with pagination support
_SHOWTIMES_URL = SENSACINE_BASE_URL + "/_/showtimes/theater-{theater_id}/d-{date}/p-{page}"


class SensaCineScraper(BaseScraper):
    """Scrapes movie showtimes from SensaCine's internal API."""

    def scrape(self, target_date: date | None = None) -> ScrapeResult:
        target_date = target_date or date.today()
        date_str = target_date.strftime("%Y-%m-%d")

        all_showtimes: list[Showtime] = []
        seen_movies: set[str] = set()
        errors: list[str] = []

        for theater_id, cinema_name in SENSACINE_THEATER_IDS.items():
            try:
                showtimes = self._scrape_theater(theater_id, cinema_name, date_str)
                for st in showtimes:
                    seen_movies.add(st.movie_title)
                all_showtimes.extend(showtimes)
            except Exception as e:
                errors.append(f"{cinema_name} ({theater_id}): {e}")

        return ScrapeResult(
            showtimes=all_showtimes,
            cinemas_count=len(SENSACINE_THEATER_IDS),
            movies_count=len(seen_movies),
            errors=errors,
        )

    def _scrape_theater(
        self, theater_id: str, cinema_name: str, date_str: str
    ) -> list[Showtime]:
        """Scrape all pages of showtimes for a single theater on a date."""
        all_showtimes: list[Showtime] = []
        page = 1
        total_pages = 1

        while page <= total_pages:
            url = _SHOWTIMES_URL.format(
                theater_id=theater_id, date=date_str, page=page
            )
            data = self._get_json(url)

            # Update pagination
            pagination = data.get("pagination", {})
            total_pages = int(pagination.get("totalPages", 1))

            showtimes = self._parse_response(data, theater_id, cinema_name, date_str)
            all_showtimes.extend(showtimes)
            page += 1

        return all_showtimes

    def _parse_response(
        self,
        data: dict,
        theater_id: str,
        cinema_name: str,
        date_str: str,
    ) -> list[Showtime]:
        """Parse the SensaCine JSON response into Showtime objects."""
        showtimes: list[Showtime] = []
        results = data.get("results", [])
        if not isinstance(results, list):
            return showtimes

        for entry in results:
            movie_data = entry.get("movie")
            if movie_data is None:
                continue

            movie_info = self._extract_movie_info(movie_data)
            seen_ids: set[int] = set()

            showtimes_dict = entry.get("showtimes", {})
            if not isinstance(showtimes_dict, dict):
                continue

            for version_key, version_sessions in showtimes_dict.items():
                if not isinstance(version_sessions, list):
                    continue

                for session in version_sessions:
                    # Deduplicate by internalId
                    internal_id = session.get("internalId")
                    if internal_id is not None and internal_id in seen_ids:
                        continue
                    if internal_id is not None:
                        seen_ids.add(internal_id)

                    time_str = self._parse_time(session.get("startsAt", ""))
                    if not time_str:
                        continue

                    diffusion = session.get("diffusionVersion", version_key)
                    language = _diffusion_to_language(diffusion)
                    screen_fmt = _version_key_to_format(version_key)

                    showtimes.append(
                        Showtime(
                            cinema_name=cinema_name,
                            cinema_id=theater_id,
                            movie_title=movie_info["title"],
                            movie_id=movie_info["id"],
                            showtime=time_str,
                            date=date_str,
                            language=language,
                            format=screen_fmt,
                            director=movie_info["director"],
                            poster_url=movie_info["poster"],
                            synopsis=movie_info["synopsis"],
                            rating=movie_info["rating"],
                            genre=movie_info["genre"],
                            duration_min=movie_info["duration"],
                        )
                    )

        return showtimes

    @staticmethod
    def _extract_movie_info(movie: dict) -> dict:
        """Extract movie metadata from the movie object."""
        title = movie.get("title") or movie.get("originalTitle") or "Unknown"

        # Director from credits array
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

        # Synopsis
        synopsis = movie.get("synopsisFull") or movie.get("synopsis", "")

        # Rating — userRating is a dict with score, or a float
        rating = None
        user_rating = movie.get("statistics", {}).get("userRating") if isinstance(movie.get("statistics"), dict) else None
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

        # Runtime in seconds → minutes
        runtime = movie.get("runtime")
        duration = None
        if isinstance(runtime, (int, float)) and runtime > 0:
            duration = int(runtime) // 60 if runtime > 300 else int(runtime)  # >300 = seconds, else already minutes

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

    @staticmethod
    def _parse_time(starts_at: str) -> str:
        """Parse 'startsAt' field → 'HH:MM' string.

        Handles: '2026-03-20T14:30:00', '14:30', epoch timestamps.
        """
        if not starts_at:
            return ""

        # ISO datetime: "2026-03-20T14:30:00"
        if "T" in starts_at:
            try:
                dt = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
                return dt.strftime("%H:%M")
            except ValueError:
                pass

        # Already "HH:MM"
        if ":" in starts_at and len(starts_at) <= 8:
            return starts_at[:5]

        # Epoch timestamp
        try:
            return datetime.fromtimestamp(int(starts_at)).strftime("%H:%M")
        except (ValueError, OSError):
            pass

        return ""


def _diffusion_to_language(diffusion: str) -> str:
    """Map diffusionVersion to readable language label."""
    d = diffusion.upper()
    if "ORIGINAL" in d or "VOSE" in d or "VOS" in d:
        return "VOSE"
    if "DUBBED" in d or "LOCAL" in d:
        return "Castellano"
    if "VO" in d:
        return "VO"
    return diffusion


def _version_key_to_format(version_key: str) -> str:
    """Infer screen format from the version key."""
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
