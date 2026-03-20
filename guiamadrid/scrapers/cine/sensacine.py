"""SensaCine scraper — uses internal JSON API for showtimes."""

from datetime import date, datetime

from guiamadrid.config import SENSACINE_SHOWTIMES_URL, SENSACINE_THEATER_IDS
from guiamadrid.scrapers.base import BaseScraper, ScrapeResult, Showtime


class SensaCineScraper(BaseScraper):
    """Scrapes movie showtimes from SensaCine's internal API.

    API endpoint: sensacine.com/_/showtimes/theater-{ID}/d-{fecha}/
    Returns JSON with movie + showtime data per theater.
    """

    def scrape(self, target_date: date | None = None) -> ScrapeResult:
        target_date = target_date or date.today()
        date_str = target_date.strftime("%Y-%m-%d")

        all_showtimes: list[Showtime] = []
        seen_movies: set[str] = set()
        errors: list[str] = []

        for theater_id, cinema_name in SENSACINE_THEATER_IDS.items():
            try:
                showtimes = self._scrape_theater(
                    theater_id, cinema_name, date_str
                )
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
        """Scrape all showtimes for a single theater on a date."""
        url = SENSACINE_SHOWTIMES_URL.format(
            theater_id=theater_id, date=date_str
        )
        data = self._get_json(url)
        return self._parse_response(data, theater_id, cinema_name, date_str)

    def _parse_response(
        self,
        data: dict,
        theater_id: str,
        cinema_name: str,
        date_str: str,
    ) -> list[Showtime]:
        """Parse the SensaCine JSON response into Showtime objects."""
        showtimes: list[Showtime] = []

        # The API returns a "results" key with movie entries
        results = data.get("results", data.get("movies", []))
        if isinstance(results, dict):
            results = results.get("movies", list(results.values()))
        if not isinstance(results, list):
            results = []

        for movie_data in results:
            movie_info = self._extract_movie_info(movie_data)
            sessions = self._extract_sessions(movie_data)

            for session in sessions:
                showtimes.append(
                    Showtime(
                        cinema_name=cinema_name,
                        cinema_id=theater_id,
                        movie_title=movie_info["title"],
                        movie_id=movie_info.get("id"),
                        showtime=session.get("time", ""),
                        date=date_str,
                        language=session.get("language", ""),
                        format=session.get("format", "2D"),
                        director=movie_info.get("director", ""),
                        poster_url=movie_info.get("poster", ""),
                        synopsis=movie_info.get("synopsis", ""),
                        rating=movie_info.get("rating"),
                        genre=movie_info.get("genre", ""),
                        duration_min=movie_info.get("duration"),
                    )
                )

        return showtimes

    def _extract_movie_info(self, movie_data: dict) -> dict:
        """Extract movie metadata from API response."""
        # Handle different possible JSON structures
        movie = movie_data.get("movie", movie_data)

        title = (
            movie.get("title")
            or movie.get("name")
            or movie.get("originalTitle", "Unknown")
        )
        rating_raw = movie.get("userRating") or movie.get("pressRating")
        rating = None
        if rating_raw is not None:
            try:
                rating = float(rating_raw)
            except (ValueError, TypeError):
                pass

        duration = movie.get("runtime") or movie.get("duration")
        if duration and isinstance(duration, str):
            # Parse "2h 15min" format
            parts = duration.replace("h", "").replace("min", "").split()
            try:
                duration = int(parts[0]) * 60 + (int(parts[1]) if len(parts) > 1 else 0)
            except (ValueError, IndexError):
                duration = None

        genres = movie.get("genre") or movie.get("genres", [])
        if isinstance(genres, list):
            genres = ", ".join(
                g.get("name", g) if isinstance(g, dict) else str(g)
                for g in genres
            )

        return {
            "id": str(movie.get("id", "")),
            "title": title,
            "director": self._get_director(movie),
            "poster": movie.get("poster", {}).get("url", "") if isinstance(movie.get("poster"), dict) else movie.get("poster", ""),
            "synopsis": movie.get("synopsis", ""),
            "rating": rating,
            "genre": genres if isinstance(genres, str) else "",
            "duration": duration if isinstance(duration, int) else None,
        }

    @staticmethod
    def _get_director(movie: dict) -> str:
        """Extract director name from movie data."""
        directors = movie.get("directors", movie.get("castingShort", {}).get("directors", ""))
        if isinstance(directors, list):
            return ", ".join(
                d.get("name", "") if isinstance(d, dict) else str(d)
                for d in directors
            )
        return str(directors) if directors else ""

    @staticmethod
    def _extract_sessions(movie_data: dict) -> list[dict]:
        """Extract individual session times from movie data."""
        sessions = []
        # Try different possible structures
        showtimes_data = (
            movie_data.get("showtimes")
            or movie_data.get("sessions")
            or movie_data.get("shows", [])
        )

        if isinstance(showtimes_data, dict):
            # Grouped by version/language
            for version_key, version_sessions in showtimes_data.items():
                lang = _parse_language(version_key)
                fmt = _parse_format(version_key)
                if isinstance(version_sessions, list):
                    for s in version_sessions:
                        time_str = s.get("time") or s.get("$time") or s.get("t", "")
                        if isinstance(time_str, str) and ":" not in time_str:
                            # Try parsing epoch
                            try:
                                time_str = datetime.fromtimestamp(int(time_str)).strftime("%H:%M")
                            except (ValueError, OSError):
                                pass
                        sessions.append({
                            "time": time_str,
                            "language": lang,
                            "format": fmt,
                        })
        elif isinstance(showtimes_data, list):
            for s in showtimes_data:
                if isinstance(s, dict):
                    sessions.append({
                        "time": s.get("time", s.get("$time", "")),
                        "language": s.get("language", s.get("version", "")),
                        "format": s.get("format", s.get("screen", "2D")),
                    })
                elif isinstance(s, str):
                    sessions.append({"time": s, "language": "", "format": "2D"})

        return sessions


def _parse_language(version_key: str) -> str:
    """Parse language from version key like 'VOSE', 'local-dubbed'."""
    key = version_key.upper()
    if "VOSE" in key or "VOS" in key:
        return "VOSE"
    if "VO" in key:
        return "VO"
    if "DUB" in key or "LOCAL" in key or "CAST" in key:
        return "Castellano"
    return version_key


def _parse_format(version_key: str) -> str:
    """Parse format from version key."""
    key = version_key.upper()
    if "IMAX" in key:
        return "IMAX"
    if "3D" in key:
        return "3D"
    if "4DX" in key:
        return "4DX"
    return "2D"
