"""Test the SensaCine parser against a saved fixture.

Usage:
1. First run: python tests/probe_sensacine_api.py  (saves real JSON to fixtures/)
2. Then run:  python tests/test_parser_with_fixture.py

Or run with the built-in mock fixture (no network needed):
    python tests/test_parser_with_fixture.py --mock
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from guiamadrid.scrapers.cine.sensacine import SensaCineScraper

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Mock fixture based on typical AlloCiné/SensaCine API structure
MOCK_FIXTURE = {
    "results": [
        {
            "movie": {
                "id": 301843,
                "title": "Capitán América: Brave New World",
                "originalTitle": "Captain America: Brave New World",
                "directors": [{"name": "Julius Onah"}],
                "castingShort": {"directors": "Julius Onah"},
                "genre": [{"name": "Acción"}, {"name": "Aventura"}],
                "runtime": "1h 58min",
                "poster": {"url": "https://fr.web.img6.acsta.net/pictures/24/01/cap4.jpg"},
                "synopsis": "Sam Wilson asume el manto del Capitán América...",
                "userRating": 3.2,
                "pressRating": 2.8,
            },
            "showtimes": {
                "dubbed": [
                    {"time": "14:30"},
                    {"time": "17:15"},
                    {"time": "20:00"},
                    {"time": "22:30"},
                ],
                "original": [
                    {"time": "16:00"},
                    {"time": "21:45"},
                ],
            },
        },
        {
            "movie": {
                "id": 295937,
                "title": "Dune: Parte Dos",
                "originalTitle": "Dune: Part Two",
                "directors": [{"name": "Denis Villeneuve"}],
                "genre": [{"name": "Ciencia ficción"}, {"name": "Drama"}],
                "runtime": "2h 46min",
                "poster": {"url": "https://fr.web.img6.acsta.net/pictures/24/dune2.jpg"},
                "synopsis": "Paul Atreides se une a los Fremen...",
                "userRating": 4.5,
            },
            "showtimes": {
                "original": [
                    {"time": "15:00"},
                    {"time": "19:30"},
                ],
            },
        },
        {
            "movie": {
                "id": 312456,
                "title": "Anora",
                "directors": [{"name": "Sean Baker"}],
                "genre": [{"name": "Drama"}, {"name": "Comedia"}],
                "runtime": "2h 19min",
                "poster": {"url": "https://fr.web.img6.acsta.net/pictures/24/anora.jpg"},
                "synopsis": "Una joven stripper de Brooklyn se casa impulsivamente...",
                "userRating": 4.1,
            },
            "showtimes": {
                "VOSE": [
                    {"time": "17:30"},
                    {"time": "20:15"},
                ],
            },
        },
    ]
}

# Alternative structure (flat list)
MOCK_FIXTURE_ALT = {
    "movies": [
        {
            "title": "Flow",
            "id": "320001",
            "directors": "Gints Zilbalodis",
            "genre": "Animación",
            "runtime": 85,
            "synopsis": "Un gato se embarca en un viaje...",
            "poster": "https://example.com/flow.jpg",
            "userRating": 4.3,
            "showtimes": [
                {"time": "12:00", "language": "VOSE", "format": "2D"},
                {"time": "16:30", "language": "Castellano", "format": "2D"},
            ],
        }
    ]
}


def test_with_fixture(data: dict, label: str):
    print(f"\n{'='*60}")
    print(f"Testing parser with: {label}")
    print(f"{'='*60}")

    scraper = SensaCineScraper()
    showtimes = scraper._parse_response(data, "C0094", "Yelmo Ideal", "2026-03-20")

    print(f"Parsed {len(showtimes)} showtimes")
    for st in showtimes:
        lang = f" ({st.language})" if st.language else ""
        fmt = f" [{st.format}]" if st.format != "2D" else ""
        print(f"  {st.movie_title} | {st.showtime}{lang}{fmt}")
        if st.director:
            print(f"    Dir: {st.director}")

    if not showtimes:
        print("  WARNING: No showtimes parsed!")
        return False

    # Validate fields
    for st in showtimes:
        assert st.cinema_name == "Yelmo Ideal", f"Bad cinema: {st.cinema_name}"
        assert st.cinema_id == "C0094", f"Bad cinema_id: {st.cinema_id}"
        assert st.movie_title, "Empty movie title"
        assert ":" in st.showtime, f"Bad time format: {st.showtime}"
        assert st.date == "2026-03-20", f"Bad date: {st.date}"

    print(f"  ALL {len(showtimes)} showtimes validated OK")
    scraper.close()
    return True


def test_with_real_fixture():
    fixture_path = FIXTURES_DIR / "sensacine_sample.json"
    if not fixture_path.exists():
        print(f"No real fixture at {fixture_path}")
        print("Run: python tests/probe_sensacine_api.py first")
        return False

    with open(fixture_path) as f:
        data = json.load(f)

    return test_with_fixture(data, f"Real API response ({fixture_path.name})")


def main():
    use_mock = "--mock" in sys.argv

    passed = 0
    failed = 0

    # Test mock fixtures
    if test_with_fixture(MOCK_FIXTURE, "Mock fixture (results/movie/showtimes by version)"):
        passed += 1
    else:
        failed += 1

    if test_with_fixture(MOCK_FIXTURE_ALT, "Mock fixture alt (movies list with inline showtimes)"):
        passed += 1
    else:
        failed += 1

    # Test real fixture (if available)
    if not use_mock:
        fixture_path = FIXTURES_DIR / "sensacine_sample.json"
        if fixture_path.exists():
            if test_with_real_fixture():
                passed += 1
            else:
                failed += 1
        else:
            print(f"\nSkipping real fixture test (run probe_sensacine_api.py first)")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
