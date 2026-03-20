"""Test the SensaCine parser against fixtures matching the REAL API structure.

The real API (confirmed via allocine-seances package) returns:
    {pagination: {page, totalPages}, results: [{movie: {...}, showtimes: {version: [{startsAt, diffusionVersion, internalId}]}}]}

Usage:
    python tests/test_parser_with_fixture.py          # mock + real fixture if available
    python tests/test_parser_with_fixture.py --mock    # mock only
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from guiamadrid.scrapers.cine.sensacine import SensaCineScraper

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Real API structure (confirmed from allocine-seances source code)
MOCK_REAL_STRUCTURE = {
    "pagination": {"page": 1, "totalPages": 1},
    "results": [
        {
            "movie": {
                "internalId": 301843,
                "title": "Capitán América: Brave New World",
                "originalTitle": "Captain America: Brave New World",
                "credits": [
                    {
                        "position": {"name": "DIRECTOR"},
                        "person": {"firstName": "Julius", "lastName": "Onah"},
                    },
                ],
                "genres": [
                    {"translate": "Acción"},
                    {"translate": "Aventura"},
                ],
                "runtime": 7080,  # seconds → 118 min
                "poster": {"url": "https://fr.web.img6.acsta.net/pictures/24/01/cap4.jpg"},
                "synopsisFull": "Sam Wilson asume el manto del Capitán América y se enfrenta a un complot internacional.",
                "statistics": {"userRating": 3.2},
                "releases": [{"name": "En cines", "releaseDate": {"date": "2025-02-14"}}],
                "languages": [{"name": "Español"}],
                "flags": {"hasDvdRelease": False},
                "customFlags": {"isPremiere": False, "weeklyOuting": False},
            },
            "showtimes": {
                "dubbed": [
                    {"internalId": 1001, "startsAt": "2026-03-20T14:30:00", "diffusionVersion": "DUBBED"},
                    {"internalId": 1002, "startsAt": "2026-03-20T17:15:00", "diffusionVersion": "DUBBED"},
                    {"internalId": 1003, "startsAt": "2026-03-20T20:00:00", "diffusionVersion": "DUBBED"},
                    {"internalId": 1004, "startsAt": "2026-03-20T22:30:00", "diffusionVersion": "DUBBED"},
                ],
                "original": [
                    {"internalId": 2001, "startsAt": "2026-03-20T16:00:00", "diffusionVersion": "ORIGINAL"},
                    {"internalId": 2002, "startsAt": "2026-03-20T21:45:00", "diffusionVersion": "ORIGINAL"},
                ],
            },
        },
        {
            "movie": {
                "internalId": 295937,
                "title": "Dune: Parte Dos",
                "originalTitle": "Dune: Part Two",
                "credits": [
                    {
                        "position": {"name": "DIRECTOR"},
                        "person": {"firstName": "Denis", "lastName": "Villeneuve"},
                    },
                ],
                "genres": [
                    {"translate": "Ciencia ficción"},
                    {"translate": "Drama"},
                ],
                "runtime": 9960,  # 166 min
                "poster": {"url": "https://fr.web.img6.acsta.net/pictures/24/dune2.jpg"},
                "synopsisFull": "Paul Atreides se une a los Fremen en una guerra contra los Harkonnen.",
                "statistics": {"userRating": 4.5},
                "releases": [],
                "languages": [],
                "flags": {},
                "customFlags": {},
            },
            "showtimes": {
                "original": [
                    {"internalId": 3001, "startsAt": "2026-03-20T15:00:00", "diffusionVersion": "ORIGINAL"},
                    {"internalId": 3002, "startsAt": "2026-03-20T19:30:00", "diffusionVersion": "ORIGINAL"},
                ],
                "imax": [
                    {"internalId": 4001, "startsAt": "2026-03-20T18:00:00", "diffusionVersion": "ORIGINAL"},
                ],
            },
        },
        {
            "movie": {
                "internalId": 312456,
                "title": "Anora",
                "credits": [
                    {
                        "position": {"name": "DIRECTOR"},
                        "person": {"firstName": "Sean", "lastName": "Baker"},
                    },
                ],
                "genres": [
                    {"translate": "Drama"},
                    {"translate": "Comedia"},
                ],
                "runtime": 8340,  # 139 min
                "poster": {"url": "https://fr.web.img6.acsta.net/pictures/24/anora.jpg"},
                "synopsisFull": "Una joven stripper de Brooklyn se casa impulsivamente con el hijo de un oligarca ruso.",
                "statistics": {"userRating": 4.1},
                "releases": [],
                "languages": [],
                "flags": {},
                "customFlags": {},
            },
            "showtimes": {
                "original": [
                    {"internalId": 5001, "startsAt": "2026-03-20T17:30:00", "diffusionVersion": "ORIGINAL"},
                    {"internalId": 5002, "startsAt": "2026-03-20T20:15:00", "diffusionVersion": "ORIGINAL"},
                ],
            },
        },
    ],
}

# Test with duplicate internalIds (should be deduplicated)
MOCK_WITH_DUPLICATES = {
    "pagination": {"page": 1, "totalPages": 1},
    "results": [
        {
            "movie": {
                "internalId": 999,
                "title": "Test Movie",
                "credits": [],
                "genres": [],
                "runtime": 5400,
            },
            "showtimes": {
                "dubbed": [
                    {"internalId": 7001, "startsAt": "2026-03-20T14:00:00", "diffusionVersion": "DUBBED"},
                    {"internalId": 7001, "startsAt": "2026-03-20T14:00:00", "diffusionVersion": "DUBBED"},  # duplicate
                ],
                "original": [
                    {"internalId": 7001, "startsAt": "2026-03-20T14:00:00", "diffusionVersion": "ORIGINAL"},  # same id, diff version
                    {"internalId": 7002, "startsAt": "2026-03-20T18:00:00", "diffusionVersion": "ORIGINAL"},
                ],
            },
        }
    ],
}


def test_with_fixture(data: dict, label: str, expected_count: int | None = None):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")

    scraper = SensaCineScraper()
    showtimes = scraper._parse_response(data, "E0621", "Yelmo Ideal", "2026-03-20")

    print(f"Parsed {len(showtimes)} showtimes")
    for st in showtimes:
        lang = f" ({st.language})" if st.language else ""
        fmt = f" [{st.format}]" if st.format != "2D" else ""
        dur = f" {st.duration_min}min" if st.duration_min else ""
        print(f"  {st.movie_title} | {st.showtime}{lang}{fmt}{dur}")
        if st.director:
            print(f"    Dir: {st.director} | {st.genre}")

    if not showtimes and expected_count != 0:
        print("  WARNING: No showtimes parsed!")
        return False

    # Validate fields
    for st in showtimes:
        assert st.cinema_name == "Yelmo Ideal", f"Bad cinema: {st.cinema_name}"
        assert st.cinema_id == "E0621", f"Bad cinema_id: {st.cinema_id}"
        assert st.movie_title, "Empty movie title"
        assert ":" in st.showtime, f"Bad time format: {st.showtime}"
        assert st.date == "2026-03-20", f"Bad date: {st.date}"

    if expected_count is not None:
        assert len(showtimes) == expected_count, f"Expected {expected_count}, got {len(showtimes)}"

    print(f"  ALL {len(showtimes)} showtimes validated OK")
    scraper.close()
    return True


def test_with_real_fixture():
    fixture_path = FIXTURES_DIR / "sensacine_sample.json"
    if not fixture_path.exists():
        print(f"\nNo real fixture at {fixture_path}")
        print("Run: python tests/probe_sensacine_api.py first")
        return None  # skip, not failure

    with open(fixture_path) as f:
        data = json.load(f)

    return test_with_fixture(data, f"Real API response ({fixture_path.name})")


def main():
    use_mock = "--mock" in sys.argv
    passed = 0
    failed = 0

    # Test 1: Real API structure
    if test_with_fixture(MOCK_REAL_STRUCTURE, "Real API structure (3 movies, 11 showtimes)", 11):
        passed += 1
    else:
        failed += 1

    # Test 2: Deduplication by internalId
    if test_with_fixture(MOCK_WITH_DUPLICATES, "Dedup by internalId (4 raw → 2 unique)", 2):
        passed += 1
    else:
        failed += 1

    # Test 3: Field validation
    print(f"\n{'='*60}")
    print("Testing: Field extraction details")
    print(f"{'='*60}")
    scraper = SensaCineScraper()
    sts = scraper._parse_response(MOCK_REAL_STRUCTURE, "E0621", "Yelmo Ideal", "2026-03-20")

    cap = [s for s in sts if "Capitán" in s.movie_title]
    assert len(cap) == 6, f"Expected 6 Cap América showtimes, got {len(cap)}"
    assert cap[0].director == "Julius Onah", f"Bad director: {cap[0].director}"
    assert cap[0].duration_min == 118, f"Bad duration: {cap[0].duration_min}"
    assert cap[0].genre == "Acción, Aventura", f"Bad genre: {cap[0].genre}"
    assert cap[0].rating == 3.2, f"Bad rating: {cap[0].rating}"
    assert "acsta.net" in cap[0].poster_url, f"Bad poster: {cap[0].poster_url}"
    dubbed = [s for s in cap if s.language == "Castellano"]
    vose = [s for s in cap if s.language == "VOSE"]
    assert len(dubbed) == 4, f"Expected 4 dubbed, got {len(dubbed)}"
    assert len(vose) == 2, f"Expected 2 VOSE, got {len(vose)}"

    dune = [s for s in sts if "Dune" in s.movie_title]
    imax = [s for s in dune if s.format == "IMAX"]
    assert len(imax) == 1, f"Expected 1 IMAX, got {len(imax)}"
    assert dune[0].duration_min == 166, f"Bad Dune duration: {dune[0].duration_min}"

    print("  All field assertions passed")
    passed += 1
    scraper.close()

    # Test 4: Real fixture if available
    if not use_mock:
        result = test_with_real_fixture()
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        # None = skipped

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
