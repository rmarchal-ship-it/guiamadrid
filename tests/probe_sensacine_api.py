"""Test script to probe the real SensaCine API and discover response structure.

Run this on your local machine (not in the cloud sandbox):
    python tests/probe_sensacine_api.py

This will:
1. Fetch ONE theater's showtimes for today
2. Print the raw JSON structure
3. Save it to tests/fixtures/sensacine_sample.json
"""

import json
import sys
from datetime import date
from pathlib import Path

import httpx

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

# Yelmo Ideal — central Madrid, always has lots of showtimes
THEATER_ID = "C0094"
DATE_STR = date.today().strftime("%Y-%m-%d")
URL = f"https://www.sensacine.com/_/showtimes/theater-{THEATER_ID}/d-{DATE_STR}/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": f"https://www.sensacine.com/cines/cine-{THEATER_ID}/",
}


def main():
    print(f"Fetching: {URL}")
    print()

    try:
        resp = httpx.get(URL, headers=HEADERS, timeout=15, follow_redirects=True)
    except httpx.HTTPError as e:
        print(f"HTTP error: {e}")
        sys.exit(1)

    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('content-type', '?')}")
    print(f"Body length: {len(resp.text)} chars")
    print()

    # Try to parse as JSON
    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("Response is NOT JSON. First 2000 chars:")
        print(resp.text[:2000])
        # Save raw response
        out_file = FIXTURES_DIR / "sensacine_raw_response.txt"
        out_file.write_text(resp.text[:10000])
        print(f"\nSaved raw response to {out_file}")
        sys.exit(1)

    # Pretty print structure
    print("=== TOP-LEVEL KEYS ===")
    if isinstance(data, dict):
        for key, value in data.items():
            vtype = type(value).__name__
            if isinstance(value, list):
                print(f"  {key}: list[{len(value)}]")
                if value:
                    print(f"    first item type: {type(value[0]).__name__}")
                    if isinstance(value[0], dict):
                        print(f"    first item keys: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())[:10]}")
            else:
                print(f"  {key}: {vtype} = {str(value)[:100]}")
    elif isinstance(data, list):
        print(f"  Root is list[{len(data)}]")
        if data and isinstance(data[0], dict):
            print(f"  First item keys: {list(data[0].keys())}")

    print()

    # Deep print first movie entry
    print("=== FIRST MOVIE ENTRY (if available) ===")
    movies = None
    if isinstance(data, dict):
        for key in ["results", "movies", "showtimes", "data"]:
            if key in data:
                val = data[key]
                if isinstance(val, list) and val:
                    movies = val
                    print(f"Found movies under key '{key}'")
                    break
                elif isinstance(val, dict):
                    # Maybe movies are nested
                    for subkey, subval in val.items():
                        if isinstance(subval, list) and subval:
                            movies = subval
                            print(f"Found movies under key '{key}.{subkey}'")
                            break
                    if movies:
                        break

    if movies:
        print(json.dumps(movies[0], indent=2, ensure_ascii=False)[:3000])
    else:
        # Just print first 3000 chars of the whole response
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])

    # Save full response
    out_file = FIXTURES_DIR / "sensacine_sample.json"
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\nFull response saved to {out_file}")
    print(f"Now run: python tests/test_parser_with_fixture.py")


if __name__ == "__main__":
    main()
