"""Test script to probe the real SensaCine API and discover response structure.

Run this on your local machine (not in the cloud sandbox):
    python tests/probe_sensacine_api.py

Or paste this one-liner in a browser console (DevTools → Console) while on sensacine.com:
    fetch('/_/showtimes/theater-E0621/d-2026-03-20/p-1').then(r=>r.json()).then(d=>console.log(JSON.stringify(d,null,2)))

This will:
1. Fetch ONE theater's showtimes for today (with pagination)
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
THEATER_ID = "E0621"
DATE_STR = date.today().strftime("%Y-%m-%d")
BASE_URL = "https://www.sensacine.com/_/showtimes/theater-{id}/d-{date}/p-{page}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": f"https://www.sensacine.com/cines/cine/{THEATER_ID}/",
}


def main():
    theater_id = sys.argv[1] if len(sys.argv) > 1 else THEATER_ID
    date_str = sys.argv[2] if len(sys.argv) > 2 else DATE_STR

    all_results = []
    page = 1
    total_pages = 1

    while page <= total_pages:
        url = BASE_URL.format(id=theater_id, date=date_str, page=page)
        print(f"Fetching page {page}: {url}")

        try:
            resp = httpx.get(url, headers=HEADERS, timeout=15, follow_redirects=True)
        except httpx.HTTPError as e:
            print(f"HTTP error: {e}")
            sys.exit(1)

        print(f"  Status: {resp.status_code}, {len(resp.text)} chars")

        # Try to parse as JSON
        try:
            data = resp.json()
        except json.JSONDecodeError:
            print("  Response is NOT JSON. First 2000 chars:")
            print(resp.text[:2000])
            out_file = FIXTURES_DIR / "sensacine_raw_response.txt"
            out_file.write_text(resp.text[:10000])
            print(f"\n  Saved raw response to {out_file}")
            sys.exit(1)

        # Update pagination
        pagination = data.get("pagination", {})
        total_pages = int(pagination.get("totalPages", 1))
        results = data.get("results", [])
        all_results.extend(results)
        print(f"  Page {page}/{total_pages}, {len(results)} movies on this page")

        if page == 1:
            # Print structure of first page
            print("\n=== TOP-LEVEL KEYS ===")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  {key}: list[{len(value)}]")
                    if value and isinstance(value[0], dict):
                        print(f"    first item keys: {list(value[0].keys())}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")

            if results:
                print("\n=== FIRST MOVIE ENTRY ===")
                print(json.dumps(results[0], indent=2, ensure_ascii=False)[:3000])

        page += 1

    # Assemble full response for fixture
    full_data = {
        "pagination": {"page": 1, "totalPages": 1},
        "results": all_results,
    }

    print(f"\n=== SUMMARY ===")
    print(f"Theater: {theater_id}")
    print(f"Date: {date_str}")
    print(f"Total movies: {len(all_results)}")
    total_showtimes = 0
    for entry in all_results:
        movie = entry.get("movie", {})
        title = movie.get("title", "?")
        showtimes_dict = entry.get("showtimes", {})
        count = sum(len(v) for v in showtimes_dict.values() if isinstance(v, list))
        total_showtimes += count
        print(f"  {title}: {count} showtimes")
    print(f"Total showtimes: {total_showtimes}")

    # Save full response
    out_file = FIXTURES_DIR / "sensacine_sample.json"
    out_file.write_text(json.dumps(full_data, indent=2, ensure_ascii=False))
    print(f"\nFull response saved to {out_file}")
    print(f"Now run: python tests/test_parser_with_fixture.py")


if __name__ == "__main__":
    main()
