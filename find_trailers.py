#!/usr/bin/env python3
"""
find_trailers.py — Busca tráilers de YouTube para las películas de la cartelera.

Usa yt-dlp para buscar en YouTube, verifica que el resultado corresponde
a la película correcta, y actualiza cartelera_standalone.html con los
IDs de vídeo encontrados.

Sin APIs de pago ni servicios capados. Solo YouTube público + yt-dlp.
"""

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HTML_FILE = Path(__file__).parent / "cartelera_standalone.html"

# ── Películas con datos de búsqueda enriquecidos para verificación ──
MOVIES = [
    {
        "title": "Amarga Navidad",
        "search": "Amarga Navidad Pedro Almodóvar trailer",
        "verify": ["almodóvar", "amarga navidad"],
        "director": "Pedro Almodóvar",
        "year": 2025,
    },
    {
        "title": "Hoppers",
        "search": "Hoppers película animación 2025 trailer español",
        "verify": ["hoppers"],
        "director": "Daniel Chong",
        "year": 2025,
    },
    {
        "title": "Torrente Presidente",
        "search": "Torrente Presidente Santiago Segura trailer",
        "verify": ["torrente"],
        "director": "Santiago Segura",
        "year": 2025,
    },
    {
        "title": "Scream VII",
        "search": "Scream VII trailer español",
        "verify": ["scream"],
        "director": "Kevin Williamson",
        "year": 2025,
    },
    {
        "title": "Greenland 2: El regreso",
        "search": "Greenland 2 El regreso Gerard Butler trailer español",
        "verify": ["greenland"],
        "director": "Ric Roman Waugh",
        "year": 2025,
    },
    {
        "title": "Whistle: El silbido del mal",
        "search": "Whistle El silbido del mal trailer español",
        "verify": ["whistle", "silbido"],
        "director": "Corin Hardy",
        "year": 2025,
    },
    {
        "title": "La novia",
        "search": "The Bride 2025 Maggie Gyllenhaal Jessie Buckley trailer",
        "verify": ["bride", "novia"],
        "director": "Maggie Gyllenhaal",
        "year": 2025,
    },
    {
        "title": "Elegir mi vida",
        "search": "Elegir mi vida Amélie Bonnin trailer español",
        "verify": ["elegir", "vida"],
        "director": "Amélie Bonnin",
        "year": 2025,
    },
    {
        "title": "La sonrisa del mal",
        "search": "La sonrisa del mal película 2025 trailer español",
        "verify": ["sonrisa", "mal"],
        "director": "",
        "year": 2025,
    },
    {
        "title": "Tafiti y sus amigos",
        "search": "Tafiti y sus amigos película trailer español",
        "verify": ["tafiti"],
        "director": "",
        "year": 2025,
    },
    {
        "title": "Una hija en Tokio",
        "search": "Una hija en Tokio película 2025 trailer español",
        "verify": ["hija", "tokio"],
        "director": "",
        "year": 2025,
    },
    {
        "title": "Perfect Blue (Reestreno)",
        "search": "Perfect Blue Satoshi Kon trailer español",
        "verify": ["perfect blue"],
        "director": "Satoshi Kon",
        "year": 1997,
    },
    {
        "title": "Your Name (Reposición)",
        "search": "Your Name Makoto Shinkai trailer español",
        "verify": ["your name", "kimi no na wa"],
        "director": "Makoto Shinkai",
        "year": 2016,
    },
    {
        "title": "Power to the People",
        "search": "Power to the People John Lennon concert documentary trailer",
        "verify": ["power", "people", "lennon"],
        "director": "",
        "year": 2025,
    },
    {
        "title": "La Grazia",
        "search": "La Grazia Paolo Sorrentino trailer",
        "verify": ["grazia", "sorrentino"],
        "director": "Paolo Sorrentino",
        "year": 2025,
    },
    {
        "title": "El mago del Kremlin",
        "search": "El mago del Kremlin Olivier Assayas trailer español",
        "verify": ["kremlin", "mago"],
        "director": "Olivier Assayas",
        "year": 2025,
    },
    {
        "title": "La última cena",
        "search": "La última cena película 2025 trailer español thriller",
        "verify": ["última cena", "ultima cena"],
        "director": "",
        "year": 2025,
    },
]


def normalize(text: str) -> str:
    """Remove accents & lowercase for fuzzy matching."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def search_youtube(query: str, max_results: int = 5) -> list[dict]:
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
                videos.append(
                    {
                        "id": data.get("id", ""),
                        "title": data.get("title", ""),
                        "channel": data.get("channel", data.get("uploader", "")),
                        "description": data.get("description", ""),
                        "duration": data.get("duration"),
                        "url": data.get("url", data.get("webpage_url", "")),
                    }
                )
            except json.JSONDecodeError:
                continue
        return videos
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ⚠ yt-dlp error: {e}")
        return []


def verify_video(video: dict, movie: dict) -> tuple[bool, str]:
    """
    Verify a YouTube result actually corresponds to the movie.
    Returns (is_valid, reason).
    """
    vtitle = normalize(video["title"])
    vchannel = normalize(video.get("channel", ""))
    vdesc = normalize(video.get("description", ""))
    combined = f"{vtitle} {vchannel} {vdesc}"

    # Check: at least one verify keyword must appear in title or description
    keyword_match = any(normalize(kw) in combined for kw in movie["verify"])
    if not keyword_match:
        return False, f"No keyword match ({movie['verify']})"

    # Check: filter out unrelated content (reactions, reviews, rankings)
    reject_patterns = [
        r"\breaccion\b",
        r"\breaction\b",
        r"\branking\b",
        r"\btop \d+\b",
        r"\bexplicaci[oó]n\b",
        r"\bresumen\b",
        r"\bcrítica\b",
        r"\breview\b",
    ]
    for pat in reject_patterns:
        if re.search(pat, vtitle):
            return False, f"Rejected: matched filter '{pat}' in title"

    # Check: prefer "trailer" or "tráiler" in title
    is_trailer = bool(re.search(r"tr[aá]iler|trailer|teaser|oficial", vtitle))

    # Check: if director is known, bonus if mentioned
    director_match = True
    if movie["director"]:
        director_norm = normalize(movie["director"].split()[-1])  # last name
        director_match = director_norm in combined

    if is_trailer and keyword_match:
        return True, "Trailer + keyword match"
    if keyword_match and director_match:
        return True, "Keyword + director match"
    if keyword_match:
        return True, "Keyword match (no trailer in title)"

    return False, "Insufficient verification"


def find_trailer(movie: dict) -> dict | None:
    """Find and verify a YouTube trailer for a movie."""
    print(f"\n🔍 Buscando: {movie['title']}")
    print(f"   Query: {movie['search']}")

    videos = search_youtube(movie["search"])
    if not videos:
        print("   ❌ Sin resultados de YouTube")
        return None

    for i, video in enumerate(videos):
        valid, reason = verify_video(video, movie)
        status = "✅" if valid else "❌"
        print(f"   {status} [{i+1}] {video['title'][:70]}")
        print(f"       → {reason}")
        if valid:
            print(f"       🎬 ID: {video['id']}")
            return video

    # If none passed strict verification, try a second search with different terms
    alt_query = f"{movie['title']} película trailer 2025"
    if movie["year"] and movie["year"] < 2020:
        alt_query = f"{movie['title']} trailer oficial"
    print(f"   🔄 Reintentando: {alt_query}")
    videos2 = search_youtube(alt_query, max_results=3)
    for video in videos2:
        valid, reason = verify_video(video, movie)
        if valid:
            print(f"   ✅ [retry] {video['title'][:70]}")
            print(f"       → {reason} | ID: {video['id']}")
            return video

    print("   ❌ No se encontró tráiler verificado")
    return None


def update_html(trailer_map: dict[str, str]):
    """Update cartelera_standalone.html with found YouTube video IDs."""
    html = HTML_FILE.read_text(encoding="utf-8")

    # Build the JS object for YOUTUBE_TRAILERS
    js_entries = []
    for title, vid_id in sorted(trailer_map.items()):
        escaped_title = title.replace('"', '\\"')
        js_entries.append(f'  "{escaped_title}": "{vid_id}"')
    js_obj = "const YOUTUBE_TRAILERS = {\n" + ",\n".join(js_entries) + "\n};"

    # Replace existing YOUTUBE_TRAILERS block
    if "const YOUTUBE_TRAILERS" in html:
        html = re.sub(
            r"const YOUTUBE_TRAILERS\s*=\s*\{[^}]*\};",
            js_obj,
            html,
        )
    else:
        # Insert after TMDB_IDS block
        html = html.replace(
            "// Genre visual themes",
            f"{js_obj}\n\n// Genre visual themes",
        )

    HTML_FILE.write_text(html, encoding="utf-8")
    print(f"\n✅ HTML actualizado con {len(trailer_map)} tráilers")


def main():
    print("=" * 60)
    print("🎬 Buscador de Tráilers — Guía Madrid Cartelera")
    print("=" * 60)

    trailer_map: dict[str, str] = {}
    failed: list[str] = []

    for movie in MOVIES:
        result = find_trailer(movie)
        if result:
            trailer_map[movie["title"]] = result["id"]
        else:
            failed.append(movie["title"])

    print("\n" + "=" * 60)
    print(f"📊 Resultados: {len(trailer_map)}/{len(MOVIES)} tráilers encontrados")
    if failed:
        print(f"❌ Sin tráiler: {', '.join(failed)}")
    print("=" * 60)

    if trailer_map:
        update_html(trailer_map)

    # Save results to JSON for reference
    results_file = Path(__file__).parent / "trailers.json"
    results_file.write_text(
        json.dumps(trailer_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"💾 Resultados guardados en {results_file}")

    return 0 if trailer_map else 1


if __name__ == "__main__":
    sys.exit(main())
