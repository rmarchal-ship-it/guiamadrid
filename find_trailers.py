#!/usr/bin/env python3
"""
find_trailers.py — Busca tráilers de YouTube para las películas de la cartelera.

Usa yt-dlp para buscar en YouTube, verifica que el resultado corresponde
a la película correcta, y actualiza cartelera_standalone.html con los
IDs de vídeo encontrados.

Fully autonomous: reads movie titles from cartelera_standalone.html,
auto-generates search queries, and updates the HTML with found trailer IDs.
No hardcoded movie list. No API keys needed. Only YouTube + yt-dlp.
"""

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

HTML_FILE = Path(__file__).parent / "cartelera_standalone.html"


def normalize(text: str) -> str:
    """Remove accents & lowercase for fuzzy matching."""
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def extract_movies_from_html() -> list[dict]:
    """Parse EMBEDDED_MOVIES from the HTML to get current movie list."""
    html = HTML_FILE.read_text(encoding="utf-8")
    match = re.search(r"const EMBEDDED_MOVIES\s*=\s*(\[.*?\]);\s*\n", html)
    if not match:
        print("⚠ Could not find EMBEDDED_MOVIES in HTML")
        return []
    return json.loads(match.group(1))


def load_existing_trailers() -> dict[str, str]:
    """Load existing YOUTUBE_TRAILERS from HTML to merge with new results."""
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


def build_search_entry(movie: dict) -> dict:
    """Auto-generate a search entry from EMBEDDED_MOVIES data."""
    title = movie["title"]
    director = movie.get("director", "")
    genre = movie.get("genre", "")

    # Clean title: remove parenthetical suffixes like "(Reestreno)", "(Reposición)"
    clean_title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()

    # Build search query
    parts = [clean_title]
    if director:
        parts.append(director)
    parts.append("trailer")
    # Add "español" for non-Spanish titles (heuristic: contains non-Spanish chars or common English words)
    if not re.search(r"[áéíóúñ]", clean_title.lower()):
        parts.append("español")
    search = " ".join(parts)

    # Build verification keywords from significant words in title (3+ chars)
    stopwords = {"del", "de", "la", "el", "los", "las", "un", "una", "en", "con", "por", "para", "que", "y"}
    words = [w for w in re.findall(r"\w+", clean_title.lower()) if len(w) >= 3 and w not in stopwords]
    # Use the most distinctive words (longest ones) as verify keywords
    verify = sorted(words, key=len, reverse=True)[:3] if words else [normalize(clean_title)]
    # Also add the full clean title as a keyword
    verify.append(normalize(clean_title))

    return {
        "title": title,
        "search": search,
        "verify": verify,
        "director": director,
        "genre": genre,
    }


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
    if movie.get("director"):
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

    # Retry with alternative query
    clean_title = re.sub(r"\s*\([^)]*\)\s*$", "", movie["title"]).strip()
    alt_query = f"{clean_title} película trailer oficial"
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

    # Auto-discover movies from HTML
    raw_movies = extract_movies_from_html()
    if not raw_movies:
        print("❌ No movies found in HTML")
        return 1
    print(f"🎬 {len(raw_movies)} películas en cartelera")

    # Build search entries automatically
    movies = [build_search_entry(m) for m in raw_movies]

    # Load existing trailers so we don't lose them
    existing = load_existing_trailers()
    print(f"📦 {len(existing)} tráilers existentes en HTML")

    trailer_map: dict[str, str] = {}
    failed: list[str] = []

    for movie in movies:
        # Skip movies that already have trailers
        if movie["title"] in existing:
            print(f"\n⏭️  {movie['title']}: ya tiene tráiler ({existing[movie['title']]})")
            continue
        result = find_trailer(movie)
        if result:
            trailer_map[movie["title"]] = result["id"]
        else:
            failed.append(movie["title"])

    print("\n" + "=" * 60)
    print(f"📊 Resultados: {len(trailer_map)} nuevos tráilers encontrados")
    if failed:
        print(f"❌ Sin tráiler: {', '.join(failed)}")
    print("=" * 60)

    # Merge new trailers with existing ones
    merged = {**existing, **trailer_map}
    if merged:
        update_html(merged)

    # Save merged results to JSON for reference
    results_file = Path(__file__).parent / "trailers.json"
    results_file.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"💾 {len(merged)} tráilers guardados en {results_file}")

    return 0 if merged else 1


if __name__ == "__main__":
    sys.exit(main())
