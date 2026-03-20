"""Email digest sender for Guía del Ocio Madrid."""

import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from guiamadrid.config import (
    EMAIL_RECIPIENT,
    EMAIL_SUBJECT_PREFIX,
    SMTP_HOST,
    SMTP_PORT,
)
from guiamadrid.db.database import get_movies_for_date, get_showtimes_for_date


def build_digest_html(target_date: str) -> str:
    """Build HTML email body with today's movie digest."""
    movies = get_movies_for_date(target_date)
    showtimes = get_showtimes_for_date(target_date)

    # Group showtimes by movie
    by_movie: dict[str, list[dict]] = {}
    for st in showtimes:
        by_movie.setdefault(st["movie"], []).append(st)

    lines = [
        "<html><body>",
        f"<h1>🎬 Cartelera Madrid — {target_date}</h1>",
        f"<p><strong>{len(movies)} películas</strong> en cartelera, "
        f"<strong>{len(showtimes)} sesiones</strong> disponibles.</p>",
        "<hr>",
    ]

    for movie_title, sessions in sorted(by_movie.items()):
        first = sessions[0]
        rating_str = f" ⭐ {first['rating']:.1f}" if first.get("rating") else ""
        duration_str = f" ({first['duration_min']} min)" if first.get("duration_min") else ""
        genre_str = f" — {first['genre']}" if first.get("genre") else ""

        lines.append(f"<h2>{movie_title}{rating_str}</h2>")
        if first.get("director"):
            lines.append(f"<p><em>Dir: {first['director']}</em>{genre_str}{duration_str}</p>")

        # Group by cinema
        by_cinema: dict[str, list[str]] = {}
        for s in sessions:
            label = s["time"]
            if s.get("language"):
                label += f" ({s['language']})"
            if s.get("format") and s["format"] != "2D":
                label += f" [{s['format']}]"
            by_cinema.setdefault(s["cinema"], []).append(label)

        lines.append("<ul>")
        for cinema, times in sorted(by_cinema.items()):
            lines.append(f"<li><strong>{cinema}</strong>: {', '.join(times)}</li>")
        lines.append("</ul>")

    lines.append("<hr>")
    lines.append("<p style='color:#888; font-size:12px;'>Guía del Ocio Madrid — datos de SensaCine</p>")
    lines.append("</body></html>")

    return "\n".join(lines)


def build_digest_text(target_date: str) -> str:
    """Build plain text email body."""
    movies = get_movies_for_date(target_date)
    showtimes = get_showtimes_for_date(target_date)

    by_movie: dict[str, list[dict]] = {}
    for st in showtimes:
        by_movie.setdefault(st["movie"], []).append(st)

    lines = [
        f"CARTELERA MADRID — {target_date}",
        f"{len(movies)} películas, {len(showtimes)} sesiones",
        "=" * 50,
        "",
    ]

    for movie_title, sessions in sorted(by_movie.items()):
        first = sessions[0]
        rating_str = f" ({first['rating']:.1f})" if first.get("rating") else ""
        lines.append(f"{movie_title}{rating_str}")
        if first.get("director"):
            lines.append(f"  Dir: {first['director']}")

        by_cinema: dict[str, list[str]] = {}
        for s in sessions:
            label = s["time"]
            if s.get("language"):
                label += f" ({s['language']})"
            by_cinema.setdefault(s["cinema"], []).append(label)

        for cinema, times in sorted(by_cinema.items()):
            lines.append(f"  {cinema}: {', '.join(times)}")
        lines.append("")

    return "\n".join(lines)


def send_digest(
    target_date: str | None = None,
    recipient: str | None = None,
) -> bool:
    """Send the daily digest email.

    Requires environment variables:
      GMAIL_USER — sender gmail address
      GMAIL_APP_PASSWORD — app-specific password (not regular password)
    """
    target = target_date or str(date.today())
    to_addr = recipient or EMAIL_RECIPIENT

    gmail_user = os.environ.get("GMAIL_USER")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_password:
        print("ERROR: Set GMAIL_USER and GMAIL_APP_PASSWORD environment variables.")
        print("Generate an app password at https://myaccount.google.com/apppasswords")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{EMAIL_SUBJECT_PREFIX} Cartelera {target}"
    msg["From"] = gmail_user
    msg["To"] = to_addr

    text_body = build_digest_text(target)
    html_body = build_digest_html(target)
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, to_addr, msg.as_string())
        print(f"Digest sent to {to_addr} for {target}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
