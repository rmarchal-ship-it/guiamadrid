"""Configuración central de Guía del Ocio Madrid."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "guiamadrid.db"

# Database
DATABASE_URL = f"sqlite:///{DB_PATH}"

# SensaCine (shares backend with allocine.fr)
SENSACINE_BASE_URL = "https://www.sensacine.com"

# Cines de Madrid — IDs de SensaCine
SENSACINE_THEATER_IDS = {
    "C0094": "Yelmo Ideal",
    "C0062": "Cinesa Proyecciones",
    "C0758": "Yelmo Palafox",
    "C0335": "Renoir Princesa",
    "C0482": "Renoir Plaza de España",
    "C0089": "Cines Verdi Madrid",
    "C0034": "Cine Doré (Filmoteca)",
    "C0816": "Cinesa Méndez Álvaro",
    "C0073": "Cinesa Manoteras",
    "C0898": "Yelmo Luxury Palacio de Hielo",
    "C0060": "Cinesa Príncipe Pío",
    "C0802": "Cinesa La Vaguada",
    "C0041": "Cines Embajadores",
    "C0814": "Cinesa Equinoccio",
    "C0801": "Cinesa Parquesur",
    "C0800": "Cinesa Xanadú",
    "C0815": "Cinesa Plenilunio",
    "C0053": "Yelmo Plaza Norte 2",
    "C0817": "Cinesa As Cancelas",
    "C0059": "Cinesa Tres Aguas",
}

# Scraper settings
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 1.0  # seconds between requests
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Email
EMAIL_RECIPIENT = "rmarchal75@gmail.com"
EMAIL_SUBJECT_PREFIX = "[Guía Madrid]"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
