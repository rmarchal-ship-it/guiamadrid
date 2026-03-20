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

# Cines de Madrid — IDs de SensaCine (verificados en sensacine.com/cines/cine/EXXXX/)
SENSACINE_THEATER_IDS = {
    # Centro Madrid
    "E0621": "Yelmo Ideal",
    "E0402": "Cinesa Proyecciones",
    "E1001": "Yelmo Luxury Palafox",
    "E0364": "Cines Princesa (Renoir)",
    "E0577": "Renoir Plaza de España",
    "E0609": "Cines Verdi Madrid",
    "G02GQ": "Cine Doré (Filmoteca)",
    "E1032": "Cines Embajadores",
    # Madrid periferia
    "E0247": "Cinesa Méndez Álvaro",
    "E0646": "Cinesa Manoteras",
    "E0432": "mk2 Palacio de Hielo",
    "E0401": "Cinesa Príncipe Pío",
    "E0459": "Yelmo La Vaguada",
    "E0475": "Yelmo Plenilunio",
    # Comunidad de Madrid
    "E0385": "Cinesa Equinoccio (Majadahonda)",
    "E0399": "Cinesa Parquesur (Leganés)",
    "E0406": "Cinesa Intu Xanadú (Arroyomolinos)",
    "E2916": "Yelmo Plaza Norte 2 (S.S. de los Reyes)",
    "E0207": "Yelmo Tres Aguas (Alcorcón)",
    # Fuera de Madrid
    "E0795": "Cinesa As Cancelas (Santiago)",
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
