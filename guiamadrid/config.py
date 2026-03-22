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
    # ── Centro Madrid ──
    "E0621": "Yelmo Ideal",
    "E0402": "Cinesa Proyecciones",
    "E1001": "Yelmo Luxury Palafox",
    "E0364": "Cines Princesa (Renoir)",
    "E0577": "Renoir Plaza de España",
    "E0609": "Cines Verdi Madrid",
    "G02GQ": "Cine Doré (Filmoteca)",
    "E1032": "Cines Embajadores",
    "E0146": "Cines Callao",
    "E0559": "Palacio de la Prensa",
    "E0347": "Golem Madrid",
    "E0578": "Renoir Retiro",
    "E0564": "mk2 Cine Paz",
    "E0792": "Artistic Metropol",
    "E0781": "Cineteca",
    "E0687": "Cine Estudio Círculo de Bellas Artes",
    "E0566": "Pequeño Cine Estudio",
    "G0FUY": "Sala Equis",
    "E0736": "Sala Berlanga",
    "E2919": "Cine Embajadores Río",
    "E0044": "Cines Acteón",
    "E0339": "Conde Duque Goya",
    "E0864": "Conde Duque Morasol",
    "E0340": "Conde Duque Santa Engracia",
    "E0338": "Conde Duque Verdi Alberto Aguilera",
    # ── Madrid periferia ──
    "E0247": "Cinesa Méndez Álvaro",
    "E0646": "Cinesa Manoteras",
    "E0432": "mk2 Palacio de Hielo",
    "E0401": "Cinesa Príncipe Pío",
    "E0459": "Yelmo La Vaguada",
    "E0475": "Yelmo Plenilunio",
    "E0681": "Yelmo Islazul",
    "G01NH": "Ocine Urban Caleido",
    "E0731": "Cinesa La Gavia",
    "E0393": "Cinesa Las Rosas",
    "E2909": "Odeon Alcalá Norte",
    "E0881": "Autocine Madrid",
    # ── Comunidad de Madrid ──
    "E0385": "Cinesa Equinoccio (Majadahonda)",
    "E0399": "Cinesa Parquesur (Leganés)",
    "E0406": "Cinesa Intu Xanadú (Arroyomolinos)",
    "E2916": "Yelmo Plaza Norte 2 (S.S. de los Reyes)",
    "E0207": "Yelmo Tres Aguas (Alcorcón)",
    "E0392": "Cinesa La Moraleja (Alcobendas)",
    "E0209": "Kinépolis Diversia (Alcobendas)",
    "E0453": "Kinépolis Ciudad de la Imagen (Pozuelo)",
    "E1004": "Ocine Urban X-Madrid (Alcorcón)",
    "E0877": "Odeon Sambil (Leganés)",
    "E0246": "Cinesa Nassica (Getafe)",
    "E0389": "Cinesa Heron City (Las Rozas)",
    "G02A1": "Odeon Gran Plaza 2 (Majadahonda)",
    "E0394": "Cinesa Plaza Loranca 2 (Fuenlabrada)",
    "E0761": "Cines Dos de Mayo (Móstoles)",
    "E0671": "Yelmo Rivas H2O (Rivas-Vaciamadrid)",
    "E2910": "Cines Plaza (Coslada)",
    "E0353": "Cines La Rambla (Coslada)",
    "E0291": "Yelmo Parque Corredor (Torrejón de Ardoz)",
    "G01P0": "Cinesa Oasiz (Torrejón de Ardoz)",
    "E0935": "Spazio Cines (Parla)",
    "E2900": "Ocine Plaza Éboli (Pinto)",
    "E0294": "Yelmo Luxury Plaza Norte (S.S. de los Reyes)",
    "E0199": "Cinebox 3C (Tres Cantos)",
    "E0815": "Odeon Tres Cantos (Tres Cantos)",
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
