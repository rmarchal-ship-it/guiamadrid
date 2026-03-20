# Guía del Ocio Madrid — Plan de desarrollo

## Fase 0: Scraper de cine (COMPLETADA)
- [x] `guiamadrid/config.py` — configuración central (URLs, IDs de cines, SMTP, etc.)
- [x] `guiamadrid/scrapers/base.py` — BaseScraper con HTTP rate-limited, dataclasses Showtime/ScrapeResult
- [x] `guiamadrid/scrapers/cine/sensacine.py` — SensaCineScraper usando API interna `/_/showtimes/theater-{ID}/d-{fecha}/`
- [x] 20 cines de Madrid configurados (Yelmo, Cinesa, Renoir, Verdi, Doré, etc.)

## Fase 1: Infraestructura backend (COMPLETADA)
- [x] **SQLite + SQLAlchemy** (`guiamadrid/db/models.py`, `guiamadrid/db/database.py`)
  - Modelos: Cinema, Movie, Showtime, ScrapeLog
  - UniqueConstraint en showtimes para idempotencia
  - store_scrape_result() con get-or-create para cinemas/movies
  - Queries: get_showtimes_for_date(), get_movies_for_date(), get_cinemas()
- [x] **FastAPI** (`guiamadrid/api/server.py`)
  - GET `/api/showtimes?fecha=YYYY-MM-DD`
  - GET `/api/movies?fecha=YYYY-MM-DD`
  - GET `/api/cinemas`
  - GET `/api/showtimes/{cinema_id}?fecha=YYYY-MM-DD`
  - CORS habilitado para SwiftUI
- [x] **Email digest** (`guiamadrid/notifications/email_sender.py`)
  - HTML + texto plano
  - Agrupado por película → cine → horarios
  - Requiere GMAIL_USER + GMAIL_APP_PASSWORD
  - Destinatario: rmarchal75@gmail.com
- [x] **CLI** (`guiamadrid/__main__.py`)
  - `python -m guiamadrid scrape [fecha]`
  - `python -m guiamadrid serve`
  - `python -m guiamadrid digest [fecha]`
  - `python -m guiamadrid stats`

## Fase 2: App SwiftUI (PENDIENTE)
- [ ] Modelo Swift que consume la API FastAPI
- [ ] Vista principal: lista de películas del día
- [ ] Detalle: horarios por cine, poster, sinopsis
- [ ] Filtros: VOSE, cine favorito, género

## Fase 3: Más fuentes (PENDIENTE)
- [ ] Scraper de teatro (ej. entradas.com, teatromadrid.com)
- [ ] Scraper de conciertos
- [ ] Scraper de exposiciones
- [ ] Unificación en modelo Event genérico
