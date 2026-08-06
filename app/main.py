"""Servicio de motor OCR — Tesseract.

Uno de los dos motores que la propuesta pone a comparar. Vive en su propio
repositorio para que el coordinador (`ocr_service`) no cargue con sus
dependencias de sistema: Tesseract es un binario que hay que instalar en la
imagen, no un paquete de pip.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.routes import router
from app import engine

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if engine.is_available():
        logger.info(
            "Motor Tesseract %s listo (idiomas: %s)",
            engine.version(),
            ", ".join(engine.languages()) or "ninguno",
        )
    else:
        logger.error("Tesseract no está disponible: /engine/run va a fallar")

    yield


app = FastAPI(
    title="Smart Inventory — Motor OCR Tesseract",
    description=(
        "Ejecuta Tesseract sobre una imagen y devuelve las palabras reconocidas "
        "con su caja y su confianza. No interpreta el resultado: la extracción de "
        "campos, la confianza por campo y la estrategia de reintentos son "
        "responsabilidad de `ocr_service`."
    ),
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    lifespan=lifespan,
)

app.include_router(router)
