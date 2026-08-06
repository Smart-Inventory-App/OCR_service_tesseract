"""Endpoints del servicio de motor.

La superficie es deliberadamente mínima: reconocer, informar qué motor hay
detrás y decir si está vivo. Todo lo demás —qué documento es, qué campos tiene,
cuánta confianza merece, si hay que reintentar— es responsabilidad del
coordinador (`ocr_service`).
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.core.config import settings
from app.core.security import verify_engine_key
from app import engine, prepare
from app.schemas import EngineInfo, EngineResponse, HealthResponse, Transform

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/engine", tags=["engine"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/run", response_model=EngineResponse)
async def run(
    file: UploadFile = File(..., description="Imagen original, sin preprocesar"),
    lang: str = Form(default=""),
    layout: str = Form(default="block", description="auto | block | sparse | line"),
    transform: str = Form(
        default="",
        description="Transformación a aplicar, en JSON. Vacío = valores por defecto.",
    ),
    _: None = Depends(verify_engine_key),
):
    """Reconoce el texto de la imagen y devuelve las palabras con su confianza."""
    content = await _read_image(file)
    spec = _parse_transform(transform)

    try:
        image = prepare.decode(content)
    except prepare.DecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # Tesseract y OpenCV son CPU-bound y bloqueantes: van a un thread aparte para
    # no trabar el event loop mientras corren.
    try:
        result = await asyncio.to_thread(
            engine.run, image, lang or settings.DEFAULT_LANG, layout, spec
        )
    except Exception as exc:
        logger.exception("Falló la ejecución del motor")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"El motor falló al procesar la imagen: {exc}",
        )

    logger.info(
        "OCR: %d palabras, layout=%s, rot=%d, binarize=%s (%d ms)",
        len(result.words),
        layout,
        spec.normalized_rotation,
        spec.binarize.value,
        result.elapsed_ms,
    )

    return result


@router.get("/info", response_model=EngineInfo)
async def info():
    """Qué motor hay detrás. El coordinador lo usa para reportar la comparativa."""
    return EngineInfo(
        engine=engine.NAME,
        version=engine.version(),
        languages=engine.languages(),
        # A Tesseract sí le sirve que el coordinador pruebe variantes de filtros:
        # su precisión depende fuertemente de ellos.
        supports_preprocessing=True,
    )


@router.get("/livez", response_model=HealthResponse)
async def livez():
    return HealthResponse(status="ok", engine=engine.NAME)


@router.get("/readyz", response_model=HealthResponse)
async def readyz():
    """Verifica que el binario de Tesseract se pueda ejecutar de verdad."""
    if not engine.is_available():
        return HealthResponse(
            status="error",
            engine=engine.NAME,
            detail="El binario de Tesseract no está disponible",
        )

    if settings.DEFAULT_LANG not in engine.languages():
        # No es un fallo: el motor cae a inglés y lo avisa. Pero conviene verlo.
        return HealthResponse(
            status="degraded",
            engine=engine.NAME,
            detail=f"Falta el paquete de idioma '{settings.DEFAULT_LANG}'",
        )

    return HealthResponse(status="ok", engine=engine.NAME)


def _parse_transform(raw: str) -> Transform:
    if not raw.strip():
        return Transform()

    try:
        return Transform(**json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"`transform` no es un JSON válido para el contrato: {exc}",
        )


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        accepted = ", ".join(sorted(ALLOWED_CONTENT_TYPES))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato '{file.content_type}' no permitido. Formatos aceptados: {accepted}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo está vacío"
        )
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"La imagen supera los {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    return content
