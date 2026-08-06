"""Ejecución de Tesseract.

La propuesta lo elige como punto de partida por ser open source, no requerir GPU
y tener bajo costo computacional ("adecuada para un MVP o entornos con recursos
limitados").

Este módulo hace una sola cosa: correr el motor sobre la imagen ya preparada y
devolver las palabras con su caja y su confianza. No interpreta el resultado, no
extrae campos y no decide reintentos: de eso se ocupa `ocr_service`.
"""

import logging
import time

import numpy as np
import pytesseract
from pytesseract import Output

from app.core.config import settings
from app.prepare import run as prepare
from app.schemas import EngineResponse, Transform, Word

logger = logging.getLogger(__name__)

NAME = "tesseract"

# `layout` es una pista agnóstica del motor que manda el coordinador; acá se
# traduce al *page segmentation mode* de Tesseract.
_PSM_BY_LAYOUT = {
    "auto": 3,     # segmentación automática: mejor en fotos con fondo y objetos
    "block": 6,    # bloque uniforme de texto (tickets, remitos, sellos)
    "sparse": 11,  # texto disperso sin orden garantizado (etiquetas)
    "line": 7,     # una única línea de texto
}
_DEFAULT_PSM = _PSM_BY_LAYOUT["block"]

_FALLBACK_LANG = "eng"

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def is_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as exc:
        logger.warning("Tesseract no disponible: %s", exc)
        return False


def version() -> str:
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return "desconocida"


def languages() -> list[str]:
    try:
        return sorted(pytesseract.get_languages(config=""))
    except Exception:
        return []


def resolve_lang(lang: str) -> str:
    """Cae a inglés si falta el paquete de idioma pedido (entornos de desarrollo)."""
    available = set(languages())
    if not available or lang in available:
        return lang

    logger.warning("Idioma '%s' no instalado; se usa '%s'", lang, _FALLBACK_LANG)
    return _FALLBACK_LANG if _FALLBACK_LANG in available else lang


def run(image: np.ndarray, lang: str, layout: str, transform: Transform) -> EngineResponse:
    """Prepara la imagen como pidió el coordinador y la pasa por Tesseract."""
    started = time.perf_counter()

    prepared = prepare(image, transform)
    resolved = resolve_lang(lang)
    psm = _PSM_BY_LAYOUT.get(layout, _DEFAULT_PSM)

    # `image_to_data` (y no `image_to_string`) porque devuelve confianza por
    # palabra: es el insumo con el que el coordinador calcula la confianza por
    # campo que exige la propuesta.
    data = pytesseract.image_to_data(
        prepared,
        lang=resolved,
        config=f"--psm {psm}",
        output_type=Output.DICT,
    )

    words = [word for word in map(_to_word, *_columns(data)) if word is not None]
    height, width = prepared.shape[:2]

    return EngineResponse(
        engine=NAME,
        text=render_text(words),
        words=words,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        image_width=width,
        image_height=height,
        lang=resolved,
    )


def render_text(words: list[Word]) -> str:
    """Texto plano reconstruido agrupando las palabras por línea."""
    grouped: dict[int, list[Word]] = {}
    for word in words:
        if word.text.strip():
            grouped.setdefault(word.line_num, []).append(word)

    return "\n".join(
        " ".join(w.text for w in sorted(line, key=lambda w: w.left))
        for _, line in sorted(grouped.items())
    )


def _columns(data: dict) -> tuple[list, ...]:
    return (
        data["text"],
        data["conf"],
        data["left"],
        data["top"],
        data["width"],
        data["height"],
        data["block_num"],
        data["par_num"],
        data["line_num"],
    )


def _to_word(
    text: str,
    conf,
    left: int,
    top: int,
    width: int,
    height: int,
    block_num: int,
    par_num: int,
    line_num: int,
) -> Word | None:
    if not str(text).strip():
        return None

    confidence = float(conf)
    if confidence < 0:  # Tesseract marca con -1 las cajas sin texto reconocido
        return None

    return Word(
        text=str(text),
        confidence=min(max(confidence / 100.0, 0.0), 1.0),
        left=int(left),
        top=int(top),
        width=int(width),
        height=int(height),
        # Identificador de línea único a lo largo de todos los bloques y
        # párrafos, monótono en orden de lectura.
        line_num=int(block_num) * 10_000 + int(par_num) * 100 + int(line_num),
    )
