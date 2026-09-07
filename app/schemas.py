"""Contrato del servicio de motor.

Es el mismo en `ocr_service_tesseract`, `ocr_service_easyocr` y
`ocr_service_paddleocr`: el coordinador (`ocr_service`) habla con los tres a
través de esta misma forma de petición y de respuesta, y por eso puede cambiar
de motor sin cambiar una línea de su código.

La petición describe **qué transformación mecánica** aplicar a la imagen antes de
reconocerla. Quién decide esa transformación es siempre el coordinador; el motor
sólo la ejecuta.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Binarize(str, Enum):
    """Modo de binarización a aplicar."""

    NONE = "none"
    OTSU = "otsu"
    """Umbral global elegido por histograma. Sirve en envases de alto contraste."""
    ADAPTIVE = "adaptive"
    """Umbral por ventana. Es el que rescata los sellos de vencimiento impresos
    con matriz de puntos, donde el umbral global se come los puntos."""


class Transform(BaseModel):
    """Cadena de preprocesamiento pedida por el coordinador.

    Todos los pasos son operaciones de OpenCV sin criterio propio: el motor las
    aplica en este orden y no decide ninguna por su cuenta.
    """

    rotate: int = Field(default=0, description="Giro en grados: 0, 90, 180 o 270")
    grayscale: bool = True
    rescale: bool = True
    min_width: int = 1600
    max_width: int = 3000
    invert: bool = True
    """Invierte si la imagen es mayormente oscura (texto claro sobre fondo oscuro)."""
    contrast: bool = True
    """Ecualización adaptativa CLAHE."""
    denoise: bool = False
    deskew: bool = True
    binarize: Binarize = Binarize.NONE

    @property
    def normalized_rotation(self) -> int:
        return self.rotate % 360


class Word(BaseModel):
    """Una palabra reconocida, con su caja y su confianza normalizada a 0-1."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    left: int
    top: int
    width: int
    height: int
    line_num: int = 0


class EngineResponse(BaseModel):
    engine: str
    text: str
    """Texto plano reconstruido a partir del agrupamiento por líneas."""

    words: list[Word] = Field(default_factory=list)
    elapsed_ms: int = 0
    image_width: int = 0
    image_height: int = 0
    lang: str = ""
    """Idioma realmente usado, que puede no ser el pedido si falta el paquete."""


class EngineInfo(BaseModel):
    engine: str
    version: str
    languages: list[str] = Field(default_factory=list)
    supports_preprocessing: bool
    """True si al motor le sirve que el coordinador varíe la cadena de filtros."""


class HealthResponse(BaseModel):
    status: str
    engine: str
    detail: str | None = None
