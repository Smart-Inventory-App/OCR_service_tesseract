"""Fixtures compartidas.

Los tests no dependen del binario de Tesseract: donde hace falta se mockea
`pytesseract`. Lo que sí se ejercita de verdad es la cadena de OpenCV, que es lo
que este servicio aporta además de invocar al motor.
"""

import io
import os

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

os.environ.setdefault("DEBUG", "true")

from app.main import app  # noqa: E402


@pytest.fixture
async def client():
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as instance:
        yield instance


def render_image(text: str = "LECHE ENTERA", size: tuple[int, int] = (600, 300)) -> bytes:
    """JPEG con texto oscuro sobre fondo claro."""
    image = Image.new("RGB", size, color=(255, 255, 255))
    ImageDraw.Draw(image).text((30, 60), text, fill=(0, 0, 0))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def image_bytes() -> bytes:
    return render_image()


@pytest.fixture
def gray_image() -> np.ndarray:
    """Escala de grises con una banda oscura, para ejercitar los filtros."""
    image = np.full((200, 600), 240, dtype=np.uint8)
    image[80:120, 50:400] = 30
    return image


@pytest.fixture
def color_image() -> np.ndarray:
    image = np.full((200, 600, 3), 240, dtype=np.uint8)
    image[80:120, 50:400] = (30, 30, 30)
    return image


@pytest.fixture
def skewed_image() -> np.ndarray:
    """Imagen con una banda de texto inclinada a propósito."""
    image = np.full((300, 700), 240, dtype=np.uint8)
    image[140:170, 60:640] = 20

    matrix = cv2.getRotationMatrix2D((350, 150), 7.0, 1.0)
    return cv2.warpAffine(image, matrix, (700, 300), borderValue=240)
