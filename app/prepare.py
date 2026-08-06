"""Preparación de la imagen con OpenCV.

La propuesta señala que Tesseract "requiere preprocesamiento (OpenCV) para
obtener buenos resultados" en escenarios reales: tipografías variadas, fondos
complejos, baja iluminación e imágenes inclinadas.

Cada paso es una función pura `np.ndarray -> np.ndarray`. **Ninguna decide nada**:
qué pasos aplicar viene en el `Transform` que manda el coordinador, que es el que
prueba variantes y se queda con la que mejor lee. Acá sólo se ejecutan.
"""

import cv2
import numpy as np

from app.schemas import Binarize, Transform

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# Mediana de gris por debajo de la cual se considera que la imagen es oscura.
_DARK_MEDIAN = 110

# Ventana y constante del umbral adaptativo. La ventana es grande a propósito:
# con ventana chica el método vacía el interior de los trazos gruesos.
_ADAPTIVE_BLOCK = 41
_ADAPTIVE_C = 15


class DecodeError(ValueError):
    """El binario recibido no es una imagen decodificable."""


def decode(content: bytes) -> np.ndarray:
    """Decodifica bytes a una matriz BGR."""
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DecodeError("El archivo no es una imagen válida")
    return image


def rotate(image: np.ndarray, degrees: int) -> np.ndarray:
    """Gira la imagen en múltiplos de 90°.

    Es la operación que permite leer las fotos de fecha de vencimiento: se sacan
    apoyando el envase de costado y llegan giradas un cuarto de vuelta.
    """
    code = _ROTATIONS.get(degrees % 360)
    return image if code is None else cv2.rotate(image, code)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rescale(image: np.ndarray, min_width: int, max_width: int) -> np.ndarray:
    """Lleva el ancho a la franja donde Tesseract rinde mejor.

    Las fotos de celular traen el texto útil muy chico (una etiqueta ocupa una
    fracción del cuadro), así que se amplía; las enormes se reducen porque no
    aportan precisión y multiplican el tiempo de proceso.
    """
    height, width = image.shape[:2]
    if width == 0:
        return image

    target = min(max(width, min_width), max_width)
    if target == width:
        return image

    factor = target / width
    return cv2.resize(
        image,
        (int(width * factor), int(height * factor)),
        # INTER_CUBIC agranda con bordes nítidos; INTER_AREA reduce sin aliasing.
        interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA,
    )


def invert_if_dark(image: np.ndarray) -> np.ndarray:
    """Invierte las imágenes de texto claro sobre fondo oscuro.

    Tesseract espera texto oscuro sobre fondo claro. Muchos envases usan lo
    contrario (etiquetas negras, packs oscuros) y sin invertir no reconoce casi
    nada.
    """
    if np.median(to_grayscale(image)) >= _DARK_MEDIAN:
        return image
    return cv2.bitwise_not(image)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Ecualización adaptativa (CLAHE): recupera texto mal iluminado sin quemar
    las zonas ya claras, que es lo que haría una ecualización global.
    """
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(to_grayscale(image))


def denoise(image: np.ndarray) -> np.ndarray:
    """Reduce ruido preservando los bordes de los caracteres."""
    if image.ndim == 2:
        return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
    return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)


def binarize(image: np.ndarray, mode: Binarize) -> np.ndarray:
    """Binariza según el modo pedido.

    - `otsu` elige un único umbral global a partir del histograma.
    - `adaptive` calcula un umbral por ventana. Es más caro, pero es el que
      levanta los sellos de vencimiento de matriz de puntos sobre envases
      brillantes, donde el umbral global funde los puntos con el fondo.
    """
    gray = to_grayscale(image)

    if mode is Binarize.OTSU:
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if mode is Binarize.ADAPTIVE:
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            _ADAPTIVE_BLOCK,
            _ADAPTIVE_C,
        )

    return gray


def deskew(image: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """Corrige la inclinación estimándola con `minAreaRect` sobre el texto.

    Sólo endereza desvíos chicos respecto de la horizontal; los giros de un
    cuarto de vuelta los resuelve `rotate`, que el coordinador pide explícitamente.
    """
    gray = to_grayscale(image)

    # El texto debe quedar en blanco sobre fondo negro para `minAreaRect`.
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(binary)
    if coords is None:
        return image

    # `minAreaRect` devuelve el ángulo en (-90, 0] o en (0, 90] según la versión
    # de OpenCV: se normaliza a la desviación respecto de la horizontal.
    angle = cv2.minAreaRect(coords)[-1] % 90
    if angle > 45:
        angle -= 90

    # Rotaciones grandes casi siempre son un error de estimación.
    if abs(angle) < 0.25 or abs(angle) > max_angle:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def run(image: np.ndarray, transform: Transform) -> np.ndarray:
    """Aplica, en orden, la cadena que pidió el coordinador."""
    result = rotate(image, transform.normalized_rotation)

    if transform.grayscale:
        result = to_grayscale(result)
    if transform.rescale:
        result = rescale(result, transform.min_width, transform.max_width)
    if transform.invert:
        result = invert_if_dark(result)
    if transform.contrast:
        result = enhance_contrast(result)
    if transform.denoise:
        result = denoise(result)
    if transform.deskew:
        result = deskew(result)
    if transform.binarize is not Binarize.NONE:
        result = binarize(result, transform.binarize)

    return result
