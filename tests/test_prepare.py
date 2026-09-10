"""Cadena de preparación con OpenCV.

Cada paso se prueba por separado, porque el coordinador los enciende y los apaga
de a uno para armar sus variantes: si uno cambia de comportamiento, cambia el
resultado de un tipo de documento entero.
"""

import cv2
import numpy as np
import pytest

from app import prepare
from app.schemas import Binarize, Transform


# ---------------------------------------------------------------------------
# Decodificación
# ---------------------------------------------------------------------------

def test_decode_returns_a_bgr_matrix(image_bytes):
    image = prepare.decode(image_bytes)

    assert image.ndim == 3
    assert image.shape[2] == 3


def test_decode_rejects_something_that_is_not_an_image():
    with pytest.raises(prepare.DecodeError):
        prepare.decode(b"esto no es una imagen")


# ---------------------------------------------------------------------------
# Giro
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degrees", [90, 270])
def test_rotate_a_quarter_turn_swaps_the_axes(degrees, gray_image):
    rotated = prepare.rotate(gray_image, degrees)

    assert rotated.shape[:2] == gray_image.shape[:2][::-1]


def test_rotate_180_keeps_the_shape(gray_image):
    assert prepare.rotate(gray_image, 180).shape == gray_image.shape


def test_rotate_zero_is_a_no_op(gray_image):
    assert np.array_equal(prepare.rotate(gray_image, 0), gray_image)


def test_rotate_normalizes_the_angle(gray_image):
    assert np.array_equal(prepare.rotate(gray_image, 360), gray_image)


def test_four_quarter_turns_return_the_original(gray_image):
    result = gray_image
    for _ in range(4):
        result = prepare.rotate(result, 90)

    assert np.array_equal(result, gray_image)


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def test_grayscale_collapses_the_channels(color_image):
    assert prepare.to_grayscale(color_image).ndim == 2


def test_grayscale_is_idempotent(gray_image):
    assert np.array_equal(prepare.to_grayscale(gray_image), gray_image)


def test_rescale_lifts_a_small_image_to_the_minimum_width(gray_image):
    result = prepare.rescale(gray_image, min_width=1600, max_width=3000)

    assert result.shape[1] == 1600


def test_rescale_shrinks_an_oversized_image():
    huge = np.full((1000, 5000), 200, dtype=np.uint8)

    assert prepare.rescale(huge, 1600, 3000).shape[1] == 3000


def test_rescale_keeps_the_aspect_ratio(gray_image):
    height, width = gray_image.shape[:2]
    result = prepare.rescale(gray_image, 1200, 3000)

    assert result.shape[0] / result.shape[1] == pytest.approx(height / width, rel=0.01)


def test_rescale_leaves_an_image_already_in_range_untouched():
    image = np.full((900, 2000), 200, dtype=np.uint8)

    assert prepare.rescale(image, 1600, 3000) is image


def test_invert_flips_a_dark_image():
    dark = np.full((100, 100), 20, dtype=np.uint8)

    assert prepare.invert_if_dark(dark)[0, 0] == 235


def test_invert_leaves_a_light_image_untouched(gray_image):
    assert np.array_equal(prepare.invert_if_dark(gray_image), gray_image)


def test_contrast_returns_a_single_channel(color_image):
    assert prepare.enhance_contrast(color_image).ndim == 2


def test_denoise_preserves_the_shape(gray_image):
    assert prepare.denoise(gray_image).shape == gray_image.shape


def test_denoise_handles_colour(color_image):
    assert prepare.denoise(color_image).shape == color_image.shape


@pytest.mark.parametrize("mode", [Binarize.OTSU, Binarize.ADAPTIVE])
def test_binarize_returns_only_black_and_white(mode, gray_image):
    result = prepare.binarize(gray_image, mode)

    assert set(np.unique(result)).issubset({0, 255})


def test_binarize_none_just_greys(color_image):
    """Sin binarizar se conserva la escala de grises, con sus valores intermedios."""
    gradient = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
    result = prepare.binarize(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR), Binarize.NONE)

    assert result.ndim == 2
    assert len(np.unique(result)) > 2
    assert prepare.binarize(color_image, Binarize.NONE).ndim == 2


def test_deskew_straightens_a_tilted_band(skewed_image):
    """El ángulo estimado sobre la imagen enderezada debe ser menor."""
    straightened = prepare.deskew(skewed_image)

    assert _tilt(straightened) < _tilt(skewed_image)


def test_deskew_leaves_a_straight_image_alone(gray_image):
    assert np.array_equal(prepare.deskew(gray_image), gray_image)


def test_deskew_survives_a_blank_image():
    blank = np.full((100, 100), 255, dtype=np.uint8)

    assert prepare.deskew(blank).shape == blank.shape


def _tilt(image: np.ndarray) -> float:
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    angle = cv2.minAreaRect(cv2.findNonZero(binary))[-1] % 90
    return abs(angle - 90 if angle > 45 else angle)


# ---------------------------------------------------------------------------
# Cadena completa
# ---------------------------------------------------------------------------

def test_run_applies_only_what_was_asked(color_image):
    """Con todo apagado, la imagen sale como entró."""
    spec = Transform(
        grayscale=False, rescale=False, invert=False, contrast=False,
        denoise=False, deskew=False, binarize=Binarize.NONE,
    )

    assert np.array_equal(prepare.run(color_image, spec), color_image)


def test_run_applies_the_rotation_first(color_image):
    spec = Transform(
        rotate=90, grayscale=False, rescale=False, invert=False,
        contrast=False, denoise=False, deskew=False,
    )

    assert prepare.run(color_image, spec).shape[:2] == color_image.shape[:2][::-1]


def test_run_with_the_label_chain_returns_grayscale(color_image):
    result = prepare.run(color_image, Transform(min_width=800, max_width=1200))

    assert result.ndim == 2
    assert result.shape[1] == 800


def test_run_with_the_stamp_chain_returns_a_binary_image(color_image):
    spec = Transform(
        min_width=800, max_width=1200, invert=False, contrast=False,
        deskew=False, binarize=Binarize.ADAPTIVE,
    )

    assert set(np.unique(prepare.run(color_image, spec))).issubset({0, 255})


# ---------------------------------------------------------------------------
# Recorte a una banda de texto
# ---------------------------------------------------------------------------

def _foto_con_renglon(ancho=1200, alto=900, y_texto=700):
    """Fondo claro con un renglón oscuro fino, como el sello sobre un envase."""
    import cv2
    import numpy as np

    imagen = np.full((alto, ancho, 3), 235, dtype=np.uint8)
    # Un bloque de texto grande arriba y el "sello" fino abajo.
    cv2.putText(imagen, "CONSERVAR EN LUGAR FRESCO", (120, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 40), 4)
    cv2.putText(imagen, "L6217 VTO 08/28", (150, y_texto),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (55, 55, 55), 2)
    return imagen


def test_text_bands_finds_the_lines_and_ranks_them():
    imagen = _foto_con_renglon()

    bandas = prepare.text_bands(imagen)

    assert len(bandas) >= 2, "tiene que encontrar los dos renglones"
    # Todas son franjas: mucho más anchas que altas.
    assert all(w / h >= 3 for _x, _y, w, h in bandas)


def test_crop_to_band_returns_a_smaller_image():
    imagen = _foto_con_renglon()

    recorte = prepare.crop_to_band(imagen, 0)

    assert recorte.shape[0] < imagen.shape[0]
    assert recorte.size > 0


def test_an_out_of_range_band_returns_the_whole_image():
    """Un índice de más cuesta un intento repetido, nunca una lectura perdida."""
    imagen = _foto_con_renglon()

    recorte = prepare.crop_to_band(imagen, 99)

    assert recorte.shape == imagen.shape


def test_no_crop_when_the_transform_does_not_ask_for_it():
    imagen = _foto_con_renglon()

    assert prepare.crop_to_band(imagen, -1).shape == imagen.shape


def test_the_crop_happens_inside_the_chain():
    """`run` tiene que aplicar el recorte, no sólo aceptarlo en el contrato."""
    imagen = _foto_con_renglon()

    entera = prepare.run(imagen, Transform(rescale=False, deskew=False))
    recortada = prepare.run(imagen, Transform(crop_band=0, rescale=False, deskew=False))

    assert recortada.shape[0] < entera.shape[0]
