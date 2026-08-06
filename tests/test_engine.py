"""Ejecución del motor y traducción de la salida de Tesseract.

`pytesseract` se mockea: lo que se verifica es que este servicio traduzca bien
lo que devuelve el binario, no que el binario sepa leer.
"""

from unittest.mock import patch

import numpy as np
import pytest

from app import engine
from app.schemas import Transform

IMAGE = np.full((120, 400, 3), 240, dtype=np.uint8)

# Formato de `image_to_data`: columnas paralelas, una entrada por caja.
TESSERACT_DATA = {
    "text": ["LECHE", "ENTERA", "", "1L"],
    "conf": ["96", "94", "-1", "88"],
    "left": [10, 120, 0, 10],
    "top": [10, 10, 0, 60],
    "width": [100, 120, 0, 40],
    "height": [30, 30, 0, 20],
    "block_num": [1, 1, 1, 1],
    "par_num": [1, 1, 1, 1],
    "line_num": [1, 1, 1, 2],
}


def run(layout: str = "block", transform: Transform | None = None):
    with patch("pytesseract.image_to_data", return_value=TESSERACT_DATA), \
         patch("pytesseract.get_languages", return_value=["spa", "eng"]):
        return engine.run(IMAGE, "spa", layout, transform or Transform())


def test_words_carry_text_box_and_confidence():
    result = run()

    assert [w.text for w in result.words] == ["LECHE", "ENTERA", "1L"]
    assert result.words[0].confidence == pytest.approx(0.96)
    assert (result.words[0].left, result.words[0].top) == (10, 10)


def test_confidence_is_normalized_to_zero_one():
    result = run()

    assert all(0.0 <= word.confidence <= 1.0 for word in result.words)


def test_boxes_without_recognized_text_are_dropped():
    """Tesseract marca con -1 las cajas donde no reconoció nada."""
    assert len(run().words) == 3


def test_text_is_rebuilt_by_lines():
    assert run().text == "LECHE ENTERA\n1L"


def test_line_numbers_are_unique_across_blocks():
    """Sin combinar bloque, párrafo y línea, dos bloques distintos se pisarían."""
    data = {key: list(value) for key, value in TESSERACT_DATA.items()}
    data["block_num"] = [1, 1, 1, 2]

    with patch("pytesseract.image_to_data", return_value=data), \
         patch("pytesseract.get_languages", return_value=["spa"]):
        result = engine.run(IMAGE, "spa", "block", Transform())

    assert len({word.line_num for word in result.words}) == 2


def test_the_response_reports_the_engine_and_the_elapsed_time():
    result = run()

    assert result.engine == "tesseract"
    assert result.elapsed_ms >= 0


def test_the_response_reports_the_prepared_image_size():
    result = run(transform=Transform(min_width=800, max_width=1200))

    assert result.image_width == 800


@pytest.mark.parametrize(
    "layout,psm",
    [("auto", 3), ("block", 6), ("sparse", 11), ("line", 7), ("desconocido", 6)],
)
def test_layout_maps_to_the_page_segmentation_mode(layout, psm):
    with patch("pytesseract.image_to_data", return_value=TESSERACT_DATA) as mock, \
         patch("pytesseract.get_languages", return_value=["spa"]):
        engine.run(IMAGE, "spa", layout, Transform())

    assert mock.call_args.kwargs["config"] == f"--psm {psm}"


def test_the_requested_transform_reaches_opencv():
    """El giro que pide el coordinador tiene que llegar a la imagen."""
    with patch("pytesseract.image_to_data", return_value=TESSERACT_DATA) as mock, \
         patch("pytesseract.get_languages", return_value=["spa"]):
        engine.run(IMAGE, "spa", "block", Transform(rotate=90, rescale=False))

    prepared = mock.call_args.args[0]
    assert prepared.shape[:2] == IMAGE.shape[:2][::-1]


# ---------------------------------------------------------------------------
# Idioma
# ---------------------------------------------------------------------------

def test_the_requested_language_is_used_when_installed():
    with patch("pytesseract.get_languages", return_value=["spa", "eng"]):
        assert engine.resolve_lang("spa") == "spa"


def test_a_missing_language_falls_back_to_english():
    """Sin el paquete español el servicio degrada con aviso; no falla."""
    with patch("pytesseract.get_languages", return_value=["eng"]):
        assert engine.resolve_lang("spa") == "eng"


def test_without_any_language_list_the_request_is_respected():
    with patch("pytesseract.get_languages", side_effect=RuntimeError("boom")):
        assert engine.resolve_lang("spa") == "spa"


# ---------------------------------------------------------------------------
# Disponibilidad
# ---------------------------------------------------------------------------

def test_available_when_the_binary_answers():
    with patch("pytesseract.get_tesseract_version", return_value="5.5.0"):
        assert engine.is_available() is True


def test_unavailable_when_the_binary_is_missing():
    with patch("pytesseract.get_tesseract_version", side_effect=OSError("not found")):
        assert engine.is_available() is False


def test_version_degrades_gracefully():
    with patch("pytesseract.get_tesseract_version", side_effect=OSError("not found")):
        assert engine.version() == "desconocida"


def test_languages_degrade_gracefully():
    with patch("pytesseract.get_languages", side_effect=OSError("not found")):
        assert engine.languages() == []
