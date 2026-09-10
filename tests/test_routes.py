"""Endpoints del servicio de motor.

El contrato tiene que ser exactamente el mismo que el de `ocr_service_paddleocr`:
es lo que le permite al coordinador tratar a los dos motores igual.
"""

import json
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.schemas import Transform
from tests.conftest import render_image

TESSERACT_DATA = {
    "text": ["LECHE"], "conf": ["96"], "left": [10], "top": [10],
    "width": [100], "height": [30], "block_num": [1], "par_num": [1], "line_num": [1],
}


def files(content_type: str = "image/jpeg"):
    return {"file": ("scan.jpg", render_image(), content_type)}


def mocked_tesseract():
    return patch.multiple(
        "pytesseract",
        image_to_data=lambda *a, **k: TESSERACT_DATA,
        get_languages=lambda *a, **k: ["spa", "eng"],
        get_tesseract_version=lambda *a, **k: "5.5.0",
    )


# ---------------------------------------------------------------------------
# /engine/run
# ---------------------------------------------------------------------------

async def test_run_returns_words_with_confidence(client):
    with mocked_tesseract():
        r = await client.post("/engine/run", files=files(), data={"lang": "spa"})

    body = r.json()

    assert r.status_code == 200
    assert body["engine"] == "tesseract"
    assert body["words"][0]["text"] == "LECHE"
    assert body["words"][0]["confidence"] == pytest.approx(0.96)


async def test_run_accepts_the_transform_as_json(client):
    spec = Transform(rotate=90, binarize="adaptive").model_dump(mode="json")

    with mocked_tesseract():
        r = await client.post(
            "/engine/run", files=files(), data={"transform": json.dumps(spec)}
        )

    assert r.status_code == 200


async def test_run_without_a_transform_uses_the_defaults(client):
    with mocked_tesseract():
        r = await client.post("/engine/run", files=files())

    assert r.status_code == 200


async def test_run_rejects_a_malformed_transform(client):
    r = await client.post("/engine/run", files=files(), data={"transform": "{no es json"})

    assert r.status_code == 400
    assert "transform" in r.json()["detail"]


async def test_run_rejects_an_unsupported_format(client):
    r = await client.post("/engine/run", files=files("application/pdf"))

    assert r.status_code == 400


async def test_run_rejects_an_empty_file(client):
    r = await client.post("/engine/run", files={"file": ("x.jpg", b"", "image/jpeg")})

    assert r.status_code == 400


async def test_run_rejects_something_that_is_not_an_image(client):
    r = await client.post(
        "/engine/run", files={"file": ("x.jpg", b"no soy una imagen", "image/jpeg")}
    )

    assert r.status_code == 400


async def test_run_rejects_an_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10)

    r = await client.post("/engine/run", files=files())

    assert r.status_code == 413


async def test_run_reports_an_engine_failure_as_500(client):
    with patch("app.engine.run", side_effect=RuntimeError("el motor falló")):
        r = await client.post("/engine/run", files=files())

    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Metadatos y salud
# ---------------------------------------------------------------------------

async def test_info_declares_that_preprocessing_helps(client):
    """Es el dato con el que el coordinador decide si le gasta variantes."""
    with mocked_tesseract():
        body = (await client.get("/engine/info")).json()

    assert body["engine"] == "tesseract"
    assert body["supports_preprocessing"] is True


async def test_livez_has_no_dependencies(client):
    r = await client.get("/engine/livez")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_readyz_is_ok_when_the_binary_and_the_language_are_there(client):
    with mocked_tesseract():
        body = (await client.get("/engine/readyz")).json()

    assert body["status"] == "ok"


async def test_readyz_reports_error_without_the_binary(client):
    with patch("app.engine.is_available", return_value=False):
        body = (await client.get("/engine/readyz")).json()

    assert body["status"] == "error"


async def test_readyz_reports_degraded_without_the_language_pack(client):
    """Sin el paquete español el motor cae a inglés: sirve, pero hay que verlo."""
    with patch("app.engine.is_available", return_value=True), \
         patch("app.engine.languages", return_value=["eng"]):
        body = (await client.get("/engine/readyz")).json()

    assert body["status"] == "degraded"


# ---------------------------------------------------------------------------
# Clave compartida
# ---------------------------------------------------------------------------

async def test_without_a_configured_key_no_header_is_required(client):
    with mocked_tesseract():
        r = await client.post("/engine/run", files=files())

    assert r.status_code == 200


async def test_a_configured_key_is_enforced(client, monkeypatch):
    monkeypatch.setattr(settings, "ENGINE_API_KEY", "s3cr3t")

    r = await client.post("/engine/run", files=files())

    assert r.status_code == 401


async def test_the_right_key_is_accepted(client, monkeypatch):
    monkeypatch.setattr(settings, "ENGINE_API_KEY", "s3cr3t")

    with mocked_tesseract():
        r = await client.post(
            "/engine/run", files=files(), headers={"X-Engine-Key": "s3cr3t"}
        )

    assert r.status_code == 200
