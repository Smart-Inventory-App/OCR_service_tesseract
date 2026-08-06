"""Control de acceso entre servicios.

Este servicio no valida los JWT de los usuarios: nunca lo llama la App. Su único
cliente es `ocr_service`, que ya autenticó al usuario antes de delegar el
reconocimiento. Lo que se protege acá es que nadie más de la red interna pueda
usar el motor como servicio de OCR gratuito, y para eso alcanza una clave
compartida.
"""

from fastapi import Header, HTTPException, status

from app.core.config import settings

HEADER_NAME = "X-Engine-Key"


async def verify_engine_key(x_engine_key: str | None = Header(default=None)) -> None:
    """Valida la clave compartida. Si no hay clave configurada, no exige nada."""
    if not settings.ENGINE_API_KEY:
        return

    if x_engine_key != settings.ENGINE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Falta o es inválida la cabecera {HEADER_NAME}",
        )
