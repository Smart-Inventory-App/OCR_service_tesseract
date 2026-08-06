from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8005
    DEBUG: bool = False

    # Binario de Tesseract. Vacío = usar el del PATH.
    TESSERACT_CMD: str = ""
    DEFAULT_LANG: str = "spa"

    # Clave compartida con el coordinador (`ocr_service`). Vacía = sin control de
    # acceso, que es lo razonable cuando el servicio sólo escucha en la red
    # interna de Docker y no está publicado hacia afuera.
    ENGINE_API_KEY: str = ""

    # Tope de tamaño del binario recibido, alineado con el del coordinador.
    MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024


settings = Settings()
