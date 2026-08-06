# Motor OCR — Tesseract

Microservicio que hace **una sola cosa**: correr Tesseract sobre una imagen y devolver las palabras reconocidas con su caja y su confianza.

No sabe qué documento está mirando, no extrae campos, no calcula confianza por campo y no decide reintentos. Todo eso vive en [`ocr_service`](../ocr_service), que es el coordinador. Acá sólo se ejecuta lo que ese coordinador pide.

```
ocr_service  ──POST /engine/run──►  este servicio
             ◄── palabras + confianza ──┘
```

---

## Motivo de la Separación

Tesseract no es un paquete de pip: es un **binario del sistema** que hay que instalar en la imagen junto con su paquete de idioma. Mientras vivía dentro del coordinador, cualquier despliegue de la lógica de extracción arrastraba esa instalación —y, en el mismo contenedor, también PyTorch por el otro motor—.

Separándolos, el coordinador bajó de ~2,5 GB a ~770 MB de imagen y de ~1,3 GB a ~30 MB de memoria en ejecución, y cada motor se puede redesplegar, apagar o escalar por su cuenta.

---

## Qué Recibe y Qué Devuelve

### `POST /engine/run`

**Body:** `multipart/form-data`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `file` | archivo (`jpeg`/`png`/`webp`) | la imagen **original**, sin preprocesar |
| `lang` | string | idioma (`spa` por defecto) |
| `layout` | string | `auto` · `block` · `sparse` · `line` |
| `transform` | JSON | qué preparación aplicar antes de reconocer |

El `transform` es lo que hace que este servicio no tenga criterio propio: el coordinador arma la cadena y acá se ejecuta tal cual.

```jsonc
{
  "rotate": 90,           // 0, 90, 180 o 270
  "grayscale": true,
  "rescale": true,
  "min_width": 2400,
  "max_width": 3600,
  "invert": true,         // invierte si la imagen es mayormente oscura
  "contrast": true,       // CLAHE
  "denoise": false,
  "deskew": true,         // endereza desvíos chicos de la horizontal
  "binarize": "adaptive"  // "none" | "otsu" | "adaptive"
}
```

**Respuesta `200`:**

```jsonc
{
  "engine": "tesseract",
  "text": "LECHE ENTERA\n1 litro",
  "words": [
    { "text": "LECHE", "confidence": 0.96,
      "left": 10, "top": 10, "width": 100, "height": 30, "line_num": 10101 }
  ],
  "elapsed_ms": 412,
  "image_width": 2400,
  "image_height": 1350,
  "lang": "spa"
}
```

`words` es lo que importa: `image_to_data` da **confianza por palabra**, y sobre eso el coordinador calcula la confianza por campo que exige la propuesta.

**Errores:** `400` formato no soportado, archivo vacío, imagen ilegible o `transform` mal formado · `401` clave compartida inválida · `413` imagen demasiado grande · `500` el motor falló.

### `GET /engine/info`

```json
{ "engine": "tesseract", "version": "5.5.0",
  "languages": ["eng", "osd", "spa"], "supports_preprocessing": true }
```

`supports_preprocessing: true` es un dato de coordinación: le dice a `ocr_service` que a este motor **sí** le sirve que le varíen los filtros, así que vale la pena gastarle reintentos. El servicio de EasyOCR declara lo contrario.

### `GET /engine/livez` · `GET /engine/readyz`

- `livez` → `{ "status": "ok" }`. Sin dependencias.
- `readyz` → verifica que el binario se pueda ejecutar. Devuelve `degraded` (no
  `error`) si falta el paquete de idioma: el motor cae a inglés y sigue sirviendo.

---

## Cadena de OpenCV

Tesseract requiere preprocesamiento (OpenCV) para obtener buenos resultados en escenarios reales. Los pasos se aplican en este orden, y cuáles se encienden lo decide el coordinador:

| Paso | Qué hace | Para qué |
|------|----------|----------|
| `rotate` | gira 0/90/180/270 | las fotos del sello de vencimiento llegan de costado |
| `grayscale` | BGR → 1 canal | el color no aporta al reconocimiento de texto |
| `rescale` | lleva el ancho a la franja pedida | el texto útil de una foto de celular es diminuto |
| `invert` | invierte si la mediana es oscura | Tesseract espera texto oscuro sobre fondo claro |
| `contrast` | CLAHE | recupera texto mal iluminado sin quemar lo ya claro |
| `denoise` | `bilateralFilter` | quita ruido preservando bordes; en fotos nítidas resta |
| `deskew` | ángulo por `minAreaRect` y rotación | corrige fotos apenas inclinadas |
| `binarize` | Otsu o adaptativo | ver abajo |

**Sobre binarizar.** No hay una respuesta única, y por eso es un parámetro y no una decisión de este servicio:

- En una **etiqueta**, binarizar empeora. Tesseract 4/5 usa un motor LSTM que binariza internamente y rinde mejor sobre grises con buen contraste; el umbral adaptativo, además, vacía el interior de los trazos gruesos de los logos.
- En un **sello de vencimiento**, el umbral adaptativo es lo único que funciona. Es tipografía de matriz de puntos sobre superficie brillante, y sin él la fecha no se lee.

---

## Ejecución

### Docker (recomendada)

Es la única vía que trae Tesseract y el idioma español ya instalados. La red `tpp-net` es externa y compartida con el resto de los servicios:

```bash
docker network create tpp-net
```

```bash
cp .env.example .env
```

```bash
docker-compose up -d --build
```

```bash
curl http://localhost:8005/engine/readyz
```

### Local

**1. Instalar Tesseract** (no se instala con pip):

- **Windows:** instalador de [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), marcando el idioma **Spanish**. Después, en `.env`:

  ```env
  TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
  ```

- **Linux:** `sudo apt-get install -y tesseract-ocr tesseract-ocr-spa`
- **macOS:** `brew install tesseract tesseract-lang`

Comprobar el idioma con `tesseract --list-langs`: debe incluir `spa`.

**2. Dependencias y arranque:**

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.main:app --reload --port 8005
```

Swagger en `http://localhost:8005/docs` (sólo con `DEBUG=true`).

---

## Configuración

| Variable | Default | Para qué |
|----------|---------|----------|
| `PORT` | `8005` | puerto del servicio |
| `TESSERACT_CMD` | *(vacío)* | ruta al binario; vacío = el del `PATH` |
| `DEFAULT_LANG` | `spa` | idioma cuando la petición no lo especifica |
| `ENGINE_API_KEY` | *(vacía)* | clave compartida con el coordinador; vacía = sin control de acceso |
| `MAX_UPLOAD_BYTES` | `15728640` | tope del binario recibido |

**Sobre `ENGINE_API_KEY`.** Este servicio no valida los JWT de los usuarios: nunca lo llama la App, sólo `ocr_service`, que ya autenticó a quien pide. Lo que protege la clave es que nadie más de la red interna use el motor como servicio de OCR gratuito. Si se despliega sin publicar el puerto hacia afuera, se puede dejar vacía.

---

## Tests

```bash
pytest
```

**66 tests, 97 % de cobertura.** No requieren el binario de Tesseract: donde hace falta se mockea `pytesseract`. La cadena de OpenCV, en cambio, se ejercita de verdad —incluido el deskew sobre una imagen inclinada a propósito— porque es la parte cuyo comportamiento el coordinador da por sentado.

| Archivo | Qué cubre |
|---------|-----------|
| `test_prepare.py` | cada paso de OpenCV y la cadena completa |
| `test_engine.py` | traducción de `image_to_data` a palabras, layouts, idioma |
| `test_routes.py` | el contrato HTTP, validaciones, salud y clave compartida |
