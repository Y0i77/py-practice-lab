#!/usr/bin/env python3
"""
onefile_convolution.py
Script "todo en uno" para aplicar la convolución RGB con el kernel
[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]].

- Funciona localmente (captura por webcam si pasas --capture y tienes OpenCV)
- Funciona en Google Colab si usas --colab y subes /content/photo.jpg
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Import cv2 solo si está disponible (evita fallos en Colab donde la webcam no funciona)
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Kernel solicitado
DEFAULT_KERNEL = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float64)

# Logging básico
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("onefile_convolution")


def load_image_rgb(path: Path) -> np.ndarray:
    """Carga una imagen y devuelve array RGB uint8 (H, W, 3)."""
    if not path.exists():
        raise FileNotFoundError(f"El archivo {path} no existe.")
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
    LOGGER.debug("Imagen cargada %s -> shape=%s", path, arr.shape)
    return arr


def save_image_rgb(array: np.ndarray, path: Path) -> None:
    """Guarda array RGB uint8 como imagen (crea directorios si es necesario)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype("uint8"), mode="RGB").save(path)
    LOGGER.info("Imagen guardada en %s", path)


def pad_image_channel(channel: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Padding reflectante para un canal 2D."""
    return np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")


def convolve2d_single_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Realiza convolución 'same' en un canal 2D (resultado uint8)."""
    if channel.ndim != 2:
        raise ValueError("channel debe ser 2D")
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = pad_image_channel(channel, pad_h, pad_w)
    out = np.zeros_like(channel, dtype=np.float64)
    kernel_rot = np.flipud(np.fliplr(kernel))
    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            region = padded[i:i + k_h, j:j + k_w]
            out[i, j] = float(np.sum(region * kernel_rot))
    out_clipped = np.clip(out, 0, 255)
    return out_clipped.astype(np.uint8)


def convolve_rgb_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Aplica la convolución por separado en cada canal RGB."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image debe ser RGB con shape (H, W, 3)")
    LOGGER.debug("Aplicando convolución RGB a imagen shape=%s", image.shape)
    channels = [convolve2d_single_channel(image[:, :, c], kernel) for c in range(3)]
    return np.stack(channels, axis=2)


def capture_image_local() -> Optional[np.ndarray]:
    """Captura una imagen desde la webcam local y devuelve la imagen en RGB.
    Requiere OpenCV y una cámara disponible. Devuelve None si falla."""
    if not _HAS_CV2:
        LOGGER.error("OpenCV no está disponible en este entorno.")
        return None
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        LOGGER.error("No se pudo abrir la cámara.")
        return None
    try:
        LOGGER.info("Se abrirá la cámara. Presiona 's' para capturar o 'q' para salir.")
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                LOGGER.error("No se obtuvo frame de la cámara.")
                return None
            cv2.imshow("Presiona 's' para guardar, 'q' para salir", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                LOGGER.info("Imagen capturada desde webcam.")
                return frame_rgb
            if key == ord("q"):
                LOGGER.info("Captura cancelada por el usuario.")
                return None
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplicar convolución RGB con kernel de realce (edge).")
    parser.add_argument("--input", type=str, default="data/photo.jpg", help="Ruta de imagen de entrada.")
    parser.add_argument("--output", type=str, default="outputs/result.png", help="Ruta de imagen de salida.")
    parser.add_argument("--capture", action="store_true", help="Capturar imagen desde webcam (solo local).")
    parser.add_argument("--colab", action="store_true", help="Modo Colab: no intenta abrir webcam y espera /content/photo.jpg.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.colab:
        # En Colab, el usuario debe subir su foto a /content/photo.jpg
        # o indicar otra ruta en --input
        LOGGER.info("Modo Colab activado. Asegúrate de subir la imagen a /content/photo.jpg o pasar --input.")
    if args.capture:
        # Intentar captura local
        image = capture_image_local()
        if image is None:
            LOGGER.error("No se capturó imagen desde la webcam. Saliendo.")
            return
    else:
        # Cargar desde archivo
        try:
            image = load_image_rgb(input_path)
        except FileNotFoundError as exc:
            LOGGER.error("No se encontró la imagen en %s: %s", input_path, exc)
            return

    # Procesar
    try:
        result = convolve_rgb_image(image, DEFAULT_KERNEL)
    except Exception as exc:
        LOGGER.exception("Error aplicando la convolución: %s", exc)
        return

    # Guardar
    try:
        save_image_rgb(result, output_path)
    except Exception as exc:
        LOGGER.exception("Error guardando la imagen: %s", exc)
        return

    LOGGER.info("Proceso finalizado. Resultado en %s", output_path)


if __name__ == "__main__":
    main()
