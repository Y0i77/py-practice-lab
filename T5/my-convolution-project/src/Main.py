"""main.py
Entrypoint: captura o carga imagen, aplica convolución y guarda resultado.
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from CameraCapture import capture_image
from IoUtils import load_image_rgb, save_image_rgb
from Convolution import convolve_rgb_image

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Kernel solicitado
DEFAULT_KERNEL = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tomar selfie / cargar imagen y aplicar convolución RGB.")
    parser.add_argument("--capture", action="store_true", help="Capturar imagen desde la webcam (presiona 's' para guardar).")
    parser.add_argument("--input", type=str, default="data/photo.jpg", help="Ruta a la imagen de entrada si no usa webcam.")
    parser.add_argument("--output", type=str, default="outputs/result.png", help="Ruta donde guardar la imagen resultante.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Inicio del proceso.")
    image = None

    if args.capture:
        LOGGER.info("Se abrirá la webcam. Presiona 's' para capturar o 'q' para cancelar.")
        image = capture_image()
        if image is None:
            LOGGER.info("No se capturó ninguna imagen, saliendo.")
            return

    else:
        input_path = Path(args.input)
        try:
            image = load_image_rgb(input_path)
        except FileNotFoundError as exc:
            LOGGER.error("No se encontró la imagen: %s", exc)
            return

    # aplicar convolución
    try:
        result = convolve_rgb_image(image, DEFAULT_KERNEL)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Error al aplicar la convolución: %s", exc)
        return

    # guardar resultado
    output_path = Path(args.output)
    save_image_rgb(result, output_path)
    LOGGER.info("Proceso finalizado. Resultado guardado en %s", output_path)


if __name__ == "__main__":
    main()
