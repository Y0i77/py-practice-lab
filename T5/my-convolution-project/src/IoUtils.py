"""io_utils.py
Funciones de carga y guardado de imágenes (Pillow + numpy).
"""
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import logging

LOGGER = logging.getLogger(__name__)


def load_image_rgb(path: Path) -> np.ndarray:
    """Carga una imagen y la devuelve como array RGB uint8 de shape (H, W, 3)."""
    LOGGER.debug("Cargando imagen desde %s", path)
    if not path.exists():
        raise FileNotFoundError(f"El archivo {path} no existe.")
    with Image.open(path) as img:
        img_conv = img.convert("RGB")
        arr = np.array(img_conv, dtype=np.uint8)
    LOGGER.debug("Imagen cargada con shape %s", arr.shape)
    return arr


def save_image_rgb(array: np.ndarray, path: Path) -> None:
    """Guarda un array RGB uint8 como imagen."""
    LOGGER.debug("Guardando imagen en %s", path)
    img = Image.fromarray(array.astype("uint8"), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    LOGGER.info("Imagen guardada en %s", path)
