"""convolution.py
Implementación de la convolución 2D por canal (sin usar SciPy).
"""
from typing import Tuple
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


def pad_image_channel(channel: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Aplica padding reflectante a un canal 2D."""
    return np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")


def convolve2d_single_channel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolución 2D 'same' para un único canal.

    :param channel: array 2D (H, W)
    :param kernel: array 2D (kH, kW)
    :return: array 2D resultante (H, W)
    """
    if channel.ndim != 2:
        raise ValueError("channel debe ser un array 2D")
    k_h, k_w = kernel.shape
    LOGGER.debug("Convolviendo canal de shape %s con kernel %s", channel.shape, kernel.shape)

    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = pad_image_channel(channel, pad_h, pad_w)
    out = np.zeros_like(channel, dtype=np.float64)

    # rotar kernel 180° para la operación de correlación -> convolución
    kernel_rot = np.flipud(np.fliplr(kernel))

    # iterar (ineficiente pero simple y transparente para sonarQube)
    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            region = padded[i:i + k_h, j:j + k_w]
            out[i, j] = np.sum(region * kernel_rot)

    # recortar y normalizar rango 0-255
    out_clipped = np.clip(out, 0, 255)
    return out_clipped.astype(np.uint8)


def convolve_rgb_image(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Aplica la convolución por separado en cada canal RGB.

    :param image: array (H, W, 3) uint8
    :param kernel: array 2D
    :return: image procesada (H, W, 3) uint8
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image debe ser RGB con shape (H, W, 3)")
    LOGGER.debug("Aplicando convolución RGB a la imagen con shape %s", image.shape)
    channels = []
    for c in range(3):
        channels.append(convolve2d_single_channel(image[:, :, c], kernel))
    # stack y devolver como uint8
    result = np.stack(channels, axis=2)
    return result
