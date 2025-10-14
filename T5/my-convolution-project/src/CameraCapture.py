"""camera.py
Captura una imagen desde la webcam.
"""
import logging
from typing import Optional

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


def capture_image(window_name: str = "Presiona 's' para guardar, 'q' para salir") -> Optional[np.ndarray]:
    """Captura una imagen de la webcam.

    Abre la webcam y devuelve la imagen capturada en formato RGB (uint8).
    Devuelve None si el usuario cancela (presiona 'q').

    :param window_name: texto para la ventana.
    :return: imagen RGB como numpy.ndarray o None si se canceló.
    """
    LOGGER.debug("Inicializando captura de webcam.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        LOGGER.error("No se pudo abrir la cámara.")
        return None

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                LOGGER.error("No se obtuvo frame de la cámara.")
                break

            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            # 's' para guardar la foto
            if key == ord("s"):
                # convertir BGR (OpenCV) a RGB (más estándar)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                LOGGER.info("Imagen capturada con éxito.")
                return frame_rgb
            # 'q' para salir sin guardar
            if key == ord("q"):
                LOGGER.info("Captura cancelada por el usuario.")
                return None
    finally:
        cap.release()
        cv2.destroyAllWindows()
