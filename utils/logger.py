"""
utils/logger.py
===============

Propósito
---------
Centraliza la creación de logs del proyecto y protege información sensible
como usuario y contraseña de la URL RTSP.

Qué hace
--------
- Crea loggers reutilizables.
- Escribe logs tanto en archivo como en terminal.
- Oculta credenciales RTSP cuando se muestran en logs.

Uso
---
from utils.logger import get_logger, mask_rtsp_url
logger = get_logger("mi_script", "logs/mi_script.log")
logger.info("Mensaje")
"""

from pathlib import Path
import logging
import sys
import re


def mask_rtsp_url(url: str) -> str:
    """
    Oculta usuario y contraseña de una URL RTSP para uso en logs.
    """
    if not url:
        return url

    pattern = r"(rtsp://)([^:]+):([^@]+)@"
    return re.sub(pattern, r"\1****:****@", url)


def get_logger(name: str, log_file: str) -> logging.Logger:
    """
    Crea y devuelve un logger configurado para archivo y terminal.
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger