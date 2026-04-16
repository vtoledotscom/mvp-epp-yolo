"""
scripts/07_train_yolo.py
========================

Propósito
---------
Entrenar un modelo YOLOv11 usando el dataset exportado desde Roboflow.

Mejoras de esta versión
-----------------------
- Verifica data.yaml
- Maneja excepción durante entrenamiento
- Logs más claros
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ultralytics import YOLO
from utils.config import DATASET_DIR, RUNS_DIR, LOG_DIR, YOLO_MODEL
from utils.logger import get_logger

logger = get_logger("train_yolo", str(LOG_DIR / "train_yolo.log"))


def main() -> None:
    """
    Entrena YOLOv11 usando configuración base del MVP.
    """
    data_yaml = DATASET_DIR / "data.yaml"

    if not data_yaml.exists():
        logger.error(f"No se encontró data.yaml en: {data_yaml}")
        return

    try:
        logger.info(f"Cargando modelo base: {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)

        logger.info("Iniciando entrenamiento YOLOv11")
        model.train(
            data=str(data_yaml),
            epochs=30,
            imgsz=640,
            batch=8,
            project=str(RUNS_DIR),
            name="epp_mvp_1",
            pretrained=True
        )
        logger.info("Entrenamiento finalizado correctamente")

    except Exception as exc:
        logger.error(f"Error durante entrenamiento: {exc}")


if __name__ == "__main__":
    main()