"""
utils/config.py
===============

Propósito
---------
Centraliza la carga de configuración desde .env y prepara las rutas base.

Qué hace
--------
- Carga variables del archivo .env
- Define rutas del proyecto
- Expone parámetros reutilizables
- Crea carpetas base si no existen
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

RTSP_URL = os.getenv("RTSP_URL", "").strip()

RAW_VIDEO_DIR = BASE_DIR / os.getenv("RAW_VIDEO_DIR", "data/raw_videos")
FRAMES_DIR = BASE_DIR / os.getenv("FRAMES_DIR", "data/extracted_frames")
DATASET_DIR = BASE_DIR / os.getenv("DATASET_DIR", "data/roboflow_export")
RUNS_DIR = BASE_DIR / os.getenv("RUNS_DIR", "data/runs")
LOG_DIR = BASE_DIR / os.getenv("LOG_DIR", "logs")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt").strip()
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.45"))
PERSON_CONFIDENCE = float(os.getenv("PERSON_CONFIDENCE", "0.45"))
HELMET_CONFIDENCE = float(os.getenv("HELMET_CONFIDENCE", "0.65"))
VEST_CONFIDENCE = float(os.getenv("VEST_CONFIDENCE", "0.45"))
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "60"))
FRAME_EVERY_N = int(os.getenv("FRAME_EVERY_N", "15"))

#Parametros de estabilidad visual para Inferencia
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5")) #mantiene consistencia entre detecciones
SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.3")) #controla cuanto suavizar la caja
MAX_MISSING_FRAMES = int(os.getenv("MAX_MISSING_FRAMES", "3")) #evita que la caja desaparezca al primer fallo
MIN_BOX_AREA = int(os.getenv("MIN_BOX_AREA", "500")) #filtra cajas demasiado pequeñas y ruidosas
MAX_CENTER_DISTANCE = int(os.getenv("MAX_CENTER_DISTANCE", "120")) #asocia una caja aunque el IoU no alcance, usando cercanía del centro

for path in [RAW_VIDEO_DIR, FRAMES_DIR, DATASET_DIR, RUNS_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

#Connect Mysql
MYSQL_ENABLED = os.getenv("MYSQL_ENABLED", "false").lower() == "true"
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1").strip()
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "").strip()
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "").strip()
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "").strip()
MYSQL_CONNECT_TIMEOUT = int(os.getenv("MYSQL_CONNECT_TIMEOUT", "5"))
