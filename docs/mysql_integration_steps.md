# Integración MariaDB/MySQL para el MVP EPP

## 1. Instalar dependencia Python
pip install mysql-connector-python

## 2. Copiar archivo nuevo
- mysql_event_repository.py -> utils/mysql_event_repository.py

## 3. Agregar variables al `.env`
MYSQL_ENABLED=true
MYSQL_HOST=192.168.20.243
MYSQL_PORT=3306
MYSQL_USER=deteccion_epp
MYSQL_PASSWORD=d373cc10n2K+1
MYSQL_DATABASE=deteccion_epp
MYSQL_CONNECT_TIMEOUT=5

## 4. Agregar variables en `utils/config.py`
MYSQL_ENABLED = os.getenv("MYSQL_ENABLED", "false").lower() == "true"
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1").strip()
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "").strip()
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "").strip()
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "").strip()
MYSQL_CONNECT_TIMEOUT = int(os.getenv("MYSQL_CONNECT_TIMEOUT", "5"))

## 5. Importar en el script principal
En scripts/06_infer_video_boxes_v4_zone_temporal_mysql_clean_draw.py agregar:

from utils.config import (
    MYSQL_ENABLED,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
    MYSQL_CONNECT_TIMEOUT,
)
from utils.mysql_event_repository import MySQLEventRepository

## 6. Inicializar repositorio dentro de main()
mysql_repo = None
if MYSQL_ENABLED:
    mysql_repo = MySQLEventRepository(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        connect_timeout=MYSQL_CONNECT_TIMEOUT,
    )

## 7. Guardar evento después de serializer.build_event(...)
if mysql_repo is not None:
    try:
        inserted = mysql_repo.save_event(event)
        logger.info(
            "MySQL save_event | event_id=%s | inserted=%s",
            event["event_id"],
            inserted,
        )

        if event["event_type"] == "violation_resolved":
            affected = mysql_repo.close_open_violation(
                camera_id=event["camera_id"],
                person_track_id=event["person_track_id"],
                scenario_id=event["scenario_id"],
                resolved_event_id=event["event_id"],
                resolved_at=event["event_confirmed_at"],
            )
            logger.info(
                "MySQL close_open_violation | event_id=%s | affected=%s",
                event["event_id"],
                affected,
            )
    except Exception as exc:
        logger.error(
            "Error guardando evento en MySQL | event_id=%s | error=%s",
            event["event_id"],
            exc,
        )

## 8. Crear tablas en la base ya existente
mysql -h 192.168.20.243 -u deteccion_epp -p deteccion_epp < mysql_schema_epp.sql

## 9. Validar inserts
mysql -h 192.168.20.243 -u deteccion_epp -p deteccion_epp

SELECT id, event_id, camera_id, person_track_id, event_type, status
FROM epp_events
ORDER BY id DESC
LIMIT 20;

SELECT id, event_id, image_full_path, image_crop_path, video_path
FROM epp_event_evidence
ORDER BY id DESC
LIMIT 20;