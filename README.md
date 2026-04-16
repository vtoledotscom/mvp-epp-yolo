# 🦺 MVP Detección de EPP con YOLO

## 📌 Descripción

Este proyecto corresponde a un MVP (Minimum Viable Product) para la detección automática de uso de Elementos de Protección Personal (EPP) mediante visión computacional usando YOLO.

El sistema permite analizar video (archivo o RTSP) para:

- Detectar personas
- Detectar casco y chaleco
- Evaluar cumplimiento según reglas de negocio
- Aplicar zonas de inspección
- Reducir falsos positivos con histéresis temporal
- Generar evidencia visual
- Persistir eventos en MariaDB/MySQL

---

## 🎯 Objetivo

Detectar automáticamente incumplimientos de seguridad en faenas (ej: minería), generando eventos auditables con evidencia.

---

## 🧠 Casos de uso soportados

El sistema soporta múltiples escenarios configurables:

- ✅ Casco y chaleco obligatorios  
- ✅ Solo casco obligatorio  
- ✅ Solo chaleco obligatorio  

Configuración basada en:

- `config/scenarios.json`
- `config/cameras.json`

---

## 🏗️ Arquitectura del sistema
YOLO (detección)
↓
ppe_associator (asociación persona ↔ EPP)
↓
business_rules (reglas de negocio por escenario)
↓
compliance_state_manager (histéresis temporal)
↓
evidence_manager (captura evidencia)
↓
event_serializer (estructura del evento)
↓
MySQL (persistencia)

---

## ⚙️ Componentes principales

### 🔹 YOLO
Detecta:
- persona
- casco
- chaleco

---

### 🔹 ppe_associator.py
Asocia:
- casco → cabeza
- chaleco → torso

---

### 🔹 business_rules.py
Evalúa cumplimiento según escenario:

Ejemplo:
```json
{
  "helmet_required": true,
  "vest_required": false
}

🔹 compliance_state_manager.py
    Evita falsos positivos mediante:
    confirmación por múltiples frames
    detección de cambios de estado

🔹 evidence_manager.py
    Genera:
        imagen completa
        imagen recortada
        video

🔹 event_serializer_mysql.py
    Construye el evento final listo para persistencia.

🔹 mysql_event_repository.py
    Inserta datos en:
        epp_events
        epp_event_evidence

📁 Estructura del proyecto
mvp-epp-yolo/
├── config/
│   ├── cameras.json
│   └── scenarios.json
├── scripts/
│   └── 06_infer_video.py
├── utils/
│   ├── business_rules.py
│   ├── compliance_state_manager.py
│   ├── ppe_associator.py
│   ├── evidence_manager_v2.py
│   ├── event_serializer_mysql.py
│   ├── mysql_event_repository.py
│   └── config.py
├── sql/
│   └── mysql_schema_epp.sql
├── logs/
├── evidence/
├── data/
├── .env.example
├── .gitignore
└── README.md

🚀 Instalación
    1. Clonar repositorio
        git clone https://github.com/TU_USUARIO/TU_REPO.git
        cd TU_REPO
    2. Crear entorno virtual
        python3 -m venv venv
        source venv/bin/activate
    3. Instalar dependencias
        pip install -r requirements.txt

🔧 Configuración
    1. Variables de entorno
        cp .env.example .env
        Editar .env según tu entorno.

    2. Configurar escenarios
        Archivo:
            config/scenarios.json
            Ejemplo:
                    {
                        "scenarios": {
                            "helmet_required": {
                            "helmet_required": true,
                            "vest_required": false
                            }
                        }
                    }
    3. Configurar cámaras
        Archivo:
            config/cameras.json
            Ejemplo:
            {
                "cameras": {
                    "cam_rtsp_002": {
                    "scenario_id": "helmet_required"
                    }
                }
            }
▶️ Ejecución
Modo básico
        python3 scripts/06_infer_video.py
        Forzando escenario

🧪 Flujo de detección
El sistema:
        Detecta objetos con YOLO
        Asocia EPP a cada persona
        Aplica reglas de negocio
        Evalúa estado temporal (histéresis)
        Confirma evento
        Genera evidencia
        Guarda en base de datos

🗄️ Base de datos (MariaDB/MySQL)
Crear tablas
    mysql -u USER -p DB_NAME < sql/mysql_schema_epp.sql

Tablas principales
    epp_events
    Contiene:
        evento
        escenario
        estado
        violaciones
    
    epp_event_evidence
    Contiene:
        rutas de imágenes
        rutas de video

📊 Tipos de eventos
    violation_started → inicio de incumplimiento
    violation_resolved → fin de incumplimiento

⚠️ Importante
El sistema NO guarda todos los frames.
    Solo guarda cuando:
        se confirma una infracción
        se resuelve una infracción

🧪 Debug
Ver logs en tiempo real:
    tail -f logs/infer_video.log

❗ Problemas comunes
    Escenario no cambia
    Revisar:
        .env
        config/cameras.json
        variable CAMERA_ID

    No se insertan datos
    Revisar:
        MYSQL_ENABLED=true
        conexión DB
        tablas creadas
    
    Falsos positivos
    Ajustar:
        min_non_compliant_frames
        min_zone_overlap

🚫 Archivos excluidos del repo
    .env
    credenciales
    videos
    evidencia generada
    modelos .pt