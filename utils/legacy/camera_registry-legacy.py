import json
from pathlib import Path


class CameraRegistry:
    """
    Registro de cámaras compatible con dos formatos de cameras.json:

    Formato 1:
    {
      "cameras": {
        "cam_rtsp_001": {...},
        "cam_rtsp_002": {...}
      }
    }

    Formato 2:
    {
      "cameras": [
        {"camera_id": "cam_rtsp_001", ...},
        {"camera_id": "cam_rtsp_002", ...}
      ]
    }
    """

    def __init__(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No existe cameras.json: {path}")

        self.data = json.loads(p.read_text(encoding="utf-8"))
        raw_cameras = self.data.get("cameras", {})

        if isinstance(raw_cameras, dict):
            self._index = {
                camera_id: {"camera_id": camera_id, **camera_data}
                for camera_id, camera_data in raw_cameras.items()
            }
        elif isinstance(raw_cameras, list):
            self._index = {}
            for item in raw_cameras:
                camera_id = item.get("camera_id")
                if not camera_id:
                    raise ValueError("Cada cámara en formato lista debe incluir camera_id.")
                self._index[camera_id] = item
        else:
            raise ValueError("El nodo 'cameras' debe ser un dict o una lista.")

    def get_camera(self, camera_id: str) -> dict:
        if camera_id not in self._index:
            raise KeyError(f"camera_id no encontrado: {camera_id}")
        return self._index[camera_id]

    def get_enabled_cameras(self):
        result = []
        for camera in self._index.values():
            if camera.get("enabled", True):
                result.append(camera)
        return result