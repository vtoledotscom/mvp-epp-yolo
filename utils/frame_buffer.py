from collections import deque


class FrameBuffer:
    """
    Buffer circular de frames para guardar segundos previos al evento.
    """

    def __init__(self, max_frames: int):
        self.frames = deque(maxlen=max_frames)

    def add_frame(self, frame):
        """
        Agrega un frame al buffer.
        """
        self.frames.append(frame.copy())

    def get_frames(self):
        """
        Retorna los frames almacenados.
        """
        return list(self.frames)

    def clear(self):
        """
        Limpia el buffer.
        """
        self.frames.clear()