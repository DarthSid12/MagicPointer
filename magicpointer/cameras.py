"""Lazy OpenCV capture with explicit resolution/FPS negotiation."""
from dataclasses import dataclass
import platform
import time


@dataclass
class Frame:
    image: object
    grabbed_at: float


class Cameras:
    def __init__(self, left=0, right=None, fps="auto", width=1920, height=1080):
        import cv2
        if left < 0 or (right is not None and (right < 0 or right == left)):
            raise ValueError("Camera indices must be nonnegative and different")
        self.cv = cv2
        self.captures = []
        self.info = []
        try:
            for index in ([left] if right is None else [left, right]):
                backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
                cap = cv2.VideoCapture(index, backend)
                self.captures.append(cap)
                if not cap.isOpened():
                    message = f"Cannot open camera {index}. "
                    if platform.system() == "Darwin":
                        message += (
                            "If macOS just requested camera access, choose Allow and rerun the command. "
                            "Open System Settings > Privacy & Security > Camera and enable the app "
                            "running Python (for example VS Code, Terminal, or iTerm). "
                            "Quit and reopen that app after changing permission. "
                            "If access is already enabled, check the camera index and connection."
                        )
                    else:
                        message += "Check its index and camera permissions."
                    raise ValueError(message)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 60 if fps == "auto" else int(fps))
                reported = cap.get(cv2.CAP_PROP_FPS)
                if fps == "auto" and reported < 55:
                    cap.set(cv2.CAP_PROP_FPS, 30)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ValueError(f"Camera {index} opened but returned no image")
                self.info.append({"index": index, "width": frame.shape[1],
                                  "height": frame.shape[0], "reported_fps": cap.get(cv2.CAP_PROP_FPS)})
        except BaseException:
            self.close()
            raise

    def read(self):
        times = []
        for cap in self.captures:
            if not cap.grab():
                raise ValueError("Camera stream stopped. Reconnect the camera and restart.")
            times.append(time.monotonic())
        frames = []
        for cap, timestamp in zip(self.captures, times):
            ok, image = cap.retrieve()
            if not ok or image is None:
                raise ValueError("Camera frame unavailable. Reconnect the camera and restart.")
            frames.append(Frame(image, timestamp))
        return frames

    def close(self):
        for capture in self.captures:
            capture.release()
        self.captures.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
