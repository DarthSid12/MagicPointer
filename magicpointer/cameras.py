"""Lazy OpenCV capture with explicit resolution/FPS negotiation.

Each camera is read continuously on its own thread, so the processing loop
always receives the newest frame instead of waiting on, or lagging behind, the
driver's buffer. Brief stream failures reopen the device in the background.
"""
from dataclasses import dataclass
import platform
import threading
import time


@dataclass
class Frame:
    image: object
    grabbed_at: float
    sequence: int = 0


class _Stream:
    """Keeps the newest frame of one capture; reopens it after brief failures."""

    # Reopen after this long without a frame; give up after `give_up` seconds.
    reopen_after, give_up, min_interval = .5, 5., .002

    def __init__(self, cap, reopen):
        self.cap, self.reopen = cap, reopen
        self.condition = threading.Condition()
        self.latest, self.sequence, self.error = None, 0, None
        self.stopped = False
        self.thread = None

    def start(self, first=None):
        if first is not None:
            self._publish(first)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _publish(self, image):
        with self.condition:
            self.sequence += 1
            self.latest = Frame(image, time.monotonic(), self.sequence)
            self.condition.notify_all()

    def _run(self):
        last_frame = last_open = time.monotonic()
        while not self.stopped:
            started = time.monotonic()
            ok, image = self.cap.read()
            if self.stopped:
                return
            now = time.monotonic()
            if ok and image is not None:
                self._publish(image)
                last_frame = now
            elif now - last_frame > self.give_up:
                with self.condition:
                    self.error = "Camera stream stopped. Reconnect the camera and restart."
                    self.condition.notify_all()
                return
            elif now - last_frame > self.reopen_after and now - last_open > self.reopen_after:
                last_open = now
                try:
                    self.cap = self.reopen(self.cap)
                except (ValueError, OSError):
                    pass
            # Real devices block in read(); this only stops a failing device spinning.
            time.sleep(max(0., self.min_interval - (time.monotonic() - started)))

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=2.)


class Cameras:
    # Wait this long for a newer frame from a lagging camera to form a tighter pair.
    pair_tolerance, pair_wait = .04, .06

    def __init__(self, left=0, right=None, fps="auto", width=1920, height=1080, *, backend="default"):
        import cv2
        if left < 0 or (right is not None and (right < 0 or right == left)):
            raise ValueError("Camera indices must be nonnegative and different")
        if width <= 0 or height <= 0:
            raise ValueError("Capture width and height must be positive")
        if backend not in ("default", "msmf"):
            raise ValueError("Camera backend must be default or msmf")
        system = platform.system()
        if backend != "default" and system != "Windows":
            raise ValueError("The Media Foundation camera backend requires Windows")
        self.backend_name = "msmf" if system == "Windows" else "system"
        self.api = cv2.CAP_MSMF if system == "Windows" else cv2.CAP_ANY
        self.cv = cv2
        self.fps, self.width, self.height = fps, width, height
        self.captures = []
        self.info = []
        self.streams = []
        self._first = []
        self._returned = []
        try:
            for index in ([left] if right is None else [left, right]):
                cap, frame = self._open(index)
                self._first.append(frame)
                self.info.append({"index": index, "backend": self.backend_name, "width": frame.shape[1],
                                  "height": frame.shape[0], "reported_fps": cap.get(cv2.CAP_PROP_FPS)})
        except BaseException:
            self.close()
            raise
        self.indices = [item["index"] for item in self.info]

    def _open(self, index):
        cv2 = self.cv
        cap = cv2.VideoCapture(index, self.api)
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
                if platform.system() == "Windows" and len(self.captures) > 1:
                    message += (f" Backend: {self.backend_name}; earlier camera is already open. "
                                "If both work individually, try lower --capture-width/--capture-height "
                                "at --fps 30 and check shared USB resources or camera drivers.")
            raise ValueError(message)
        # Let Media Foundation negotiate its format, matching the working B13 diagnostic.
        if self.backend_name != "msmf":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 60 if self.fps == "auto" else int(self.fps))
        reported = cap.get(cv2.CAP_PROP_FPS)
        if self.fps == "auto" and reported < 55:
            cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Camera {index} opened but returned no image")
        return cap, frame

    def _reopener(self, index):
        def reopen(old):
            old.release()
            if old in self.captures:
                self.captures.remove(old)
            cap, _ = self._open(index)
            return cap
        return reopen

    def _start(self):
        for cap, index, first in zip(list(self.captures), self.indices, self._first):
            stream = _Stream(cap, self._reopener(index))
            stream.start(first)
            self.streams.append(stream)
        self._returned = [0] * len(self.streams)

    def _wait(self, streams, newer_than, deadline):
        """Frames newer than the given sequence numbers, or None at the deadline."""
        frames = []
        for stream, sequence in zip(streams, newer_than):
            with stream.condition:
                while stream.error is None and stream.sequence <= sequence:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    stream.condition.wait(remaining)
                if stream.error:
                    raise ValueError(stream.error)
                frames.append(stream.latest)
        return frames

    def read(self):
        """Newest frame from every camera, each newer than the last one returned."""
        if not self.streams:
            self._start()
        frames = self._wait(self.streams, self._returned, time.monotonic() + _Stream.give_up + 1.)
        if frames is None:
            raise ValueError("Camera stream stopped. Reconnect the camera and restart.")
        if len(frames) == 2 and abs(frames[0].grabbed_at - frames[1].grabbed_at) > self.pair_tolerance:
            # Free-running cameras: give the older view one frame to catch up.
            older = 0 if frames[0].grabbed_at < frames[1].grabbed_at else 1
            newer = self._wait([self.streams[older]], [frames[older].sequence], time.monotonic() + self.pair_wait)
            if newer is not None:
                frames[older] = newer[0]
                other = self.streams[1 - older]
                with other.condition:
                    frames[1 - older] = other.latest
        self._returned = [frame.sequence for frame in frames]
        return frames

    def close(self):
        for stream in self.streams:
            stream.stop()
        self.streams.clear()
        for capture in self.captures:
            capture.release()
        self.captures.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
