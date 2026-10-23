"""Native primary-display input. Imported only for real stereo control."""
import ctypes
import math
import platform
import threading
import time

# Approximate macOS pixels per scroll line for continuous (pixel-unit) scrolling.
MAC_LINE_PIXELS = 12


class Desktop:
    def __init__(self):
        self.system = platform.system()
        if self.system == "Windows":
            from ctypes import wintypes
            self.user32 = ctypes.WinDLL("user32", use_last_error=True)
            # Work in physical coordinates, including displays with >100% scaling.
            try:
                ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
            except (OSError, AttributeError):
                self.user32.SetProcessDPIAware()
            self.width, self.height = self.user32.GetSystemMetrics(0), self.user32.GetSystemMetrics(1)
            self.offset = (0, 0)

            class MouseInput(ctypes.Structure):
                _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                            ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                            ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

            class KeyboardInput(ctypes.Structure):
                _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                            ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                            ("dwExtraInfo", ctypes.c_size_t)]

            class HardwareInput(ctypes.Structure):
                _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

            class InputUnion(ctypes.Union):
                _fields_ = [("mi", MouseInput), ("ki", KeyboardInput), ("hi", HardwareInput)]

            class Input(ctypes.Structure):
                _anonymous_ = ("u",)
                _fields_ = [("type", wintypes.DWORD), ("u", InputUnion)]

            self.Input, self.MouseInput = Input, MouseInput
            self.user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int)
            self.user32.SendInput.restype = wintypes.UINT
        elif self.system == "Darwin":
            try:
                import Quartz
            except ImportError as error:
                raise ValueError("macOS control needs: pip install -e '.[macos]'") from error
            self.quartz = Quartz
            if not Quartz.CGPreflightPostEventAccess():
                raise ValueError("Enable Accessibility access for your terminal/Python in macOS System Settings, then restart")
            bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            self.width, self.height = bounds.size.width, bounds.size.height
            self.offset = (bounds.origin.x, bounds.origin.y)
        else:
            raise ValueError("Desktop control supports Windows and optional macOS. Use --debug for camera preview.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Cannot determine primary display size")

    def pixel(self, point):
        return (self.offset[0] + round(point[0] * (self.width - 1)),
                self.offset[1] + round(point[1] * (self.height - 1)))

    @property
    def wheel_step(self):
        """Smallest scroll increment, in lines: one wheel unit or one macOS pixel."""
        return 1 / 120 if self.system == "Windows" else 1 / MAC_LINE_PIXELS

    def move(self, point):
        x, y = self.pixel(point)
        if self.system == "Windows":
            if not self.user32.SetCursorPos(int(x), int(y)):
                raise OSError("Windows could not move the cursor")
        else:
            q = self.quartz
            q.CGEventPost(q.kCGHIDEventTap, q.CGEventCreateMouseEvent(None, q.kCGEventMouseMoved, (x, y), q.kCGMouseButtonLeft))

    def _send(self, flags, data=0):
        item = self.Input(type=0, mi=self.MouseInput(0, 0, data & 0xffffffff, flags, 0, 0))
        if self.user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(item)) != 1:
            raise OSError("Windows rejected pointer input (check app privileges)")

    def click(self, button):
        if self.system == "Windows":
            down, up = (0x0002, 0x0004) if button == "left" else (0x0008, 0x0010)
            try:
                self._send(down)
            finally:
                self._send(up)
        else:
            q = self.quartz
            pos = q.CGEventGetLocation(q.CGEventCreate(None))
            down, up, code = ((q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp, q.kCGMouseButtonLeft)
                              if button == "left" else (q.kCGEventRightMouseDown, q.kCGEventRightMouseUp, q.kCGMouseButtonRight))
            try:
                q.CGEventPost(q.kCGHIDEventTap, q.CGEventCreateMouseEvent(None, down, pos, code))
            finally:
                q.CGEventPost(q.kCGHIDEventTap, q.CGEventCreateMouseEvent(None, up, pos, code))

    def scroll(self, lines):
        if self.system == "Windows":
            self._send(0x0800, int(lines) * 120)
        else:
            q = self.quartz
            q.CGEventPost(q.kCGHIDEventTap, q.CGEventCreateScrollWheelEvent(None, q.kCGScrollEventUnitLine, 1, int(lines)))

    def scroll_fine(self, lines):
        """Fractional scroll: high-resolution wheel units on Windows, pixels on macOS."""
        if self.system == "Windows":
            self._send(0x0800, round(lines * 120))
        else:
            q = self.quartz
            q.CGEventPost(q.kCGHIDEventTap, q.CGEventCreateScrollWheelEvent(
                None, q.kCGScrollEventUnitPixel, 1, round(lines * MAC_LINE_PIXELS)))


class SmoothDesktop:
    """Glides the OS pointer between vision frames and spreads wheel steps over time.

    Vision runs at 10-30 Hz, so moving the pointer only on each frame looks
    jerky. A 120 Hz thread eases the pointer toward the latest target with a
    time constant of half the observed frame interval, and drains scroll lines
    as small wheel increments. Clicks first land the pointer exactly on target.
    """

    rate = 120.

    def __init__(self, native, *, smooth_scroll=True, clock=time.monotonic, start=True):
        self.native, self.smooth_scroll, self.clock = native, smooth_scroll, clock
        self.lock = threading.Lock()
        self.target = self.position = self.last_target = None
        self.interval = 1 / 30
        self.pending = self.carry = 0.
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True) if start else None
        if self.thread:
            self.thread.start()

    def _raise_pending(self):
        if self.error is not None:
            error, self.error = self.error, None
            raise error

    def move(self, point):
        self._raise_pending()
        with self.lock:
            now = self.clock()
            if self.last_target is not None:
                self.interval += .2 * (min(.2, now - self.last_target) - self.interval)
            self.last_target, self.target = now, tuple(point)
            if self.position is None:
                self.position = self.target
                self.native.move(self.position)

    def click(self, button):
        self._raise_pending()
        with self.lock:
            if self.target is not None and self.position != self.target:
                self.position = self.target
                self.native.move(self.position)
            self.native.click(button)

    def scroll(self, lines):
        self._raise_pending()
        if not self.smooth_scroll:
            self.native.scroll(lines)
            return
        with self.lock:
            self.pending += lines

    def tick(self, dt):
        with self.lock:
            if self.target is not None and self.position != self.target:
                tau = min(.06, max(.008, .5 * self.interval))
                alpha = 1 - math.exp(-max(dt, 0.) / tau)
                position = tuple(p + alpha * (t - p) for p, t in zip(self.position, self.target))
                if max(abs(t - p) for t, p in zip(self.target, position)) < 1e-4:
                    position = self.target
                if self.native.pixel(position) != self.native.pixel(self.position):
                    self.native.move(position)
                self.position = position
            if self.pending:
                drained = self.pending if abs(self.pending) < .02 else self.pending * (1 - math.exp(-max(dt, 0.) / .08))
                self.pending -= drained
                self.carry += drained
                step = self.native.wheel_step
                units = math.trunc(self.carry / step)
                if units:
                    self.native.scroll_fine(units * step)
                    self.carry -= units * step

    def _run(self):
        last = self.clock()
        while not self.stopped.wait(1 / self.rate):
            now = self.clock()
            try:
                self.tick(now - last)
            except OSError as error:
                self.error = error  # Re-raised by the next main-loop input call.
            last = now

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=1.)
