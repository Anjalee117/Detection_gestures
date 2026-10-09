"""Thread-safe confidence and consecutive-frame capture policy."""
from dataclasses import dataclass
from threading import RLock
import math
import time


@dataclass(frozen=True)
class Recognition:
    letter: str = ""
    confidence: float = 0.0
    frames: int = 0
    ready: bool = False
    status: str = "NO HAND"


class CaptureGate:
    def __init__(self, threshold=0.7, stable_frames=5, max_age=0.5, clock=time.monotonic):
        if not 0 <= threshold <= 1 or stable_frames < 1 or max_age <= 0:
            raise ValueError("Invalid capture threshold, frame count or maximum age")
        self.threshold = threshold
        self.stable_frames = stable_frames
        self.max_age = max_age
        self.clock = clock
        self._lock = RLock()
        self._state = Recognition()
        self._observed_at = 0.0

    def observe(self, letter=None, confidence=0.0):
        with self._lock:
            self._observed_at = self.clock()
            if not letter:
                self._state = Recognition()
            elif not math.isfinite(confidence) or confidence < self.threshold:
                self._state = Recognition(confidence=confidence, status="UNCERTAIN")
            else:
                frames = self._state.frames + 1 if letter == self._state.letter else 1
                frames = min(frames, self.stable_frames)
                ready = frames >= self.stable_frames
                self._state = Recognition(letter, confidence, frames, ready,
                                          "READY" if ready else "HOLD STEADY")
            return self._state

    def snapshot(self):
        with self._lock:
            if self.clock() - self._observed_at > self.max_age:
                self._state = Recognition()
            return self._state

    def capture(self):
        with self._lock:
            state = self.snapshot()
            if not state.ready:
                return ""
            self._state = Recognition()
            return state.letter

    def reset(self):
        with self._lock:
            self._state = Recognition()
            self._observed_at = 0.0
