"""Background camera/inference and system-speech workers; no Tk calls."""
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import sys
import time
import subprocess
from types import SimpleNamespace


def put_latest(queue, value):
    try:
        queue.put_nowait(value)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put_nowait(value)


class CameraStream:
    """Reopen failed cameras with a backoff; only the camera thread owns capture."""
    def __init__(self, device=0, width=1050, height=740, retry_delay=1.0,
                 capture_factory=None, clock=time.monotonic):
        self.device, self.width, self.height = device, width, height
        self.retry_delay, self.clock = retry_delay, clock
        self.capture_factory = capture_factory or self._open
        self.capture = None
        self.next_attempt = 0.0

    @staticmethod
    def _open(device):
        import cv2 as cv
        backend = cv.CAP_AVFOUNDATION if sys.platform == "darwin" else cv.CAP_ANY
        return cv.VideoCapture(device, backend)

    def close(self):
        if self.capture is not None:
            try:
                self.capture.release()
            finally:
                self.capture = None

    def select(self, device):
        self.close()
        self.device = device
        self.next_attempt = 0.0

    def _fail(self, status):
        self.close()
        self.next_attempt = self.clock() + self.retry_delay
        return None, status

    def read(self):
        if self.capture is None and self.clock() < self.next_attempt:
            return None, "CAMERA OFFLINE / RETRYING"
        try:
            if self.capture is None:
                self.capture = self.capture_factory(self.device)
                if not self.capture.isOpened():
                    return self._fail(f"CAMERA {self.device} OFFLINE / CHECK ACCESS OR SELECT ANOTHER CAMERA")
                # OpenCV property IDs are stable; injected captures can implement set().
                self.capture.set(3, self.width)
                self.capture.set(4, self.height)
            ok, frame = self.capture.read()
            if not ok or frame is None:
                return self._fail("NO CAMERA FRAME / REOPENING")
            return frame, "CAMERA LIVE"
        except Exception as error:
            return self._fail(f"CAMERA ERROR / {error}")


@dataclass(frozen=True)
class CameraPacket:
    generation: int
    device: int
    created_at: float
    status: str
    image: object = None


class CameraWorker(Thread):
    def __init__(self, processor_factory, gate, device=0, width=1050, height=740,
                 capture_factory=None):
        super().__init__(name="camera-inference", daemon=True)
        self.processor_factory, self.gate = processor_factory, gate
        self.width, self.height = width, height
        self.capture_factory = capture_factory
        self.packets = Queue(maxsize=1)
        self.stop_event = Event()
        self._lock = Lock()
        self._device, self._generation = device, 0

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def select(self, device):
        with self._lock:
            self._device = device
            self._generation += 1
            self.gate.reset()

    def stop(self):
        self.stop_event.set()
        self.gate.reset()

    def run(self):
        stream = CameraStream(width=self.width, height=self.height,
                              capture_factory=self.capture_factory)
        processor = None
        active_generation = -1
        try:
            processor = self.processor_factory()
            while not self.stop_event.is_set():
                with self._lock:
                    device, generation = self._device, self._generation
                if generation != active_generation:
                    stream.select(device)
                    self.gate.reset()
                    active_generation = generation
                frame, status = stream.read()
                image = None
                if frame is None:
                    self.gate.reset()
                else:
                    try:
                        image = processor.process(frame)
                    except Exception as error:
                        self.gate.reset()
                        status = f"RECOGNITION ERROR / {error}"
                # Ignore inference from a camera selected before the latest request.
                with self._lock:
                    if generation != self._generation:
                        self.gate.reset()
                        continue
                    put_latest(self.packets, CameraPacket(generation, device, time.monotonic(), status, image))
                self.stop_event.wait(0.05 if image is None else 0.001)
        except Exception as error:
            self.gate.reset()
            with self._lock:
                put_latest(self.packets, CameraPacket(self._generation, self._device,
                           time.monotonic(), f"STARTUP ERROR / {error}"))
        finally:
            stream.close()
            if processor is not None:
                processor.close()


class MacSpeechEngine:
    """Use the native say process; pyttsx3's NSSS loop can return prematurely."""
    def __init__(self, stop_event, voices=None, default_voice=None, process_factory=subprocess.Popen):
        if voices is None:
            from AppKit import NSSpeechSynthesizer
            voices = []
            for voice_id in NSSpeechSynthesizer.availableVoices():
                attrs = NSSpeechSynthesizer.attributesForVoice_(voice_id)
                voices.append(SimpleNamespace(id=voice_id, name=attrs.get("VoiceName", voice_id)))
            default_voice = NSSpeechSynthesizer.alloc().initWithVoice_(None).voice()
        self.voices, self.voice = voices, default_voice
        self.stop_event, self.process_factory = stop_event, process_factory
        self.rate, self.text, self.process = 125, "", None

    def getProperty(self, key):
        return self.voices if key == "voices" else self.voice

    def setProperty(self, key, value):
        if key == "voice":
            self.voice = value
        elif key == "rate":
            self.rate = value

    def say(self, text):
        self.text = text

    def runAndWait(self):
        command = ["/usr/bin/say", "-r", str(self.rate)]
        if self.voice:
            command.extend(["-v", str(self.voice)])
        self.process = self.process_factory(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        input_text = self.text
        while True:
            if self.stop_event.is_set() and self.process.poll() is None:
                self.process.terminate()
            try:
                _, error = self.process.communicate(input=input_text, timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                input_text = None
        if self.process.returncode and not self.stop_event.is_set():
            raise RuntimeError(error.strip() or "Native speech command failed")
        self.process = None

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=2)


class SpeechWorker(Thread):
    def __init__(self, engine_factory=None):
        super().__init__(name="speech", daemon=True)
        self.engine_factory = engine_factory
        self.events = Queue()
        self.requests = Queue(maxsize=1)
        self.stop_event = Event()

    def speak(self, text, voice_id=None):
        if text.strip() and not self.stop_event.is_set():
            put_latest(self.requests, (text, voice_id))

    def stop(self):
        self.stop_event.set()

    def run(self):
        engine = None
        try:
            if self.engine_factory is None and sys.platform == "darwin":
                engine = MacSpeechEngine(self.stop_event)
            elif self.engine_factory is None:
                import pyttsx3
                engine = pyttsx3.init()
            else:
                engine = self.engine_factory()
            voices = [(voice.id, voice.name or voice.id)
                      for voice in (engine.getProperty("voices") or [])]
            self.events.put(("voices", voices, engine.getProperty("voice")))
            while not self.stop_event.is_set():
                try:
                    text, voice_id = self.requests.get(timeout=0.1)
                except Empty:
                    continue
                self.events.put(("status", "SPEAKING"))
                try:
                    if voice_id:
                        engine.setProperty("voice", voice_id)
                    engine.setProperty("rate", 125)
                    engine.say(text)
                    engine.runAndWait()
                    self.events.put(("status", "SPEECH READY"))
                except Exception as error:
                    self.events.put(("error", f"SPEECH ERROR / {error}"))
        except Exception as error:
            self.events.put(("error", f"SPEECH UNAVAILABLE / {error}"))
        finally:
            if engine is not None:
                engine.stop()
