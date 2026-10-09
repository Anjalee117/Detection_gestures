from queue import Queue
from types import SimpleNamespace
from threading import Event

from utils.recognition import CaptureGate
from utils.workers import CameraStream, CameraWorker, MacSpeechEngine, SpeechWorker, put_latest


class FakeCapture:
    def __init__(self, opened=True, frames=None):
        self.opened, self.frames = opened, iter(frames or [(True, "frame")])
        self.released = False
        self.properties = {}

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.properties[key] = value

    def read(self):
        return next(self.frames, (False, None))

    def release(self):
        self.released = True


def test_camera_open_failure_backs_off_then_recovers():
    now = [0.0]
    bad, good = FakeCapture(opened=False), FakeCapture()
    captures = iter([bad, good])
    stream = CameraStream(capture_factory=lambda _: next(captures), clock=lambda: now[0])
    assert stream.read()[0] is None and bad.released
    assert stream.read()[0] is None
    now[0] = 1
    assert stream.read()[0] == "frame"
    assert good.properties == {3: 1050, 4: 740}
    stream.close()
    assert good.released


def test_camera_read_failure_and_manual_selection_reopen():
    failed, good = FakeCapture(frames=[(False, None)]), FakeCapture()
    captures = iter([failed, good]); devices = []
    def open_camera(device):
        devices.append(device)
        return next(captures)
    stream = CameraStream(capture_factory=open_camera)
    assert stream.read()[0] is None and failed.released
    stream.select(2)
    assert stream.read()[0] == "frame" and devices == [0, 2]
    stream.close()


def test_camera_worker_reports_startup_error_without_crashing_ui():
    def fail():
        raise FileNotFoundError("missing model")
    worker = CameraWorker(fail, CaptureGate())
    worker.start()
    packet = worker.packets.get(timeout=2)
    worker.join(timeout=2)
    assert "STARTUP ERROR / missing model" == packet.status
    assert not worker.is_alive()


def test_camera_worker_processes_on_background_thread_and_closes_resources():
    import threading
    calls = []; ready = Event()
    capture = FakeCapture()
    class Processor:
        def process(self, frame):
            calls.append(threading.current_thread().name)
            ready.set()
            return frame
        def close(self):
            calls.append("closed")
    worker = CameraWorker(Processor, CaptureGate(), capture_factory=lambda _: capture)
    worker.start()
    assert ready.wait(2)
    worker.stop(); worker.join(timeout=2)
    assert calls[0] == "camera-inference" and calls[-1] == "closed"
    assert capture.released and not worker.is_alive()


def test_latest_frame_replaces_backlog():
    queue = Queue(maxsize=1)
    put_latest(queue, "old"); put_latest(queue, "new")
    assert queue.get_nowait() == "new"


def test_speech_is_asynchronous_uses_voice_id_and_stops():
    speaking, finish = Event(), Event()
    class Engine:
        def __init__(self):
            self.properties = {}; self.stopped = False
        def getProperty(self, key):
            return [SimpleNamespace(id="voice-id", name="Real installed name")] if key == "voices" else "voice-id"
        def setProperty(self, key, value):
            self.properties[key] = value
        def say(self, text):
            self.text = text
        def runAndWait(self):
            speaking.set()
            assert finish.wait(2)
        def stop(self):
            self.stopped = True
    engine = Engine(); worker = SpeechWorker(lambda: engine)
    worker.start()
    assert worker.events.get(timeout=2) == ("voices", [("voice-id", "Real installed name")], "voice-id")
    worker.speak("sentence", "voice-id")
    assert speaking.wait(2)  # Caller remains free while the engine is blocked.
    assert worker.events.get(timeout=2) == ("status", "SPEAKING")
    assert engine.properties["voice"] == "voice-id" and engine.text == "sentence"
    finish.set()
    assert worker.events.get(timeout=2) == ("status", "SPEECH READY")
    worker.stop(); worker.join(timeout=2)
    assert not worker.is_alive() and engine.stopped


def test_speech_engine_startup_error_is_reported():
    def fail():
        raise RuntimeError("no voice engine")
    worker = SpeechWorker(fail); worker.start()
    assert worker.events.get(timeout=2) == ("error", "SPEECH UNAVAILABLE / no voice engine")
    worker.join(timeout=2)


def test_native_speech_waits_and_passes_sentence_on_stdin():
    import subprocess
    calls = []
    class Process:
        returncode = 0
        def __init__(self):
            self.communications = 0
        def poll(self):
            return None if not self.communications else 0
        def communicate(self, input=None, timeout=None):
            calls.append((input, timeout))
            self.communications += 1
            if self.communications == 1:
                raise subprocess.TimeoutExpired("say", timeout)
            return "", ""
    command = []
    def start(args, **kwargs):
        command.extend(args)
        assert kwargs["stdin"] == subprocess.PIPE and "shell" not in kwargs
        return Process()
    engine = MacSpeechEngine(Event(), voices=[], default_voice="installed-id", process_factory=start)
    engine.say("A sentence with $() and `literal` characters")
    engine.runAndWait()
    assert command == ["/usr/bin/say", "-r", "125", "-v", "installed-id"]
    assert calls == [("A sentence with $() and `literal` characters", 0.1), (None, 0.1)]


def test_native_speech_failure_is_reported():
    import pytest
    class Process:
        returncode = 1
        def communicate(self, **kwargs):
            return "", "Voice unavailable"
    engine = MacSpeechEngine(Event(), voices=[], process_factory=lambda *a, **k: Process())
    engine.say("text")
    with pytest.raises(RuntimeError, match="Voice unavailable"):
        engine.runAndWait()
