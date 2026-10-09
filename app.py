from pathlib import Path
import csv
import argparse
import sys
import time

try:
    import cv2 as cv
    import numpy as np

    import tkinter as tk
    from tkinter import ttk
    from tkinter import StringVar
except ImportError as error:
    raise SystemExit(f"STARTUP ERROR / {error}. Install requirements.txt and ensure Tkinter is available.")

from utils.preview import VideoPreview
from utils.landmarks import pre_process_landmark
from utils.recognition import CaptureGate
from utils.workers import CameraWorker, SpeechWorker
from queue import Empty
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

PROJECT_ROOT = Path(__file__).resolve().parent

def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def device_index(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("camera index must be nonnegative")
    return number


def probability(value):
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a finite value between 0 and 1")
    return number


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Static ASL alphabet recognition")
    parser.add_argument("--device", type=device_index, default=0)
    parser.add_argument("--width", type=positive_int, default=1050)
    parser.add_argument("--height", type=positive_int, default=740)
    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", type=probability, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=probability, default=0.5)
    parser.add_argument("--confidence_threshold", type=probability, default=0.7)
    parser.add_argument("--stable_frames", type=positive_int, default=5)
    return parser.parse_args(argv)


class FrameProcessor:
    """Created and used exclusively by the camera/inference worker."""
    def __init__(self, args, gate):
        import mediapipe as mp
        self.gate = gate
        with (PROJECT_ROOT / "model/keypoint_classifier/keypoint_classifier_label.csv").open(encoding="utf-8-sig") as stream:
            self.labels = [row[0] for row in csv.reader(stream)]
        if self.labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            raise ValueError("Label file must contain A–Z in model output order")
        self.classifier = KeyPointClassifier()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=args.use_static_image_mode, max_num_hands=1,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

    def process(self, frame):
        image = cv.flip(frame, 1)
        results = self.hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            self.gate.observe()
            return image
        hand = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0]
        landmarks = calc_landmark_list(image, hand)
        index, confidence = self.classifier.predict(pre_process_landmark(landmarks))
        state = self.gate.observe(self.labels[index], confidence)
        bounds = calc_bounding_rect(image, hand)
        draw_bounding_rect(True, image, bounds)
        draw_landmarks(image, landmarks)
        draw_info_text(image, bounds, handedness, state.letter or state.status)
        return image

    def close(self):
        self.hands.close()


def main():
    args = get_args()
    sentence = ""
    gate = CaptureGate(args.confidence_threshold, args.stable_frames)
    camera_worker = None
    speech_worker = SpeechWorker()
    voice_ids = []
    last_packet = None
    closing = False
    camera_requested = False
    dashboard_opened = False

    # Brutalist dashboard palette
    ink = "#171717"
    paper = "#F4F0E8"
    white = "#FFFFFF"
    accent = "#C9F264"
    muted = "#68675F"
    danger = "#FF6B57"

    root = tk.Tk()
    root.title("Sign Language Studio")
    root.geometry("1180x780")
    root.minsize(940, 660)
    root.configure(bg=paper)

    def show_window():
        if sys.platform == "darwin":
            try:
                from AppKit import NSApplication, NSApplicationActivationPolicyRegular
                application = NSApplication.sharedApplication()
                application.setActivationPolicy_(NSApplicationActivationPolicyRegular)
                application.activateIgnoringOtherApps_(True)
            except Exception as error:
                print(f"Window activation: {error}", file=sys.stderr)
        root.deiconify()
        root.lift()
        name_entry.focus_set()

    root.after(100, show_window)


    parentFrame = tk.Frame(root, bg=paper)
    loading = tk.Frame(root, bg=ink)
    formFrame = tk.Frame(parentFrame, bg=paper)
    mainFrame = tk.Frame(parentFrame, bg=paper)

    loading.pack(fill="both", expand=True)
    loading_label = tk.Label(
        loading, text="SIGN / STUDIO", font=("Arial", 28, "bold"),
        bg=ink, fg=accent,
    )
    loading_label.pack(expand=True)

    def countdown(remaining):
        if remaining:
            root.after(250, countdown, remaining - 1)
            return
        loading.pack_forget()
        parentFrame.pack(fill="both", expand=True)
        if not dashboard_opened:
            formFrame.pack(fill="both", expand=True)

    root.after(250, countdown, 1)

    formFrame.columnconfigure(0, weight=1)
    formFrame.rowconfigure(0, weight=1)
    welcome_card = tk.Frame(
        formFrame, bg=white, padx=42, pady=36,
        highlightbackground=ink, highlightthickness=3,
    )
    welcome_card.grid(row=0, column=0)
    tk.Label(
        welcome_card, text="SIGN / STUDIO", font=("Arial", 32, "bold"),
        bg=white, fg=ink,
    ).pack(anchor="w")
    tk.Label(
        welcome_card, text="ASL alphabet recognition", font=("Arial", 16),
        bg=white, fg=muted,
    ).pack(anchor="w", pady=(4, 24))
    tk.Label(
        welcome_card,
        text="1. Open the dashboard to start your camera.\n"
             "2. Hold one hand still until the status says READY.\n"
             "3. Press G to capture, Space between words, S to speak.",
        font=("Arial", 12), bg=white, fg=ink, justify="left", wraplength=420,
    ).pack(anchor="w", pady=(0, 20))

    tk.Label(
        welcome_card, text="YOUR NAME", font=("Arial", 11, "bold"),
        bg=white, fg=ink,
    ).pack(anchor="w")
    name_var = StringVar()
    name_entry = tk.Entry(
        welcome_card, textvariable=name_var, font=("Arial", 18),
        relief="solid", bd=2, width=30,
    )
    name_entry.pack(fill="x", pady=(6, 18), ipady=8)
    tk.Label(
        welcome_card, text="SPEECH VOICE", font=("Arial", 11, "bold"),
        bg=white, fg=ink,
    ).pack(anchor="w")
    voice_var = StringVar(value="Loading installed voices…")
    voice_picker = ttk.Combobox(welcome_card, textvariable=voice_var, state="disabled", width=38)
    voice_picker.pack(fill="x", pady=(6, 8))
    speech_status = tk.Label(welcome_card, text="SPEECH / STARTING", bg=white, fg=muted, wraplength=360)
    speech_status.pack(anchor="w", pady=(0, 16))

    def start_camera():
        nonlocal camera_worker
        if closing or not camera_requested:
            return
        if camera_worker is not None and camera_worker.is_alive() and camera_worker.stop_event.is_set():
            root.after(100, start_camera)
            return
        if camera_worker is None or not camera_worker.is_alive():
            camera_worker = CameraWorker(lambda: FrameProcessor(args, gate), gate,
                                         device=int(camera_var.get()), width=args.width, height=args.height)
            camera_worker.start()

    def on_submit():
        nonlocal camera_requested, dashboard_opened
        camera_requested = True
        dashboard_opened = True
        loading.pack_forget()
        parentFrame.pack(fill="both", expand=True)
        username = name_var.get().strip() or "Guest"
        label1.config(text=f"SESSION / {username.upper()}")
        formFrame.pack_forget()
        mainFrame.pack(fill="both", expand=True)
        name_entry.focus_set()
        root.focus_set()
        start_camera()

    tk.Button(
        welcome_card, text="OPEN DASHBOARD  →", command=on_submit,
        font=("Arial", 14, "bold"), bg=accent, fg=ink,
        activebackground=accent, relief="solid", bd=2, padx=14, pady=10,
    ).pack(fill="x")

    dashboard_header = tk.Frame(mainFrame, bg=ink, padx=20, pady=14)
    dashboard_header.pack(fill="x", padx=18, pady=(18, 10))
    tk.Label(
        dashboard_header, text="SIGN / STUDIO", font=("Arial", 20, "bold"),
        bg=ink, fg=accent,
    ).pack(side="left")
    label1 = tk.Label(
        dashboard_header, text="SESSION / GUEST", font=("Arial", 12, "bold"),
        bg=ink, fg=white,
    )
    label1.pack(side="right")

    dashboard_body = tk.Frame(mainFrame, bg=paper, padx=18, pady=4)
    dashboard_body.pack(fill="both", expand=True)
    dashboard_body.columnconfigure(0, weight=1, minsize=320)
    dashboard_body.columnconfigure(1, weight=0, minsize=310)
    dashboard_body.rowconfigure(0, weight=1)
    camera_panel = tk.Frame(
        dashboard_body, bg=white, padx=14, pady=14,
        highlightbackground=ink, highlightthickness=3,
    )
    camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
    camera_panel.columnconfigure(0, weight=1)
    camera_panel.rowconfigure(1, weight=1)
    side_panel = tk.Frame(
        dashboard_body, bg=white, width=310, padx=12, pady=12,
        highlightbackground=ink, highlightthickness=3,
    )
    side_panel.grid(row=0, column=1, sticky="ns")
    side_panel.pack_propagate(False)

    tk.Label(
        camera_panel, text="01 / LIVE CAMERA", font=("Arial", 14, "bold"),
        bg=white, fg=ink,
    ).grid(row=0, column=0, sticky="w", pady=(0, 10))
    canvas = VideoPreview(camera_panel, bg=ink)
    canvas.grid(row=1, column=0, sticky="nsew")
    camera_status = tk.Label(
        camera_panel, text="CAMERA / STARTING", font=("Arial", 10, "bold"),
        bg=white, fg=muted, anchor="w",
    )
    camera_status.grid(row=2, column=0, sticky="ew", pady=(10, 0))
    camera_controls = tk.Frame(camera_panel, bg=white)
    camera_controls.grid(row=3, column=0, sticky="ew", pady=(6, 0))
    tk.Label(camera_controls, text="CAMERA INDEX", bg=white, fg=ink).pack(side="left")
    camera_var = StringVar(value=str(args.device))
    camera_picker = tk.Spinbox(camera_controls, from_=0, to=99, textvariable=camera_var, width=4)
    camera_picker.pack(side="left", padx=8)

    def select_camera():
        nonlocal last_packet, camera_requested
        try:
            device = device_index(camera_var.get())
        except (ValueError, argparse.ArgumentTypeError):
            camera_status.configure(text="CAMERA INDEX MUST BE A NONNEGATIVE INTEGER", fg=danger)
            return
        camera_requested = True
        gate.reset()
        last_packet = None
        camera_status.configure(text=f"CAMERA {device} / CONNECTING", fg=muted)
        canvas.show_message("CONNECTING CAMERA")
        if camera_worker is not None and camera_worker.is_alive() and not camera_worker.stop_event.is_set():
            camera_worker.select(device)
        else:
            start_camera()

    tk.Button(camera_controls, text="SELECT / RETRY", command=select_camera).pack(side="left")

    def stop_camera():
        nonlocal camera_requested, last_packet
        camera_requested = False
        gate.reset()
        last_packet = None
        if camera_worker is not None:
            camera_worker.stop()
        canvas.show_message("CAMERA STOPPED\nSelect / Retry to start again")
        prediction.configure(text="—")
        capture_button.configure(state="disabled")
        camera_status.configure(text="CAMERA STOPPED / SELECT RETRY TO START", fg=muted)

    tk.Button(camera_controls, text="STOP CAMERA", command=stop_camera).pack(side="left", padx=8)
    tk.Label(camera_panel, text="Hold one hand still → wait for READY → press G to capture.",
             font=("Arial", 11), bg=white, fg=ink).grid(row=4, column=0, sticky="w", pady=(6, 0))



    tk.Label(
        side_panel, text="02 / RECOGNITION", font=("Arial", 14, "bold"),
        bg=white, fg=ink,
    ).pack(anchor="w")
    prediction_card = tk.Frame(
        side_panel, bg=accent, padx=14, pady=8,
        highlightbackground=ink, highlightthickness=2,
    )
    prediction_card.pack(fill="x", pady=(10, 18))
    tk.Label(
        prediction_card, text="CURRENT LETTER", font=("Arial", 10, "bold"),
        bg=accent, fg=ink,
    ).pack(anchor="w")
    prediction = tk.Label(
        prediction_card, text="—", font=("Arial", 32, "bold"),
        bg=accent, fg=ink,
    )
    prediction.pack(anchor="w")
    tk.Label(
        side_panel, text="03 / SENTENCE", font=("Arial", 14, "bold"),
        bg=white, fg=ink,
    ).pack(anchor="w")
    sentence_frame = tk.Frame(side_panel, bg=paper)
    sentence_frame.pack(fill="x", pady=(8, 10))
    label = tk.Text(sentence_frame, font=("Arial", 14), height=3, width=20,
                    wrap="word", relief="solid", bd=2, padx=6, pady=6,
                    bg=paper, fg=ink, state="disabled")
    sentence_scroll = ttk.Scrollbar(sentence_frame, command=label.yview)
    label.configure(yscrollcommand=sentence_scroll.set)
    sentence_scroll.pack(side="right", fill="y")
    label.pack(side="left", fill="both", expand=True)

    def show_sentence():
        label.configure(state="normal")
        label.delete("1.0", "end")
        label.insert("1.0", sentence)
        label.configure(state="disabled")
        label.see("end")

    tk.Label(
        side_panel, text="ACTIONS / G · SPACE · D · C · S",
        font=("Arial", 9, "bold"), bg=white, fg=muted,
    ).pack(anchor="w", pady=(0, 8))
    controls_frame = tk.Frame(side_panel, bg=white)
    controls_frame.pack(fill="x")

    def capture_character():
        nonlocal sentence
        if camera_worker is None or last_packet is None or last_packet.generation != camera_worker.generation:
            return
        letter = gate.capture()
        if letter:
            sentence += letter
            show_sentence()
            capture_button.configure(state="disabled")

    def add_space():
        nonlocal sentence
        gate.reset()
        sentence += " "
        show_sentence()

    def clear_text():
        nonlocal sentence
        gate.reset()
        sentence = ""
        show_sentence()

    def delete_text():
        nonlocal sentence
        gate.reset()
        sentence = sentence[:-1]
        show_sentence()

    def text_to_speech():
        if sentence.strip():
            index = voice_picker.current()
            voice_id = voice_ids[index] if 0 <= index < len(voice_ids) else None
            speech_worker.speak(sentence, voice_id)

    def on_key_press(event):
        if not mainFrame.winfo_ismapped() or isinstance(event.widget, (tk.Entry, tk.Spinbox, ttk.Combobox)):
            return
        if event.char == 'g' or event.char == 'G':
            capture_character()
        elif event.char == 'd' or event.char == 'D':
            delete_text()
        elif event.char == 'c' or event.char == 'C':
            clear_text()
        elif event.char == 's' or event.char == 'S':
            text_to_speech()
        elif event.char == ' ':
            add_space()

    root.bind("<Key>", on_key_press)

    def action_button(text, bg, command):
        return tk.Button(
            controls_frame, text=text, command=command,
            font=("Arial", 12, "bold"), bg=bg, fg=ink,
            activebackground=bg, relief="solid", bd=2, anchor="w",
            padx=10, pady=4,
        )

    capture_button = action_button("CAPTURE LETTER  /  G", accent, capture_character)
    capture_button.configure(state="disabled")
    capture_button.pack(fill="x", pady=2)
    space_button = action_button("INSERT SPACE  /  SPACE", paper, add_space)
    space_button.pack(fill="x", pady=2)
    delete_button = action_button("BACKSPACE  /  D", paper, delete_text)
    delete_button.pack(fill="x", pady=2)
    clear_button = action_button("CLEAR SENTENCE  /  C", danger, clear_text)
    clear_button.pack(fill="x", pady=2)
    speak_button = action_button("SPEAK SENTENCE  /  S", ink, text_to_speech)
    speak_button.configure(fg=white, activeforeground=white)
    speak_button.configure(state="disabled")
    speak_button.pack(fill="x", pady=2)
    dashboard_speech_status = tk.Label(side_panel, text="SPEECH / STARTING", bg=white, fg=muted, wraplength=260)
    dashboard_speech_status.pack(fill="x", pady=6)


    def update_frame():
        nonlocal last_packet
        if closing:
            return
        while True:
            try:
                event = speech_worker.events.get_nowait()
            except Empty:
                break
            if event[0] == "voices":
                voices, default_id = event[1:]
                voice_ids[:] = [voice_id for voice_id, _ in voices]
                names = [name for _, name in voices]
                voice_picker.configure(values=names, state="readonly" if names else "disabled")
                if names:
                    voice_picker.current(voice_ids.index(default_id) if default_id in voice_ids else 0)
                else:
                    voice_var.set("System default")
                message = "SPEECH READY"
                speak_button.configure(state="normal")
            else:
                message = event[1]
                speak_button.configure(state="disabled" if event[0] == "error" or message == "SPEAKING" else "normal")
            color = danger if event[0] == "error" else muted
            speech_status.configure(text=message, fg=color)
            dashboard_speech_status.configure(text=message, fg=color)
        if camera_worker is not None and camera_requested and not camera_worker.stop_event.is_set():
            try:
                packet = camera_worker.packets.get_nowait()
            except Empty:
                packet = None
            if packet is not None and packet.generation == camera_worker.generation:
                last_packet = packet
                if packet.image is None:
                    canvas.show_message(packet.status)
            state = gate.snapshot()
            prediction.configure(text=state.letter or ("?" if state.status == "UNCERTAIN" else "—"))
            capture_button.configure(state="normal" if state.ready else "disabled")
            if last_packet is not None:
                status = last_packet.status
                if last_packet.image is not None:
                    status += f" / {state.status} / {state.confidence:.0%} / {state.frames}/{args.stable_frames}"
                    if time.monotonic() - last_packet.created_at > 1:
                        status = "CAMERA / WAITING FOR FRAME"
                    elif packet is not None:
                        canvas.show_frame(last_packet.image)
                camera_status.configure(text=status, fg=muted if last_packet.image is not None else danger)
        root.after(30, update_frame)

    def close():
        nonlocal closing, camera_requested
        closing = True
        camera_requested = False
        gate.reset()
        if camera_worker is not None:
            camera_worker.stop()
        speech_worker.stop()
        root.destroy()

    def report_ui_error(error_type, error, traceback):
        nonlocal camera_requested
        camera_requested = False
        gate.reset()
        if camera_worker is not None:
            camera_worker.stop()
        capture_button.configure(state="disabled")
        camera_status.configure(text=f"PREVIEW ERROR / {error}", fg=danger)
        print(f"PREVIEW ERROR / {error_type.__name__}: {error}", file=sys.stderr)
        root.after(30, update_frame)

    root.report_callback_exception = report_ui_error
    root.protocol("WM_DELETE_WINDOW", close)
    speech_worker.start()
    root.after(30, update_frame)
    try:
        root.mainloop()
    finally:
        if camera_worker is not None:
            camera_worker.stop()
            camera_worker.join(timeout=2)
        speech_worker.stop()
        speech_worker.join(timeout=2)
        cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 65 <= key <= 90:  # A ~ B
        number = key - 65
    if key == 110:  # n (Inference Mode)
        mode = 0
    if key == 107:  # k (Capturing Landmark From Camera Mode)
        mode = 1
    if key == 100:  # d (Capturing Landmarks From Provided Dataset Mode)
        mode = 2
    return number, mode





def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if (mode == 1 or mode == 2) and (0 <= number <= 35):
        csv_path = PROJECT_ROOT / "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # Middle finger
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # Ring finger
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # Little finger
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    return image


def draw_info(image, fps, mode, number):
    cv.putText(
        image,
        # "FPS:" + str(fps),
        "Press Q/Esc to Exit...",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        # "FPS:" + str(fps),
        "Press Q/Esc to Exit...",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = [
        "Logging Key Point",
        "Capturing Landmarks From Provided Dataset Mode",
    ]
    if 1 <= mode <= 2:
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1],
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                "NUM:" + str(number),
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return image



# Execute main
if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"STARTUP ERROR / {error}", file=sys.stderr)
        sys.exit(1)
