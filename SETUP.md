# Setup Guide

Two parts here. **Part A** is the one-time installation you do on a fresh machine.
**Part B** is the short version you use every time you come back to the project afterwards.

If you just want to get it running, do Part A once, then Part B forever.

---

## Part A - First time installation

### A1. Check what you have

| Requirement | Why | How to check |
|---|---|---|
| Python 3.9 (64-bit) | MediaPipe and TensorFlow 2.16.1 wheels are built for it | `python --version` |
| pip | Installs the packages | `pip --version` |
| A working webcam | The whole app is a camera loop | Open the Camera app on Windows |
| Visual C++ Redistributable (Windows only) | MediaPipe and OpenCV ship native DLLs that need it | Try the install, you will see a DLL error if it is missing |

If `python --version` shows something other than 3.9, that is usually fine for running the app, but
TensorFlow 2.16.1 is not published for very new Python versions and pip will fail to find it.
Python 3.9 to 3.11 is the safe range.

Downloads if you need them:

- Python 3.9.13 (64-bit): https://www.python.org/downloads/release/python-3913/
- Visual C++ Redistributable x64: https://aka.ms/vs/17/release/vc_redist.x64.exe

When installing Python on Windows, tick **Add Python to PATH** on the first screen. It saves a lot of pain later.

### A2. Get the code

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Or download the ZIP from GitHub and extract it, then open a terminal inside the extracted folder.

### A3. Make a virtual environment

This keeps the project's packages separate from the rest of your system, so nothing you install here
can break another project.

Windows (PowerShell or CMD):

```bash
python -m venv venv
```

Linux or macOS:

```bash
python3 -m venv venv
```

### A4. Activate it

Windows (CMD):

```bash
venv\Scripts\activate
```

Windows (PowerShell):

```bash
venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source venv/bin/activate
```

Your prompt should now start with `(venv)`. That is how you know it worked.

> PowerShell may refuse with "running scripts is disabled on this system". Fix it once with
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and then activate again.

### A5. Install the packages

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

This pulls in TensorFlow, MediaPipe, OpenCV, Pillow, pyttsx3 and the training libraries.
It downloads a few hundred megabytes and takes several minutes on a normal connection. Let it finish.

### A6. Tkinter check

Tkinter draws the interface. It comes bundled with the official Python installer on Windows and macOS,
so you almost certainly already have it. On Debian or Ubuntu it is a separate package:

```bash
sudo apt install python3-tk
```

Verify it either way:

```bash
python -c "import tkinter; print('tkinter ok')"
```

### A7. Linux only - a voice for the speech output

`pyttsx3` speaks through the system speech engine. Windows has SAPI5 built in and macOS has NSSpeech,
so both work out of the box. Linux needs espeak:

```bash
sudo apt install espeak
```

### A8. Run it

```bash
python app.py
```

A loading screen appears for about five seconds, then the welcome form. Type a name, choose a voice,
press **Start Conversation**, and the camera window opens with live hand tracking.

Installation is done. From now on, use Part B.

---

## Part B - Starting it again next time

Every session after the first is three commands. The virtual environment already exists, the packages
are already installed. You only need to switch it on again.

Windows:

```bash
cd path\to\<repo-name>
```

```bash
venv\Scripts\activate
```

```bash
python app.py
```

Linux or macOS:

```bash
cd path/to/<repo-name> && source venv/bin/activate && python app.py
```

That is the whole routine. If the prompt does not show `(venv)`, the activate step did not take and
Python will not find the packages, so check that first before assuming something is broken.

### Useful flags

| Flag | Default | What it is for |
|---|---|---|
| `--device` | `0` | Camera index. Use `1` or `2` if you have more than one camera |
| `--width` | `1050` | Capture width |
| `--height` | `740` | Capture height |
| `--min_detection_confidence` | `0.7` | Raise it if it detects hands that are not there |
| `--min_tracking_confidence` | `0.5` | Raise it for steadier tracking, lower it if tracking keeps dropping |

Example:

```bash
python app.py --device 1 --min_detection_confidence 0.8
```

### Leaving the environment

```bash
deactivate
```

---

## Retraining the model

Only needed if you change the dataset. The trained model is already committed, so the app runs without this.

```bash
pip install jupyter
```

```bash
jupyter notebook keypoint_classification.ipynb
```

Run the cells from top to bottom. The notebook reads `model/keypoint_classifier/keypoint.csv`,
trains the network, prints the confusion matrix and classification report, and writes both
`keypoint_classifier.keras` and the quantised `keypoint_classifier.tflite` that the app loads at startup.

---

## Troubleshooting

**The camera window is black or the app closes immediately.**
Another program is holding the camera. Close Zoom, Teams, Meet or the Camera app and try again.
If you have more than one camera, try `python app.py --device 1`.

**`ImportError: DLL load failed while importing ...` on Windows.**
The Visual C++ Redistributable is missing. Install `vc_redist.x64.exe` from the link in step A1 and reboot.

**`ModuleNotFoundError: No module named 'cv2'` (or mediapipe, or pyttsx3).**
The virtual environment is not active, or the install did not finish. Activate it and run
`pip install -r requirements.txt` again.

**`ERROR: Could not find a version that satisfies the requirement tensorflow==2.16.1`.**
Your Python is too new or is 32-bit. Check with `python -c "import platform; print(platform.python_version(), platform.architecture())"`.
You want 3.9 to 3.11 and 64-bit.

**Nothing is spoken when I press S.**
On Linux, install espeak (step A7). On Windows, check that Windows has at least one voice installed under
Settings, Time and language, Speech. The app picks voice index 0 for male and index 1 for female, so if
your system only has one voice installed, the female option falls back to whatever is there.

**The TFLite conversion cell in the notebook fails.**
Use TensorFlow 2.16.1 exactly. Later versions changed the converter API and that cell breaks.

**Detection is jumpy or predicts the wrong letter.**
Improve the lighting, put some distance between your hand and the background, and keep only one hand in
frame. Hold the sign still until the label on the video settles, then press `G` to capture it.

**`FileNotFoundError: ./assets/banner.jpg`.**
You are running the app from the wrong folder. All paths are relative to the project root, so
`cd` into the folder that contains `app.py` before running it.
