# Setup

Static alphabet recognition; J/Z motion and full sign-language translation are unsupported.
The trained runtime model is included. Training is optional.

## First installation

Use 64-bit Python 3.9–3.11, Tkinter, and a webcam. This stack was validated on
macOS arm64 with Python 3.9.6 (training) and Python 3.11.17 / Tk 9.0 (repaired GUI).
Windows and Linux still require platform testing.
For a fresh macOS installation, use python.org Python with bundled Tkinter.

```bash
git clone https://github.com/Anjalee117/Detection_gestures.git
cd Detection_gestures
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tkinter; print('Tkinter available')"
python app.py
```

On Windows, use `python` instead of `python3`, then activate with
`venv\Scripts\activate` (CMD) or `venv\Scripts\Activate.ps1` (PowerShell).
Windows native DLL errors may require the Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe.

On Debian/Ubuntu, install Tkinter and the speech engine before running:

```bash
sudo apt install python3-tk espeak
```

Enter a name, choose an actual installed voice, and open the dashboard. Voice
loading happens in the background. Speech errors appear in the speech status.

## Open the visible window on macOS

The repaired local GUI uses `.venv` with project-local Python 3.11 and modern Tk.
Activate it with `source .venv/bin/activate`. `run.command` prefers `.venv` and
falls back to `venv` if the GUI environment is absent. Apple system Python/Tk 8.5
can produce a black preview; use a modern Python/Tk installation for the GUI.

Double-click `run.command` in Finder, or run `open run.command` from the project
folder. Terminal launches the app and the welcome window is brought forward.
The camera stays off until you press **OPEN DASHBOARD**. Use **STOP CAMERA** to
release it; **SELECT / RETRY** starts it again. Close the window to exit.

## Subsequent runs

From the project folder on this repaired macOS installation:

```bash
source .venv/bin/activate
python app.py
```

Windows:

```bat
venv\Scripts\activate
python app.py
```

Model and label paths resolve relative to `app.py`; launching with an absolute
path also works. Leave the environment with `deactivate`.

## Recognition and controls

| Key | Action |
|---|---|
| G | Capture only when the prediction is READY |
| Space | Insert a space |
| D | Delete the last character |
| C | Clear the sentence |
| S | Speak using the selected installed voice |

The app tracks one active hand. A letter must exceed the confidence threshold
for N consecutive frames. A changed letter, missing hand, uncertain prediction,
camera failure, or stale frame resets the capture streak. Each capture starts a
new streak, so held keys cannot commit unvalidated letters.

| Flag | Default | Validation |
|---|---|---|
| `--device` | 0 | Nonnegative camera index |
| `--width` | 1050 | Positive capture width |
| `--height` | 740 | Positive capture height |
| `--min_detection_confidence` | 0.7 | Finite value in [0, 1] |
| `--min_tracking_confidence` | 0.5 | Finite value in [0, 1] |
| `--confidence_threshold` | 0.7 | Finite value in [0, 1] |
| `--stable_frames` | 5 | Positive consecutive-frame count |
| `--use_static_image_mode` | Off | Detect independently on each frame |

```bash
python app.py --device 1 --confidence_threshold 0.8 --stable_frames 7
```

The dashboard's camera-index field and **SELECT / RETRY** button switch or reopen
cameras. Failed opens/reads automatically release and retry after one second.
Camera reads and inference run on a worker; speech uses a separate worker.
macOS playback uses the native `say` process with the selected voice ID; Windows/Linux use pyttsx3.

## Training and evaluation

```bash
python -m pip install -r requirements-training.txt
python -m ipykernel install --prefix venv --name sign-language
python training/run_notebook.py
python -m pytest -q tests
python tests/gui_smoke.py  # Desktop checks; simulated camera only
```

Run these from the project root. `training/run_notebook.py` executes every cell
and saves outputs in `keypoint_classification.ipynb`. It expects the virtual
environment to be named `venv` as above. VS Code can also run the notebook using
that environment's kernel.

The loader requires exactly 43 finite columns per row: integer class 0–25 plus
42 normalized features. The joined 85-column legacy row 35,522 was removed;
36,401 valid samples remain. Malformed rows now stop training with a line-number
error.

The default split is 70% training, 15% validation, and 15% untouched test.
Training-only class weights balance the class counts. Early stopping selects
weights using validation loss only, then restores the best checkpoint.
Keras is saved and reloaded in the current environment. Frozen inference export
avoids the previous converter failure. TFLite is written only after at least
99% agreement with Keras over the test partition; quantization is not exact.

Artifacts in `model/keypoint_classifier/`:

- `keypoint_classifier.keras`, `keypoint_classifier.tflite`: matching trained exports.
- `evaluation.json`: split method, accuracy, class precision/recall, weights and hashes.
- `test_per_class.csv`, `confusion_matrix.png`: test report; U/D/L/T marked for review.
- `split_manifest.npz`: exact sample indices and dataset hash for each partition.

Legacy signer/session IDs are unknown. Do not invent them. Current results are
sample-split accuracy, not unseen-signer/session accuracy. Complete, genuine
`sample_metadata.csv` enables whole-session holdout automatically; its schema
and validation rules are in [training/PROVENANCE.md](training/PROVENANCE.md).

## Troubleshooting

**Camera offline:** close other apps using the webcam, check the selected index,
and press SELECT / RETRY. On macOS, allow Terminal or VS Code under
System Settings > Privacy & Security > Camera. Reopen that app after changing access.

**Startup error:** the dashboard reports missing/broken model or label files.
Restore the included artifacts or rerun training. Missing imports at launch report
an installation error; activate the environment and reinstall `requirements.txt`.

**Speech unavailable:** check installed system voices; on Linux install espeak.
The welcome screen lists real voice names rather than assumed male/female indexes.

**OpenCV conflicts:** only `opencv-contrib-python` is required. If both OpenCV
packages are installed, repair the environment:

```bash
python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install -r requirements.txt
```

**Environment warnings:** this macOS environment reports an old-system-Python
LibreSSL warning. Some universal MediaPipe/gRPC wheels contain Intel-only tags
inside their metadata, so `pip check` can report unsupported architecture despite
successful native runtime checks. Fresh installs and other platforms still need
validation; do not treat those warnings as proof of a clean dependency check.
