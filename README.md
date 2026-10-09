<div align="center">

<img src="docs/title.svg" alt="Sign Language Detection Using Deep-Learning" width="880">

<br>

<img src="https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9">
<img src="https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow 2.16.1">
<img src="https://img.shields.io/badge/MediaPipe-Hands-0097A7?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe Hands">
<img src="https://img.shields.io/badge/OpenCV-Live%20Video-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/Interface-Tkinter-0A364D?style=for-the-badge" alt="Tkinter">
<img src="https://img.shields.io/badge/License-MIT-3c8ce7?style=for-the-badge" alt="MIT License">

<br><br>

**Static alphabet recognition; J/Z motion and full sign-language translation are unsupported.**

Capture stable letters at your webcam, build a sentence, and speak it using a system voice.

Runs offline on an ordinary laptop. No GPU, no internet, no cloud API.

</div>

<img src="docs/divider.svg" width="100%" alt="">

## About

A person who is deaf or mute can sign fluently. Most people around them cannot read a single sign.
That gap is the whole reason this project exists.

This app sits in the middle of that conversation. It watches your hand through a normal webcam,
recognises the American Sign Language letter you are making, lets you stack those letters into words and
sentences, and then speaks the finished sentence using an installed system voice. The person on the other side
never has to learn a sign.

It was built as our final year B.Tech project in Computer Science & Engineering, and the approach behind
it was published as a review paper in IRJMETS in November 2024.

<img src="docs/divider.svg" width="100%" alt="">

## Interface

**[Open the live web app](https://sign-language-live.azure-vine-9967.chatgpt.site)** —
anyone can access it. Press **Start camera** and allow webcam access. Camera
processing stays on your device. See [web/README.md](web/README.md) for web setup.

The welcome screen asks for a name and lists installed system voices. The dashboard
shows one active hand, prediction confidence, consecutive-frame stability, the sentence,
and camera selection/retry/stop controls. Capture is disabled until the prediction is READY.
On macOS, double-click `run.command` to open the welcome window. The camera stays off
until you open the dashboard.

Previous screenshots and the demo GIF show the old interface and are no longer embedded.

<img src="docs/divider.svg" width="100%" alt="">

## Features

| Feature | What it means in practice |
|---|---|
| **Static recognition** | 26 trained pose classes; actual J/Z motion is unsupported |
| **Reliable capture** | Confidence threshold plus consecutive-frame stability; uncertain predictions cannot be captured |
| **Hand tracking overlay** | 21 landmarks, a bounding box, the detected hand and the predicted letter drawn live |
| **Sentence building** | Capture letters one at a time, add spaces, backspace a mistake, clear and start over |
| **Speech output** | Installed voice names; background playback keeps the dashboard responsive |
| **Camera recovery** | Select a camera index, retry manually, or let failed captures reopen automatically |
| **Keyboard driven** | Every action has a shortcut, because one hand is already busy signing |
| **Fully offline** | Nothing leaves the machine. The model is 25 KB and runs on CPU |

<img src="docs/divider.svg" width="100%" alt="">

## How it works

<div align="center">

<img src="docs/architecture.svg" alt="Pipeline: camera, MediaPipe, normalisation, classifier, letter, speech" width="920">

</div>

The trick is that raw camera pixels never reach the neural network.

**MediaPipe Hands** locates the hand and returns 21 landmarks as (x, y) pairs. Those points are then
re-expressed relative to the wrist and normalised, so the same sign produces roughly the same numbers
whether your hand is close or far, left or right, big or small. What the classifier finally sees is just
42 numbers.

That is why the model can be tiny and still keep up with a live video feed. MediaPipe has already solved
the hard vision problem; the network only has to learn the shape of each letter in landmark space.

### The model

```
Input (42)
   |
   +-- BatchNormalization
   +-- Dense 128   mish   L2 0.01   -> Dropout 0.5
   +-- Dense  64   mish   L2 0.01   -> Dropout 0.5
   +-- Dense  32   mish   L2 0.01
   +-- Dense  26   softmax
```

Optimiser Adam, sparse categorical crossentropy, up to 1000 epochs with early stopping (patience 20).
Exported to TFLite with default quantisation, which is the 25 KB file the app actually loads.

### Results

| Measure | Value |
|---|---|
| Dataset | 36,401 validated landmark samples |
| Classes | 26 (A to Z) |
| Split | 70 / 15 / 15 train / validation / untouched test |
| Test accuracy | **84.12% TFLite**, 84.20% Keras on 5,461 untouched test samples |
| Keras/TFLite agreement | **99.32%**; quantization causes small prediction differences |
| Model size | 243 KB Keras, 25 KB quantised TFLite |

Early stopping uses validation only; class weights are computed from training data only.
Legacy signer/session IDs are unavailable, so these are sample-split results and do not establish
accuracy on unseen people or recording sessions. Complete provenance metadata enables session holdout;
see [training/PROVENANCE.md](training/PROVENANCE.md).

Watch-class recalls: **U 62.1%, D 93.8%, L 78.9%, T 78.7%**. Full precision/recall and reproducible
split/model hashes are in [evaluation.json](model/keypoint_classifier/evaluation.json),
[test_per_class.csv](model/keypoint_classifier/test_per_class.csv), and the notebook.

<img src="docs/divider.svg" width="100%" alt="">

## Quick start

```bash
git clone https://github.com/Anjalee117/Detection_gestures.git
cd Detection_gestures
python -m venv venv
venv\Scripts\activate          # Linux or macOS: source venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

You need **Python 3.9–3.11** and a webcam. On Windows you also need the Visual C++ Redistributable, and on
Linux you need `python3-tk` plus `espeak` for the voice.

Full walkthrough, including how to start it again on later days and what to do when something breaks:
**[SETUP.md](SETUP.md)**.

### Controls

| Key | Button | Action |
|:---:|---|---|
| `G` | Capture | Adds the letter currently on screen to the sentence |
| `Space` | Space | Adds a space between words |
| `D` | Delete | Removes the last character |
| `C` | Clear | Wipes the sentence |
| `S` | Speak | Reads the sentence out loud |

Capture is enabled only after five consecutive confident frames (default threshold 0.7).
Press `G` or Capture when the status shows `READY`; `UNCERTAIN` displays `?`.
Use `--confidence_threshold 0.8 --stable_frames 7` to change the policy.
Camera reads, inference, and speech run on background workers.

<img src="docs/divider.svg" width="100%" alt="">

## Training and checks

```bash
python -m pip install -r requirements-training.txt
python -m ipykernel install --prefix venv --name sign-language
python training/run_notebook.py
python -m pytest -q tests
```

Validated locally on macOS arm64 with Python 3.9.6; Windows/Linux need platform validation.
The notebook rejects malformed rows, restores the best validation checkpoint, reloads Keras,
and requires at least 99% test-prediction agreement before writing the TFLite export.

## Project structure

```
app.py                                  Tkinter dashboard and frame processor
keypoint_classification.ipynb           Training and evaluation notebook
requirements.txt                        Pinned runtime dependencies
requirements-training.txt               Pinned training and test dependencies
SETUP.md                                Installation and day-to-day run guide

model/keypoint_classifier/
    keypoint.csv                        Landmark dataset, ~36k samples
    keypoint_classifier_label.csv       Class labels, A to Z
    keypoint_classifier.py              TFLite inference wrapper
    keypoint_classifier.keras           Trained Keras model
    keypoint_classifier.tflite          Quantised model used at runtime

utils/landmarks.py                      Landmark normalization
utils/recognition.py                    Confidence/stability capture gate
utils/workers.py                        Camera recovery, inference and speech workers
training/                               Strict loader, session splits and notebook runner
tests/                                  pytest coverage
assets/                                 Banner and loading animation
docs/                                   README images and diagrams
```

<img src="docs/divider.svg" width="100%" alt="">

## Known limitations

- **Static alphabet recognition; J/Z and full translation unsupported.** J/Z dataset classes are static poses and cannot recognize the required motion.
- **Lighting matters.** A dim room, or a background the same colour as your hand, and detection suffers.
- **One active hand only.** The first tracked hand supplies the prediction.
- **Letters, not words.** The app spells out sentences. It does not recognise word-level signs or grammar.
- **System voices.** Speech quality is whatever your operating system provides.

<img src="docs/divider.svg" width="100%" alt="">

## License

MIT. See [LICENSE.md](LICENSE.md).

<div align="center">
<br>
<img src="docs/divider.svg" width="100%" alt="">
<sub>Built to make one conversation easier.</sub>
</div>
