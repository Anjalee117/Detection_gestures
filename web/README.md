# Sign Language Live

Browser version of the desktop app. Static alphabet recognition only; J/Z motion
and full sign-language translation are unsupported. The classifier uses the
trained Keras inference weights, with matching batch normalization, Mish and
softmax operations. Hand tracking and classification run in a Web Worker.
Camera frames never leave the visitor's device. No training records are deployed.

Serve `dist/` over HTTPS (or localhost for development):

```sh
python3 -m http.server 8000 --directory web/dist
```

Open `http://localhost:8000`, press **Start camera**, and allow camera permission.
Use a current Chrome, Edge or Safari browser with WebAssembly, OffscreenCanvas
and Web Workers. Installed speech voices depend on the browser and device.
Capture requires 70% confidence and five consecutive matching frames. A stopped,
disconnected or hidden camera cannot capture letters. Use Start camera to retry.

## Checks and model updates

```sh
node web/tests.mjs
python web/export_model.py
```

Model export requires the project's training environment. Export again after
retraining. The export publishes only layer weights and alphabet labels.

Bundled hand tracking: Google MediaPipe Tasks Vision 0.10.14 (Apache-2.0;
see `dist/vendor/LICENSE`) and Google's Hand Landmarker float16 model v1.
Sources: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js
and https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

The Sites hosting manifest is in `.openai/hosting.json`. Public deployment serves
only `dist/`; it does not publish the desktop project, dataset or test fixtures.
