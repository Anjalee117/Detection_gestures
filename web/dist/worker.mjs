// MediaPipe 0.10.14 loads its WASM glue with importScripts, requiring a classic
// worker. Dynamic imports still let us share the tested ES-module classifier.
const dependencies = Promise.all([import('./vendor/vision_bundle.mjs'), import('./core.mjs')]);

let detector, model, surface, context;
self.onmessage = async ({data}) => {
  try {
    const [{HandLandmarker, FilesetResolver}, {normalize, predict}] = await dependencies;
    if (data.type === 'init') {
      const response = await fetch('./model.json');
      if (!response.ok) throw new Error('Recognition model could not load.');
      model = await response.json();
      const vision = await FilesetResolver.forVisionTasks('./vendor/wasm');
      detector = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {modelAssetPath: './hand_landmarker.task', delegate: 'CPU'},
        runningMode: 'VIDEO', numHands: 1, minHandDetectionConfidence: .5,
        minHandPresenceConfidence: .5, minTrackingConfidence: .5
      });
      self.postMessage({type: 'ready'});
    } else if (data.type === 'frame' && detector) {
      const {bitmap, timestamp} = data;
      try {
        if (!surface || surface.width !== bitmap.width || surface.height !== bitmap.height) {
          surface = new OffscreenCanvas(bitmap.width, bitmap.height); context = surface.getContext('2d');
        }
        context.setTransform(-1, 0, 0, 1, surface.width, 0);
        context.drawImage(bitmap, 0, 0);
        const result = detector.detectForVideo(surface, timestamp);
        const hand = result.landmarks[0];
        const points = hand?.map(p => [Math.min(Math.trunc(p.x * surface.width), surface.width - 1), Math.min(Math.trunc(p.y * surface.height), surface.height - 1)]);
        self.postMessage({type: 'result', timestamp, landmarks: hand || [], ...(points ? predict(model, normalize(points)) : {letter: '', confidence: 0})});
      } finally { bitmap.close(); }
    }
  } catch (error) { self.postMessage({type: 'error', message: error.message || 'Recognition failed.'}); }
};
