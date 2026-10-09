export function normalize(points) {
  if (points.length !== 21 || points.some(p => p.length !== 2 || p.some(v => !Number.isFinite(v)))) throw new Error('Expected 21 finite landmark pairs');
  const values = points.flatMap(p => [p[0] - points[0][0], p[1] - points[0][1]]);
  const scale = Math.max(...values.map(Math.abs));
  return values.map(v => scale ? v / scale : v);
}

export function predict(model, features) {
  if (features.length !== 42 || features.some(v => !Number.isFinite(v))) throw new Error('Invalid model input');
  let values = features;
  for (const layer of model.layers) {
    if (layer.kind === 'BatchNormalization') {
      const [gamma, beta, mean, variance] = layer.weights;
      values = values.map((v, i) => gamma[i] * (v - mean[i]) / Math.sqrt(variance[i] + layer.epsilon) + beta[i]);
    } else if (layer.kind === 'Dense') {
      const [kernel, bias] = layer.weights;
      values = bias.map((b, j) => kernel.reduce((sum, row, i) => sum + row[j] * values[i], b));
      if (layer.activation === 'mish') values = values.map(x => x * Math.tanh(x > 20 ? x : Math.log1p(Math.exp(x))));
      else if (layer.activation === 'softmax') {
        const max = Math.max(...values), exp = values.map(v => Math.exp(v - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        values = exp.map(v => v / sum);
      } else throw new Error('Unsupported model activation');
    } else throw new Error('Unsupported model layer');
  }
  if (values.length !== 26 || values.some(v => !Number.isFinite(v))) throw new Error('Invalid model output');
  const index = values.indexOf(Math.max(...values));
  return {letter: model.labels[index], confidence: values[index], probabilities: values};
}

export class CaptureGate {
  constructor(threshold = .7, frames = 5, clock = () => performance.now()) {
    this.threshold = threshold; this.frames = frames; this.clock = clock; this.reset();
  }
  reset() { this.state = {letter: '', confidence: 0, frames: 0, ready: false, status: 'NO HAND'}; this.at = 0; }
  observe(letter, confidence = 0) {
    this.at = this.clock();
    if (!letter) this.reset();
    else if (!Number.isFinite(confidence) || confidence < this.threshold || letter === 'J' || letter === 'Z') {
      this.state = {letter: '', confidence, frames: 0, ready: false, status: letter === 'J' || letter === 'Z' ? 'MOTION UNSUPPORTED' : 'UNCERTAIN'};
    } else {
      const frames = Math.min(this.frames, this.state.letter === letter ? this.state.frames + 1 : 1);
      this.state = {letter, confidence, frames, ready: frames === this.frames, status: frames === this.frames ? 'READY TO CAPTURE' : 'HOLD STEADY'};
    }
    return this.state;
  }
  snapshot() { if (this.clock() - this.at > 500) this.reset(); return this.state; }
  capture() { const state = this.snapshot(), letter = state.ready ? state.letter : ''; if (letter) this.reset(); return letter; }
}
