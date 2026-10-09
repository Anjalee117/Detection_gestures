import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {CaptureGate, normalize, predict} from './dist/core.mjs';

let now = 1000;
const gate = new CaptureGate(.7, 5, () => now);
for (let i = 0; i < 4; i++) assert.equal(gate.observe('A', .9).ready, false);
assert.equal(gate.observe('A', .9).ready, true);
assert.equal(gate.capture(), 'A'); assert.equal(gate.capture(), '');
gate.observe('B', .6); assert.equal(gate.state.status, 'UNCERTAIN');
for (let i = 0; i < 5; i++) gate.observe('B', .9);
now += 501; assert.equal(gate.capture(), '');
gate.observe('J', .99); assert.equal(gate.state.status, 'MOTION UNSUPPORTED');
gate.observe('Z', .99); assert.equal(gate.capture(), '');
gate.observe('A', .9); gate.observe('', 0); assert.equal(gate.state.frames, 0);
const points = Array.from({length:21}, (_, i) => [i * 3 + 10, i * 2 + 20]);
const normalized = normalize(points); assert.equal(normalized.length, 42);
assert.deepEqual(normalized.slice(0, 2), [0,0]); assert.equal(Math.max(...normalized), 1);
assert.deepEqual(normalize(Array.from({length:21}, () => [1,1])), Array(42).fill(0));
assert.throws(() => normalize([[0,0]])); assert.throws(() => normalize(points.map(() => [NaN,0])));
const model = JSON.parse(await readFile(new URL('./dist/model.json', import.meta.url)));
const output = predict(model, normalized);
assert.equal(output.probabilities.length, 26);
assert.ok(Math.abs(output.probabilities.reduce((a,b) => a+b,0) - 1) < 1e-12);
assert.throws(() => predict(model, [1])); assert.throws(() => predict(model, Array(42).fill(Infinity)));
if (process.argv[2]) {
  const fixtures = JSON.parse(await readFile(process.argv[2])); let maxError = 0;
  for (const {features, probabilities} of fixtures) {
    const result = predict(model, features).probabilities;
    for (let i = 0; i < 26; i++) maxError = Math.max(maxError, Math.abs(result[i] - probabilities[i]));
    assert.equal(result.indexOf(Math.max(...result)), probabilities.indexOf(Math.max(...probabilities)));
  }
  assert.ok(maxError < .0001, `Prediction error: ${maxError}`);
  console.log(`PASS: ${fixtures.length} Keras / JavaScript predictions agree; maximum probability error ${maxError}`);
}
console.log('PASS: normalization, model I/O, stability, uncertainty, stale capture, J/Z unsupported');
