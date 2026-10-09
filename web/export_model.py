"""Export inference weights only; never publish training samples."""
import csv
import json
from pathlib import Path

import tensorflow as tf

root = Path(__file__).resolve().parents[1]
source = root / 'model/keypoint_classifier'
model = tf.keras.models.load_model(source / 'keypoint_classifier.keras')
layers = []
for layer in model.layers:
    kind = type(layer).__name__
    if kind == 'Dropout':
        continue
    if kind not in ('BatchNormalization', 'Dense'):
        raise ValueError(f'Unsupported layer: {kind}')
    config = layer.get_config()
    layers.append(dict(kind=kind, weights=[w.tolist() for w in layer.get_weights()],
                       activation=config.get('activation'), epsilon=config.get('epsilon')))
with (source / 'keypoint_classifier_label.csv').open(encoding='utf-8-sig') as f:
    labels = [row[0] for row in csv.reader(f)]
target = Path(__file__).parent / 'dist/model.json'
target.write_text(json.dumps(dict(labels=labels, layers=layers), separators=(',', ':')))
print(f'Exported {len(layers)} inference layers, {len(labels)} labels to {target}')
