import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from training.data import load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/keypoint_classifier"


@pytest.fixture(scope="module")
def classifier():
    return KeyPointClassifier()


def test_model_contract_and_location_independent_loading(classifier, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    other = KeyPointClassifier()
    assert tuple(other.input_details[0]["shape"]) == (1, 42)
    assert tuple(other.output_details[0]["shape"]) == (1, 26)
    index, confidence = classifier.predict(np.zeros(42))
    assert 0 <= index < 26 and 0 <= confidence <= 1
    assert other(np.zeros(42)) == index


@pytest.mark.parametrize("features", [np.zeros(41), np.zeros(43), np.zeros((1, 42)),
                                        np.full(42, np.nan), np.full(42, np.inf)])
def test_invalid_model_inputs_are_rejected(classifier, features):
    with pytest.raises(ValueError, match="42 finite"):
        classifier.predict(features)


def test_exported_models_and_recorded_evaluation_agree(classifier):
    import tensorflow as tf

    evaluation = json.loads((MODEL / "evaluation.json").read_text())
    assert hashlib.sha256((MODEL / "keypoint.csv").read_bytes()).hexdigest() == evaluation["dataset_sha256"]
    assert hashlib.sha256((MODEL / "keypoint_classifier.tflite").read_bytes()).hexdigest() == evaluation["model_sha256"]
    assert hashlib.sha256((MODEL / "split_manifest.npz").read_bytes()).hexdigest() == evaluation["split_manifest_sha256"]
    x, _ = load_dataset(MODEL / "keypoint.csv")
    with np.load(MODEL / "split_manifest.npz") as manifest:
        test = manifest["test"]
    keras = tf.keras.models.load_model(MODEL / "keypoint_classifier.keras")
    scores = keras(x[test], training=False).numpy()
    predictions = np.asarray([classifier(row) for row in x[test]])
    agreement = np.mean(predictions == scores.argmax(axis=1))
    assert agreement >= 0.99
    assert agreement == pytest.approx(evaluation["keras_tflite_agreement"])
