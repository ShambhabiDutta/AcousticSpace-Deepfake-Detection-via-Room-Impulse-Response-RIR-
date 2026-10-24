
import numpy as np
import pytest

from audio_ml.evaluate import evaluate_predictions


def test_evaluation_metrics_are_correct(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = evaluate_predictions(y_true, y_pred, tmp_path)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(0.8)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]

    assert (tmp_path / "confusion_matrix.npy").is_file()
    assert (tmp_path / "evaluation_metrics.txt").is_file()


def test_evaluation_rejects_empty_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_predictions(np.array([]), np.array([]))


def test_evaluation_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_predictions(np.array([0, 1]), np.array([0]))


def test_evaluation_rejects_invalid_labels():
    with pytest.raises(ValueError, match="True labels"):
        evaluate_predictions(np.array([0, 2]), np.array([0, 1]))
