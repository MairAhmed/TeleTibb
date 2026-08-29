import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from teletibb_signal import (
    build_feature_vector,
    chrom_dehaan,
    encode_gender,
    estimate_heart_rate,
    feature_names,
    ica_poh,
    parse_signal,
    pos_wang,
    resample_rgb_trace,
)
from train_models import participant_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_rgb_trace(
    heart_rate_bpm: float = 72,
    sample_rate: float = 30,
    duration: float = 30,
) -> tuple[np.ndarray, float]:
    times = np.arange(int(sample_rate * duration)) / sample_rate
    pulse = np.sin(2 * np.pi * heart_rate_bpm / 60 * times)
    trace = np.column_stack(
        [
            120 + 0.5 * pulse,
            90 + 1.8 * pulse,
            70 + 0.2 * pulse,
        ]
    )
    return trace, sample_rate


def test_gender_mapping_is_stable() -> None:
    assert encode_gender("F") == 0
    assert encode_gender("M") == 1
    with pytest.raises(ValueError):
        encode_gender("unknown")


def test_signal_parser_does_not_execute_python() -> None:
    with pytest.raises((ValueError, SyntaxError)):
        parse_signal("__import__('os').system('echo unsafe')")


def test_timestamp_resampling_uses_measured_rate() -> None:
    times = np.arange(0, 10, 1 / 24)
    jittered_times = times + 0.002 * np.sin(times)
    rgb = np.column_stack([times, times * 2, times * 3])
    resampled, sample_rate = resample_rgb_trace(jittered_times, rgb)

    assert 23.5 < sample_rate < 24.5
    assert resampled.shape[1] == 3
    assert resampled.shape[0] > 200


def test_rppg_consensus_recovers_synthetic_heart_rate() -> None:
    rgb, sample_rate = synthetic_rgb_trace()
    signals = {
        "pos": pos_wang(rgb, sample_rate),
        "chrom": chrom_dehaan(rgb, sample_rate),
        "ica": ica_poh(rgb, sample_rate),
    }
    heart_rate, quality, estimates = estimate_heart_rate(
        signals,
        sample_rate,
    )

    assert heart_rate == pytest.approx(72, abs=3)
    assert quality >= 0.8
    assert set(estimates) == {"pos", "chrom", "ica"}


def test_feature_schema_is_finite_and_stable() -> None:
    rgb, sample_rate = synthetic_rgb_trace()
    signals = {
        "pos": pos_wang(rgb, sample_rate),
        "chrom": chrom_dehaan(rgb, sample_rate),
        "ica": ica_poh(rgb, sample_rate),
    }
    vector = build_feature_vector(
        age=40,
        gender="M",
        ica_signal=signals["ica"],
        chrome_signal=signals["chrom"],
        pos_signal=signals["pos"],
        sample_rate=sample_rate,
    )

    assert vector.shape == (len(feature_names()),)
    assert np.isfinite(vector).all()
    assert vector[-2] == 40
    assert vector[-1] == 1


def test_participant_holdout_has_no_guid_overlap() -> None:
    groups = np.repeat(np.asarray([f"person-{index}" for index in range(20)]), 2)
    train_indices, test_indices = participant_split(groups)

    assert not set(groups[train_indices]).intersection(groups[test_indices])


@pytest.mark.parametrize(
    "artifact_name",
    ["bp_systolic_model.pkl", "bp_diastolic_model.pkl"],
)
def test_model_bundle_matches_shared_features(artifact_name: str) -> None:
    frame = pd.read_csv(PROJECT_ROOT / "DataLoader.csv", nrows=1)
    row = frame.iloc[0]
    vector = build_feature_vector(
        age=row["Age"],
        gender=row["Gender"],
        ica_signal=row["ICA_Output"],
        chrome_signal=row["CHROME_Output"],
        pos_signal=row["POS_Output"],
    )
    bundle = joblib.load(PROJECT_ROOT / artifact_name)

    assert bundle["schema_version"] == 1
    assert bundle["feature_names"] == feature_names()
    prediction = bundle["model"].predict(vector.reshape(1, -1))
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_capture_extracts_rgb_not_opencv_bgr() -> None:
    spec = importlib.util.spec_from_file_location(
        "teletibb_app",
        PROJECT_ROOT / "Teletibb-main.py",
    )
    app = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(app)

    grid = np.indices((200, 200)).sum(axis=0) % 2
    frame = np.empty((200, 200, 3), dtype=np.uint8)
    frame[..., 0] = 30 + 20 * grid
    frame[..., 1] = 80 + 20 * grid
    frame[..., 2] = 180 + 20 * grid
    rgb, quality, _, _ = app._extract_rgb_sample(
        frame,
        np.asarray([0, 0, 200, 200], dtype=float),
        None,
    )

    assert quality >= 0.3
    assert rgb is not None
    assert rgb[0] > rgb[1] > rgb[2]
