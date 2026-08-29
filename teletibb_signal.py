import ast
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.decomposition import FastICA


GENDER_MAPPING = {"F": 0.0, "M": 1.0}
SIGNAL_COLUMNS = ("ICA_Output", "CHROME_Output", "POS_Output")
SIGNAL_NAMES = ("ica", "chrom", "pos")
PULSE_BAND_HZ = (0.7, 3.0)

SIGNAL_FEATURE_NAMES = (
    "mean_absolute_amplitude",
    "interquartile_range",
    "peak_to_peak",
    "median_absolute_difference",
    "skew",
    "kurtosis",
    "derivative_rms",
    "dominant_frequency_hz",
    "dominant_frequency_bpm",
    "peak_power_ratio",
    "pulse_snr_db",
    "spectral_entropy",
    "peak_interval_mean_seconds",
    "peak_interval_std_seconds",
    "peak_interval_cv",
)


def parse_signal(values: Sequence[float] | str) -> np.ndarray:
    if isinstance(values, str):
        values = ast.literal_eval(values)
    parsed = np.asarray(values, dtype=float).reshape(-1)
    parsed = parsed[np.isfinite(parsed)]
    if parsed.size < 32:
        raise ValueError("A physiological signal needs at least 32 finite samples.")
    return parsed


def encode_gender(gender: str) -> float:
    try:
        return GENDER_MAPPING[str(gender).upper()]
    except KeyError as exc:
        raise ValueError("Gender must be 'F' or 'M'.") from exc


def preprocess_signal(
    values: Sequence[float] | str,
    sample_rate: float,
) -> np.ndarray:
    if sample_rate <= 2 * PULSE_BAND_HZ[1]:
        raise ValueError("Sample rate is too low for the configured pulse band.")

    filtered = parse_signal(values)
    filtered = signal.medfilt(filtered, kernel_size=3)
    filtered = signal.detrend(filtered, type="linear")

    sos = signal.butter(
        3,
        PULSE_BAND_HZ,
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, filtered)
    standard_deviation = float(np.std(filtered))
    if standard_deviation <= np.finfo(float).eps:
        raise ValueError("The physiological signal has no usable variation.")
    return (filtered - np.mean(filtered)) / standard_deviation


def _spectral_measurements(
    filtered: np.ndarray,
    sample_rate: float,
) -> dict[str, float]:
    frequencies, power = signal.welch(
        filtered,
        fs=sample_rate,
        nperseg=min(256, filtered.size),
    )
    pulse_mask = (
        (frequencies >= PULSE_BAND_HZ[0])
        & (frequencies <= PULSE_BAND_HZ[1])
    )
    pulse_power = power[pulse_mask]
    pulse_frequencies = frequencies[pulse_mask]
    if pulse_power.size == 0 or float(np.sum(pulse_power)) <= 0:
        raise ValueError("No pulse-band spectral power was found.")

    peak_index = int(np.argmax(pulse_power))
    dominant_frequency = float(pulse_frequencies[peak_index])
    peak_power_ratio = float(pulse_power[peak_index] / np.sum(pulse_power))

    resolution = (
        float(pulse_frequencies[1] - pulse_frequencies[0])
        if pulse_frequencies.size > 1
        else 0.1
    )
    peak_mask = np.abs(pulse_frequencies - dominant_frequency) <= max(
        0.1,
        resolution,
    )
    peak_power = float(np.sum(pulse_power[peak_mask]))
    noise_power = max(
        float(np.sum(pulse_power[~peak_mask])),
        np.finfo(float).eps,
    )
    pulse_snr_db = float(10 * np.log10(peak_power / noise_power))

    normalized_power = pulse_power / np.sum(pulse_power)
    spectral_entropy = float(
        -np.sum(normalized_power * np.log(normalized_power + np.finfo(float).eps))
        / np.log(normalized_power.size)
    )
    return {
        "dominant_frequency_hz": dominant_frequency,
        "dominant_frequency_bpm": dominant_frequency * 60,
        "peak_power_ratio": peak_power_ratio,
        "pulse_snr_db": pulse_snr_db,
        "spectral_entropy": spectral_entropy,
    }


def extract_signal_features(
    values: Sequence[float] | str,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    filtered = preprocess_signal(values, sample_rate)
    spectral = _spectral_measurements(filtered, sample_rate)

    minimum_peak_distance = max(1, int(sample_rate * 60 / 180))
    peaks, _ = signal.find_peaks(
        filtered,
        distance=minimum_peak_distance,
        prominence=0.25,
    )
    peak_intervals = np.diff(peaks) / sample_rate
    valid_intervals = peak_intervals[
        (peak_intervals >= 60 / 180) & (peak_intervals <= 60 / 40)
    ]
    interval_mean = (
        float(np.mean(valid_intervals)) if valid_intervals.size else 0.0
    )
    interval_std = (
        float(np.std(valid_intervals)) if valid_intervals.size else 0.0
    )
    interval_cv = interval_std / interval_mean if interval_mean else 0.0

    differences = np.diff(filtered)
    features = np.asarray(
        [
            np.mean(np.abs(filtered)),
            np.percentile(filtered, 75) - np.percentile(filtered, 25),
            np.ptp(filtered),
            np.median(np.abs(differences)),
            skew(filtered),
            kurtosis(filtered),
            np.sqrt(np.mean(np.square(differences))),
            spectral["dominant_frequency_hz"],
            spectral["dominant_frequency_bpm"],
            spectral["peak_power_ratio"],
            spectral["pulse_snr_db"],
            spectral["spectral_entropy"],
            interval_mean,
            interval_std,
            interval_cv,
        ],
        dtype=float,
    )
    return np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ), filtered


def feature_names() -> list[str]:
    names = [
        f"{signal_name}_{feature_name}"
        for signal_name in SIGNAL_NAMES
        for feature_name in SIGNAL_FEATURE_NAMES
    ]
    names.extend(
        [
            "ica_chrom_correlation",
            "ica_pos_correlation",
            "chrom_pos_correlation",
            "heart_rate_median_bpm",
            "heart_rate_spread_bpm",
            "peak_power_ratio_mean",
            "pulse_snr_mean_db",
            "age",
            "gender",
        ]
    )
    return names


def _aligned_correlation(left: np.ndarray, right: np.ndarray) -> float:
    length = min(left.size, right.size)
    axis = np.linspace(0, 1, length)
    left_aligned = np.interp(
        axis,
        np.linspace(0, 1, left.size),
        left,
    )
    right_aligned = np.interp(
        axis,
        np.linspace(0, 1, right.size),
        right,
    )
    return float(np.corrcoef(left_aligned, right_aligned)[0, 1])


def build_feature_vector(
    age: float,
    gender: str,
    ica_signal: Sequence[float] | str,
    chrome_signal: Sequence[float] | str,
    pos_signal: Sequence[float] | str,
    sample_rate: float = 30.0,
) -> np.ndarray:
    feature_blocks = []
    filtered_signals = []
    for values in (ica_signal, chrome_signal, pos_signal):
        features, filtered = extract_signal_features(values, sample_rate)
        feature_blocks.append(features)
        filtered_signals.append(filtered)

    dominant_bpm_index = SIGNAL_FEATURE_NAMES.index("dominant_frequency_bpm")
    peak_power_index = SIGNAL_FEATURE_NAMES.index("peak_power_ratio")
    pulse_snr_index = SIGNAL_FEATURE_NAMES.index("pulse_snr_db")
    heart_rates = [
        block[dominant_bpm_index]
        for block in feature_blocks
    ]
    correlations = [
        _aligned_correlation(filtered_signals[0], filtered_signals[1]),
        _aligned_correlation(filtered_signals[0], filtered_signals[2]),
        _aligned_correlation(filtered_signals[1], filtered_signals[2]),
    ]
    aggregate_features = np.asarray(
        [
            *correlations,
            np.median(heart_rates),
            np.ptp(heart_rates),
            np.mean([block[peak_power_index] for block in feature_blocks]),
            np.mean([block[pulse_snr_index] for block in feature_blocks]),
            float(age),
            encode_gender(gender),
        ],
        dtype=float,
    )
    combined = np.concatenate([*feature_blocks, aggregate_features])
    if combined.size != len(feature_names()):
        raise RuntimeError("The feature schema is internally inconsistent.")
    return np.nan_to_num(
        combined,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def build_feature_matrix(
    frame: pd.DataFrame,
    sample_rate: float = 30.0,
) -> np.ndarray:
    required_columns = {"Age", "Gender", *SIGNAL_COLUMNS}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing training columns: {sorted(missing_columns)}"
        )
    return np.vstack(
        [
            build_feature_vector(
                row.Age,
                row.Gender,
                row.ICA_Output,
                row.CHROME_Output,
                row.POS_Output,
                sample_rate,
            )
            for row in frame.itertuples(index=False)
        ]
    )


def estimate_heart_rate(
    signals: Mapping[str, Sequence[float]],
    sample_rate: float,
) -> tuple[float, float, dict[str, float]]:
    estimates = {}
    weights = {}
    for name, values in signals.items():
        features, _ = extract_signal_features(values, sample_rate)
        bpm = float(
            features[SIGNAL_FEATURE_NAMES.index("dominant_frequency_bpm")]
        )
        peak_power_ratio = float(
            features[SIGNAL_FEATURE_NAMES.index("peak_power_ratio")]
        )
        estimates[name] = bpm
        weights[name] = max(peak_power_ratio, 0.01)

    estimate_values = np.asarray(list(estimates.values()))
    median_estimate = float(np.median(estimate_values))
    inlier_names = [
        name
        for name, bpm in estimates.items()
        if abs(bpm - median_estimate) <= 12
    ]
    if len(inlier_names) < 2:
        inlier_names = list(estimates)

    inlier_estimates = np.asarray([estimates[name] for name in inlier_names])
    inlier_weights = np.asarray([weights[name] for name in inlier_names])
    consensus = float(np.average(inlier_estimates, weights=inlier_weights))
    spread = float(np.ptp(inlier_estimates))
    spectral_quality = min(
        1.0,
        float(np.mean(inlier_weights)) / 0.35,
    )
    agreement_quality = max(0.0, 1.0 - spread / 20)
    quality = spectral_quality * agreement_quality
    return consensus, quality, estimates


def resample_rgb_trace(
    timestamps: Sequence[float],
    rgb_trace: Sequence[Sequence[float]],
) -> tuple[np.ndarray, float]:
    timestamp_array = np.asarray(timestamps, dtype=float)
    trace_array = np.asarray(rgb_trace, dtype=float)
    if timestamp_array.size != trace_array.shape[0]:
        raise ValueError("Each RGB sample must have one timestamp.")
    if timestamp_array.size < 32 or trace_array.shape[1:] != (3,):
        raise ValueError("At least 32 timestamped RGB samples are required.")

    unique_mask = np.concatenate(
        ([True], np.diff(timestamp_array) > np.finfo(float).eps)
    )
    timestamp_array = timestamp_array[unique_mask]
    trace_array = trace_array[unique_mask]
    intervals = np.diff(timestamp_array)
    if intervals.size == 0:
        raise ValueError("Capture timestamps do not span a usable duration.")

    sample_rate = float(np.clip(1 / np.median(intervals), 8.0, 60.0))
    uniform_times = np.arange(
        timestamp_array[0],
        timestamp_array[-1],
        1 / sample_rate,
    )
    resampled = np.column_stack(
        [
            np.interp(uniform_times, timestamp_array, trace_array[:, channel])
            for channel in range(3)
        ]
    )
    return resampled, sample_rate


def _bandpass(values: np.ndarray, sample_rate: float) -> np.ndarray:
    sos = signal.butter(
        3,
        PULSE_BAND_HZ,
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return signal.sosfiltfilt(sos, np.asarray(values, dtype=float))


def pos_wang(rgb_trace: np.ndarray, sample_rate: float) -> np.ndarray:
    rgb = np.asarray(rgb_trace, dtype=float)
    window_length = max(2, int(np.ceil(1.6 * sample_rate)))
    output = np.zeros(rgb.shape[0])
    for end in range(window_length, rgb.shape[0] + 1):
        start = end - window_length
        window = rgb[start:end]
        means = np.maximum(np.mean(window, axis=0), np.finfo(float).eps)
        normalized = window / means
        projected = np.asarray(
            [[0, 1, -1], [-2, 1, 1]],
            dtype=float,
        ) @ normalized.T
        denominator = max(np.std(projected[1]), np.finfo(float).eps)
        pulse = projected[0] + np.std(projected[0]) / denominator * projected[1]
        output[start:end] += pulse - np.mean(pulse)
    return _bandpass(signal.detrend(output), sample_rate)


def chrom_dehaan(rgb_trace: np.ndarray, sample_rate: float) -> np.ndarray:
    rgb = np.asarray(rgb_trace, dtype=float)
    window_length = max(2, int(np.ceil(1.6 * sample_rate)))
    if window_length % 2:
        window_length += 1
    step = window_length // 2
    output = np.zeros(rgb.shape[0])
    weights = np.zeros(rgb.shape[0])

    for start in range(0, rgb.shape[0] - window_length + 1, step):
        end = start + window_length
        window = rgb[start:end]
        means = np.maximum(np.mean(window, axis=0), np.finfo(float).eps)
        normalized = window / means
        x_component = 3 * normalized[:, 0] - 2 * normalized[:, 1]
        y_component = (
            1.5 * normalized[:, 0]
            + normalized[:, 1]
            - 1.5 * normalized[:, 2]
        )
        x_filtered = _bandpass(x_component, sample_rate)
        y_filtered = _bandpass(y_component, sample_rate)
        denominator = max(np.std(y_filtered), np.finfo(float).eps)
        pulse = x_filtered - np.std(x_filtered) / denominator * y_filtered
        window_weights = np.hanning(window_length)
        output[start:end] += pulse * window_weights
        weights[start:end] += window_weights

    valid_weights = weights > np.finfo(float).eps
    output[valid_weights] /= weights[valid_weights]
    return _bandpass(signal.detrend(output), sample_rate)


def ica_poh(rgb_trace: np.ndarray, sample_rate: float) -> np.ndarray:
    rgb = np.asarray(rgb_trace, dtype=float)
    detrended = signal.detrend(rgb, axis=0)
    standard_deviation = np.std(detrended, axis=0)
    normalized = detrended / np.maximum(
        standard_deviation,
        np.finfo(float).eps,
    )
    sources = FastICA(
        n_components=3,
        whiten="unit-variance",
        random_state=42,
        max_iter=1000,
        tol=0.0005,
    ).fit_transform(normalized)

    peak_power_ratios = []
    for component in sources.T:
        filtered = _bandpass(component, sample_rate)
        peak_power_ratios.append(
            _spectral_measurements(filtered, sample_rate)["peak_power_ratio"]
        )
    selected = sources[:, int(np.argmax(peak_power_ratios))]
    return _bandpass(selected, sample_rate)
