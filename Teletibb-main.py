from pathlib import Path
import time

import cv2
import joblib
import numpy as np
import streamlit as st

from teletibb_signal import (
    build_feature_vector,
    chrom_dehaan,
    estimate_heart_rate,
    feature_names,
    ica_poh,
    pos_wang,
    resample_rgb_trace,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_CAPTURE_SECONDS = 30
MAX_CAPTURE_SECONDS = 50
MIN_VALID_FRAMES = 300
MIN_CAPTURE_QUALITY = 0.4
MIN_SIGNAL_QUALITY = 0.25


class MeasurementError(RuntimeError):
    pass


@st.cache_resource
def load_model_bundle(filename: str) -> dict:
    bundle = joblib.load(PROJECT_ROOT / filename)
    required_keys = {
        "schema_version",
        "sample_rate",
        "feature_names",
        "model",
        "validation",
    }
    if required_keys.difference(bundle):
        raise MeasurementError(f"{filename} is not a valid TeleTibb model.")
    if bundle["schema_version"] != 1:
        raise MeasurementError(f"{filename} uses an unsupported schema.")
    if bundle["feature_names"] != feature_names():
        raise MeasurementError(
            f"{filename} does not match the current feature pipeline."
        )
    return bundle


def _intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    intersection_left = max(left_x, right_x)
    intersection_top = max(left_y, right_y)
    intersection_right = min(
        left_x + left_width,
        right_x + right_width,
    )
    intersection_bottom = min(
        left_y + left_height,
        right_y + right_height,
    )
    intersection = max(0, intersection_right - intersection_left) * max(
        0,
        intersection_bottom - intersection_top,
    )
    union = left_width * left_height + right_width * right_height - intersection
    return float(intersection / union) if union else 0.0


def _select_and_smooth_face(
    faces: np.ndarray,
    previous_face: np.ndarray | None,
) -> np.ndarray:
    candidates = np.asarray(faces, dtype=float)
    if previous_face is None:
        selected = max(candidates, key=lambda face: face[2] * face[3])
        return np.asarray(selected, dtype=float)

    selected = max(
        candidates,
        key=lambda face: (
            _intersection_over_union(face, previous_face),
            face[2] * face[3],
        ),
    )
    smoothing = 0.25
    return previous_face * (1 - smoothing) + selected * smoothing


def _region_coordinates(
    face: np.ndarray,
    frame_shape: tuple[int, ...],
) -> list[tuple[int, int, int, int]]:
    x, y, width, height = face
    normalized_regions = (
        (0.25, 0.12, 0.75, 0.32),
        (0.10, 0.48, 0.38, 0.75),
        (0.62, 0.48, 0.90, 0.75),
    )
    frame_height, frame_width = frame_shape[:2]
    regions = []
    for left, top, right, bottom in normalized_regions:
        x1 = int(np.clip(x + left * width, 0, frame_width - 1))
        y1 = int(np.clip(y + top * height, 0, frame_height - 1))
        x2 = int(np.clip(x + right * width, x1 + 1, frame_width))
        y2 = int(np.clip(y + bottom * height, y1 + 1, frame_height))
        regions.append((x1, y1, x2, y2))
    return regions


def _extract_rgb_sample(
    frame: np.ndarray,
    face: np.ndarray,
    previous_gray_face: np.ndarray | None,
) -> tuple[np.ndarray | None, float, np.ndarray, list[tuple[int, int, int, int]]]:
    x, y, width, height = np.rint(face).astype(int)
    frame_height, frame_width = frame.shape[:2]
    x = int(np.clip(x, 0, frame_width - 1))
    y = int(np.clip(y, 0, frame_height - 1))
    width = int(np.clip(width, 1, frame_width - x))
    height = int(np.clip(height, 1, frame_height - y))
    face_crop = frame[y : y + height, x : x + width]
    if face_crop.size == 0:
        return None, 0.0, np.empty((0, 0), dtype=np.uint8), []

    gray_face = cv2.resize(
        cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY),
        (96, 96),
    )
    blur = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    brightness = float(np.mean(gray_face))
    clipped_fraction = float(
        np.mean((gray_face <= 5) | (gray_face >= 250))
    )
    motion = (
        float(np.mean(cv2.absdiff(gray_face, previous_gray_face)))
        if previous_gray_face is not None
        else 0.0
    )

    blur_score = float(np.clip((blur - 15) / 85, 0, 1))
    exposure_score = float(1 - min(abs(brightness - 127.5) / 110, 1))
    clipping_score = float(np.clip(1 - clipped_fraction / 0.15, 0, 1))
    motion_score = float(np.clip(1 - motion / 35, 0, 1))
    quality = float(
        np.mean(
            [
                blur_score,
                exposure_score,
                clipping_score,
                motion_score,
            ]
        )
    )

    regions = _region_coordinates(face, frame.shape)
    pixels = [
        frame[top:bottom, left:right].reshape(-1, 3)
        for left, top, right, bottom in regions
    ]
    combined_pixels = np.vstack(pixels)
    pixel_brightness = np.mean(combined_pixels, axis=1)
    usable_pixels = combined_pixels[
        (pixel_brightness >= 20) & (pixel_brightness <= 235)
    ]
    if usable_pixels.shape[0] < 100 or quality < 0.3:
        return None, quality, gray_face, regions

    lower = np.percentile(usable_pixels, 5, axis=0)
    upper = np.percentile(usable_pixels, 95, axis=0)
    robust_pixels = np.clip(usable_pixels, lower, upper)
    bgr_mean = np.mean(robust_pixels, axis=0)
    rgb_mean = bgr_mean[::-1]
    return rgb_mean, quality, gray_face, regions


def capture_rgb_trace() -> tuple[np.ndarray, float, float, dict[str, int]]:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise MeasurementError("The default webcam could not be opened.")

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    preview = st.image([])
    progress = st.progress(0, text="Waiting for a stable face")
    timestamps = []
    rgb_samples = []
    quality_scores = []
    rejection_counts = {"no_face": 0, "low_quality": 0}
    previous_face = None
    previous_gray_face = None
    capture_started = time.monotonic()
    first_valid_time = None

    try:
        while time.monotonic() - capture_started < MAX_CAPTURE_SECONDS:
            success, frame = camera.read()
            if not success:
                break
            now = time.monotonic()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            regions = []
            current_quality = 0.0
            if len(faces):
                previous_face = _select_and_smooth_face(
                    faces,
                    previous_face,
                )
                (
                    rgb_sample,
                    current_quality,
                    current_gray_face,
                    regions,
                ) = _extract_rgb_sample(
                    frame,
                    previous_face,
                    previous_gray_face,
                )
                previous_gray_face = current_gray_face
                if rgb_sample is not None:
                    if first_valid_time is None:
                        first_valid_time = now
                    timestamps.append(now)
                    rgb_samples.append(rgb_sample)
                    quality_scores.append(current_quality)
                else:
                    rejection_counts["low_quality"] += 1
            else:
                rejection_counts["no_face"] += 1

            display_frame = frame.copy()
            if previous_face is not None:
                face_x, face_y, face_width, face_height = np.rint(
                    previous_face
                ).astype(int)
                color = (
                    (0, 180, 0)
                    if current_quality >= 0.3
                    else (0, 0, 255)
                )
                cv2.rectangle(
                    display_frame,
                    (face_x, face_y),
                    (face_x + face_width, face_y + face_height),
                    color,
                    2,
                )
                for left, top, right, bottom in regions:
                    cv2.rectangle(
                        display_frame,
                        (left, top),
                        (right, bottom),
                        (255, 180, 0),
                        1,
                    )
            preview.image(
                cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            )

            valid_duration = (
                now - first_valid_time if first_valid_time is not None else 0
            )
            progress.progress(
                min(valid_duration / TARGET_CAPTURE_SECONDS, 1.0),
                text=(
                    f"Stable capture: {valid_duration:.1f}/"
                    f"{TARGET_CAPTURE_SECONDS} seconds · "
                    f"{len(rgb_samples)} valid frames"
                ),
            )
            if (
                valid_duration >= TARGET_CAPTURE_SECONDS
                and len(rgb_samples) >= MIN_VALID_FRAMES
            ):
                break
    finally:
        camera.release()

    if len(rgb_samples) < MIN_VALID_FRAMES:
        raise MeasurementError(
            "Not enough high-quality frames were captured. Improve lighting, "
            "keep one face still, and retry."
        )
    if timestamps[-1] - timestamps[0] < TARGET_CAPTURE_SECONDS:
        raise MeasurementError(
            "The stable capture was too short. Keep your face visible and retry."
        )

    intervals = np.diff(timestamps)
    median_interval = float(np.median(intervals))
    gap_fraction = float(np.mean(intervals > 2.5 * median_interval))
    mean_capture_quality = float(np.mean(quality_scores))
    if gap_fraction > 0.15 or mean_capture_quality < MIN_CAPTURE_QUALITY:
        raise MeasurementError(
            "Capture timing or image quality was too inconsistent for a "
            "reliable estimate. Please retry."
        )

    resampled_trace, sample_rate = resample_rgb_trace(
        timestamps,
        rgb_samples,
    )
    progress.empty()
    return (
        resampled_trace,
        sample_rate,
        mean_capture_quality,
        rejection_counts,
    )


def predict_blood_pressure(
    bundle: dict,
    age: float,
    gender: str,
    signals: dict[str, np.ndarray],
    sample_rate: float,
) -> tuple[float, float]:
    features = build_feature_vector(
        age=age,
        gender=gender,
        ica_signal=signals["ica"],
        chrome_signal=signals["chrom"],
        pos_signal=signals["pos"],
        sample_rate=sample_rate,
    ).reshape(1, -1)
    prediction = float(bundle["model"].predict(features)[0])
    error_p90 = float(
        bundle["validation"]["holdout_metrics"]["absolute_error_p90"]
    )
    return prediction, error_p90


def main() -> None:
    st.title("TeleTibb")
    st.subheader(
        "Research prototype for camera-based heart rate and blood pressure"
    )
    st.warning(
        "TeleTibb is not a medical device. Do not use its output for diagnosis "
        "or treatment decisions."
    )

    age = st.number_input(
        "Age",
        min_value=8,
        max_value=100,
        value=30,
        step=1,
    )
    gender = st.selectbox("Gender used by the research model", ["F", "M"])
    st.caption(
        "Use even lighting, keep one face centered, and remain still for "
        f"{TARGET_CAPTURE_SECONDS} seconds."
    )

    if not st.button("Start measurement", type="primary"):
        return

    try:
        rgb_trace, sample_rate, capture_quality, rejection_counts = (
            capture_rgb_trace()
        )
        with st.spinner("Extracting pulse signals and estimating vitals"):
            signals = {
                "pos": pos_wang(rgb_trace, sample_rate),
                "chrom": chrom_dehaan(rgb_trace, sample_rate),
                "ica": ica_poh(rgb_trace, sample_rate),
            }
            heart_rate, signal_quality, method_estimates = estimate_heart_rate(
                signals,
                sample_rate,
            )
            if signal_quality < MIN_SIGNAL_QUALITY:
                raise MeasurementError(
                    "The pulse methods did not agree strongly enough. Remain "
                    "still, improve lighting, and retry."
                )

            systolic_bundle = load_model_bundle("bp_systolic_model.pkl")
            diastolic_bundle = load_model_bundle("bp_diastolic_model.pkl")
            systolic, systolic_uncertainty = predict_blood_pressure(
                systolic_bundle,
                age,
                gender,
                signals,
                sample_rate,
            )
            diastolic, diastolic_uncertainty = predict_blood_pressure(
                diastolic_bundle,
                age,
                gender,
                signals,
                sample_rate,
            )

        quality = min(capture_quality, signal_quality)
        st.success("Measurement completed")
        heart_rate_column, systolic_column, diastolic_column = st.columns(3)
        heart_rate_column.metric("Heart rate", f"{heart_rate:.0f} bpm")
        systolic_column.metric(
            "Systolic BP",
            f"{systolic:.0f} mmHg",
        )
        diastolic_column.metric(
            "Diastolic BP",
            f"{diastolic:.0f} mmHg",
        )
        st.progress(
            quality,
            text=f"Signal quality: {quality * 100:.0f}%",
        )
        st.info(
            "Participant-level holdout uncertainty (90th-percentile absolute "
            f"error): systolic ±{systolic_uncertainty:.0f} mmHg, "
            f"diastolic ±{diastolic_uncertainty:.0f} mmHg."
        )
        with st.expander("Measurement diagnostics"):
            st.write(f"Measured sampling rate: {sample_rate:.1f} FPS")
            st.write(
                "Heart-rate estimates: "
                + ", ".join(
                    f"{name.upper()} {value:.1f} bpm"
                    for name, value in method_estimates.items()
                )
            )
            st.write(
                "Rejected frames: "
                + ", ".join(
                    f"{reason.replace('_', ' ')} {count}"
                    for reason, count in rejection_counts.items()
                )
            )
    except (MeasurementError, ValueError, RuntimeError) as error:
        st.error(str(error))
    except Exception:
        st.error(
            "The measurement could not be completed. Check the camera and "
            "model files, then retry."
        )


if __name__ == "__main__":
    main()
