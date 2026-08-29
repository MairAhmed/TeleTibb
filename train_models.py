import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from teletibb_signal import build_feature_matrix, feature_names


RANDOM_STATE = 42
TARGETS = {
    "systolic": ("BP_Sys", "bp_systolic_model.pkl"),
    "diastolic": ("BP_Dia", "bp_diastolic_model.pkl"),
}


def participant_split(
    groups: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(
        splitter.split(np.zeros(groups.size), groups=groups)
    )
    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])
    if train_groups.intersection(test_groups):
        raise RuntimeError("Participant leakage detected in the holdout split.")
    return train_indices, test_indices


def candidate_searches(
    grouped_cv: GroupKFold,
    quick: bool,
) -> dict[str, GridSearchCV]:
    tree_count = 120 if quick else 400
    candidates = {
        "ridge": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            {"model__alpha": [1.0, 10.0, 100.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=tree_count,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [6, None],
                "model__min_samples_leaf": [2, 4],
                "model__max_features": [0.7],
            },
        ),
        "extra_trees": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=tree_count,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [6, None],
                "model__min_samples_leaf": [2, 4],
                "model__max_features": [0.7],
            },
        ),
        "gradient_boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingRegressor(
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.05],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [3],
            },
        ),
    }
    return {
        name: GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring="neg_mean_absolute_error",
            cv=grouped_cv,
            n_jobs=-1,
            refit=True,
        )
        for name, (estimator, parameters) in candidates.items()
    }


def regression_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    errors = predicted - expected
    correlation = (
        float(np.corrcoef(expected, predicted)[0, 1])
        if np.std(expected) > 0 and np.std(predicted) > 0
        else 0.0
    )
    return {
        "mae": float(mean_absolute_error(expected, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(expected, predicted))),
        "bias": float(np.mean(errors)),
        "error_standard_deviation": float(np.std(errors)),
        "correlation": correlation,
        "absolute_error_p90": float(np.percentile(np.abs(errors), 90)),
    }


def evaluate_baselines(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    grouped_cv: GroupKFold,
) -> dict[str, dict[str, float]]:
    baseline_models = {
        "mean": (DummyRegressor(strategy="mean"), features),
        "demographics_random_forest": (
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            features[:, -2:],
        ),
    }
    results = {}
    for name, (model, model_features) in baseline_models.items():
        scores = -cross_val_score(
            model,
            model_features,
            target,
            groups=groups,
            cv=grouped_cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        results[name] = {
            "grouped_cv_mae_mean": float(np.mean(scores)),
            "grouped_cv_mae_standard_deviation": float(np.std(scores)),
        }
    return results


def train_target(
    name: str,
    target: np.ndarray,
    features: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    sample_rate: float,
    output_path: Path,
    quick: bool,
) -> dict:
    development_features = features[train_indices]
    development_target = target[train_indices]
    development_groups = groups[train_indices]
    grouped_cv = GroupKFold(n_splits=5)

    searches = candidate_searches(grouped_cv, quick)
    candidate_results = {}
    for model_name, search in searches.items():
        search.fit(
            development_features,
            development_target,
            groups=development_groups,
        )
        candidate_results[model_name] = {
            "grouped_cv_mae_mean": float(-search.best_score_),
            "grouped_cv_mae_standard_deviation": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
            "best_parameters": search.best_params_,
        }

    selected_name = min(
        candidate_results,
        key=lambda candidate: candidate_results[candidate][
            "grouped_cv_mae_mean"
        ],
    )
    selected_search = searches[selected_name]
    holdout_predictions = selected_search.predict(features[test_indices])
    holdout_metrics = regression_metrics(
        target[test_indices],
        holdout_predictions,
    )

    deployment_model = clone(selected_search.best_estimator_)
    deployment_model.fit(features, target)
    metadata = {
        "target": name,
        "selected_model": selected_name,
        "candidate_results": candidate_results,
        "holdout_metrics": holdout_metrics,
    }
    bundle = {
        "schema_version": 1,
        "sample_rate": sample_rate,
        "feature_names": feature_names(),
        "model": deployment_model,
        "validation": metadata,
        "scikit_learn_version": sklearn.__version__,
        "training_rows": int(features.shape[0]),
        "training_participants": int(np.unique(groups).size),
    }
    joblib.dump(bundle, output_path, compress=3)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train reproducible TeleTibb BP models with participant-grouped "
            "validation."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("DataLoader.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer trees for a faster development run.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.data).drop_duplicates().reset_index(drop=True)
    if frame["GUID"].isna().any():
        raise ValueError("Every training row must have a participant GUID.")

    features = build_feature_matrix(frame, args.sample_rate)
    groups = frame["GUID"].astype(str).to_numpy()
    train_indices, test_indices = participant_split(groups)
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 1,
        "data_file": str(args.data),
        "sample_rate": args.sample_rate,
        "rows_after_exact_deduplication": int(frame.shape[0]),
        "participants": int(np.unique(groups).size),
        "feature_count": int(features.shape[1]),
        "feature_names": feature_names(),
        "split": {
            "development_rows": int(train_indices.size),
            "holdout_rows": int(test_indices.size),
            "development_participants": int(
                np.unique(groups[train_indices]).size
            ),
            "holdout_participants": int(
                np.unique(groups[test_indices]).size
            ),
            "participant_overlap": 0,
        },
        "targets": {},
    }

    development_cv = GroupKFold(n_splits=5)
    for name, (target_column, artifact_name) in TARGETS.items():
        target = frame[target_column].to_numpy(dtype=float)
        report["targets"][name] = {
            "baselines": evaluate_baselines(
                features[train_indices],
                target[train_indices],
                groups[train_indices],
                development_cv,
            ),
            **train_target(
                name=name,
                target=target,
                features=features,
                groups=groups,
                train_indices=train_indices,
                test_indices=test_indices,
                sample_rate=args.sample_rate,
                output_path=output_directory / artifact_name,
                quick=args.quick,
            ),
        }

    report_path = output_directory / "model_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
