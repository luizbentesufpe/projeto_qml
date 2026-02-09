# compare_pca_only.py
from __future__ import annotations

import time
import numpy as np
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from rl_and_qml_in_clinical_images.util import Logger, set_seeds
from rl_and_qml_in_clinical_images.dataset import load_chestmnist_pool_flatten
from rl_and_qml_in_clinical_images.modeling.train import split_holdout
from rl_and_qml_in_clinical_images.rl.env import find_threshold, _rates_from_thr
from rl_and_qml_in_clinical_images.rl.rl_config import Config, get_thr_targets


def _safe_auc(y01: np.ndarray, probs: np.ndarray) -> float:
    y01 = np.asarray(y01).astype(int).reshape(-1)
    probs = np.asarray(probs).astype(float).reshape(-1)
    if len(np.unique(y01)) < 2:
        return 0.5
    try:
        a = float(roc_auc_score(y01, probs))
        return 0.5 if not np.isfinite(a) else a
    except Exception:
        return 0.5


def _calib_split_indices(y01: np.ndarray, calib_frac: float, rng: np.random.Generator):
    y01 = np.asarray(y01).astype(int).reshape(-1)
    idx = np.arange(len(y01))

    idx_pos = idx[y01 == 1]
    idx_neg = idx[y01 == 0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    n_cal_pos = int(max(1, round(calib_frac * len(idx_pos)))) if len(idx_pos) else 0
    n_cal_neg = int(max(1, round(calib_frac * len(idx_neg)))) if len(idx_neg) else 0

    cal_idx = np.concatenate([idx_pos[:n_cal_pos], idx_neg[:n_cal_neg]]) if (n_cal_pos + n_cal_neg) > 0 else np.array([], dtype=int)
    fit_idx = np.setdiff1d(idx, cal_idx) if cal_idx.size else idx
    return fit_idx, cal_idx


def _thr_star_from_cal(cfg: Config, y01_cal: np.ndarray, probs_cal: np.ndarray, logits_cal: np.ndarray | None, logger: Logger):
    sens_tgt, spec_min, fpr_max = get_thr_targets(cfg)
    thr_star = find_threshold(
        y01_cal.astype(int),
        probs_cal.astype(np.float32),
        mode=str(cfg.thr_mode),
        sens_target=float(sens_tgt),
        grid_size=int(cfg.grid_size),
        fpr_max=float(fpr_max),
        spec_min=float(spec_min),
        lam_spec=float(getattr(cfg, "thr_lam_spec", 2.0)),
        lam_fpr=float(getattr(cfg, "thr_lam_fpr", 2.0)),
        lam_sens=float(getattr(cfg, "thr_lam_sens", 1.0)),
        logits=(None if logits_cal is None else logits_cal.astype(float)),
        logger=logger,
        return_info=False
    )
    return float(thr_star)


def _model_pca_lr(pca_dim: int, seed: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=int(pca_dim), random_state=int(seed))),
        ("clf", LogisticRegression(
            max_iter=5000, solver="lbfgs",
            class_weight="balanced",
            random_state=int(seed),
        ))
    ])


def _model_pca_svm_calibrated(pca_dim: int, seed: int):
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=int(pca_dim), random_state=int(seed))),
        ("svm", LinearSVC(class_weight="balanced", random_state=int(seed))),
    ])
    # calibra para obter probs (necessário p/ thr*)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def run_pca_baseline_holdout(
    X_train_all: np.ndarray,
    Y_train_all: np.ndarray,
    X_holdout: np.ndarray,
    Y_holdout: np.ndarray,
    cfg: Config,
    logger: Logger,
    seed: int,
    kind: str,
    pca_dim: int,
):
    y01_tr = (np.asarray(Y_train_all).reshape(-1) > 0.5).astype(int)
    y01_ho = (np.asarray(Y_holdout).reshape(-1) > 0.5).astype(int)

    rng = np.random.default_rng(int(getattr(cfg, "thr_calib_seed", 12345)) + 7777 * int(seed))
    fit_idx, cal_idx = _calib_split_indices(
        y01_tr,
        calib_frac=float(getattr(cfg, "thr_calib_frac", 0.20)),
        rng=rng
    )
    X_fit, y_fit = X_train_all[fit_idx], y01_tr[fit_idx]
    X_cal, y_cal = X_train_all[cal_idx], y01_tr[cal_idx]

    t0 = time.perf_counter()

    if kind == "pca_lr":
        model = _model_pca_lr(pca_dim=pca_dim, seed=seed)
        model.fit(X_fit, y_fit)
        probs_cal = model.predict_proba(X_cal)[:, 1]
        logits_cal = None
        try:
            logits_cal = model.decision_function(X_cal)
        except Exception:
            pass
        thr_star = _thr_star_from_cal(cfg, y_cal, probs_cal, logits_cal, logger)
        probs_ho = model.predict_proba(X_holdout)[:, 1]

    elif kind == "pca_svm":
        model = _model_pca_svm_calibrated(pca_dim=pca_dim, seed=seed)
        model.fit(X_fit, y_fit)
        probs_cal = model.predict_proba(X_cal)[:, 1]
        thr_star = _thr_star_from_cal(cfg, y_cal, probs_cal, None, logger)
        probs_ho = model.predict_proba(X_holdout)[:, 1]
    else:
        raise ValueError("kind must be 'pca_lr' or 'pca_svm'")

    dt = time.perf_counter() - t0

    auc = _safe_auc(y01_ho, probs_ho)
    sens, spec, fpr = _rates_from_thr(y01_ho, probs_ho, float(thr_star))

    return {
        "kind": kind,
        "pca_dim": int(pca_dim),
        "holdout": {
            "auc": float(auc),
            "thr_star": float(thr_star),
            "sens@thr*": float(sens),
            "spec@thr*": float(spec),
            "fpr@thr*": float(fpr),
        },
        "time_sec": float(dt),
    }


def main_compare_pca_once():
    SEED = 0
    set_seeds(SEED)
    cfg = Config()

    out_dir = "publication_out_compare_pca_only"
    logger = Logger(out_dir)

    # 1) pool total (eval percent) + 2) holdout intocado
    X_full, Y_full = load_chestmnist_pool_flatten(cfg, percent_total=int(cfg.percent_eval), seed=int(SEED))
    tr_idx, ho_idx = split_holdout(X_full, Y_full, frac=float(cfg.holdout_frac), seed=int(SEED))
    X_train_all, Y_train_all = X_full[tr_idx], Y_full[tr_idx]
    X_holdout,  Y_holdout    = X_full[ho_idx], Y_full[ho_idx]

    D = int(X_train_all.shape[1])
    pca_dim = 49 if D >= 49 else max(2, min(16, D))  # regra simples

    lr = run_pca_baseline_holdout(
        X_train_all, Y_train_all, X_holdout, Y_holdout,
        cfg=cfg, logger=logger, seed=int(SEED),
        kind="pca_lr", pca_dim=int(pca_dim)
    )
    svm = run_pca_baseline_holdout(
        X_train_all, Y_train_all, X_holdout, Y_holdout,
        cfg=cfg, logger=logger, seed=int(SEED),
        kind="pca_svm", pca_dim=int(pca_dim)
    )

    def fmt(x): return f"{float(x):.4f}"

    print("\n==================== PCA BASELINES (ONE SPLIT) ====================")
    print(f"Seed={SEED} | Holdout_frac={cfg.holdout_frac} | PCA_dim={pca_dim} | thr_mode={cfg.thr_mode}")
    print("\n--- PCA + LR (holdout) ---")
    print(f"AUC={fmt(lr['holdout']['auc'])}  thr*={lr['holdout']['thr_star']:.3f}  "
          f"SENS={fmt(lr['holdout']['sens@thr*'])}  SPEC={fmt(lr['holdout']['spec@thr*'])}  "
          f"FPR={fmt(lr['holdout']['fpr@thr*'])}  time={lr['time_sec']:.2f}s")

    print("\n--- PCA + SVM(cal) (holdout) ---")
    print(f"AUC={fmt(svm['holdout']['auc'])}  thr*={svm['holdout']['thr_star']:.3f}  "
          f"SENS={fmt(svm['holdout']['sens@thr*'])}  SPEC={fmt(svm['holdout']['spec@thr*'])}  "
          f"FPR={fmt(svm['holdout']['fpr@thr*'])}  time={svm['time_sec']:.2f}s")

    print("\n--- Compare with your RL best-episode log (same metrics) ---")
    print("Take from RL log: AUC, SENS, SPEC, THR* (holdout). Then apply your rule:")
    print("If PCA >> RL: don't migrate to PCA; fix proxy/search & separability (e.g., train alpha/beta during search).")
    print("If PCA ~ RL: PCA is baseline; focus on your method.")

    return {"pca_lr": lr, "pca_svm": svm}


if __name__ == "__main__":
    main_compare_pca_once()
