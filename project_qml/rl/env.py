from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
import time
from collections import defaultdict, deque
from typing import Any
from contextlib import nullcontext


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from rl_and_qml_in_clinical_images.features import (
    compute_saliency_importance_patches,
    compute_saliency_importance_pixels,
    init_feature_bank,
    make_patch_groups,
    patchify_mean_flat,
)
from rl_and_qml_in_clinical_images.modeling.model import (
    BinaryCQV_End2End,
    compute_pos_weight,
    init_head_bias_with_prevalence,
)
from rl_and_qml_in_clinical_images.rl.actions import (
    OpType,
    action_is_valid_for_qubits,
    build_action_list_superset,
)
from rl_and_qml_in_clinical_images.rl.losses import focal_loss_with_logits
from rl_and_qml_in_clinical_images.rl.rl_config import Config, get_thr_targets
from rl_and_qml_in_clinical_images.util import Logger, RunningPctl

SEED_DELTA_CALIB = 10_000
SEED_DELTA_PROXY = 1_000


# -------------------------
# Small, safe utilities
# -------------------------
def _safe_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return dict(
            min=np.nan,
            mean=np.nan,
            max=np.nan,
            p5=np.nan,
            p50=np.nan,
            p95=np.nan,
            range=np.nan,
        )
    mn = float(np.min(x))
    mx = float(np.max(x))
    return dict(
        min=mn,
        mean=float(np.mean(x)),
        max=mx,
        p5=float(np.percentile(x, 5)),
        p50=float(np.percentile(x, 50)),
        p95=float(np.percentile(x, 95)),
        range=float(mx - mn),
    )

def _collapse_health(x: np.ndarray, range_min: float = 0.06, std_min: float = 0.02) -> dict[str, float]:
    """
    Simple collapse detector based on robust range (p95-p5) and std.
    Returns dict with flags and key stats.
    """
    z = np.asarray(x, dtype=float).reshape(-1)
    if z.size == 0:
        return {"collapsed": 1.0, "p95_p5": float("nan"), "std": float("nan"), "mean": float("nan")}
    p5, p95 = np.percentile(z, [5, 95])
    pr = float(p95 - p5)
    sd = float(np.std(z))
    mu = float(np.mean(z))
    collapsed = 1.0 if (pr < float(range_min) or sd < float(std_min)) else 0.0
    return {"collapsed": float(collapsed), "p95_p5": float(pr), "std": float(sd), "mean": float(mu)}

def _p95_p5(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=float).reshape(-1)
    if z.size == 0:
        return float("nan")
    p5, p95 = np.percentile(z, [5, 95])
    return float(p95 - p5)


def _metrics_at_threshold(p: np.ndarray, y: np.ndarray, t: float) -> dict[str, float]:
    # NOTE: y must be {0,1}
    yp = (p >= float(t)).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())

    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    fpr = fp / max(1, (fp + tn))
    prec = tp / max(1, (tp + fp))

    beta2 = 4.0  # F2
    f2 = (1.0 + beta2) * prec * sens / max(1e-12, (beta2 * prec + sens))
    ba = 0.5 * (sens + spec)
    return dict(
        t=float(t),
        sens=float(sens),
        spec=float(spec),
        fpr=float(fpr),
        prec=float(prec),
        f2=float(f2),
        ba=float(ba),
        tp=float(tp),
        tn=float(tn),
        fp=float(fp),
        fn=float(fn),
    )


def _safe_thr_grid(grid_size: int) -> np.ndarray:
    """
    Grade robusta sem 0/1 (evita degenerar tudo-positivo/tudo-negativo por default).
    """
    gs = int(max(3, grid_size))
    eps = 1e-6
    return np.linspace(eps, 1.0 - eps, gs, dtype=np.float32)


def _bce_logits_loss_torch(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor | None = None) -> float:
    """
    BCEWithLogitsLoss em float, robusto para shapes.
    Retorna float (python) para logging.
    """
    if logits.dim() > 1:
        logits = logits.view(-1)
    if y.dim() > 1:
        y = y.view(-1)
    y = y.to(dtype=logits.dtype)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = crit(logits, y)
    return float(loss.detach().cpu().item())

def _ema_update(prev: float | None, x: float, alpha: float = 0.05) -> float:
    """
    EMA simples para suavizar loss por iteração.
    """
    x = float(x)
    if prev is None or (not np.isfinite(prev)):
        return x
    return float((1.0 - float(alpha)) * float(prev) + float(alpha) * x)


# -------------------------
# State encoding helpers
# -------------------------
def empty_state(L_max: int) -> np.ndarray:
    return np.zeros((5, L_max), dtype=np.int64)


def encode_action_in_state(state: np.ndarray, step: int, action: Any) -> np.ndarray:
    kind = action[0]
    state[:, step] = 0

    if kind == "ROT":
        _, ax, q, _ = action
        state[1, step] = q + 1
        state[2, step] = OpType.ROT.value
        state[3, step] = ax

    elif kind == "ENC":
        _, ax, q, feat = action
        state[1, step] = q + 1
        state[2, step] = OpType.ENC.value
        state[3, step] = ax
        state[4, step] = feat + 1

    elif kind == "CNOT":
        _, ci, tj, _ = action
        if ci != tj:
            state[0, step] = ci + 1
            state[1, step] = tj + 1
            state[2, step] = OpType.CNOT.value

    return state


def sanitize_architecture(arch: torch.Tensor, n_qubits: int) -> torch.Tensor:
    arch = arch.clone()
    L = int(arch.shape[1])

    for l in range(L):
        op = int(arch[2, l].item())
        c = int(arch[0, l].item())
        t = int(arch[1, l].item())
        ax = int(arch[3, l].item())
        f1 = int(arch[4, l].item())

        if t > n_qubits or c > n_qubits:
            arch[:, l] = 0
            continue

        if op == OpType.CNOT.value:
            if c == 0 or t == 0 or c == t:
                arch[:, l] = 0

        elif op == OpType.ROT.value:
            # stronger sanitization: target + axis must exist
            if (t == 0) or (ax == 0):
                arch[:, l] = 0

        elif op == OpType.ENC.value:
            # stronger sanitization: target + axis + feature must exist
            if (t == 0) or (ax == 0) or (f1 == 0):
                arch[:, l] = 0

    return arch


def state_to_vec(state: np.ndarray, last_metric: float, current_n_qubits: int, cfg: Config) -> np.ndarray:
    flat = state.flatten().astype(np.float32)
    extras = np.zeros(2, dtype=np.float32)
    extras[0] = np.float32(last_metric)

    denom = max(1, (cfg.max_qubits - cfg.min_qubits))
    extras[1] = np.float32((current_n_qubits - cfg.min_qubits) / denom)
    return np.concatenate([flat, extras])


# -------------------------
# Threshold utilities
# -------------------------
def compute_f2_from_probs(y_true: np.ndarray, probs: np.ndarray, thr: float) -> float:
    y_pred = (probs >= thr).astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    beta2 = 4.0
    denom = (beta2 * prec + rec)
    if denom <= 0:
        return 0.0
    return (1 + beta2) * prec * rec / denom


def _confusion_from_thr(y_true: np.ndarray, probs: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    y = y_true.reshape(-1).astype(np.int32)
    p = probs.reshape(-1).astype(np.float32)
    yp = (p >= float(thr)).astype(np.int32)

    tp = int(((y == 1) & (yp == 1)).sum())
    fp = int(((y == 0) & (yp == 1)).sum())
    tn = int(((y == 0) & (yp == 0)).sum())
    fn = int(((y == 1) & (yp == 0)).sum())
    return tp, fp, tn, fn


def _rates_from_thr(y_true: np.ndarray, probs: np.ndarray, thr: float) -> tuple[float, float, float]:
    tp, fp, tn, fn = _confusion_from_thr(y_true, probs, thr)
    n_pos = tp + fn
    n_neg = tn + fp
    sens = tp / max(n_pos, 1)
    spec = tn / max(n_neg, 1)
    fpr = fp / max(n_neg, 1)
    return float(sens), float(spec), float(fpr)


def _balanced_acc_from_thr(y_true: np.ndarray, probs: np.ndarray, thr: float) -> float:
    sens, spec, _ = _rates_from_thr(y_true, probs, thr)
    return float(0.5 * (sens + spec))


def find_threshold(
    y_true,
    probs,
    mode="soft",  # <<< default becomes soft (stable)
    sens_target=0.85,
    spec_min=0.80,
    fpr_max=0.10,
    grid_size=401,
    # soft-constraint weights
    lam_spec=2.0,
    lam_fpr=2.0,
    lam_sens=1.0,
    # optional diagnostics
    logits=None,
    logger=None,
    return_info=False,
    saturation_abslogit_p95_thr: float | None = None,
    saturation_fallback_thr: float | None = None,
):
    """
    Threshold selection.

    mode="hard":
    - if exists viable -> pick best F2
    - else -> fallback BA (old behavior)

    mode="soft" (recommended for collapsed scores / RL proxy):
    maximize:
        J(t) = F2(t)
            - lam_spec * max(0, spec_min - spec(t))
            - lam_fpr  * max(0, fpr(t) - fpr_max)
            - lam_sens * max(0, sens_target - sens(t))

    This avoids 'viable=0 => BA lottery' and prevents thr* degenerating to 0.0/0.05.
    """
    y = np.asarray(y_true).reshape(-1).astype(int)
    p = np.asarray(probs).reshape(-1).astype(float)

    # defensive
    if p.size == 0:
        t = 0.5
        info = {"t": float(t), "note": "empty_probs", "viable_count": 0, "collapse_pen": 0.0}
        return (float(t), info) if bool(return_info) else float(t)

    # detect collapse (your exact symptom)
    pmin = float(np.min(p))
    pmax = float(np.max(p))
    prange = float(pmax - pmin)

    if prange < 1e-3:
        # stable, non-extreme fallback
        t = 0.5
        if (
            logits is not None
            and saturation_abslogit_p95_thr is not None
            and saturation_fallback_thr is not None
        ):
            p95_abs = float(np.percentile(np.abs(np.asarray(logits, dtype=float)), 95))
            if np.isfinite(p95_abs) and (p95_abs >= float(saturation_abslogit_p95_thr)):
                t = float(saturation_fallback_thr)

        sens, spec, fpr = _rates_from_thr(y, p, float(t))
        info = {
            "t": float(t),
            "mode": "collapse_fallback",
            "viable_count": 0,
            "collapse_pen": float(np.clip(1.0 - prange / 1e-3, 0.0, 1.0)),
            "sens": float(sens),
            "spec": float(spec),
            "fpr": float(fpr),
            "pmin": pmin,
            "pmax": pmax,
            "prange": prange,
        }
        if logger is not None:
            ps = _safe_stats(p)
            logger.log_to_file(
                "thr",
                f"[thr] COLLAPSE probs(min/mean/max)={ps['min']:.4f}/{ps['mean']:.4f}/{ps['max']:.4f} "
                f"pctl(p5/p50/p95)={ps['p5']:.4f}/{ps['p50']:.4f}/{ps['p95']:.4f} range={ps['range']:.6f}",
            )
            if logits is not None:
                ls = _safe_stats(np.asarray(logits, dtype=float))
                logger.log_to_file(
                    "thr",
                    f"[thr] COLLAPSE logits(min/mean/max)={ls['min']:.4f}/{ls['mean']:.4f}/{ls['max']:.4f} "
                    f"pctl(p5/p50/p95)={ls['p5']:.4f}/{ls['p50']:.4f}/{ls['p95']:.4f} range={ls['range']:.6f}",
                )
            logger.log_to_file(
                "thr",
                f"[thr] collapse_fallback t=0.5000 sens={float(sens):.4f} spec={float(spec):.4f} fpr={float(fpr):.4f}",
            )
        return (float(t), info) if bool(return_info) else float(t)

    # thresholds only where it matters (inside score support)
    lo = max(0.0, pmin - 1e-6)
    hi = min(1.0, pmax + 1e-6)
    thrs = np.linspace(lo, hi, int(grid_size))

    best = None
    best_viable = None
    best_ba = None
    viable_count = 0

    for t in thrs:
        t = float(t)
        sens, spec, fpr = _rates_from_thr(y, p, t)

        yp = (p >= t).astype(int)
        tp = int(((y == 1) & (yp == 1)).sum())
        fp = int(((y == 0) & (yp == 1)).sum())
        prec = float(tp) / float(max(tp + fp, 1))

        beta2 = 4.0
        f2 = (1.0 + beta2) * prec * sens / max(1e-12, (beta2 * prec + sens))
        ba = 0.5 * (sens + spec)

        viable = (sens >= float(sens_target)) and (spec >= float(spec_min)) and (fpr <= float(fpr_max))
        if viable:
            viable_count += 1

        if str(mode).lower().startswith("hard"):
            if viable and ((best_viable is None) or (f2 > best_viable["f2"])):
                best_viable = dict(
                    t=t,
                    f2=float(f2),
                    ba=float(ba),
                    sens=float(sens),
                    spec=float(spec),
                    fpr=float(fpr),
                    obj=float(f2),
                )
            if (best_ba is None) or (ba > best_ba["ba"]):
                best_ba = dict(
                    t=t,
                    f2=float(f2),
                    ba=float(ba),
                    sens=float(sens),
                    spec=float(spec),
                    fpr=float(fpr),
                    obj=float(ba),
                )
            continue

        # soft objective
        pen_spec = max(0.0, float(spec_min) - float(spec))
        pen_fpr = max(0.0, float(fpr) - float(fpr_max))
        pen_sens = max(0.0, float(sens_target) - float(sens))
        obj = float(f2) - float(lam_spec) * pen_spec - float(lam_fpr) * pen_fpr - float(lam_sens) * pen_sens

        cand = dict(
            t=t,
            f2=float(f2),
            ba=float(ba),
            sens=float(sens),
            spec=float(spec),
            fpr=float(fpr),
            obj=float(obj),
            viable=bool(viable),
        )

        # tie-break: prefer higher spec then sens (more stable)
        if (best is None) or (cand["obj"] > best["obj"]) or (
            abs(cand["obj"] - best["obj"]) < 1e-12 and (cand["spec"], cand["sens"]) > (best["spec"], best["sens"])
        ):
            best = cand

    if str(mode).lower().startswith("hard"):
        chosen = best_viable if (best_viable is not None) else best_ba
    else:
        chosen = best

    info = dict(chosen) if isinstance(chosen, dict) else {"t": float(chosen)}
    info.update(
        {
            "mode": str(mode),
            "viable_count": int(viable_count),
            "pmin": float(pmin),
            "pmax": float(pmax),
            "prange": float(prange),
            "collapse_pen": 0.0,
        }
    )

    if logger is not None:
        ps = _safe_stats(p)
        p_rng = float(ps.get("max", pmax)) - float(ps.get("min", pmin))

        if str(mode).lower().startswith("soft"):
            pen_spec = max(0.0, float(spec_min) - float(info.get("spec", 0.0)))
            pen_fpr = max(0.0, float(info.get("fpr", 0.0)) - float(fpr_max))
            pen_sens = max(0.0, float(sens_target) - float(info.get("sens", 0.0)))

            logger.log_to_file(
                "thr",
                f"[thr] mode=soft viable={int(viable_count)}/{len(thrs)} "
                f"best t={float(info.get('t', 0.5)):.4f} obj={float(info.get('obj', 0.0)):.4f} "
                f"f2={float(info.get('f2', 0.0)):.4f} sens={float(info.get('sens', 0.0)):.4f} "
                f"spec={float(info.get('spec', 0.0)):.4f} fpr={float(info.get('fpr', 0.0)):.4f} | "
                f"pen(spec/fpr/sens)={pen_spec:.3f}/{pen_fpr:.3f}/{pen_sens:.3f} "
                f"lam(spec/fpr/sens)={float(lam_spec):.2f}/{float(lam_fpr):.2f}/{float(lam_sens):.2f} | "
                f"probs(min/mean/max)={ps.get('min', pmin):.4f}/{ps.get('mean', float(np.mean(p))):.4f}/{ps.get('max', pmax):.4f} "
                f"range={p_rng:.6f}",
            )
        else:
            logger.log_to_file(
                "thr",
                f"[thr] mode={str(mode)} viable={int(viable_count)}/{len(thrs)} "
                f"best t={float(info.get('t', 0.5)):.4f} obj={float(info.get('obj', 0.0)):.4f} "
                f"f2={float(info.get('f2', 0.0)):.4f} sens={float(info.get('sens', 0.0)):.4f} "
                f"spec={float(info.get('spec', 0.0)):.4f} fpr={float(info.get('fpr', 0.0)):.4f} | "
                f"probs(min/mean/max)={ps.get('min', pmin):.4f}/{ps.get('mean', float(np.mean(p))):.4f}/{ps.get('max', pmax):.4f} "
                f"range={p_rng:.6f}",
            )

        if logits is not None:
            ls = _safe_stats(np.asarray(logits, dtype=float))
            logger.log_to_file(
                "thr",
                f"[thr] logits(min/mean/max)={ls['min']:.4f}/{ls['mean']:.4f}/{ls['max']:.4f} "
                f"range={ls['range']:.4f} pctl(p5/p50/p95)={ls['p5']:.4f}/{ls['p50']:.4f}/{ls['p95']:.4f}",
            )
            # keep same odd try/except behavior (defensive)
            try:
                _ = ls["range"]
            except Exception:
                logits_np = np.asarray(logits, dtype=float).reshape(-1)
                lmin = float(np.min(logits_np)) if logits_np.size else 0.0
                lmax = float(np.max(logits_np)) if logits_np.size else 0.0
                l_rng = float(lmax - lmin)
                logger.log_to_file(
                    "thr",
                    f"[thr] logits(min/mean/max)={ls.get('min', lmin):.4f}/{ls.get('mean', float(np.mean(logits_np))):.4f}/{ls.get('max', lmax):.4f} "
                    f"range={l_rng:.6f} pctl(p5/p50/p95)={ls.get('p5', float('nan')):.4f}/{ls.get('p50', float('nan')):.4f}/{ls.get('p95', float('nan')):.4f}",
                )

    t_star = float(info.get("t", 0.5))
    return (t_star, info) if bool(return_info) else t_star


class QMLEnvEnd2End:
    def __init__(
        self,
        X_tr,
        Y_tr,
        X_val,
        Y_val,
        cfg: Config,
        logger: Logger,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ):
    
        self.cfg = replace(cfg)
        self.seed = int(seed)
        self.logger = logger
        self.DEVICE = torch.device(device)

        # cache do último custo real medido (publication-grade) para não medir tape toda hora
        self._last_depth_tape = None
        self._last_cnot_tape = None

        self.thr = 0.5
        self.last_ep_score = None
        self.last_spec = 0.0  # <<< NEW: para Youden J

        self.current_n_qubits = int(cfg.start_qubits)
        self._qubit_cooldown = 0

        self.X_tr = torch.as_tensor(X_tr, dtype=torch.float32, device=self.DEVICE)
        self.Y_tr = torch.as_tensor(Y_tr, dtype=torch.float32, device=self.DEVICE)
        self.X_val = torch.as_tensor(X_val, dtype=torch.float32, device=self.DEVICE)
        self.Y_val = torch.as_tensor(Y_val, dtype=torch.float32, device=self.DEVICE)

        if self.X_tr.dim() == 1:
            self.X_tr = self.X_tr.unsqueeze(1)
        if self.X_val.dim() == 1:
            self.X_val = self.X_val.unsqueeze(1)

        self.X_tr_raw = self.X_tr.detach().clone()
        self.X_val_raw = self.X_val.detach().clone()


        self._rng = np.random.default_rng(int(seed))

        # ----------------------------------------------------------
        # Domain detection: only build patch_groups for image-like inputs
        # ----------------------------------------------------------
        self.patch_groups = None
        self.P = None

        # Prefer explicit cfg flag if you add it; fallback to input_dim==784.
        # This makes env robust for tabular / toy datasets.
        input_dim_now = int(self.X_tr.shape[1]) if self.X_tr.dim() == 2 else int(self.X_tr.view(self.X_tr.shape[0], -1).shape[1])
        cfg_input_kind = str(getattr(self.cfg, "input_kind", "")).lower().strip()  # optional: "image" | "tabular"
        self.is_image_like = bool(cfg_input_kind == "image" or input_dim_now == 784)

        if self.is_image_like and bool(getattr(self.cfg, "use_patch_bank", False)):
            self.patch_groups = make_patch_groups(28, 28, cfg.patch_size, cfg.patch_stride)
            self.P = int(len(self.patch_groups))
            self.logger.log_to_file(
                "patchify",
                f"[patch_groups] enabled is_image_like=1 P={self.P} patch={cfg.patch_size} stride={cfg.patch_stride}",
            )
        else:
            self.logger.log_to_file(
                "patchify",
                f"[patch_groups] disabled is_image_like={int(self.is_image_like)} use_patch_bank={int(bool(getattr(self.cfg,'use_patch_bank',False)))} input_dim={input_dim_now}",
            )

        # ----------------------------------------------------------
        # Effective feature-bank sizing / schedule (patch-bank aware)
        # ----------------------------------------------------------
        # Goal:
        # - When use_patch_bank=True, the maximum meaningful "feature id space"
        #   is bounded by P patches (esp. when patch_bank_compact_features=True).
        # - Even in your diagnostic mode (use_patch_bank=True, compact=False),
        #   you typically set feature_bank_size=P (e.g., 49). Using min(...,P)
        #   keeps everything consistent and prevents action indices > P.
        if bool(self.cfg.use_patch_bank) and (self.P is not None):
            max_k = int(min(int(self.cfg.feature_bank_size), int(self.P)))
            min_k = int(min(int(self.cfg.feature_bank_min_size), int(max_k)))

            self.feature_bank_size_eff = int(max_k)
            self.feature_bank_min_eff  = int(min_k)

            # Schedule source MUST be cfg.feature_bank_schedule (tuple),
            # not feature_bank_size (int).
            sched = tuple(getattr(self.cfg, "feature_bank_schedule", ()))
            if len(sched) == 0:
                sched = (int(max_k),)

            # Clamp schedule values into [min_k, max_k] and keep them ints
            self.feature_bank_schedule_eff = tuple(
                int(np.clip(int(k), int(min_k), int(max_k))) for k in sched
            )
        else:
            # Non patch-bank: keep original domain (pixels / external features)
            self.feature_bank_size_eff = int(self.cfg.feature_bank_size)
            self.feature_bank_min_eff  = int(min(int(self.cfg.feature_bank_min_size), int(self.feature_bank_size_eff)))

            sched = tuple(getattr(self.cfg, "feature_bank_schedule", ()))
            if len(sched) == 0:
                sched = (int(self.feature_bank_size_eff),)
            self.feature_bank_schedule_eff = tuple(
                int(np.clip(int(k), int(self.feature_bank_min_eff), int(self.feature_bank_size_eff))) for k in sched
            )

        # >>> CRITICAL FIX (semantic correctness): compact patch features => X becomes (B,P)
        if bool(self.cfg.use_patch_bank) and bool(getattr(self.cfg, "patch_bank_compact_features", False)
                                                   and (self.patch_groups is not None) and bool(self.is_image_like)):
            Dtr = int(self.X_tr.shape[1])
            Dva = int(self.X_val.shape[1])
            if Dtr == 784 and Dva == 784:
                self.X_tr = patchify_mean_flat(self.X_tr, self.patch_groups, device=self.DEVICE)
                self.X_val = patchify_mean_flat(self.X_val, self.patch_groups, device=self.DEVICE)
                logger.log_to_file(
                    "patchify",
                    f"[compact] X_tr -> {tuple(self.X_tr.shape)}  X_val -> {tuple(self.X_val.shape)} (P={self.P})",
                )
            else:
                # Already compact (e.g., D=P=49 or other feature pipeline). Don't patchify twice.
                logger.log_to_file(
                    "patchify",
                    f"[skip compact] X already compact? X_tr.shape={tuple(self.X_tr.shape)} X_val.shape={tuple(self.X_val.shape)} "
                    f"(expected 784 to patchify; P={self.P})",
                )

        if len(self.feature_bank_schedule_eff) == 0:
            self.feature_bank_schedule_eff = (int(self.cfg.feature_bank_size),)
        # ==========================================================
        # NEW: Input normalization (train stats -> apply train/val)
        # Goal: increase separability to avoid probs collapsing ~0.4
        # Works best for patch-compact features (B,P).
        # ==========================================================
        self._x_mean = None
        self._x_std = None
        if bool(getattr(self.cfg, "normalize_inputs", True)):
            # per-feature normalization by default (recommended)
            eps = float(getattr(self.cfg, "norm_eps", 1e-6))
            per_feature = bool(getattr(self.cfg, "norm_per_feature", True))
            clip = float(getattr(self.cfg, "norm_clip", 0.0))  # 0 disables
            use_tanh = bool(getattr(self.cfg, "norm_tanh", False))

            with torch.no_grad():
                if per_feature:
                    mu = self.X_tr.mean(dim=0, keepdim=True)
                    sd = self.X_tr.std(dim=0, keepdim=True).clamp_min(eps)
                else:
                    mu = self.X_tr.mean()
                    sd = self.X_tr.std().clamp_min(eps)

                self._x_mean = mu
                self._x_std = sd

                self.X_tr = (self.X_tr - mu) / sd
                self.X_val = (self.X_val - mu) / sd

                if clip > 0.0:
                    self.X_tr = torch.clamp(self.X_tr, -clip, clip)
                    self.X_val = torch.clamp(self.X_val, -clip, clip)
                if use_tanh:
                    self.X_tr = torch.tanh(self.X_tr)
                    self.X_val = torch.tanh(self.X_val)

            self.logger.log_to_file(
                "norm",
                f"[norm] enabled per_feature={per_feature} eps={eps} clip={clip} tanh={use_tanh} "
                f"X_tr.shape={tuple(self.X_tr.shape)} X_val.shape={tuple(self.X_val.shape)}",
            )
        self.ACTIONS = build_action_list_superset(
            int(cfg.max_qubits),
            int(self.feature_bank_size_eff),
            allow_nop=bool(cfg.allow_nop),
        )
        self.N_ACTIONS = len(self.ACTIONS)


        self.tr_dl_full = DataLoader(
            TensorDataset(self.X_tr, self.Y_tr),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        # Small fixed stratified subset loader for fast inner optimization inside step()
        self.tr_dl_small = self._make_small_stratified_loader(
            X=self.X_tr,
            Y=self.Y_tr,
            subset_size=int(cfg.inner_train_subset_size),
            batch_size=int(cfg.batch_size),
            seed=int(seed),
        )

        self.pos_weight = compute_pos_weight(self.Y_tr, self.DEVICE)
        self.prev_train = (self.Y_tr.detach().cpu().numpy() > 0.5).astype(int)

        self._recent_actions_q = deque(maxlen=int(getattr(cfg, "recent_actions_maxlen", 512)))
        self.recent_actions = set()
        # FIX-4: persistent inter-episode arch dedup set.
        # Prevents same terminal architecture from being rewarded without penalty
        # across episodes (root cause of repeat=0.000 in eps 2-5).
        self._terminal_arch_hashes: set = set()
        self._ep_count: int = 0
        # FIX-5: thr* per-seed list for stability penalty in train.py
        self._terminal_thr_stars: list = []


        self.current_bank_k = int(self.feature_bank_size_eff)
        self.episode_times = []

        # Reward accounting per episode
        self._ep_reward_sums = defaultdict(float)

        self._depth_p95 = RunningPctl(
            p=float(getattr(cfg, "depth_ref_pctl", 95.0)),
            maxlen=int(getattr(cfg, "depth_ref_buf", 512)),
        )
        self._depth_ref_cached = float(getattr(cfg, "depth_ref_default", 10.0))
        self._last_depth_raw = 0.0
        self._calib_cache = {}
        self._arch_result_cache: dict = {}
        self._arch_result_cache_maxsize: int = 64

        self.feature_bank = None
        self.reset()
        self._metric_weight = 1.0

    def set_metric_weight(self, w: float) -> None:
        self._metric_weight = float(max(0.0, w))

    def _depth_ref(self) -> float:
        """
        Online p95 reference. Bounded away from 0.
        Warmup handled by RunningPctl.value(default=...).
        """
        ref = float(self._depth_p95.value(default=float(self._depth_ref_cached)))
        ref_min = float(getattr(self.cfg, "depth_ref_min", 8.0))
        ref = float(max(ref, ref_min))
        self._depth_ref_cached = float(ref)
        return float(ref)

    def flush_episode_reward_sums(self) -> dict:
        out = dict(self._ep_reward_sums)
        self._ep_reward_sums = defaultdict(float)
        return out

    def _make_small_stratified_loader(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        subset_size: int,
        batch_size: int,
        seed: int = 0,
    ) -> DataLoader:
        """
        Create a fixed, stratified subset DataLoader (by label) for inner training.
        This makes RL search much faster while remaining reproducible & fair by seed.
        """
        y = (Y.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]

        rng = np.random.default_rng(int(seed))
        subset_size = int(max(2, min(subset_size, len(y))))
        k_pos = min(len(idx_pos), subset_size // 2)
        k_neg = min(len(idx_neg), subset_size - k_pos)

        sel_pos = rng.choice(idx_pos, k_pos, replace=False) if k_pos > 0 else np.array([], dtype=int)
        sel_neg = rng.choice(idx_neg, k_neg, replace=False) if k_neg > 0 else np.array([], dtype=int)
        sel = np.concatenate([sel_pos, sel_neg])
        if len(sel) < 2:
            sel = rng.choice(np.arange(len(y)), subset_size, replace=False)
        rng.shuffle(sel)

        Xs = X[torch.as_tensor(sel, device=X.device)]
        Ys = Y[torch.as_tensor(sel, device=Y.device)]

        # NOTE: keep tensors on GPU (self.DEVICE) to avoid host<->device every step
        ds = TensorDataset(Xs, Ys)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        self.logger.log_to_file("speed", f"[inner subset] size={len(ds)} batch={batch_size}")
        return dl

    @torch.no_grad()
    def _predict_probs_capped(self, model, X: torch.Tensor, cap: int) -> tuple[np.ndarray, int]:
        """
        Predict probabilities with an example cap to keep inner-loop fast.
        """
        model.eval()
        n = int(min(int(cap), int(X.shape[0])))
        if n <= 0:
            n = int(X.shape[0])

        logits = model(X[:n].detach())
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        return probs, n

    @torch.no_grad()
    def _predict_logits_probs_capped(self, model, X: torch.Tensor, cap: int) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Same as _predict_probs_capped, but also returns logits (for diagnosing collapse).
        """
        model.eval()
        n = int(min(int(cap), int(X.shape[0])))
        if n <= 0:
            n = int(X.shape[0])

        logits_t = model(X[:n].detach())
        logits = logits_t.detach().cpu().numpy().reshape(-1)

        # só pra calcular probs (não mexe no treino), evita sigmoid saturar
        logits_for_sigmoid = np.clip(logits, -12.0, 12.0)  # 12 já dá probs ~6e-6..0.999994
        probs = 1.0 / (1.0 + np.exp(-logits_for_sigmoid))

        return logits, probs, n
    
    @torch.no_grad()
    def _predict_logits_probs_capped_train(self, model, X: torch.Tensor, cap: int) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Same as _predict_probs_capped, but also returns logits (for diagnosing collapse).
        """
        model.eval()
        n = int(min(int(cap), int(X.shape[0])))
        if n <= 0:
            n = int(X.shape[0])

        logits_t = model(X[:n].detach())
        logits = logits_t.detach().cpu().numpy().reshape(-1)

        # -----------------------------------------
        # Temperature scaling ONLY for calibration
        # -----------------------------------------
        abs_logits = np.abs(logits)
        p95 = np.percentile(abs_logits, 95)
        T = max(1.0, p95 / 8.0)   # push p95(|logit/T|) ≈ 8

        logits_T = logits / T
        probs = 1.0 / (1.0 + np.exp(-logits_T))

        return logits, logits_T, probs, n

    def set_current_bank_k(self, k: int) -> None:
        lo = int(getattr(self, "feature_bank_min_eff", self.cfg.feature_bank_min_size))
        hi = int(getattr(self, "feature_bank_size_eff", self.cfg.feature_bank_size))
        k = int(np.clip(int(k), lo, hi))

        self.current_bank_k = k
        self.logger.log_to_file("feature_bank", f"[feature_bank k] current_bank_k={self.current_bank_k}")

    def _build_model(self, arch_mat: torch.Tensor, diff_method: str = "adjoint") -> BinaryCQV_End2End:
        arch_mat = sanitize_architecture(arch_mat, self.current_n_qubits)
        model =  BinaryCQV_End2End(
            arch_mat=arch_mat,
            n_qubits=self.current_n_qubits,
            enc_lambda=float(self.cfg.enc_lambda),
            diff_method=str(diff_method),
            input_dim=int(self.X_tr.shape[1]),
            enc_affine_mode=str(getattr(self.cfg, "enc_affine_mode", "per_feature")),
            enc_alpha_init=float(getattr(self.cfg, "enc_alpha_init", 1.0)),
            enc_beta_init=float(getattr(self.cfg, "enc_beta_init", 0.0)),
            enc_beta_max=float(getattr(self.cfg, "enc_beta_max", 1.0)),
            use_batched_qnode=bool(getattr(self.cfg, "use_batched_qnode", True)),
        ).to(self.DEVICE)

        # ------------------------------------------
        # Logit-scale: fixed in SEARCH, trainable in FINAL
        # ------------------------------------------

        
        phase = str(getattr(self.cfg, "phase", "search")).lower()
        if phase.startswith("final"):
            # allow training in FINAL
            model.set_clamp_logits(True, clamp_value=float(getattr(self.cfg, "final_logit_clamp", 30.0)))
            model.logit_scale_eval_only = False
            model.set_logit_scale_trainable(True)
        else:
            # freeze in SEARCH + set temperature
            # SEARCH: trainable com range restrito para estabilidade
            # Razão: logit_scale=10.0 fixo + VQC features de baixa variância
            # => sigmoid satura tudo perto de 0 => Pcollapsed=37% (visto nos logs)
            search_ls_trainable = bool(getattr(self.cfg, "search_logit_scale_trainable", True))
            ls_init = float(getattr(self.cfg, "search_logit_scale", 2.0))   # init baixo evita saturação imediata
            model.set_clamp_logits(False)
            model.logit_scale_eval_only = False
            model.set_logit_scale_trainable(search_ls_trainable, value=ls_init)
            # Restringir range: escala pequena no início -> o modelo calibra via gradiente
            model.logit_scale_min = float(getattr(self.cfg, "search_logit_scale_min", 0.5))
            model.logit_scale_max = float(getattr(self.cfg, "search_logit_scale_max", 8.0))


        return model


    def _count_ops(self, arch_mat: torch.Tensor) -> dict[str, int]:
        ops = arch_mat[2, :].detach().cpu().numpy()
        return {
            "ENC": int(np.sum(ops == OpType.ENC.value)),
            "ROT": int(np.sum(ops == OpType.ROT.value)),
            "CNOT": int(np.sum(ops == OpType.CNOT.value)),
        }

    def _budget_excess(self, counts: dict[str, int]) -> dict[str, int]:
        return {
            "ENC": max(0, counts["ENC"] - int(self.cfg.ENC_budget)),
            "ROT": max(0, counts["ROT"] - int(self.cfg.ROT_budget)),
            "CNOT": max(0, counts["CNOT"] - int(self.cfg.CNOT_budget)),
        }

    def _would_exceed_budget(self, counts: dict[str, int], action) -> bool:
        kind = action[0]
        if kind == "ENC":
            return counts["ENC"] >= int(self.cfg.ENC_budget)
        if kind == "ROT":
            return counts["ROT"] >= int(self.cfg.ROT_budget)
        if kind == "CNOT":
            return counts["CNOT"] >= int(self.cfg.CNOT_budget)
        return False

    def valid_action_mask(self) -> np.ndarray:
        mask = np.array(
            [action_is_valid_for_qubits(a, self.current_n_qubits, self.current_bank_k) for a in self.ACTIONS],
            dtype=bool,
        )
        if not self.cfg.hard_block_budget:
            return mask

        arch = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        counts = self._count_ops(arch)
        for i, a in enumerate(self.ACTIONS):
            if mask[i] and self._would_exceed_budget(counts, a):
                mask[i] = False
        return mask

    def reset(self) -> np.ndarray:
        self.step_idx = 0
        self.state = empty_state(self.cfg.L_max)
        self.last_auc = 0.5
        self.last_sens = 0.0
        self.last_spec = 0.0
        self.last_proxy_score = 0.5

        

        # mantenha thr inicial estável
        self.thr = float(getattr(self.cfg, "thr_init", 0.5)) if hasattr(self.cfg, "thr_init") else 0.5
        # zera custo real (tape) do episódio
        self._last_depth_tape = None
        self._last_cnot_tape = None

        self.current_n_qubits = int(self.cfg.start_qubits)
        self._qubit_cooldown = 0
        # self._recent_actions_q.clear()
        # self.recent_actions.clear()
        # FIX-4: clear per-episode action history only after episode 0.
        # The guard prevents early-episode "forgetting" of inter-episode arch hashes.
        if self._ep_count > 0:
            self._recent_actions_q.clear()
            self.recent_actions.clear()
        # _terminal_arch_hashes is intentionally NEVER cleared here.
        self._terminal_thr_stars = []   # reset thr* list each episode
        self._ep_count += 1


        # if bool(self.cfg.use_patch_bank) and bool(self.cfg.patch_bank_compact_features):
        #     X_bank = self.X_tr.detach().cpu().numpy()  # já (B,P)
        #     # init_feature_bank precisa suportar "patch" direto; se não suportar,
        #     # crie init_feature_bank_patches_simple que retorna range(P) ou saliency patch-based
        # else:
        #     X_bank = self.X_tr_raw.detach().cpu().numpy()  # (B,784)
        X_bank = self.X_tr.detach().cpu().numpy().astype(np.float32)

        if self.feature_bank is None:
            # # Use the same domain the policy will act on:
            # # - compact patch bank: X_tr is (B,P)
            # # - else: raw is (B,784)
            # X_raw_np = self.X_tr_raw.detach().cpu().numpy().astype(np.float32)  # raw domain (before normalization)
            y_np = (self.Y_tr.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)

            # if bool(self.cfg.use_patch_bank) and bool(getattr(self.cfg, "patch_bank_compact_features", False)):
            #     # raw pixels -> patch means (non-negative if pixels are in [0,1])
            #     # NOTE: patchify_mean_flat returns torch.Tensor; convert to numpy
            #     Xp = patchify_mean_flat(X_raw_np, self.patch_groups, device=self.DEVICE).detach().cpu().numpy().astype(np.float32)
            #     X_np_for_fb = Xp
            # else:
            #     X_np_for_fb = X_raw_np
            X_np_for_fb = X_bank

            self.feature_bank = init_feature_bank(
                X_np_for_fb,
                y_np,
                k_max=int(getattr(self, "feature_bank_size_eff", self.cfg.feature_bank_size)),
                mode=str(self.cfg.feature_bank_update),
                seed=int(self.seed),
                use_patch_bank=bool(self.cfg.use_patch_bank and (self.patch_groups is not None) and self.is_image_like),
                patch_groups=(self.patch_groups if (self.patch_groups is not None and self.is_image_like) else None),
            )
            self.logger.log_to_file(
                "feature_bank",
                f"[init] mode={self.cfg.feature_bank_update} k={len(self.feature_bank)} use_patch_bank={self.cfg.use_patch_bank}",
            )


        return self.state.copy()

    def _dead_qubit_count(self, arch_mat: torch.Tensor) -> int:
        """
        Count qubits (1..nq) that are never targeted/controlled by any op.
        If a qubit is never used, measuring it is wasted capacity.
        """
        nq = int(self.current_n_qubits)
        if nq <= 0:
            return 0

        used = np.zeros(nq, dtype=bool)
        a = arch_mat.detach().cpu().numpy()

        for l in range(a.shape[1]):
            op = int(a[2, l])
            c = int(a[0, l])
            t = int(a[1, l])

            if op == OpType.CNOT.value:
                if c > 0 and c <= nq:
                    used[c - 1] = True
                if t > 0 and t <= nq:
                    used[t - 1] = True
            elif op in (OpType.ENC.value, OpType.ROT.value):
                if t > 0 and t <= nq:
                    used[t - 1] = True

        return int((~used).sum())

    @torch.no_grad()
    def _eval_metrics(
        self,
        model: BinaryCQV_End2End,
        X: torch.Tensor,
        Y: torch.Tensor,
        thr: float | None = None,
    ) -> tuple[float, float, float]:
        model.eval()

        probs, n = self._predict_probs_capped(model, X, cap=int(self.cfg.inner_eval_batch_cap))

        yt_full = (Y.detach().cpu().numpy().reshape(-1) > 0.5).astype(int)
        yt = yt_full[:n]  # <-- CRITICAL: match probs length

        n_pos = int(yt.sum())
        n_neg = int(len(yt) - n_pos)

        if (n_pos == 0) or (n_neg == 0):
            auc = 0.5
        else:
            try:
                auc = float(roc_auc_score(yt, probs))
                if not np.isfinite(auc):
                    auc = 0.5
            except Exception:
                auc = 0.5

        thr_use = float(thr) if (thr is not None) else float(self.thr)
        # If model outputs are nearly constant, fallback to 0.5 to avoid random thresholds
        if (float(probs.max()) - float(probs.min())) < 1e-3:
            thr_use = 0.5

        tp, fp, tn, fn = _confusion_from_thr(yt, probs, float(thr_use))
        sens = float(tp) / float(max(n_pos, 1))
        spec = float(tn) / float(max(n_neg, 1))
        return float(auc), float(sens), float(spec)

    def compute_reward(self, auc, sens, arch_mat, depth, cnot_count, action_key):
        #used_rot = bool(torch.any(arch_mat[2, :] == OpType.ROT.value).item())
        n_rot = int(torch.sum(arch_mat[2, :] == OpType.ROT.value).item())
        
        # IMPORTANTE: durante step() barato, auc/sens podem ficar “stale” (do último terminal_evaluate).
        # Então não deixe recompensa depender demais de auc/sens aqui (mantenha alpha/beta pequenos
        # ou use apenas shaping). O “reward de verdade” vem no terminal_evaluate().
        if np.isnan(auc):
            auc = 0.0
        if np.isnan(sens):
            sens = 0.0

        auc01 = float(np.clip(float(auc), 0.0, 1.0))

        denom = max(1, (self.cfg.max_qubits - self.cfg.min_qubits))
        qnorm = (self.current_n_qubits - self.cfg.min_qubits) / denom
        qubit_cost = self.cfg.qubit_penalty * qnorm

        # components (for logging)
        # ---- NEW: proxy metric = Youden J (avoid pure SENS incentives) ----
        sens01 = float(np.clip(float(sens), 0.0, 1.0))
        # Use SPEC/Youden to prevent degenerate "all-positive" solutions.
        last_spec = float(np.clip(float(getattr(self, "last_spec", 0.0)), 0.0, 1.0))
        youden = float(np.clip(sens01 + last_spec - 1.0, -1.0, 1.0))
        youden01 = float(0.5 * (youden + 1.0))  # map [-1,1] -> [0,1]
        proxy_metric = 0.5 * auc01 + 0.5 * youden01  # [0,1]

        mscale = float(getattr(self.cfg, "metric_shaping_scale", 0.20))
        w = float(getattr(self, "_metric_weight", 1.0))
        comp_metric = float(w * mscale * (proxy_metric - 0.5))

        depth_raw = float(depth)
        if depth_raw < 0:
            depth_raw = float(abs(depth_raw))
        self._last_depth_raw = float(depth_raw)

        depth_ref = float(self._depth_ref())
        depth_norm = float(min(depth_raw / max(depth_ref, 1e-6), 1.0))  # [0,1]
        comp_depth = float(-float(self.cfg.depth_penalty) * depth_norm)  # bounded in [-depth_penalty, 0]

        comp_cnot = float(-self.cfg.cnot_penalty * float(cnot_count))
        comp_rot = float(-float(self.cfg.rot_penalty) * float(n_rot))
        comp_qubit = float(-float(qubit_cost))

        reward = comp_metric + comp_depth + comp_cnot + comp_rot + comp_qubit

        # budget shaping (soft)
        dead_q = self._dead_qubit_count(arch_mat)
        counts = self._count_ops(arch_mat)
        excess = self._budget_excess(counts)

        comp_dead = float(-float(getattr(self.cfg, "dead_qubit_penalty", 0.01)) * float(dead_q))
        comp_budget = float(
            -float(self.cfg.budget_penalty) * float(excess["ENC"] + excess["ROT"] + excess["CNOT"])
        )
        reward += comp_dead + comp_budget

        # ---- NEW: anti-collapse shaping (light) ----
        # If SPEC is too low, add a small penalty even during step-shaping.
        spec_floor = float(getattr(self.cfg, "spec_floor_shaping", 0.10))
        if last_spec < spec_floor:
            pen = float(getattr(self.cfg, "spec_collapse_penalty", 0.01)) * float(spec_floor - last_spec) / max(
                spec_floor, 1e-6
            )
            reward += float(-pen)
            self._ep_reward_sums["collapse"] += float(-pen)

        comp_repeat = 0.0
        if action_key in self.recent_actions:
            rep = float(getattr(self.cfg, "repeat_penalty", 0.01))
            comp_repeat = -abs(rep)
            reward += float(comp_repeat)

        # accumulate per-episode
        self._ep_reward_sums["metric"] += float(comp_metric)
        self._ep_reward_sums["depth"] += float(comp_depth)
        self._ep_reward_sums["cnot"] += float(comp_cnot)
        self._ep_reward_sums["qubit"] += float(comp_qubit)
        self._ep_reward_sums["rot"] += float(comp_rot)
        self._ep_reward_sums["dead"] += float(comp_dead)
        self._ep_reward_sums["budget"] += float(comp_budget)
        self._ep_reward_sums["repeat"] += float(comp_repeat)

        return float(reward)

    def step(self, action_idx: int):
        t0 = time.perf_counter()
        action = self.ACTIONS[action_idx]

        def _info_base(**extra):
            youden = float(self.last_sens + float(getattr(self, "last_spec", 0.0)) - 1.0)
            youden = float(np.clip(youden, -1.0, 1.0))

            info = {
                "auc_val": float(self.last_auc),
                "sens_val": float(self.last_sens),
                "spec_val": float(getattr(self, "last_spec", 0.0)),
                "youden_val": float(youden),
                "steps": int(self.step_idx),
                "thr": float(self.thr),
            }
            info.update(extra)
            return info

        arch_before = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        counts_before = self._count_ops(arch_before)

        # ==========================================================
        # 1) HARD budget blocking (early exits)
        # ==========================================================
        if self.cfg.hard_block_budget:
            saturated = (
                counts_before["ENC"] >= self.cfg.ENC_budget
                and counts_before["ROT"] >= self.cfg.ROT_budget
                and counts_before["CNOT"] >= self.cfg.CNOT_budget
            )
            if saturated:
                info = _info_base(
                    depth=int(self.step_idx),
                    cnot=int(counts_before["CNOT"]),
                    budget_saturated=True,
                    counts=counts_before,
                    step_time_s=float(time.perf_counter() - t0),
                )
                return self.state.copy(), 0.0, True, info

            if self._would_exceed_budget(counts_before, action):
                info = _info_base(
                    depth=int(self.step_idx),
                    cnot=int(counts_before["CNOT"]),
                    budget_blocked=True,
                    counts=counts_before,
                    action=action,
                    step_time_s=float(time.perf_counter() - t0),
                )
                return self.state.copy(), -float(self.cfg.budget_penalty), False, info

        # ==========================================================
        # 2) cooldown de qubits
        # ==========================================================
        if self._qubit_cooldown > 0:
            self._qubit_cooldown -= 1

        # ==========================================================
        # 3) ações de qubits (early exits)
        # ==========================================================
        kind = action[0]

        if kind == "ADD_QUBIT":
            if (self._qubit_cooldown == 0) and (self.current_n_qubits < self.cfg.max_qubits):
                self.current_n_qubits += 1
                self._qubit_cooldown = int(self.cfg.qubit_change_cooldown)
                reward = -0.01
            else:
                reward = -0.02

            info = _info_base(
                depth=0,
                cnot=0,
                n_qubits=int(self.current_n_qubits),
                step_time_s=float(time.perf_counter() - t0),
            )
            return self.state.copy(), float(reward), False, info

        if kind == "REMOVE_QUBIT":
            if (self._qubit_cooldown == 0) and (self.current_n_qubits > self.cfg.min_qubits):
                self.current_n_qubits -= 1
                self._qubit_cooldown = int(self.cfg.qubit_change_cooldown)
                reward = -0.005
            else:
                reward = -0.02

            info = _info_base(
                depth=0,
                cnot=0,
                n_qubits=int(self.current_n_qubits),
                step_time_s=float(time.perf_counter() - t0),
            )
            return self.state.copy(), float(reward), False, info

        # ==========================================================
        # 4) validação da ação (qubits + bank_k)
        # ==========================================================
        if not action_is_valid_for_qubits(action, self.current_n_qubits, self.current_bank_k):
            info = _info_base(depth=0, cnot=0, n_qubits=int(self.current_n_qubits))
            return self.state.copy(), -0.02, False, info

        # ==========================================================
        # 5) map bank index -> feature global (ENC)
        # ==========================================================
        if kind == "ENC":
            ax, q, b = action[1], action[2], action[3]

            if b is None:
                return self.state.copy(), -0.02, False, {"bad_action": action, "reason": "ENC b is None"}

            if int(b) < 0 or int(b) >= int(self.current_bank_k):
                # inconsistente com ACTIONS/mask => erro forte como você tinha
                raise RuntimeError(f"ENC index out of range: b={b} bank_k={self.current_bank_k}")

            # feat_global = int(self.feature_bank[int(b)])
            # Domain-agnostic mapping:
            # - if feature_bank is defined: bank idx -> selected global feature idx
            # - else: treat b directly as a feature index (tabular/identity fallback)
            if (self.feature_bank is None) or (len(self.feature_bank) == 0):
                feat_global = int(b)
            else:
                feat_global = int(self.feature_bank[int(b)])

            # final safety: keep feature in range of current input dim
            input_dim = int(self.X_tr.shape[1])
            if feat_global < 0 or feat_global >= input_dim:
                raise RuntimeError(
                    f"ENC feature out of input range: feat_global={feat_global} input_dim={input_dim} "
                    f"(bank_k={self.current_bank_k}, b={b})"
                )
            action = ("ENC", ax, q, feat_global)

        # ==========================================================
        # 6) NOP (fim imediato)
        # ==========================================================
        # Prevent premature termination early in the episode.
        # If NOP is selected before min_steps_before_nop, we apply a small
        # penalty and continue without ending the episode.
        if kind == "NOP":
            # Prevent premature termination early in the episode.
            # If NOP is selected before min_steps_before_nop, apply small penalty
            # and continue without ending the episode (do not modify state).
            min_k = int(getattr(self.cfg, "min_steps_before_nop", 0))
            if int(self.step_idx) < int(min_k):
                info = _info_base(
                    nop_blocked=True,
                    min_steps_before_nop=int(min_k),
                    depth=int(self.step_idx),
                    cnot=int(counts_before["CNOT"]),
                    n_qubits=int(self.current_n_qubits),
                    step_time_s=float(time.perf_counter() - t0),
                )
                return (
                    self.state.copy(),
                    -float(getattr(self.cfg, "nop_penalty", 0.01)),
                    False,
                    info,
                )

            # Allowed to end episode
            info = _info_base(
                depth=int(self.step_idx),
                cnot=int(counts_before["CNOT"]),
                n_qubits=int(self.current_n_qubits),
                step_time_s=float(time.perf_counter() - t0),
            )
            return self.state.copy(), 0.0, True, info

        # ==========================================================
        # 7) aplica ação no estado / incrementa step
        # ==========================================================
        encode_action_in_state(self.state, self.step_idx, action)
        self.step_idx += 1

        # ==========================================================
        # 8) STEP BARATO: NÃO treina, NÃO calibra thr, NÃO avalia modelo aqui.
        # Mantém auc/sens do último terminal_evaluate().
        # ==========================================================
        arch = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        auc, sens = float(self.last_auc), float(self.last_sens)

        # ==========================================================
        # 13) depth/cnot caro (opcional) ou proxy
        # ==========================================================
        depth = int(self.step_idx)
        cnot_count = int(self._count_ops(arch)["CNOT"])

        if bool(self.cfg.depth_use_proxy_when_skip):
            counts_now = self._count_ops(arch)
            n_ops = int(counts_now["ENC"] + counts_now["ROT"] + counts_now["CNOT"])
            depth = int(max(1, round(self.cfg.proxy_depth_per_op * n_ops)))
            cnot_count = int(self.cfg.proxy_cnot_per_cnot * counts_now["CNOT"])

        # ==========================================================
        # 14) reward + done + info
        # ==========================================================
        action_key = tuple(action)
        reward = self.compute_reward(auc, sens, arch, int(depth), int(cnot_count), action_key)

        if len(self._recent_actions_q) == self._recent_actions_q.maxlen:
            old = self._recent_actions_q.popleft()
            self.recent_actions.discard(old)
        self._recent_actions_q.append(action_key)
        self.recent_actions.add(action_key)

        done = bool(self.step_idx >= int(self.cfg.L_max))
        counts_now = self._count_ops(arch)
        excess_now = self._budget_excess(counts_now)

        info = {
            "auc_val": float(auc),
            "sens_val": float(sens),
            "steps": int(self.step_idx),
            "depth": int(depth),
            "cnot": int(cnot_count),
            "thr": float(self.thr),
            "counts": counts_now,
            "excess": excess_now,
            "n_qubits": int(self.current_n_qubits),
            "step_time_s": float(time.perf_counter() - t0),
        }
        return self.state.copy(), float(reward), bool(done), info

    def maybe_rescore_feature_bank_saliency(self, arch_use: np.ndarray) -> None:
        if str(getattr(self.cfg, "feature_bank_update", "none")).lower() != "saliency":
            return
        if self.X_tr is None or self.Y_tr is None:
            return

        model = self._build_model(
            torch.tensor(arch_use, dtype=torch.int64, device=self.DEVICE),
            diff_method="backprop",
        )

        

        # if bool(self.cfg.use_patch_bank):
        #     sal = compute_saliency_importance_patches(
        #         model,
        #         self.X_tr,
        #         self.Y_tr,
        #         top_k=int(self.cfg.feature_bank_size),
        #     )
        #     self.feature_bank = sal[: int(self.cfg.feature_bank_size)]
        # else:
        #     sal = compute_saliency_importance_pixels(
        #         model,
        #         self.X_tr,
        #         self.Y_tr,
        #         top_k=self.cfg.feature_bank_size,
        #     )
        #     self.feature_bank = sal[: self.cfg.feature_bank_size]
        # If patch-bank saliency is requested, require a coherent patch domain.
        if bool(getattr(self.cfg, "use_patch_bank", False)):
            if (self.patch_groups is None) or (not bool(getattr(self, "is_image_like", False))):
                # No-op: cannot compute patch saliency without patch domain
                return
            sal = compute_saliency_importance_patches(
                model,
                self.X_tr,
                self.Y_tr,
                top_k=int(self.feature_bank_size_eff),
            )
            self.feature_bank = np.asarray(sal[: int(self.feature_bank_size_eff)], dtype=np.int64)
            return

        # Pixel / generic saliency (works for any vector input)
        sal = compute_saliency_importance_pixels(
            model,
            self.X_tr,
            self.Y_tr,
            top_k=int(self.feature_bank_size_eff),
        )
        self.feature_bank = np.asarray(sal[: int(self.feature_bank_size_eff)], dtype=np.int64)

    def terminal_evaluate(self):
        """
        Executa 1x por episódio (no RUN) para ser viável e publicável:
        - build model
        - warmup head (opcional)
        - proxy-train (poucos batches)
        - calibra thr*
        - avalia AUC/SENS (val cap)
        - mede depth/cnot via tape (1x) OU média em poucos samples (publication-grade)
        """
        sens_tgt, spec_min, fpr_max = get_thr_targets(self.cfg)
        t0 = time.perf_counter()
        _arch_cache_key = hash(self.state.tobytes())
        if _arch_cache_key in self._arch_result_cache:
            _cached = self._arch_result_cache[_arch_cache_key]
            self.last_auc, self.last_sens, self.last_spec = float(_cached[0]), float(_cached[1]), float(_cached[2])
            self.thr = float(_cached[3])
            try:
                self.logger.log_to_file("terminal_splits", f"[arch_cache_HIT] auc={_cached[0]:.4f}")
            except Exception:
                pass
            return _cached
        
        arch = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
        try:
            _yv = (self.Y_val.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
            _nv = int(len(_yv))
            self.logger.log_to_file(
                "terminal_splits",
                f"[N_val] total={_nv} pos={int(_yv.sum())} neg={_nv - int(_yv.sum())} "
                f"val_frac={getattr(self.cfg,'val_frac_search',0.40):.2f} "
                f"percent_search={getattr(self.cfg,'percent_search',60)}"
            )
        except Exception:
            pass
        sens_tgt, spec_min, fpr_max = get_thr_targets(self.cfg)
        thr_sens_target = float(sens_tgt)
        thr_spec_min = float(spec_min)
        thr_fpr_max = float(fpr_max)

        # ----------------------------------------------------------
        # Phase-aware threshold constraints
        # ----------------------------------------------------------
        phase = str(getattr(self.cfg, "phase", "search")).lower()  # "search" | "final"
        if phase.startswith("final"):
            thr_sens_target = float(getattr(self.cfg, "final_sens_target", getattr(self.cfg, "sens_target", 0.85)))
            thr_spec_min = float(getattr(self.cfg, "final_thr_spec_min", getattr(self.cfg, "thr_spec_min", 0.80)))
            thr_fpr_max = float(getattr(self.cfg, "final_thr_fpr_max", getattr(self.cfg, "thr_fpr_max", 0.10)))
        else:
            thr_sens_target = float(getattr(self.cfg, "search_sens_target", 0.42))
            thr_spec_min = float(getattr(self.cfg, "search_thr_spec_min", 0.60))
            thr_fpr_max = float(getattr(self.cfg, "search_thr_fpr_max", 0.30))
            try:
                self.logger.log_to_file(
                    "thr",
                    f"[phase] phase={phase} targets sens/spec/fpr={thr_sens_target:.3f}/{thr_spec_min:.3f}/{thr_fpr_max:.3f}",
                )
            except Exception:
                pass

        # ----------------------------------------------------------
        # Threshold config (shared by proxy runs)
        # ----------------------------------------------------------
        thr_mode = str(getattr(self.cfg, "thr_mode", "soft")).lower().strip()
        if thr_mode not in ("soft", "hard"):
            thr_mode = "soft"

        thr_policy = str(getattr(self.cfg, "thr_policy", "f2")).lower().strip()
        if thr_policy not in ("f2", "sens", "youden"):
            thr_policy = "f2"

        # Base lambdas (phase-aware)
        base_lam_spec = float(
            getattr(
                self.cfg,
                "final_thr_lam_spec" if phase.startswith("final") else "search_thr_lam_spec",
                getattr(self.cfg, "thr_lam_spec", 2.0),
            )
        )
        base_lam_fpr = float(
            getattr(
                self.cfg,
                "final_thr_lam_fpr" if phase.startswith("final") else "search_thr_lam_fpr",
                getattr(self.cfg, "thr_lam_fpr", 2.0),
            )
        )
        base_lam_sens = float(
            getattr(
                self.cfg,
                "final_thr_lam_sens" if phase.startswith("final") else "search_thr_lam_sens",
                getattr(self.cfg, "thr_lam_sens", 1.0),
            )
        )

        lam_spec = float(base_lam_spec)
        lam_fpr = float(base_lam_fpr)
        lam_sens = float(base_lam_sens)

        # Policy adjustments
        if thr_policy == "sens":
            lam_sens *= float(getattr(self.cfg, "thr_policy_sens_mult", 2.0))
            lam_spec *= float(getattr(self.cfg, "thr_policy_spec_mult", 1.0))
            lam_fpr  *= float(getattr(self.cfg, "thr_policy_fpr_mult", 1.0))
        elif thr_policy == "youden":
            lam_spec *= float(getattr(self.cfg, "thr_policy_youden_spec_mult", 1.5))
            lam_sens *= float(getattr(self.cfg, "thr_policy_youden_sens_mult", 1.5))
            lam_fpr  *= float(getattr(self.cfg, "thr_policy_fpr_mult", 1.0))

        # ----------------------------------------------------------
        # NEW: proxy evaluation with 1 or 2 seeds (reduce gap RL->train)
        # returns mean and std over seeds
        # ----------------------------------------------------------
        def _one_proxy_run(seed_offset: int):
            # seed control
            torch.manual_seed(int(self.seed + seed_offset))
            np.random.seed(int(self.seed + seed_offset))

            # ----------------------------------------------------------
            # SEARCH HEAD-ONLY MODE:
            # During SEARCH, train ONLY the classification head.
            # Circuit parameters (theta/enc) are frozen so the circuit changes
            # ONLY through architecture actions, not gradients.
            # ----------------------------------------------------------
            phase_local = str(getattr(self.cfg, "phase", "search")).lower()
            # search_head_only = bool(getattr(self.cfg, "search_head_only", True)) and (not phase_local.startswith("final"))
            # # Optional: allow a small number of VQC batches even in SEARCH (default 0)
            # search_vqc_batches_override = int(getattr(self.cfg, "search_inner_train_batches_vqc_override", 0))
            # CORREÇÃO: default False — VQC precisa de gradiente no search
            # Evidência: TEST4 mostrou VQC congelado AUC=0.573 vs end2end AUC=0.798
            # Com search_head_only=True o VQC recebia 0 batches de gradiente em 291 episódios
            search_head_only = bool(getattr(self.cfg, "search_head_only", False)) and (not phase_local.startswith("final"))
            # batches VQC no search: usa inner_train_batches_vqc diretamente (sem override para 0)
            search_vqc_batches_override = int(getattr(self.cfg, "search_inner_train_batches_vqc_override", -1))
            # -1 = "não usar override, usar n_vqc_batches do config"


            def _freeze_vqc_params(model):
                # Freeze everything first...
                for p in model.parameters():
                    p.requires_grad_(False)
                # ...then unfreeze head
                for p in model.head.parameters():
                    p.requires_grad_(True)

            def _unfreeze_all_params(model):
                for p in model.parameters():
                    p.requires_grad_(True)
 
            # ----------------------------------------------------------
            # Calib set for thr*: do NOT calibrate on dl_small (too noisy).
            # Instead, use a larger stratified cap from full train, fixed by seed.
            # ----------------------------------------------------------
            def _stratified_cap_from_train(cap: int, seed_for_cap: int) -> tuple[torch.Tensor, torch.Tensor]:
                """
                Returns (X_cal, Y_cal) from self.X_tr/self.Y_tr, stratified and capped.
                Tensors returned are already on self.DEVICE.
                Cached to keep thr* stability across attempts within same run.
                """
                cap = int(max(2, cap))
                # key = (int(cap), int(seed_offset))
                key = (int(cap), int(seed_for_cap))
                if key in self._calib_cache:
                    return self._calib_cache[key]

                y = (self.Y_tr.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
                n = int(y.shape[0])
                cap_eff = int(min(cap, n))

                idx_pos = np.where(y == 1)[0]
                idx_neg = np.where(y == 0)[0]

                rng = np.random.default_rng(int(seed_for_cap))
                rng.shuffle(idx_pos)
                rng.shuffle(idx_neg)

                # stratified half/half as much as possible
                k_pos = int(min(len(idx_pos), cap_eff // 2))
                k_neg = int(min(len(idx_neg), cap_eff - k_pos))

                sel = []
                if k_pos > 0:
                    sel.append(idx_pos[:k_pos])
                if k_neg > 0:
                    sel.append(idx_neg[:k_neg])
                if len(sel) == 0:
                    # extreme edge case: empty? fallback uniform
                    sel_idx = rng.choice(np.arange(n), size=cap_eff, replace=False)
                else:
                    sel_idx = np.concatenate(sel, axis=0)
                    if sel_idx.size < cap_eff:
                        # fill remainder from whichever class has leftovers
                        rem = cap_eff - int(sel_idx.size)
                        pool = np.setdiff1d(np.arange(n), sel_idx, assume_unique=False)
                        if pool.size > 0:
                            extra = rng.choice(pool, size=min(rem, pool.size), replace=False)
                            sel_idx = np.concatenate([sel_idx, extra], axis=0)
                rng.shuffle(sel_idx)

                sel_t = torch.as_tensor(sel_idx, device=self.DEVICE, dtype=torch.long)
                X_cal = self.X_tr.index_select(0, sel_t)
                Y_cal = self.Y_tr.index_select(0, sel_t)
                seed_for_cap = (
                    int(self.seed)
                    + SEED_DELTA_CALIB
                    + int(seed_offset)
                )
                self._calib_cache[key] = (X_cal, Y_cal)
                # try:
                #     ycal = (Y_cal.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
                #     pos = int(ycal.sum())
                #     tot = int(ycal.shape[0])
                #     self.logger.log_to_file(
                #         "thr",
                #         f"[calib] fixed_stratified cap={cap_eff} seed_off={seed_offset} "
                #         f"pos={int(pos)}/{int(tot)}",
                #     )

                #     self.logger.log_to_file(
                #         "calib",
                #         f"[calib] fixed_stratified cap={cap_eff} "
                #         f"seed_base={self.seed} "
                #         f"seed_off={seed_offset} "
                #         f"seed_for_cap={seed_for_cap} "
                #         f"pos={pos}/{cap_eff}"
                #     )

                # except Exception:
                #     pass
                # FIX-C: confirm FIX-3 — log N_calib, seed lineage, structural overlap.
                # calib ⊂ X_tr and val ⊂ X_val → structurally disjoint → overlap always 0.
                # Logging the seed lineage lets you verify calib independence after the fact.
                try:
                    ycal = (Y_cal.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
                    pos  = int(ycal.sum())
                    tot  = int(ycal.shape[0])
                    _csd = int(getattr(self.cfg, "calib_seed_delta", 99991))
                    self.logger.log_to_file(
                        "terminal_splits",
                        f"[N_calib] total={tot} pos={pos} neg={tot-pos} "
                        f"seed_base={self.seed} seed_off={seed_offset} "
                        f"calib_seed_delta={_csd} seed_for_cap={seed_for_cap} "
                        f"calib∩val_overlap=0(structural:X_tr⊥X_val)"
                    )
                    # legacy log kept for backwards compatibility
                    self.logger.log_to_file(
                        "thr",
                        f"[calib] fixed_stratified cap={cap_eff} seed_off={seed_offset} "
                        f"pos={pos}/{tot} calib_seed_delta={_csd}",
                    )
                except Exception:
                    pass
                return X_cal, Y_cal


            # ----------------------------------------------------------
            # SEARCH separability knobs (head + circuit) - defaults safe
            # ----------------------------------------------------------
            phase_local = str(getattr(self.cfg, "phase", "search")).lower()
            is_final = phase_local.startswith("final")

            # HEAD: faster in SEARCH to avoid logits/probs stuck ~0
            # (kept small enough to not fully dominate architecture ranking)
            lr_head_eff = float(
                getattr(
                    self.cfg,
                    "final_lr_head" if is_final else "search_lr_head",
                    getattr(self.cfg, "lr_head", 3e-3),
                )
            )
            head_epochs_eff = int(
                getattr(
                    self.cfg,
                    "final_head_epochs" if is_final else "search_head_epochs",
                    1 if is_final else 8,
                )
            )
            # SEARCH-only: lightly penalize bias (relative) without touching FINAL.
            # This pushes the head to use weights when possible.
            wd_head_bias_eff = float(
                getattr(
                    self.cfg,
                    "final_wd_head_bias" if is_final else "search_wd_head_bias",
                    0.0 if is_final else 1e-4,
                )
            )


            def _split_head_params(head_module):
                """
                Return (weights, biases) param lists for a nn.Linear-like head.
                Biases are ndim==1; weights are ndim>=2.
                """
                w, b = [], []
                for p in head_module.parameters():
                    (b if p.ndim == 1 else w).append(p)
                return w, b



            # allow bias to learn in SEARCH (helps recenter logits)
            allow_bias_learn_search = bool(getattr(self.cfg, "search_allow_bias_learn", True))



            def _collapse_decision(
                logits_np: np.ndarray,
                probs_np: np.ndarray,
                auc_val: float | None,
                viable_count: int | None,
                fail_streak_clinical: int,
                fail_streak_auc_floor: int,
            ) -> tuple[bool, dict]:
                logits_np = np.asarray(logits_np, dtype=float).reshape(-1)
                probs_np  = np.asarray(probs_np,  dtype=float).reshape(-1)

                logit_margin = _p95_p5(logits_np)
                prob_std = float(np.std(probs_np)) if probs_np.size else float("nan")

                logit_margin_min = float(getattr(self.cfg, "collapse_logit_margin_min", 0.05))
                prob_std_weak    = float(getattr(self.cfg, "collapse_prob_std_weak", 0.01))
                auc_eps          = float(getattr(self.cfg, "collapse_auc_eps", 0.01))
                streak_k         = int(getattr(self.cfg, "collapse_fail_streak_k", 2))

                # 1) Strong signal: logits nearly constant
                collapsed_strong = bool(np.isfinite(logit_margin) and (logit_margin < logit_margin_min))

                # 2) Clinical fail: AUC ~ 0.5 AND no viable thresholds
                auc_near_05 = (
                    (auc_val is not None)
                    and np.isfinite(auc_val)
                    and (abs(float(auc_val) - 0.5) <= auc_eps)
                )
                no_viable = (viable_count is not None) and (int(viable_count) == 0)
                clinical_fail = bool(auc_near_05 and no_viable)

                # 3) Weak signal: probs too flat, but only after streak and gated by logit margin
                logit_gate = float(getattr(self.cfg, "collapse_logit_gate_mult", 2.0))
                logit_gate_thr = float(logit_gate) * float(logit_margin_min)
                collapsed_weak = bool(
                    np.isfinite(prob_std)
                    and (prob_std < prob_std_weak)
                    and (fail_streak_auc_floor >= streak_k)
                    and (np.isfinite(logit_margin) and (logit_margin < logit_gate_thr))
                )

                # 4) SEARCH AUC floor gate (optional collapse)
                phase_l = str(getattr(self.cfg, "phase", "search")).lower()
                is_search = phase_l.startswith("search")
                auc_floor = float(getattr(self.cfg, "search_auc_floor_collapse",
                                        getattr(self.cfg, "search_auc_floor", 0.50)))
                auc_floor_streak = int(getattr(self.cfg, "search_auc_floor_streak", 2))

                auc_low = bool(
                    is_search
                    and (auc_val is not None)
                    and np.isfinite(auc_val)
                    and (float(auc_val) < auc_floor)
                )
                # Só “collapse” por AUC floor se também estiver fraco de separabilidade
                auc_floor_fail = bool(
                    auc_low
                    and (fail_streak_auc_floor >= auc_floor_streak)
                    and (
                        (np.isfinite(logit_margin) and (logit_margin < logit_gate_thr))
                        or (np.isfinite(prob_std) and (prob_std < prob_std_weak))
                    )
                )

                # 5) Final collapsed decision (single source of truth)
                collapsed = bool(
                    collapsed_strong
                    or (clinical_fail and fail_streak_clinical >= streak_k)
                    or collapsed_weak
                    or auc_floor_fail
                )

                if collapsed:
                    if collapsed_strong:
                        reason = "logit_margin"
                    elif auc_floor_fail:
                        reason = f"auc_floor<{auc_floor:.3f}"
                    elif (clinical_fail and fail_streak_clinical >= streak_k):
                        reason = "auc~0.5+viable=0_streak"
                    elif collapsed_weak:
                        reason = "prob_std_weak+streak+logit_gate"
                    else:
                        reason = "collapsed_other"
                else:
                    reason = f"warn_auc_floor<{auc_floor:.3f}" if auc_low else "ok"

                dbg = {
                    "collapsed": int(collapsed),
                    "reason": str(reason),
                    "logit_p95_p5": float(logit_margin) if np.isfinite(logit_margin) else float("nan"),
                    "prob_std": float(prob_std) if np.isfinite(prob_std) else float("nan"),
                    "auc": (None if auc_val is None else float(auc_val)),
                    "viable_count": (None if viable_count is None else int(viable_count)),
                    "fail_streak_clinical": int(fail_streak_clinical),
                    "fail_streak_auc_floor": int(fail_streak_auc_floor),
                    "auc_floor": (float(auc_floor) if is_search else None),
                    "auc_low": bool(auc_low),
                }

                return collapsed, dbg


            # ----------------------------------------------------------
            # Choose calibration source: SAME subset used for proxy training (recommended)
            # ----------------------------------------------------------
            # def _get_calib_tensors_from_loader(dl: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
            #     # TensorDataset(Xs, Ys) -> get tensors directly (already on device)
            #     """
            #     PATCH: stop calibrating thr* on dl_small (too small / unstable).
            #     Use full-train stratified cap instead (fixed per seed_offset).

            #     Config knobs (optional):
            #       - calib_cap (default 1024)
            #       - calib_seed_delta (default 99991)  # separates from training seed
            #     """
            #     cap = int(getattr(self.cfg, "calib_cap", 1024))
            #     seed_delta = int(getattr(self.cfg, "calib_seed_delta", 99991))
            #     seed_for_cap = int(self.seed + seed_delta + seed_offset)
            #     #seed_for_cap = int(self.seed + seed_delta)
            #     return _stratified_cap_from_train(cap=cap, seed_for_cap=seed_for_cap)
            def _get_calib_tensors_from_loader(dl: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
                """
                FIX-3: Calibrate thr* on a stratified cap from full train, NOT dl_small.

                calib_seed_delta (default 99991) >> max proxy seed offset (2*1337=2674),
                so the calib subset can never coincide with a proxy evaluation subset.

                This eliminates the systematic val_bce < calib_bce observed in all 5
                spike episodes (ep047, ep243, ep214, ep158, ep170).

                Config knobs:
                  - calib_cap        (default 1024)
                  - calib_seed_delta (default 99991)
                """
                # COST-D: smaller calib cap during SEARCH, full cap during FINAL.
                _is_final = str(getattr(self.cfg, "phase", "search")).lower().startswith("final")
                cap = int(
                    getattr(self.cfg, "final_calib_cap",  getattr(self.cfg, "calib_cap", 1024))
                    if _is_final else
                    getattr(self.cfg, "search_calib_cap", getattr(self.cfg, "calib_cap", 256))
                )
                seed_delta    = int(getattr(self.cfg, "calib_seed_delta", 99991))
                seed_for_cap  = int(self.seed) + int(seed_delta) + int(seed_offset)
                return _stratified_cap_from_train(cap=cap, seed_for_cap=seed_for_cap)

            # ----------------------------------------------------------
            # knobs per phase
            # ----------------------------------------------------------
            if phase.startswith("final"):
                term_diff = str(getattr(self.cfg, "final_terminal_diff_method", getattr(self.cfg, "terminal_diff_method", "adjoint")))
                n_head = int(getattr(self.cfg, "final_inner_train_batches_head", getattr(self.cfg, "inner_train_batches_head", 0)))
                n_vqc_batches = int(getattr(self.cfg, "final_inner_train_batches_vqc", getattr(self.cfg, "inner_train_batches_vqc", 1)))
                n_epochs = int(getattr(self.cfg, "final_inner_epochs_classif", getattr(self.cfg, "inner_epochs_classif", 1)))
                head_epochs = int(getattr(self.cfg, "final_head_epochs", head_epochs_eff))
            else:
                term_diff = str(getattr(self.cfg, "search_terminal_diff_method", getattr(self.cfg, "terminal_diff_method", "backprop")))
                n_head = int(getattr(self.cfg, "search_inner_train_batches_head", getattr(self.cfg, "inner_train_batches_head", 16)))
                n_vqc_batches = int(getattr(self.cfg, "search_inner_train_batches_vqc", getattr(self.cfg, "inner_train_batches_vqc", 1)))
                n_epochs = int(getattr(self.cfg, "search_inner_epochs_classif", getattr(self.cfg, "inner_epochs_classif", 1)))
                head_epochs = int(getattr(self.cfg, "search_head_epochs", head_epochs_eff))


            # ----------------------------------------------------------
            # pos_weight handling:
            # - FINAL: use dataset prevalence (pos_weight)
            # - SEARCH (stratified 50/50 subsets): disable by default
            # ----------------------------------------------------------
            use_pos_weight_search = bool(getattr(self.cfg, "search_use_pos_weight", False))
            pos_weight_eff = self.pos_weight if (phase.startswith("final") or use_pos_weight_search) else None
            n_head = int(max(0, n_head))
            n_vqc_batches = int(max(0, n_vqc_batches))
            n_epochs = int(max(1, n_epochs))
            head_epochs = int(max(1, head_epochs))
            model = self._build_model(arch, diff_method=str(term_diff))
            if search_head_only:
                _freeze_vqc_params(model)

            # # SEARCH: boost logit scale to avoid logits stuck near 0
            # if (not phase.startswith("final")) and hasattr(model, "logit_scale"):
            #     init_ls = float(getattr(self.cfg, "search_logit_scale_init", 3.0))
            #     with torch.no_grad():
            #         try:
            #             model.logit_scale.data.fill_(init_ls)
            #         except Exception:
            #             pass
            

            
            if phase.startswith("final"):
                init_head_bias_with_prevalence(model, self.prev_train)
            else:
                # SEARCH: force neutral head
                init_head_bias_with_prevalence(model, self.prev_train, force_p=0.5)
                #torch.nn.init.zeros_(model.head.bias)
                torch.nn.init.normal_(model.head.weight, mean=0.0, std=0.02)
            # resample small loader (cheap) to reduce subset overfit
            dl_small = self._make_small_stratified_loader(
                X=self.X_tr,
                Y=self.Y_tr,
                subset_size=int(self.cfg.inner_train_subset_size),
                batch_size=int(self.cfg.batch_size),
                seed=int(self.seed + seed_offset),
            )

            X_cal, Y_cal = _get_calib_tensors_from_loader(dl_small)

            # 1) warmup head
            # In SEARCH head-only we already froze VQC; in FINAL we freeze here as usual.
            #if not search_head_only:
            if True: 
                for p in model.parameters():
                    p.requires_grad_(False)
                for p in model.head.parameters():
                    p.requires_grad_(True)
                # SEARCH: allow bias to learn (default True) to improve separability/recentering
                # FINAL: keep your previous behavior (bias frozen during head-only) unless user overrides
                if model.head.bias is not None:
                    if (not phase.startswith("final")) and bool(allow_bias_learn_search):
                        model.head.bias.requires_grad_(True)
                    else:
                        model.head.bias.requires_grad_(False)
                # no weight decay on bias
                
                hw, hb = _split_head_params(model.head)
                groups = []
                if hw:
                    groups.append({"params": hw, "lr": lr_head_eff, "weight_decay": getattr(self.cfg, "wd_head", 0.0)})
                if hb:
                    groups.append({"params": hb, "lr": lr_head_eff, "weight_decay": wd_head_bias_eff})
                opt_head = torch.optim.Adam(groups)

                #crit = torch.nn.BCEWithLogitsLoss()
                crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_eff)
                model.train()
                for _he in range(int(head_epochs)):
                    for bi, (xb, yb) in enumerate(dl_small):
                        xb = xb.detach()
                        xb.requires_grad_(False)

                        if xb.dim() == 0:
                            xb = xb.view(1, 1)
                        elif xb.dim() == 1:
                            xb = xb.unsqueeze(0) if xb.numel() == model.input_dim else xb.view(-1, 1)
                        elif xb.dim() > 2:
                            xb = xb.view(xb.shape[0], -1)

                        if yb.dim() == 0:
                            yb = yb.view(1, 1)
                        elif yb.dim() == 1:
                            yb = yb.view(-1, 1)

                        logits_b = model(xb)
                        loss = crit(logits_b, yb)
                        opt_head.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                        opt_head.step()
                        # ---------------------------
                        # LEVEL 1: batch loss logging (head warmup)
                        # ---------------------------
                        if bool(getattr(self.cfg, "log_loss_per_iter", True)):
                            log_every = int(getattr(self.cfg, "loss_log_every", 10))
                            if ((bi % log_every) == 0) or (bi == 0):
                                loss_f = float(loss.detach().cpu().item())
                                # EMA para ficar legível
                                if not hasattr(self, "_ema_head_loss"):
                                    self._ema_head_loss = None
                                self._ema_head_loss = _ema_update(self._ema_head_loss, loss_f, alpha=float(getattr(self.cfg, "loss_ema_alpha", 0.05)))
                                self.logger.log_to_file(
                                    "loss_iter",
                                    f"[head][iter] seed_off={seed_offset} he={_he} bi={bi} "
                                    f"loss={loss_f:.6f} ema={float(self._ema_head_loss):.6f}"
                                )
                        if (not phase.startswith("final")) and (model.head.bias is not None):
                            bmax = float(getattr(self.cfg, "search_head_bias_clamp", 2.0))
                            with torch.no_grad():
                                model.head.bias.data.clamp_(-bmax, bmax)
                        if (bi + 1) >= n_head:
                            break

            # ----------------------------------------------------------
            # SEARCH-only: recenter logits after head warmup (optional).
            # This improves sigmoid symmetry and reduces "bias steals the work".
            # It does NOT change ranking, only shifts logits (bias).
            # ----------------------------------------------------------
            if (not phase.startswith("final")) and bool(getattr(self.cfg, "search_recenter_logits", True)):
                try:
                    with torch.no_grad():
                        model.eval()
                        # use the same calib cap set (stable) for centering
                        logits_center, _, n_center = self._predict_logits_probs_capped(
                            model,
                            X_cal,
                            cap=int(min(int(getattr(self.cfg, "recenter_cap", 1024)), int(X_cal.shape[0]))),
                        )
                        if int(n_center) > 0 and hasattr(model, "head") and getattr(model.head, "bias", None) is not None:
                            mu = float(np.mean(np.asarray(logits_center, dtype=float)))
                            model.head.bias.data.sub_(float(mu))
                            self.logger.log_to_file("head", f"[recenter] phase=search seed_off={seed_offset} mu_logit={mu:.6f}")
                except Exception:
                    pass

            # ----------------------------------------------------------
            # 2) proxy train
            # FINAL: unfreeze all and train head+enc+theta (existing behavior)
            # SEARCH head-only: keep VQC frozen (no gradient changes to circuit),
            # train ONLY the head for stronger separability.
            # ----------------------------------------------------------
            if phase.startswith("final"):
                _unfreeze_all_params(model)
                if model.head.bias is not None:
                    model.head.bias.requires_grad_(True)
            else:
                # SEARCH
                if search_head_only:
                    # ensure VQC stays frozen, head trainable (incl bias)
                    _freeze_vqc_params(model)
                    if model.head.bias is not None:
                        model.head.bias.requires_grad_(True)
                    # override VQC batches to 0 by default (hard head-only)
                # Aplica override APENAS se explicitamente definido (>= 0)
                # Com search_head_only=False (novo default) este bloco não executa
                if int(search_vqc_batches_override) >= 0:
                    n_vqc_batches = int(search_vqc_batches_override)
                # else: mantém n_vqc_batches do config (inner_train_batches_vqc)
                else:
                    # legacy SEARCH behavior: allow full unfreeze
                    _unfreeze_all_params(model)
                    if model.head.bias is not None:
                        model.head.bias.requires_grad_(True)
            # ----------------------------------------------------------
            # NEW: Optimizer param groups (head / enc / theta)
            # SEARCH defaults: WD=0 to improve separability and avoid collapse
            # ----------------------------------------------------------
            # Use explicit SEARCH knobs if present

            # param_groups = []

            # lr_head = float(lr_head_eff)
            # lr_enc  = float(
            #     getattr(
            #         self.cfg,
            #         "final_lr_enc" if is_final else "search_lr_enc",
            #         getattr(self.cfg, "lr_enc", 2e-3),
            #     )
            # )
            # lr_th   = float(
            #     getattr(
            #         self.cfg,
            #         "final_lr_theta" if is_final else "search_lr_theta",
            #         getattr(self.cfg, "lr_theta", float(getattr(self.cfg, "lr_vqc", 1e-3))),
            #     )
            # )
            # # SEARCH: default WD=0 unless user overrides search_wd_*
            # wd_head = float(getattr(self.cfg, "wd_head", 0.0))
            # wd_enc  = float(
            #     getattr(
            #         self.cfg,
            #         "final_wd_enc" if is_final else "search_wd_enc",
            #         float(getattr(self.cfg, "wd_enc", 0.0) if is_final else 0.0),
            #     )
            # )
            # wd_th   = float(
            #     getattr(
            #         self.cfg,
            #         "final_wd_theta" if is_final else "search_wd_theta",
            #         float(getattr(self.cfg, "wd_theta", 0.0) if is_final else 0.0),
            #     )
            # )
            # head_w, head_b = _split_head_params(model.head)
            # head_w = _uniq(head_w)
            # head_b = _uniq(head_b)

            # if head_w:
            #     param_groups.append({"params": head_w, "lr": lr_head, "weight_decay": wd_head})
            # if head_b:
            #     param_groups.append({"params": head_b, "lr": lr_head, "weight_decay": wd_head_bias_eff})

            # # avoid duplicates: build id-set
            # seen = set()
            # def _uniq(ps):
            #     out = []
            #     for p in ps:
            #         if p is None:
            #             continue
            #         if id(p) in seen:
            #             continue
            #         seen.add(id(p))
            #         out.append(p)
            #     return out

            # head_params = _uniq(head_params)
            # enc_params  = _uniq(enc_params)
            # theta_params= _uniq(theta_params)

            # if len(head_params) > 0:
            #     param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": wd_head})

            # # In SEARCH head-only, do NOT add enc/theta groups (keep circuit frozen)
            # if not (search_head_only and (not phase_local.startswith("final"))):
            #     if len(enc_params) > 0:
            #         param_groups.append({"params": enc_params, "lr": lr_enc, "weight_decay": wd_enc})
            #     if len(theta_params) > 0:
            #         param_groups.append({"params": theta_params, "lr": lr_th, "weight_decay": wd_th})
 
            # # fallback: if something went wrong, still train all params
            # if len(param_groups) == 0:
            #     param_groups = [{"params": model.parameters(), "lr": float(getattr(self.cfg, "lr_vqc", 1e-3)), "weight_decay": float(getattr(self.cfg, "wd_vqc", 0.0))}]

            # opt = torch.optim.Adam(param_groups)

            # ----------------------------------------------------------
            # Optimizer param groups (head / enc / theta)
            # - no WD on head bias
            # - avoid duplicated params across groups
            # - SEARCH head-only: only train head (optional)
            # ----------------------------------------------------------
            param_groups = []

            # avoid duplicates across groups
            _seen = set()
            def _uniq(ps):
                out = []
                for p in ps:
                    if p is None:
                        continue
                    if id(p) in _seen:
                        continue
                    _seen.add(id(p))
                    out.append(p)
                return out

            # LRs / WDs (phase-aware)
            lr_head = float(lr_head_eff)
            lr_enc  = float(
                getattr(
                    self.cfg,
                    "final_lr_enc" if is_final else "search_lr_enc",
                    getattr(self.cfg, "lr_enc", 2e-3),
                )
            )
            lr_th   = float(
                getattr(
                    self.cfg,
                    "final_lr_theta" if is_final else "search_lr_theta",
                    getattr(self.cfg, "lr_theta", float(getattr(self.cfg, "lr_vqc", 1e-3))),
                )
            )

            wd_head = float(getattr(self.cfg, "wd_head", 0.0))
            wd_enc  = float(
                getattr(
                    self.cfg,
                    "final_wd_enc" if is_final else "search_wd_enc",
                    float(getattr(self.cfg, "wd_enc", 0.0) if is_final else 0.0),
                )
            )
            wd_th   = float(
                getattr(
                    self.cfg,
                    "final_wd_theta" if is_final else "search_wd_theta",
                    float(getattr(self.cfg, "wd_theta", 0.0) if is_final else 0.0),
                )
            )

            # ---- collect params explicitly ----
            # head split
            head_w, head_b = _split_head_params(model.head)
            head_w = _uniq(head_w)
            head_b = _uniq(head_b)

            # enc params
            enc_params = []
            if hasattr(model, "enc_alpha_raw") and isinstance(model.enc_alpha_raw, torch.nn.Parameter):
                enc_params.append(model.enc_alpha_raw)
            if hasattr(model, "enc_beta_raw") and isinstance(model.enc_beta_raw, torch.nn.Parameter):
                enc_params.append(model.enc_beta_raw)
            enc_params = _uniq(enc_params)

            # theta params
            theta_params = []
            if hasattr(model, "theta") and isinstance(model.theta, torch.nn.Parameter):
                theta_params.append(model.theta)
            theta_params = _uniq(theta_params)
            # Onde você monta os param_groups do otimizador, após theta_params:
            if hasattr(model, 'logit_scale') and model.logit_scale.requires_grad:
                param_groups.append({
                    'params': [model.logit_scale],
                    'lr': lr_head,
                    'weight_decay': 0.0
                })
            # ---- build groups ----
            if head_w:
                param_groups.append({"params": head_w, "lr": lr_head, "weight_decay": wd_head})
            if head_b:
                param_groups.append({"params": head_b, "lr": lr_head, "weight_decay": wd_head_bias_eff})

            # In SEARCH head-only, do NOT add enc/theta groups (keep circuit frozen)
            phase_local_now = str(getattr(self.cfg, "phase", "search")).lower()
            if not (search_head_only and (not phase_local_now.startswith("final"))):
                if enc_params:
                    param_groups.append({"params": enc_params, "lr": lr_enc, "weight_decay": wd_enc})
                if theta_params:
                    param_groups.append({"params": theta_params, "lr": lr_th, "weight_decay": wd_th})

            # fallback safety
            if len(param_groups) == 0:
                param_groups = [
                    {
                        "params": model.parameters(),
                        "lr": float(getattr(self.cfg, "lr_vqc", 1e-3)),
                        "weight_decay": float(getattr(self.cfg, "wd_vqc", 0.0)),
                    }
                ]

            opt = torch.optim.Adam(param_groups)

            for _ in range(n_epochs):
                epoch_losses = []
                model.train()
                for bi, (xb, yb) in enumerate(dl_small):
                    # if (search_head_only and (not phase_local.startswith("final")) and int(n_vqc_batches) == 0):
                    #     break
                    xb = xb.detach()
                    xb.requires_grad_(False)

                    if xb.dim() == 0:
                        xb = xb.view(1, 1)
                    elif xb.dim() == 1:
                        xb = xb.unsqueeze(0) if xb.numel() == model.input_dim else xb.view(-1, 1)
                    elif xb.dim() > 2:
                        xb = xb.view(xb.shape[0], -1)

                    if yb.dim() == 0:
                        yb = yb.view(1, 1)
                    elif yb.dim() == 1:
                        yb = yb.view(-1, 1)

                    logits = model(xb)
                    if bool(self.cfg.use_focal):
                        loss = focal_loss_with_logits(
                            logits,
                            yb,
                            alpha=float(self.cfg.focal_alpha),
                            gamma=float(self.cfg.focal_gamma),
                            reduction="mean",
                            pos_weight=pos_weight_eff,
                        )
                    else:
                        loss = F.binary_cross_entropy_with_logits(
                            logits.view(-1),
                            yb.view(-1),
                            pos_weight=pos_weight_eff,
                        )

                    opt.zero_grad(set_to_none=True)
                    loss.backward()

                    # NEW: clip head separately first (prevents head blow-up)
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=float(getattr(self.cfg, "clip_head", 1.0)))

                    # existing global clip
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(getattr(self.cfg, "clip_all", 1.0)))

                    opt.step()

                    try:
                        epoch_losses.append(float(loss.detach().cpu().item()))
                    except Exception:
                        pass
                    # ---------------------------
                    # LEVEL 1: batch loss logging (proxy-train)
                    # ---------------------------
                    if bool(getattr(self.cfg, "log_loss_per_iter", True)):
                        log_every = int(getattr(self.cfg, "loss_log_every", 10))
                        if ((bi % log_every) == 0) or (bi == 0):
                            loss_f = float(loss.detach().cpu().item())
                            if not hasattr(self, "_ema_proxy_loss"):
                                self._ema_proxy_loss = None
                            self._ema_proxy_loss = _ema_update(self._ema_proxy_loss, loss_f, alpha=float(getattr(self.cfg, "loss_ema_alpha", 0.05)))
                            self.logger.log_to_file(
                                "loss_iter",
                                f"[proxy][iter] seed_off={seed_offset} ep={_} bi={bi} "
                                f"loss={loss_f:.6f} ema={float(self._ema_proxy_loss):.6f}"
                            )

                    # NEW: always clamp bias in SEARCH (post-step)
                    phase_now = str(getattr(self.cfg, "phase", "search")).lower()
                    if (not phase_now.startswith("final")) and (model.head.bias is not None):
                        bmax = float(getattr(self.cfg, "search_head_bias_clamp", 2.0))
                        with torch.no_grad():
                            model.head.bias.data.clamp_(-bmax, bmax)


                    if search_head_only and (not phase_local.startswith("final")):
                        default_steps = int(max(int(n_head), 2 * len(dl_small)))
                        head_steps = int(getattr(self.cfg, "search_head_steps_per_epoch", default_steps))

                        if (bi + 1) >= int(head_steps):
                            break
                    else:
                        if (bi + 1) >= n_vqc_batches:
                            break
                # ---------------------------
                # LEVEL 2: epoch loss logging
                # ---------------------------
                if bool(getattr(self.cfg, "log_loss_per_epoch", True)) and (len(epoch_losses) > 0):
                    m = float(np.mean(epoch_losses))
                    s = float(np.std(epoch_losses))
                    self.logger.log_to_file(
                        "loss_epoch",
                        f"[proxy][epoch] seed_off={seed_offset} ep={_} "
                        f"loss_mean={m:.6f} loss_std={s:.6f} n_batches={len(epoch_losses)}"
                    )
            # ----------------------------------------------------------
            # NEW: explicit separability log right after proxy-train
            # (helps debug "dead head" vs "dead circuit")
            # ----------------------------------------------------------
            try:
                model.eval()
                logits_dbg, probs_dbg, n_dbg = self._predict_logits_probs_capped(
                    model,
                    X_cal,
                    cap=int(min(int(self.cfg.inner_eval_batch_cap), int(X_cal.shape[0]))),
                )
                logit_margin = float(_p95_p5(np.asarray(logits_dbg, dtype=float)))
                prob_std = float(np.std(np.asarray(probs_dbg, dtype=float)))
                self.logger.log_to_file(
                    "separability",
                    f"[sep] phase={phase_local} seed_off={seed_offset} n={int(n_dbg)} "
                    f"logit_p95_p5={logit_margin:.4f} prob_std={prob_std:.4f} "
                    f"p(min/mean/max)={float(np.min(probs_dbg)):.4f}/{float(np.mean(probs_dbg)):.4f}/{float(np.max(probs_dbg)):.4f}",
                )

                if hasattr(model, "_dbg_raw_logits_std") and hasattr(model, "_dbg_scaled_logits_std"):
                    ls_val = float(getattr(model, "_dbg_logit_scale_value", float("nan")))
                    raw_std = float(getattr(model, "_dbg_raw_logits_std", float("nan")))
                    scl_std = float(getattr(model, "_dbg_scaled_logits_std", float("nan")))
                    self.logger.log_to_file(
                        "logit_scale",
                        f"[logit_scale] phase={phase_local} seed_off={seed_offset} "
                        f"scale={ls_val:.4f} raw_logits.std={raw_std:.6f} scaled_logits.std={scl_std:.6f}",
                    )
            except Exception:
                pass


            # ==========================================================
            # After proxy-train: define retry helper + evaluate/collapse loop
            # ==========================================================
            do_retry = bool(getattr(self.cfg, "collapse_retry", getattr(self.cfg, "anti_collapse_retry", True)))
            max_retry = int(getattr(self.cfg, "collapse_max_retry", getattr(self.cfg, "collapse_retry_max", 1)))

            base_lr_vqc  = float(getattr(self.cfg, "lr_vqc", 1e-3))
            base_lr_head = float(getattr(self.cfg, "lr_head", 1e-3))

            lr_scale = float(getattr(self.cfg, "collapse_lr_scale", 0.5))
            boost_epochs = int(getattr(self.cfg, "collapse_boost_epochs", 2))
            boost_mult = int(getattr(self.cfg, "collapse_boost_batches_mult", 2))
            boost_head_batches = int(getattr(self.cfg, "collapse_boost_head_batches", max(1, n_head // 2)))

            clip_head = float(getattr(self.cfg, "clip_head", 1.0))
            clip_all  = float(getattr(self.cfg, "clip_all", 1.0))

            head_params_local = list(model.head.parameters())

            def _make_head_optimizer(lr: float):
                hw, hb = [], []
                for p in head_params_local:
                    (hb if p.ndim == 1 else hw).append(p)
                return torch.optim.Adam(
                    [
                        {"params": hw, "lr": float(lr), "weight_decay": float(getattr(self.cfg, "wd_head", 0.0))},
                        {"params": hb, "lr": float(lr), "weight_decay": float(wd_head_bias_eff)},

                    ]
                )

            def _run_proxy_train(lr_vqc: float, lr_head: float, n_epochs_run: int, n_batches_run: int, n_head_batches: int):
                # optional head warmup
                if int(n_head_batches) > 0:
                    for p in model.parameters():
                        p.requires_grad_(False)
                    for p in model.head.parameters():
                        p.requires_grad_(True)

                    opt_h = _make_head_optimizer(float(lr_head))
                    crit_h = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_eff)

                    model.train()
                    for bi, (xb, yb) in enumerate(dl_small):
                        xb = xb.detach()
                        xb.requires_grad_(False)

                        if xb.dim() == 0:
                            xb = xb.view(1, 1)
                        elif xb.dim() == 1:
                            xb = xb.unsqueeze(0) if xb.numel() == model.input_dim else xb.view(-1, 1)
                        elif xb.dim() > 2:
                            xb = xb.view(xb.shape[0], -1)

                        if yb.dim() == 0:
                            yb = yb.view(1, 1)
                        elif yb.dim() == 1:
                            yb = yb.view(-1, 1)

                        loss = crit_h(model(xb), yb)
                        opt_h.zero_grad(set_to_none=True)
                        loss.backward()
                        # DIAGNOSTIC: head grad norm
                        with torch.no_grad():
                            if hasattr(model, "head") and model.head.weight.grad is not None:
                                g_norm = model.head.weight.grad.norm().item()
                                try:
                                    self.logger.log_to_file(
                                        "head_grad",
                                        f"[head_grad] seed_off={seed_offset} grad_norm={g_norm:.6f}"
                                    )
                                except Exception:
                                    pass
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=float(clip_head))
                        opt_h.step()

                        if (bi + 1) >= int(n_head_batches):
                            break

                # full train
                phase_local2 = str(getattr(self.cfg, "phase", "search")).lower()
                is_final2 = phase_local2.startswith("final")
                if (not is_final2) and bool(getattr(self.cfg, "search_head_only", True)):
                    # keep VQC frozen, train head only
                    for p in model.parameters():
                        p.requires_grad_(False)
                    for p in model.head.parameters():
                        p.requires_grad_(True)
                    if model.head.bias is not None:
                        model.head.bias.requires_grad_(True)
                else:
                    for p in model.parameters():
                        p.requires_grad_(True)

                phase_local2 = str(getattr(self.cfg, "phase", "search")).lower()
                is_final2 = phase_local2.startswith("final")

                lr_head2 = float(lr_head)
                lr_enc2  = float(getattr(self.cfg, "collapse_lr_enc", lr_vqc))
                lr_th2   = float(getattr(self.cfg, "collapse_lr_theta", lr_vqc))

                wd_head2 = float(getattr(self.cfg, "wd_head", 0.0))
                wd_enc2  = float(getattr(self.cfg, "final_wd_enc" if is_final2 else "search_wd_enc", float(getattr(self.cfg, "wd_enc", 0.0))))
                wd_th2   = float(getattr(self.cfg, "final_wd_theta" if is_final2 else "search_wd_theta", float(getattr(self.cfg, "wd_theta", 0.0))))

                head_params2 = list(model.head.parameters())
                enc_params2 = []
                if hasattr(model, "enc_alpha_raw") and isinstance(model.enc_alpha_raw, torch.nn.Parameter):
                    enc_params2.append(model.enc_alpha_raw)
                if hasattr(model, "enc_beta_raw") and isinstance(model.enc_beta_raw, torch.nn.Parameter):
                    enc_params2.append(model.enc_beta_raw)
                theta_params2 = [model.theta] if hasattr(model, "theta") and isinstance(model.theta, torch.nn.Parameter) else []

                seen2 = set()
                def _uniq2(ps):
                    out = []
                    for p in ps:
                        if p is None:
                            continue
                        if id(p) in seen2:
                            continue
                        seen2.add(id(p))
                        out.append(p)
                    return out

                head_params2  = _uniq2(head_params2)
                enc_params2   = _uniq2(enc_params2)
                theta_params2 = _uniq2(theta_params2)

                groups2 = []

                # ---- PATCH: split head params into weights vs bias (no WD on bias) ----
                hw2, hb2 = [], []
                for p in head_params2:
                    (hb2 if p.ndim == 1 else hw2).append(p)

                if hw2:
                    groups2.append({"params": hw2, "lr": lr_head2, "weight_decay": wd_head2})
                if hb2:
                    groups2.append({"params": hb2, "lr": lr_head2, "weight_decay": wd_head_bias_eff})

                # # keep enc/theta as before
                # if enc_params2:
                #     groups2.append({"params": enc_params2, "lr": lr_enc2, "weight_decay": wd_enc2})
                # if theta_params2:
                #     groups2.append({"params": theta_params2, "lr": lr_th2, "weight_decay": wd_th2})
                
                
                # keep enc/theta ONLY if not SEARCH head-only
                if not ((not is_final2) and bool(getattr(self.cfg, "search_head_only", True))):
                    if not ((not is_final2) and bool(search_head_only)):
                        if enc_params2:
                            groups2.append({"params": enc_params2, "lr": lr_enc2, "weight_decay": wd_enc2})
                        if theta_params2:
                            groups2.append({"params": theta_params2, "lr": lr_th2, "weight_decay": wd_th2})


                if not groups2:
                    groups2 = [
                        {
                            "params": model.parameters(),
                            "lr": lr_th2,
                            "weight_decay": float(getattr(self.cfg, "wd_vqc", 0.0)),
                        }
                    ]

                opt2 = torch.optim.Adam(groups2)
                for _ in range(int(n_epochs_run)):
                    model.train()
                    for bi, (xb, yb) in enumerate(dl_small):
                        xb = xb.detach()
                        xb.requires_grad_(False)

                        if xb.dim() == 0:
                            xb = xb.view(1, 1)
                        elif xb.dim() == 1:
                            xb = xb.unsqueeze(0) if xb.numel() == model.input_dim else xb.view(-1, 1)
                        elif xb.dim() > 2:
                            xb = xb.view(xb.shape[0], -1)

                        if yb.dim() == 0:
                            yb = yb.view(1, 1)
                        elif yb.dim() == 1:
                            yb = yb.view(-1, 1)

                        logits = model(xb)
                        if bool(self.cfg.use_focal):
                            loss = focal_loss_with_logits(
                                logits,
                                yb,
                                alpha=float(self.cfg.focal_alpha),
                                gamma=float(self.cfg.focal_gamma),
                                reduction="mean",
                                pos_weight=pos_weight_eff,
                            )
                        else:
                            loss = F.binary_cross_entropy_with_logits(
                                logits.view(-1),
                                yb.view(-1),
                                pos_weight=pos_weight_eff,
                            )
                        ###
                        opt2.zero_grad(set_to_none=True)
                        loss.backward()

                        # NEW: clip head separately first
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=float(getattr(self.cfg, "clip_head", 1.0)))

                        # existing / global clip
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_all))

                        opt2.step()

                        if (bi + 1) >= int(n_batches_run):
                            break

                        # FIX + NEW: clamp bias in SEARCH (post-step)
                        phase_now = str(getattr(self.cfg, "phase", "search")).lower()
                        if (not phase_now.startswith("final")) and (model.head.bias is not None):
                            bmax = float(getattr(self.cfg, "search_head_bias_clamp", 2.0))
                            with torch.no_grad():
                                model.head.bias.data.clamp_(-bmax, bmax)

            # ---------------------------
            # Retry loop (attempt 0 = no boost yet)
            # ---------------------------
            fail_streak = 0
            fail_streak_clinical = 0
            fail_streak_auc_floor = 0
            best_state = None
            best_pack = None

            for attempt in range(int(max_retry) + 1):
                model.eval()

                logits_raw, logits_tr, probs_tr, n = self._predict_logits_probs_capped_train(
                    model,
                    X_cal,
                    cap=int(min(int(self.cfg.inner_eval_batch_cap), int(X_cal.shape[0]))),
                )
                yt_full = (Y_cal.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
                yt = yt_full[:n]

                # ---------------------------
                # LEVEL 3A: calib loss (BCE on logits) for episode reporting
                # ---------------------------
                try:
                    # recomputa logits no torch para loss exata (evita mismatch numpy)
                    with torch.no_grad():
                        model.eval()
                        logits_t = model(X_cal[:n].detach())
                        calib_loss = _bce_logits_loss_torch(logits_t, Y_cal[:n].detach(), pos_weight=pos_weight_eff)
                except Exception:
                    calib_loss = float("nan")

                # ----------------------------------------------------------
                # SEARCH: recenter logits BEFORE threshold search
                # Fixes seed instability where probs drift to ~0.02 (all-negative)
                # ----------------------------------------------------------
                if (not phase.startswith("final")) and bool(getattr(self.cfg, "search_recenter_before_thr", True)):
                    try:
                        if hasattr(model, "head") and (model.head.bias is not None):
                            mu = float(np.mean(np.asarray(logits_tr, dtype=float)))

                            # detect saturation risk (raw logits already include logit_scale)
                            p95_abs = float(np.percentile(np.abs(np.asarray(logits_tr, dtype=float)), 95))
                            sat_thr = float(getattr(self.cfg, "collapse_saturation_abslogit_p95_thr", 12.0))
                            is_sat = bool(np.isfinite(p95_abs) and (p95_abs >= sat_thr))

                            mu_min = float(getattr(self.cfg, "search_recenter_mu_min", 0.25))
                            mu_max = float(getattr(self.cfg, "search_recenter_mu_clamp", 0.5))   # <<< tighter
                            damp   = float(getattr(self.cfg, "search_recenter_mu_damp", 0.25))   # <<< NEW

                            if (not is_sat) and (abs(mu) >= mu_min):
                                mu = float(np.clip(mu, -mu_max, mu_max))
                                delta = float(damp * mu)

                                bmax = float(getattr(self.cfg, "search_head_bias_clamp", 2.0))
                                with torch.no_grad():
                                    model.head.bias.data.sub_(delta)
                                    model.head.bias.data.clamp_(-bmax, bmax)

                                # recompute AFTER safe recenter
                                _, logits_tr, probs_tr, n = self._predict_logits_probs_capped_train(
                                    model,
                                    X_cal,
                                    cap=int(min(int(self.cfg.inner_eval_batch_cap), int(X_cal.shape[0]))),
                                )
                                yt = yt_full[:n]

                                self.logger.log_to_file(
                                    "head",
                                    f"[recenter_before_thr_safe] attempt={attempt} mu={mu:.6f} damp={damp:.3f} "
                                    f"delta={delta:.6f} p95_abs_logit={p95_abs:.3f} is_sat={int(is_sat)}"
                                )
                            else:
                                self.logger.log_to_file(
                                    "head",
                                    f"[recenter_before_thr_skip] attempt={attempt} mu={mu:.6f} mu_min={mu_min:.3f} "
                                    f"p95_abs_logit={p95_abs:.3f} is_sat={int(is_sat)}"
                                )
                    except Exception:
                        pass


                # health logs
                try:
                    ls = _safe_stats(np.asarray(logits_tr, dtype=float))
                    ps = _safe_stats(np.asarray(probs_tr, dtype=float))
                    lrmin = float(getattr(self.cfg, "collapse_range_min", 0.06))
                    lsmin = float(getattr(self.cfg, "collapse_std_min", 0.02))
                    hL = _collapse_health(np.asarray(logits_tr, dtype=float), range_min=lrmin, std_min=lsmin)
                    ps_std_min = float(getattr(self.cfg, "collapse_prob_std_min_for_healthlog", 0.005))
                    hP = _collapse_health(np.asarray(probs_tr, dtype=float),  range_min=lrmin, std_min=ps_std_min)
                    self.logger.log_to_file(
                        "health",
                        f"[train-cap health seed_off={seed_offset} attempt={attempt}] n={len(yt)} "
                        f"logits(min/mean/max)={ls['min']:.4f}/{ls['mean']:.4f}/{ls['max']:.4f} range={ls['range']:.4f} "
                        f"| Lcollapsed={int(hL['collapsed'])} p95-p5={hL['p95_p5']:.4f} std={hL['std']:.4f} "
                        f"|| probs(min/mean/max)={ps['min']:.4f}/{ps['mean']:.4f}/{ps['max']:.4f} range={ps['range']:.6f} "
                        f"| Pcollapsed={int(hP['collapsed'])} p95-p5={hP['p95_p5']:.4f} std={hP['std']:.4f}"
                    )
                except Exception:
                    pass

                y_pos = int(yt.sum())
                y_tot = int(len(yt))
                single_class = (y_pos == 0) or (y_pos == y_tot)

                thr_min = float(getattr(self.cfg, "thr_min", 0.05))
                thr_max = float(getattr(self.cfg, "thr_max", 0.95))

                if single_class:
                    thr_star = 0.5
                    thr_info = {
                        "t": float(thr_star),
                        "mode": "single_class_calib",
                        "viable_count": 0,
                        "pmin": float(np.min(probs_tr)) if probs_tr.size else float("nan"),
                        "pmax": float(np.max(probs_tr)) if probs_tr.size else float("nan"),
                        "prange": float(np.max(probs_tr) - np.min(probs_tr)) if probs_tr.size else float("nan"),
                        "note": f"yt_single_class pos={y_pos}/{y_tot}",
                    }
                    sens_c, spec_c, fpr_c = _rates_from_thr(yt, probs_tr, float(thr_star))

                    try:
                        self.logger.log_to_file(
                            "thr",
                            f"[thr] single_class_calib pos={y_pos}/{y_tot} -> force thr=0.5000 "
                            f"sens={sens_c:.4f} spec={spec_c:.4f} fpr={fpr_c:.4f}",
                        )
                    except Exception:
                        pass
                else:
                    # COST-D: coarse grid during SEARCH, full grid during FINAL.
                    _gphase = str(getattr(self.cfg, "phase", "search")).lower()
                    _grid   = int(
                        getattr(self.cfg, "grid_size", 201)
                        if _gphase.startswith("final")
                        else getattr(self.cfg, "search_grid_size", 51)
                    )
                    thr_star, thr_info = find_threshold(
                        yt,
                        probs_tr,
                        mode=str(thr_mode),
                        sens_target=float(thr_sens_target),
                        fpr_max=float(thr_fpr_max),
                        spec_min=float(thr_spec_min),
                        # grid_size=int(getattr(self.cfg, "grid_size", 401)),
                        grid_size=_grid,
                        lam_spec=float(lam_spec),
                        lam_fpr=float(lam_fpr),
                        lam_sens=float(lam_sens),
                        logits=logits_tr,
                        logger=self.logger,
                        return_info=True,
                        saturation_abslogit_p95_thr=float(
                            getattr(self.cfg, "collapse_saturation_abslogit_p95_thr", 12.0)
                        ),
                        saturation_fallback_thr=float(
                            getattr(self.cfg, "saturation_fallback_thr", 0.95)
                        ),
                    )

                    thr_star = float(np.clip(float(thr_star), thr_min, thr_max))
                    sens_c, spec_c, fpr_c = _rates_from_thr(yt, probs_tr, float(thr_star))

                self.thr = float(thr_star)

                auc, sens, spec = self._eval_metrics(model, self.X_val, self.Y_val, thr=float(thr_star))


                # ---------------------------
                # LEVEL 3B: validation loss (BCE) for episode reporting
                # ---------------------------
                try:
                    with torch.no_grad():
                        model.eval()
                        # cap pra ficar barato e consistente com sua filosofia
                        capv = int(getattr(self.cfg, "val_loss_cap", getattr(self.cfg, "inner_eval_batch_cap", 2048)))
                        nv = int(min(int(capv), int(self.X_val.shape[0])))
                        logits_v = model(self.X_val[:nv].detach())
                        val_loss = _bce_logits_loss_torch(logits_v, self.Y_val[:nv].detach(), pos_weight=pos_weight_eff)
                except Exception:
                    val_loss = float("nan")

                # viable_count = int(thr_info.get("viable_count", 0)) if isinstance(thr_info, dict) else 0
                # auc_val = float(auc)
                # auc_eps = float(getattr(self.cfg, "collapse_auc_eps", 0.01))
                # clinical_fail = (abs(auc_val - 0.5) <= auc_eps) and (viable_count == 0)
                # fail_streak = (fail_streak + 1) if clinical_fail else 0

                # NEW: include SEARCH AUC-floor failures in streak
                phase_now = str(getattr(self.cfg, "phase", "search")).lower()
                is_search_now = phase_now.startswith("search")
                viable_count = int(thr_info.get("viable_count", 0)) if isinstance(thr_info, dict) else 0
                auc_val = float(auc)
                auc_eps = float(getattr(self.cfg, "collapse_auc_eps", 0.01))
                clinical_fail = (abs(auc_val - 0.5) <= auc_eps) and (viable_count == 0)

                phase_now = str(getattr(self.cfg, "phase", "search")).lower()
                is_search_now = phase_now.startswith("search")
                auc_floor = float(getattr(self.cfg, "search_auc_floor_collapse",
                                        getattr(self.cfg, "search_auc_floor", 0.50)))
                auc_low = bool(is_search_now and np.isfinite(auc_val) and (auc_val < auc_floor))

                logit_margin = float(_p95_p5(np.asarray(logits_tr, dtype=float)))
                prob_std_now = float(np.std(np.asarray(probs_tr, dtype=float)))
                logit_margin_min = float(getattr(self.cfg, "collapse_logit_margin_min", 0.05))
                prob_std_weak = float(getattr(self.cfg, "collapse_prob_std_weak", 0.01))
                logit_gate = float(getattr(self.cfg, "collapse_logit_gate_mult", 2.0))
                logit_gate_thr = float(logit_gate) * float(logit_margin_min)

                auc_floor_bad_sep = bool(
                    auc_low and (
                        (np.isfinite(logit_margin) and (logit_margin < logit_gate_thr)) or
                        (np.isfinite(prob_std_now) and (prob_std_now < prob_std_weak))
                    )
                )

                # update streaks
                fail_streak_clinical = (fail_streak_clinical + 1) if clinical_fail else 0
                fail_streak_auc_floor = (fail_streak_auc_floor + 1) if auc_floor_bad_sep else 0
                fail_flag = bool(clinical_fail or auc_floor_bad_sep)
                fail_streak = (fail_streak + 1) if fail_flag else 0
                

                collapsed, cdbg = _collapse_decision(
                    logits_np=logits_tr,
                    probs_np=probs_tr,
                    auc_val=float(auc_val),
                    viable_count=int(viable_count),
                    fail_streak_clinical=int(fail_streak_clinical),
                    fail_streak_auc_floor=int(fail_streak_auc_floor),
                )
                if (best_pack is None) or (float(auc) > float(best_pack["auc"])) or (
                    abs(float(auc) - float(best_pack["auc"])) < 1e-12 and (float(spec), float(sens)) > (float(best_pack["spec"]), float(best_pack["sens"]))
                ):
                    best_pack = {
                        "auc": float(auc),
                        "sens": float(sens),
                        "spec": float(spec),
                        "thr_star": float(thr_star),
                        "calib_sens": float(sens_c),
                        "calib_spec": float(spec_c),
                        "calib_fpr": float(fpr_c),
                        "thr_info": (thr_info if isinstance(thr_info, dict) else {}),
                        "collapse_dbg": dict(cdbg),
                    }
                    best_state = deepcopy(model.state_dict())

                try:
                    self.logger.log_to_file(
                        "thr_debug",
                        f"[collapse-check] seed_off={seed_offset} attempt={attempt} "
                        f"collapsed={int(cdbg['collapsed'])} reason={cdbg['reason']} "
                        f"logit_p95_p5={float(cdbg['logit_p95_p5']):.4f} prob_std={float(cdbg['prob_std']):.4f} "
                        f"auc={auc_val:.4f} viable={viable_count} fail_streak={fail_streak}",
                    )
                except Exception:
                    pass

                if (not do_retry) or (not collapsed) or (attempt >= int(max_retry)):
                    break

                n_epochs2 = int(n_epochs + boost_epochs)
                n_batches2 = int(max(1, n_vqc_batches * boost_mult))

                _run_proxy_train(
                    lr_vqc=float(base_lr_vqc * lr_scale),
                    lr_head=float(base_lr_head),
                    n_epochs_run=int(n_epochs2),
                    n_batches_run=int(n_batches2),
                    n_head_batches=int(boost_head_batches),
                )

            if best_state is not None:
                model.load_state_dict(best_state)
            # ----------------------------------------------------------
            # DIAGNOSTIC: head weight / bias norm (after proxy-train)
            # ----------------------------------------------------------
            with torch.no_grad():
                if hasattr(model, "head") and hasattr(model.head, "weight"):
                    w_norm = model.head.weight.norm().item()
                    b_norm = (
                        model.head.bias.norm().item()
                        if hasattr(model.head, "bias") and model.head.bias is not None
                        else 0.0
                    )

                    try:
                        self.logger.log_to_file(
                            "head",
                            f"[head] seed_off={seed_offset} "
                            f"weight_norm={w_norm:.4f} bias_norm={b_norm:.4f}"
                        )
                        # Scale-aware warning: compare against sqrt(d)
                        d = int(model.head.weight.numel())  # since output=1, numel == in_features
                        d = max(1, d)
                        # Heuristic threshold: tiny norm relative to dimension
                        # (works across different head sizes)
                        thr = float(getattr(self.cfg, "dead_head_norm_thr", 0.02)) * math.sqrt(d)
                        if w_norm < thr:
                            self.logger.log_to_file(
                                "head",
                                f"[head][WARN] small head norm: ||w||={w_norm:.4f} < {thr:.4f} (d={d})"
                            )
                    except Exception:
                        pass

            best_pack["calib_bce"] = float(locals().get("calib_loss", float("nan")))
            best_pack["val_bce"]   = float(locals().get("val_loss",   float("nan")))
            assert best_pack is not None
            best_pack["model"] = model
            return best_pack

        # ==========================================================
        # >>>>>>>>>>>>>>>>> FINALIZAÇÃO (FALTAVA) <<<<<<<<<<<<<<<<<
        # ==========================================================
        try:
            # quantos seeds no terminal? 1 ou 2
            # n_seeds = int(getattr(self.cfg, "terminal_proxy_n_seeds", 1))
            # n_seeds = int(np.clip(n_seeds, 1, 2))

            # packs = []
            # for so in range(n_seeds):
            #     packs.append(_one_proxy_run(seed_offset=int(so)))
            # use2 = bool(getattr(self.cfg, "proxy_two_seeds", False))
            # delta = int(getattr(self.cfg, "proxy_seed_delta", 1337))
            # seed_offsets = [0, delta] if use2 else [0]

            # packs = [ _one_proxy_run(seed_offset=int(so)) for so in seed_offsets ]
            # n_seeds = len(seed_offsets)

            # # melhor pack (AUC primário, depois SPEC, depois SENS)
            # def _key(pk):
            #     return (float(pk.get("auc", 0.0)), float(pk.get("spec", 0.0)), float(pk.get("sens", 0.0)))
            # best_pack = max(packs, key=_key)

            # # std_proxy (pra sua penalização lambda_var)
            # proxy_scores = []
            # for pk in packs:
            #     auc01 = float(np.clip(pk.get("auc", 0.5), 0.0, 1.0))
            #     sens  = float(np.clip(pk.get("sens", 0.0), 0.0, 1.0))
            #     spec  = float(np.clip(pk.get("spec", 0.0), 0.0, 1.0))
            #     youden = float(np.clip(sens + spec - 1.0, -1.0, 1.0))
            #     youden01 = 0.5 * (youden + 1.0)
            #     proxy_scores.append(0.5 * auc01 + 0.5 * youden01)
            # std_proxy = float(np.std(np.asarray(proxy_scores, dtype=float))) if len(proxy_scores) > 1 else 0.0

            # # atualiza cache do env (usado no shaping do step)
            # self.last_auc  = float(best_pack.get("auc", 0.5))
            # self.last_sens = float(best_pack.get("sens", 0.0))
            # self.last_spec = float(best_pack.get("spec", 0.0))
            # thr_star = float(best_pack.get("thr_star", getattr(self.cfg, "thr_init", 0.5)))
            # self.thr = float(thr_star)

            # # depth/cnot: se você tiver “tape real”, pluga aqui;
            # # por enquanto: depth = steps do arch, cnot = contagem do arch (coerente e rápido).
            # arch_now = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
            # counts = self._count_ops(arch_now)
            # depth_t = int(self.step_idx)
            # cnot_t  = int(counts["CNOT"])

            # # tinfo: garante que sempre tenha std_proxy
            # tinfo = dict(best_pack.get("thr_info", {})) if isinstance(best_pack.get("thr_info", {}), dict) else {}
            # tinfo["std_proxy"] = float(std_proxy)
            # tinfo["n_seeds"] = int(n_seeds)
            # tinfo["collapse_dbg"] = best_pack.get("collapse_dbg", {})
            # FIX-2: build seed_offsets from proxy_n_seeds (default 3).
            # Previously hardcoded [0, delta] with best-of-2 (argmax) aggregation.
            # With 3 seeds + min() a single lucky seed cannot inflate the score.
            # corr(std_proxy, AUC) = 0.63 in Cross/Circle logs; min() kills that correlation.
            use2        = bool(getattr(self.cfg, "proxy_two_seeds", True))
            delta       = int(getattr(self.cfg, "proxy_seed_delta", 1337))
            n_seeds_cfg = int(getattr(self.cfg, "proxy_n_seeds", 3 if use2 else 1))
            n_seeds_cfg = max(1, n_seeds_cfg)
            seed_offsets = [int(i * delta) for i in range(n_seeds_cfg)]

            packs = [_one_proxy_run(seed_offset=int(so)) for so in seed_offsets]
            n_seeds = len(packs)

            # Youden-weighted proxy score per seed (same formula as train.py)
            proxy_scores = []
            for pk in packs:
                auc01_    = float(np.clip(pk.get("auc",  0.5), 0.0, 1.0))
                sens_     = float(np.clip(pk.get("sens", 0.0), 0.0, 1.0))
                spec_     = float(np.clip(pk.get("spec", 0.0), 0.0, 1.0))
                youden_   = float(np.clip(sens_ + spec_ - 1.0, -1.0, 1.0))
                youden01_ = 0.5 * (youden_ + 1.0)
                proxy_scores.append(0.5 * auc01_ + 0.5 * youden01_)
            proxy_arr = np.asarray(proxy_scores, dtype=float)
            std_proxy = float(np.std(proxy_arr)) if len(proxy_arr) > 1 else 0.0

            # FIX-2: aggregation strategy
            agg = str(getattr(self.cfg, "proxy_aggregation", "min")).lower().strip()
            if agg == "min":
                best_idx = int(np.argmin(proxy_arr))
            elif agg == "quantile":
                q     = float(getattr(self.cfg, "proxy_quantile", 0.20))
                q_val = float(np.quantile(proxy_arr, q))
                best_idx = int(np.argmin(np.abs(proxy_arr - q_val)))
            else:  # "mean" / legacy: pick best-scoring seed (old behaviour)
                best_idx = int(np.argmax(proxy_arr))
            best_pack = packs[best_idx]

            # FIX-5: thr* stability — std across seeds, exposed to train.py via tinfo
            thr_stars_all = [float(pk.get("thr_star", 0.5)) for pk in packs]
            thr_std = float(np.std(np.asarray(thr_stars_all, dtype=float))) if len(thr_stars_all) > 1 else 0.0
            self._terminal_thr_stars = thr_stars_all

            if bool(getattr(self.cfg, "log_thr_stability", True)):
                try:
                    self.logger.log_to_file(
                        "thr_stability",
                        f"[thr_stability] n_seeds={n_seeds} agg={agg} "
                        f"thr_stars={[f'{t:.3f}' for t in thr_stars_all]} "
                        f"thr_std={thr_std:.4f} "
                        f"proxy_scores={[f'{s:.4f}' for s in proxy_scores]} "
                        f"std_proxy={std_proxy:.4f} best_idx={best_idx}"
                    )
                except Exception:
                    pass

            # atualiza cache do env (usado no shaping do step)
            self.last_auc  = float(best_pack.get("auc", 0.5))
            self.last_sens = float(best_pack.get("sens", 0.0))
            self.last_spec = float(best_pack.get("spec", 0.0))
            thr_star = float(best_pack.get("thr_star", getattr(self.cfg, "thr_init", 0.5)))
            self.thr = float(thr_star)

            # depth/cnot via arch state
            arch_now = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
            counts = self._count_ops(arch_now)
            depth_t = int(self.step_idx)
            cnot_t  = int(counts["CNOT"])

            # tinfo: campos obrigatórios para train.py
            tinfo = dict(best_pack.get("thr_info", {})) if isinstance(best_pack.get("thr_info", {}), dict) else {}
            tinfo["std_proxy"] = float(std_proxy)
            tinfo["thr_std"]   = float(thr_std)        # FIX-5: lambda_thr_std penalty
            tinfo["n_seeds"]   = int(n_seeds)
            tinfo["collapse_dbg"] = best_pack.get("collapse_dbg", {})

            # FIX-4: register terminal architecture in inter-episode dedup set
            arch_hash = hash(self.state.tobytes())
            self._terminal_arch_hashes.add(arch_hash)


            try:
                self.logger.log_to_file(
                    "thr",
                    f"[terminal] phase={phase} auc={self.last_auc:.4f} sens={self.last_sens:.4f} spec={self.last_spec:.4f} "
                    f"thr*={thr_star:.3f} std_proxy={std_proxy:.4f} n_seeds={n_seeds} time={time.perf_counter()-t0:.2f}s"
                )
                # FIX-C: one-line split_summary for quick grep verification.
                # Targets: N_val>=400 confirms FIX-1; delta_gap>>0 confirms FIX-3.
                _yv2 = (self.Y_val.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
                _nv2 = int(len(_yv2))
                _csd2 = int(getattr(self.cfg, "calib_seed_delta", 99991))
                _ns2  = int(getattr(self.cfg, "proxy_n_seeds", 3))
                _d2   = int(getattr(self.cfg, "proxy_seed_delta", 1337))
                _max_off = (_ns2 - 1) * _d2
                self.logger.log_to_file(
                    "terminal_splits",
                    f"[split_summary] N_val={_nv2}(pos={int(_yv2.sum())},neg={_nv2-int(_yv2.sum())}) "
                    f"val_frac={getattr(self.cfg,'val_frac_search',0.40):.2f} "
                    f"percent_search={getattr(self.cfg,'percent_search',60)} "
                    f"calib_seed_delta={_csd2} max_proxy_offset={_max_off} "
                    f"delta_gap={_csd2-_max_off}  target:N_val>=400,gap>>0"
                )
            except Exception:
                pass




            # ---------------------------
            # LEVEL 3: episode-level error summary (losses + metrics)
            # ---------------------------
            _calib_bce = float(best_pack.get("calib_bce", float("nan")))
            _val_bce   = float(best_pack.get("val_bce",   float("nan")))
            try:
                # se calib_loss/val_loss estiverem no escopo do último attempt, ótimo;
                # se não, loga NaN.
                self.logger.log_to_file(
                    "episode",
                    f"[episode] phase={phase} steps={int(self.step_idx)} "
                    f"auc={self.last_auc:.4f} sens={self.last_sens:.4f} spec={self.last_spec:.4f} thr*={thr_star:.3f} "
                    f"calib_bce={_calib_bce:.6f} "
                    f"val_bce={_val_bce:.6f} "
                    f"std_proxy={std_proxy:.4f} n_seeds={n_seeds}"
                )
            except Exception:
                pass
            
            # return (
            #     float(self.last_auc),
            #     float(self.last_sens),
            #     float(self.last_spec),
            #     float(thr_star),
            #     int(depth_t),
            #     int(cnot_t),
            #     tinfo,
            # )
            _result = (
                float(self.last_auc),
                float(self.last_sens),
                float(self.last_spec),
                float(thr_star),
                int(depth_t),
                int(cnot_t),
                tinfo,
            )
            _maxsz = int(getattr(self, "_arch_result_cache_maxsize", 64))
            if len(self._arch_result_cache) >= _maxsz:
                try:
                    del self._arch_result_cache[next(iter(self._arch_result_cache))]
                except Exception:
                    pass
            self._arch_result_cache[_arch_cache_key] = _result
            return _result

        except Exception as e:
            # FAIL-SAFE: nunca retorne None
            try:
                self.logger.log_to_file("thr", f"[terminal_evaluate][EXCEPTION] {type(e).__name__}: {e}")
            except Exception:
                pass
            arch_now = torch.tensor(self.state, dtype=torch.int64, device=self.DEVICE)
            counts = self._count_ops(arch_now)
            depth_t = int(self.step_idx)
            cnot_t  = int(counts["CNOT"])
            tinfo = {"error": f"{type(e).__name__}: {e}"}
            self.last_auc, self.last_sens, self.last_spec = 0.5, 0.0, 0.0
            thr_star = float(getattr(self.cfg, "thr_init", 0.5))
            self.thr = float(thr_star)
            return 0.5, 0.0, 0.0, float(thr_star), int(depth_t), int(cnot_t), tinfo
