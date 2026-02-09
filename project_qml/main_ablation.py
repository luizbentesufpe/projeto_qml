
"""
Ablation runner for your RL-QCV pipeline.

What this adds on top of your current main():
- ABLATIONS list (scenarios) with Config overrides
- Runs: for each scenario -> for each seed -> for each nq in QUBITS_LIST
- Keeps protocol:
  * RL search uses percent_search (reduced)
  * final reporting uses percent_eval (full)
  * holdout is created from full data and NEVER used in RL-search
  * nested CV on train_all, then one final holdout evaluation
  * measured cost via tape-based mean depth/CNOT
- Saves:
  * one JSON per scenario
  * one final aggregate JSON across scenarios

Drop-in: paste below replacing your main() block (or create main_ablation()).
"""

# -------------------------
# Main Ablation Runner
# -------------------------

from datetime import datetime
import json
from rl_and_qml_in_clinical_images.util import Logger
from pathlib import Path

from rl_and_qml_in_clinical_images.dataset import _make_search_splits_from_train_all, load_chestmnist_flatten, load_chestmnist_pool_flatten
from rl_and_qml_in_clinical_images.modeling.baselines import kfold_baselines_calibrated
from rl_and_qml_in_clinical_images.modeling.ci import mean_ci_bootstrap, mean_ci_t
from rl_and_qml_in_clinical_images.modeling.cost import measure_cost_from_arch, pareto_front
from rl_and_qml_in_clinical_images.modeling.train import make_cfg_for_qubits, nested_cv_eval_fixed_arch, run_arch_search_end2end, split_holdout, train_final_model_end2end
from rl_and_qml_in_clinical_images.rl.env import sanitize_architecture
from rl_and_qml_in_clinical_images.rl.rl_config import Config, apply_overrides, build_ablation_scenarios, scenario_tag
from rl_and_qml_in_clinical_images.util import dump_run_metadata, set_seeds
import numpy as np

import torch, os
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import csv

def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 2: return float("nan")
    x = x - x.mean(); y = y - y.mean()
    den = (np.sqrt((x*x).sum()) * np.sqrt((y*y).sum()) + 1e-12)
    return float((x*y).sum() / den)

def _spearman(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 2: return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return _pearson(rx, ry)


def main_ablation():
    try:
        import torch

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"

    print("[HB] entered main_ablation()")
    root_out = Path("publication_out_ablation")
    root_out.mkdir(parents=True, exist_ok=True)

    # Base config
    cfg0 = Config()
    
    SEEDS = [0, 1, 2, 3, 4]
    QUBITS_LIST = [4, 6, 8, 10]

    PERCENT_SEARCH = int(cfg0.percent_search)
    PERCENT_EVAL   = int(cfg0.percent_eval)

    scenarios = build_ablation_scenarios()

    all_scenarios_results = []

    for sc in scenarios:
        sc_name = str(sc["name"])
        sc_over = dict(sc.get("overrides", {}))
        sc_tag  = scenario_tag(sc_name)
        sc_sem  = str(sc.get("semantics_flag", "strong"))
        sc_note = str(sc.get("notes", ""))


        sc_dir = root_out / sc_tag
        (sc_dir / "logs").mkdir(parents=True, exist_ok=True)

        logger = Logger(sc_dir / "logs")

        cfg_base = apply_overrides(cfg0, sc_over)

        # Save scenario meta
        dump_run_metadata(
            logger, cfg_base,
            extra={
                "scenario": sc_name,
                "scenario_tag": sc_tag,
                "semantics_flag": sc_sem,
                "notes": sc_note,
                "overrides": sc_over,
                "percent_search": PERCENT_SEARCH,
                "percent_eval": PERCENT_EVAL,
                "seeds": SEEDS,
                "qubits_list": QUBITS_LIST,
            }
        )

        scenario_runs = []

        # NÍVEL 3 (7) — correlation RL proxy vs final AUC (per scenario)
        corr_csv = sc_dir / "rl_proxy_vs_final.csv"
        if not corr_csv.exists():
            with open(corr_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["scenario_tag", "seed", "nq_requested", "best_nq", "best_proxy_rl", "final_auc_nested"])


        for seed in SEEDS:
            set_seeds(seed)
            dump_run_metadata(
                logger, cfg_base,
                extra={"scenario": sc_name, "seed": seed, "percent_search": PERCENT_SEARCH, "percent_eval": PERCENT_EVAL}
            )

            # 1) Full data for evaluation (100%) – used ONLY for final reporting
            # (XtrF, YtrF), (XvaF, YvaF), (XteF, YteF) = load_chestmnist_flatten(
            #     cfg_base, percentage_each_split=PERCENT_EVAL, seed=seed
            # )
            # X_full = np.concatenate([XtrF, XvaF, XteF], axis=0)
            # Y_full = np.concatenate([YtrF, YvaF, YteF], axis=0)

            X_full, Y_full = load_chestmnist_pool_flatten(cfg_base, percent_total=int(PERCENT_EVAL), seed=int(seed))

            # 2) Final untouched holdout
            tr_idx, ho_idx = split_holdout(X_full, Y_full, frac=float(cfg_base.holdout_frac), seed=seed)
            X_train_all, Y_train_all = X_full[tr_idx], Y_full[tr_idx]
            X_holdout,  Y_holdout    = X_full[ho_idx], Y_full[ho_idx]

            # 3) Reduced data for RL search ONLY (no leakage with holdout)
            # (XtrS, YtrS), (XvaS, YvaS), (XteS, YteS) = load_chestmnist_flatten(
            #     cfg_base, percentage_each_split=PERCENT_SEARCH, seed=seed, allowed_indices=tr_idx
            # )
            (XtrS, YtrS), (XvaS, YvaS) = _make_search_splits_from_train_all(
                X_train_all, Y_train_all,
                percent_search=int(cfg_base.percent_search),
                seed=int(seed),
                val_frac=0.20,
            )
            # # Baselines: evaluated on TRAIN ONLY (holdout untouched),
            # # with fold-train threshold calibration (thr*)
            # baselines = kfold_baselines_calibrated(
            #     X_train_all, Y_train_all, cfg=cfg_base, seed=seed, n_splits=5
            # )

            rlqcv = {}
            pareto_points = []

            for nq in QUBITS_LIST:
                # Important: make_cfg_for_qubits may change feature_bank sizes (pixels domain)
                # For patch-bank compact, apply_overrides already set domain to P patches,
                # and make_cfg_for_qubits keeps it stable (because use_patch_bank True).
                cfg_nq = make_cfg_for_qubits(cfg_base, int(nq))

                cfg_nq.start_qubits = int(nq)
                cfg_nq.min_qubits   = max(4, int(nq) - 2)
                cfg_nq.max_qubits   = min(10, int(nq) + 2)

                dump_run_metadata(
                    logger, cfg_nq,
                    extra={"scenario": sc_name, "seed": seed, "n_qubits": int(nq),
                           "percent_search": PERCENT_SEARCH, "percent_eval": PERCENT_EVAL}
                )

                # RL search on reduced splits only
                arch_mat, best_nq, best_proxy = run_arch_search_end2end(
                    XtrS, YtrS, XvaS, YvaS, cfg=cfg_nq, logger=logger, seed=seed, device=DEVICE
                )

                arch_mat = sanitize_architecture(arch_mat, int(best_nq))

                # Nested CV on TRAIN_ALL (holdout untouched)
                nested = nested_cv_eval_fixed_arch(
                    arch_mat, X_train_all, Y_train_all, cfg=cfg_nq, n_qubits=int(best_nq), logger=logger, seed=seed, device=DEVICE
                )

                # --- log RL-proxy vs final AUC for correlation figure ---
                try:
                    with open(corr_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([str(sc_tag), int(seed), int(nq), int(best_nq),
                                    ("" if best_proxy is None else float(best_proxy)),
                                    float(nested["auc_mean"])])
                except Exception as e:
                    logger.log_to_file("corr", f"[WARN] could not append corr csv: {e}")


                # Freeze arch; train on TRAIN_ALL and evaluate once on HOLDOUT
                auc_ho, sens_ho, thr_ho = train_final_model_end2end(
                    arch_mat, int(best_nq),
                    X_train_all, Y_train_all,
                    X_holdout,  Y_holdout,
                    cfg_nq, logger, device=DEVICE
                )

                # Measured cost (tape-based)
                X_ref = X_train_all[:max(64, int(cfg_nq.cost_measure_samples))]
                cost_obj = measure_cost_from_arch(arch_mat, int(best_nq), cfg_nq, X_ref, seed=seed)
                cost = float(cost_obj["cost"])

                # Perf metric: nested CV mean (not holdout)
                perf = 0.5 * (nested["auc_mean"] + nested["sens_mean"])

                rlqcv[str(nq)] = {
                    "best_n_qubits": int(best_nq),
                    "best_proxy_rl": (None if best_proxy is None else float(best_proxy)),
                    "arch_mat": arch_mat.cpu().numpy().tolist(),
                    "nested_cv": nested,
                    "holdout": {"thr_star": float(thr_ho), "auc": float(auc_ho), "sens@thr*": float(sens_ho)},
                    "perf": float(perf),
                    "cost": float(cost),
                    "budgets": {"ENC": int(cfg_nq.ENC_budget), "ROT": int(cfg_nq.ROT_budget), "CNOT": int(cfg_nq.CNOT_budget)},
                    "measured_cost": cost_obj
                }
                pareto_points.append({"seed": int(seed), "n_qubits": int(nq), "cost": float(cost), "perf": float(perf)})

            # pick best nq per seed by perf
            best = None
            for nq, obj in rlqcv.items():
                if best is None or obj["perf"] > best["perf"]:
                    best = {"seed": int(seed), "n_qubits": int(nq), **obj}

            scenario_runs.append({
                "scenario": sc_name,
                "seed": int(seed),
                "percent_search": int(PERCENT_SEARCH),
                "percent_eval": int(PERCENT_EVAL),
                "holdout_frac": float(cfg_base.holdout_frac),
                # "baselines": baselines,
                "rlqcv": rlqcv,
                "best_rlqcv": best,
                "pareto": pareto_points
            })

        # Aggregate per-scenario
        best_per_seed = [r["best_rlqcv"] for r in scenario_runs]
        perf_list = [b["perf"] for b in best_per_seed if b is not None]

        if str(cfg_base.ci_method).lower() == "bootstrap":
            perf_mean, perf_lo, perf_hi = mean_ci_bootstrap(
                perf_list, B=int(cfg_base.bootstrap_B), alpha=float(cfg_base.ci_alpha), seed=0
            )
        else:
            perf_mean, perf_lo, perf_hi = mean_ci_t(perf_list, alpha=float(cfg_base.ci_alpha))

        base_aggr = {}
        # for bn in base_names:
        #     vals = [
        #         0.5 * (np.mean(r["baselines"][bn]["aucs"]) + np.mean(r["baselines"][bn]["sens@thr*"]))
        #         for r in scenario_runs
        #     ]
        #     if str(cfg_base.ci_method).lower() == "bootstrap":
        #         m, lo, hi = mean_ci_bootstrap(vals, B=int(cfg_base.bootstrap_B), alpha=float(cfg_base.ci_alpha), seed=0)
        #     else:
        #         m, lo, hi = mean_ci_t(vals, alpha=float(cfg_base.ci_alpha))
        #     base_aggr[bn] = {"perf_mean": float(m), "ci95": [float(lo), float(hi)], "per_seed": [float(v) for v in vals]}

        # Pareto front over all seed*nq points
        all_points = [p for r in scenario_runs for p in r["pareto"]]
        pf = pareto_front(all_points)

        scenario_result = {
            "scenario": {"name": sc_name, "tag": sc_tag, "overrides": sc_over,
                "semantics_flag": sc_sem, "notes": sc_note},
            "meta": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seeds": SEEDS,
                "percent_search": int(PERCENT_SEARCH),
                "percent_eval": int(PERCENT_EVAL),
                "holdout_frac": float(cfg_base.holdout_frac),
                "qubits_list": QUBITS_LIST,
                "metric": "0.5*(AUC_mean + SENS_mean) for RLQCV (nested CV), baselines use SENS@thr*",
                "ci_method": str(cfg_base.ci_method),
                "ci_alpha": float(cfg_base.ci_alpha),
                "bootstrap_B": int(cfg_base.bootstrap_B),
            },
            "best_rlqcv_per_seed": best_per_seed,
            "rlqcv_perf_ci95": {"mean": float(perf_mean), "ci95": [float(perf_lo), float(perf_hi)], "per_seed": [float(x) for x in perf_list]},
            "baselines": base_aggr,
            "pareto_front": pf,
            "runs": scenario_runs
        }

                # NÍVEL 3 (7) — compute correlation over the CSV we built
        try:
            rows = []
            with open(corr_csv, "r") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        bp = float(row["best_proxy_rl"])
                        fa = float(row["final_auc_nested"])
                        if np.isfinite(bp) and np.isfinite(fa):
                            rows.append((bp, fa))
                    except Exception:
                        pass
            if len(rows) >= 2:
                xs = [a for a, b in rows]
                ys = [b for a, b in rows]
                pr = _pearson(xs, ys)
                sr = _spearman(xs, ys)
                logger.log_to_file("corr", f"[CORR] scenario={sc_tag} n={len(rows)} pearson={pr:.3f} spearman={sr:.3f} (rl_proxy vs final_auc)")
                scenario_result["corr_rlproxy_vs_finalauc"] = {"n": int(len(rows)), "pearson": float(pr), "spearman": float(sr)}
        except Exception as e:
            logger.log_to_file("corr", f"[WARN] correlation computation failed: {e}")


        out_path = sc_dir / f"results_{sc_tag}.json"
        out_path.write_text(json.dumps(scenario_result, indent=2), encoding="utf-8")
        print(f"[OK] Saved scenario: {out_path}")

        all_scenarios_results.append({
            "scenario": {"name": sc_name, "tag": sc_tag, "overrides": sc_over,
                "semantics_flag": sc_sem, "notes": sc_note},
            "summary": {
                "rlqcv_perf_ci95": scenario_result["rlqcv_perf_ci95"],
                "baselines": scenario_result["baselines"],
                "pareto_front": scenario_result["pareto_front"],
            },
            "results_file": str(out_path)
        })

    # Final aggregate index
    agg = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_scenarios": len(all_scenarios_results),
        },
        "scenarios": all_scenarios_results
    }
    agg_path = root_out / "ALL_SCENARIOS_INDEX.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"[OK] Saved aggregate index: {agg_path}")


