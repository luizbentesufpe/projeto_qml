from datetime import datetime
import json
from rl_and_qml_in_clinical_images.util import Logger
from pathlib import Path
import traceback
from typing import Any, List

from typing import Dict

from rl_and_qml_in_clinical_images.dataset import _make_search_splits_from_train_all, load_chestmnist_flatten
from rl_and_qml_in_clinical_images.modeling.baselines import kfold_baselines_calibrated
from rl_and_qml_in_clinical_images.modeling.ci import mean_ci_bootstrap, mean_ci_t
from rl_and_qml_in_clinical_images.modeling.cost import measure_cost_from_arch, pareto_front
from rl_and_qml_in_clinical_images.modeling.train import make_cfg_for_qubits, nested_cv_eval_fixed_arch, run_arch_search_end2end, split_holdout, train_final_model_end2end
from rl_and_qml_in_clinical_images.rl.env import sanitize_architecture
from rl_and_qml_in_clinical_images.rl.rl_config import Config, apply_overrides, build_ablation_scenarios, scenario_tag
from rl_and_qml_in_clinical_images.util import dump_run_metadata, set_seeds
import numpy as np


def _minimalize_cfg_for_debug(cfg: Config, fixed_nq: int) -> Config:
    """
    Forces minimal compute while still exercising the full pipeline.
    """
    # RL search
    cfg.episodes = 1
    cfg.L_max = 4
    cfg.n_steps = 2  # if your agent uses it

    # data percentages (tiny)
    cfg.percent_search = 1
    cfg.percent_eval = 1

    # inner loop (fast)
    cfg.inner_train_subset_size = 32
    cfg.inner_train_batches_head = 1
    cfg.inner_train_batches_vqc = 1
    cfg.inner_eval_batch_cap = 32
    cfg.inner_epochs_classif = 1

    # final training (fast)
    cfg.final_epochs = 1
    cfg.cost_measure_samples = 4  # minimal cost probe

    # robustness for debug
    cfg.use_focal = False
    cfg.hard_block_budget = False

    # stabilize qubits (kills many edge bugs)
    cfg.start_qubits = fixed_nq
    cfg.min_qubits = fixed_nq
    cfg.max_qubits = fixed_nq

    # nested cv minimal but non-trivial
    cfg.nested_cv_splits_outer = 2
    cfg.nested_cv_splits_inner = 2  # if used anywhere else

    # CI lightweight
    cfg.ci_method = str(cfg.ci_method)  # keep whatever you have
    cfg.bootstrap_B = int(min(int(cfg.bootstrap_B), 200))
    cfg.ci_alpha = float(cfg.ci_alpha)

    return cfg



def main_debug_ablation() -> None:
    """
    DEBUG ONLY.
    Runs all scenarios with minimal compute but exercises ALL steps from main_ablation.
    """
    print("[DEBUG] entered main_debug_ablation()")

    # Always decide device HERE, then pass it down (padronizado: device vem do train/main)
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    root_out = Path("DEBUG_ABLATION")
    root_out.mkdir(parents=True, exist_ok=True)

    # Base config
    cfg0 = Config()
    DEBUG_NQ = 4
    cfg0 = _minimalize_cfg_for_debug(cfg0, fixed_nq=DEBUG_NQ)

    SEEDS = [0]
    QUBITS_LIST = [DEBUG_NQ]  # fixed for debug

    scenarios = build_ablation_scenarios()
    all_scenarios_index: List[Dict[str, Any]] = []

    for sc in scenarios:
        sc_name = str(sc["name"])
        sc_over = dict(sc.get("overrides", {}))
        sc_tag = scenario_tag(sc_name)
        sc_sem = str(sc.get("semantics_flag", "strong"))
        sc_note = str(sc.get("notes", ""))

        print(f"\n[DEBUG] Scenario: {sc_name} (tag={sc_tag}, semantics={sc_sem})")

        sc_dir = root_out / sc_tag
        logs_dir = sc_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        logger = Logger(logs_dir)

        # Apply overrides to base cfg, then re-minimalize to avoid a scenario override making it huge
        cfg_base = apply_overrides(cfg0, sc_over)
        cfg_base = _minimalize_cfg_for_debug(cfg_base, fixed_nq=DEBUG_NQ)

        dump_run_metadata(
            logger,
            cfg_base,
            device=device,
            extra={
                "mode": "DEBUG",
                "scenario": sc_name,
                "scenario_tag": sc_tag,
                "semantics_flag": sc_sem,
                "notes": sc_note,
                "overrides": sc_over,
                "seeds": SEEDS,
                "qubits_list": QUBITS_LIST,
                "percent_search": int(cfg_base.percent_search),
                "percent_eval": int(cfg_base.percent_eval),
            },
        )

        scenario_runs: List[Dict[str, Any]] = []
        pareto_points: List[Dict[str, Any]] = []

        for seed in SEEDS:
            set_seeds(seed)

            # -------------------------
            # (A) FULL DATA (percent_eval) -> HOLDOUT SPLIT
            # -------------------------
            (XtrF, YtrF), (XvaF, YvaF), (XteF, YteF) = load_chestmnist_flatten(
                cfg_base, percentage_each_split=int(cfg_base.percent_eval), seed=seed
            )
            X_full = np.concatenate([XtrF, XvaF, XteF], axis=0)
            Y_full = np.concatenate([YtrF, YvaF, YteF], axis=0)

            tr_idx, ho_idx = split_holdout(X_full, Y_full, frac=float(cfg_base.holdout_frac), seed=seed)
            X_train_all, Y_train_all = X_full[tr_idx], Y_full[tr_idx]
            X_holdout, Y_holdout = X_full[ho_idx], Y_full[ho_idx]

            # -------------------------
            # (B) SEARCH DATA (percent_search) -> RL SEARCH ONLY
            # -------------------------
            (XtrS, YtrS), (XvaS, YvaS) = _make_search_splits_from_train_all(
                X_train_all, Y_train_all,
                percent_search=int(cfg_base.percent_search),
                seed=int(seed),
                val_frac=0.20,
            )

            print(f"[DEBUG] seed={seed} | TRAIN_ALL size={len(X_train_all)} | HOLDOUT size={len(X_holdout)}")
            # -------------------------
            # (C) BASELINES on TRAIN_ALL (holdout untouched)
            # -------------------------
            # keep minimal folds but still run calibration path
            baselines = kfold_baselines_calibrated(
                X_train_all,
                Y_train_all,
                cfg=cfg_base,
                seed=seed,
                n_splits=2,
            )

            rlqcv_by_nq: Dict[str, Any] = {}
            seed_status = "OK"
            seed_error = None

            for nq in QUBITS_LIST:
                cfg_nq = make_cfg_for_qubits(cfg_base, int(nq))

                # debug: fix qubits
                cfg_nq.start_qubits = int(nq)
                cfg_nq.min_qubits = int(nq)
                cfg_nq.max_qubits = int(nq)

                dump_run_metadata(
                    logger,
                    cfg_nq,
                    device=device,
                    extra={"mode": "DEBUG", "scenario": sc_name, "seed": int(seed), "n_qubits": int(nq)},
                )

                #try:
                # -------------------------
                # (D) RL SEARCH (minimal)
                    # -------------------------
                # RL search on reduced splits only
                arch_mat, best_nq, best_proxy = run_arch_search_end2end(
                    XtrS, YtrS, XvaS, YvaS, cfg=cfg_nq, logger=logger, seed=seed, device="cpu"
                )
                arch_mat = sanitize_architecture(arch_mat, int(best_nq))

                # -------------------------
                # (E) NESTED CV on TRAIN_ALL (holdout untouched)
                # -------------------------
                nested = nested_cv_eval_fixed_arch(
                    arch_mat,
                    X_train_all,
                    Y_train_all,
                    cfg=cfg_nq,
                    n_qubits=int(best_nq),
                    logger=logger,
                    seed=seed,
                    device=device,
                )

                # -------------------------
                # (F) FINAL TRAIN on TRAIN_ALL -> EVAL on HOLDOUT
                # -------------------------
                auc_ho, sens_ho, thr_ho = train_final_model_end2end(
                    arch_mat,
                    int(best_nq),
                    X_train_all,
                    Y_train_all,
                    X_holdout,
                    Y_holdout,
                    cfg_nq,
                    logger,
                    device=device,
                )

                # -------------------------
                # (G) COST (tape-based) minimal
                # -------------------------
                X_ref = X_train_all[: max(8, int(cfg_nq.cost_measure_samples))]
                cost_obj = measure_cost_from_arch(arch_mat, int(best_nq), cfg_nq, X_ref, seed=seed, device=device)
                cost = float(cost_obj.get("cost", float("nan")))

                # perf metric used in your main_ablation
                perf = 0.5 * (float(nested["auc_mean"]) + float(nested["sens_mean"]))

                rlqcv_by_nq[str(nq)] = {
                    "best_n_qubits": int(best_nq),
                    "nested_cv": nested,
                    "holdout": {"thr_star": float(thr_ho), "auc": float(auc_ho), "sens@thr*": float(sens_ho)},
                    "perf": float(perf),
                    "cost": float(cost),
                    "measured_cost": cost_obj,
                    "arch_mat_shape": list(getattr(arch_mat, "shape", [])),
                }
                pareto_points.append(
                    {"seed": int(seed), "n_qubits": int(nq), "cost": float(cost), "perf": float(perf)}
                )

                print(
                    f"[DEBUG] seed={seed} nq={nq} OK | perf={perf:.4f} holdout_auc={auc_ho:.4f} holdout_sens={sens_ho:.4f}"
                )

                # except Exception as e:
                #     seed_status = "FAIL"
                #     seed_error = {
                #         "scenario": sc_name,
                #         "seed": int(seed),
                #         "n_qubits": int(nq),
                #         "error": str(e),
                #         "traceback": traceback.format_exc(),
                #     }
                #     print(f"[ERROR] seed={seed} nq={nq} failed: {e}")
                #     break

            # best nq by perf
            best_rlqcv = None
            for nq_str, obj in rlqcv_by_nq.items():
                if best_rlqcv is None or float(obj["perf"]) > float(best_rlqcv["perf"]):
                    best_rlqcv = {"n_qubits": int(nq_str), **obj}

            scenario_runs.append(
                {
                    "scenario": sc_name,
                    "seed": int(seed),
                    "status": seed_status,
                    "error": seed_error,
                    "baselines": baselines,
                    "rlqcv": rlqcv_by_nq,
                    "best_rlqcv": best_rlqcv,
                    "pareto_points": pareto_points,
                }
            )

        # -------------------------
        # Scenario aggregation (CI + Pareto)
        # -------------------------
        best_per_seed = [r["best_rlqcv"] for r in scenario_runs if r["best_rlqcv"] is not None]
        perf_list = [float(b["perf"]) for b in best_per_seed]

        if len(perf_list) == 0:
            perf_ci = {"mean": None, "ci95": [None, None], "per_seed": []}
        else:
            if str(cfg_base.ci_method).lower() == "bootstrap":
                m, lo, hi = mean_ci_bootstrap(
                    perf_list,
                    B=int(cfg_base.bootstrap_B),
                    alpha=float(cfg_base.ci_alpha),
                    seed=0,
                )
            else:
                m, lo, hi = mean_ci_t(perf_list, alpha=float(cfg_base.ci_alpha))
            perf_ci = {"mean": float(m), "ci95": [float(lo), float(hi)], "per_seed": [float(x) for x in perf_list]}

        pf = pareto_front([p for r in scenario_runs for p in r.get("pareto_points", [])])

        scenario_result = {
            "scenario": {
                "name": sc_name,
                "tag": sc_tag,
                "semantics_flag": sc_sem,
                "notes": sc_note,
                "overrides": sc_over,
            },
            "meta": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "DEBUG",
                "device": str(device),
                "seeds": SEEDS,
                "qubits_list": QUBITS_LIST,
                "percent_search": int(cfg_base.percent_search),
                "percent_eval": int(cfg_base.percent_eval),
                "holdout_frac": float(cfg_base.holdout_frac),
                "metric": "0.5*(AUC_mean + SENS_mean) (nested CV), holdout uses SENS@thr*",
                "ci_method": str(cfg_base.ci_method),
                "ci_alpha": float(cfg_base.ci_alpha),
                "bootstrap_B": int(cfg_base.bootstrap_B),
            },
            "rlqcv_perf_ci95": perf_ci,
            "pareto_front": pf,
            "runs": scenario_runs,
        }

        out_path = root_out / sc_tag / f"DEBUG_results_{sc_tag}.json"
        out_path.write_text(json.dumps(scenario_result, indent=2), encoding="utf-8")
        print(f"[DEBUG] Saved: {out_path}")

        all_scenarios_index.append(
            {
                "scenario": {
                    "name": sc_name,
                    "tag": sc_tag,
                    "semantics_flag": sc_sem,
                    "notes": sc_note,
                },
                "results_file": str(out_path),
                "summary": {"rlqcv_perf_ci95": perf_ci, "pareto_front": pf},
            }
        )

    agg = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "DEBUG",
            "device": str(device),
            "n_scenarios": len(all_scenarios_index),
        },
        "scenarios": all_scenarios_index,
    }
    agg_path = root_out / "DEBUG_ALL_SCENARIOS_INDEX.json"
    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n[DEBUG] Finished main_debug_ablation() | Aggregate: {agg_path}")



