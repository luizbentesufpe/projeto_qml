from typing import Any
from rl_and_qml_in_clinical_images.util import Logger, RunningStd, save_circuit_image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from rl_and_qml_in_clinical_images.features import num_patches
from rl_and_qml_in_clinical_images.modeling.model import BinaryCQV_End2End, compute_pos_weight, init_head_bias_with_prevalence
from rl_and_qml_in_clinical_images.rl.agent import DDQNAgent
from rl_and_qml_in_clinical_images.rl.env import QMLEnvEnd2End, _rates_from_thr, find_threshold, sanitize_architecture, state_to_vec
from rl_and_qml_in_clinical_images.rl.losses import focal_loss_with_logits
from rl_and_qml_in_clinical_images.rl.rl_config import Config, get_thr_targets
from rl_and_qml_in_clinical_images.util import set_seeds
import math
from collections import defaultdict

def make_cfg_for_qubits(base_cfg: Config, n_qubits: int) -> Config:
    cfg = Config(**base_cfg.__dict__)
    # cfg.n_qubits = int(n_qubits)

    nq = int(n_qubits)
    cfg.n_qubits = nq
    cfg.start_qubits = int(np.clip(nq, int(cfg.min_qubits), int(cfg.max_qubits)))
    # Publication fix:
    # If patch bank is enabled, the feature bank domain is patches (P),
    # and must NOT be scaled by pixels/qubits heuristics.
    if bool(cfg.use_patch_bank):
        P = num_patches(28, 28, int(cfg.patch_size), int(cfg.patch_stride))
        cfg.feature_bank_size = int(P)
        cfg.feature_bank_min_size = int(min(cfg.feature_bank_min_size, P))
        cfg.feature_bank_schedule = tuple(int(min(int(k), P)) for k in cfg.feature_bank_schedule) if len(cfg.feature_bank_schedule) else (P,)
    else:
        bank_start = int(min(784, max(32, 2 * (2 ** cfg.n_qubits))))
        bank_min   = int(max(32, bank_start // 4))
        cfg.feature_bank_size = int(max(cfg.feature_bank_size, bank_start))
        cfg.feature_bank_min_size = int(min(cfg.feature_bank_min_size, bank_min))
        cfg.feature_bank_schedule = (
            cfg.feature_bank_size,
            int(max(cfg.feature_bank_min_size, round(0.75 * cfg.feature_bank_size))),
            int(max(cfg.feature_bank_min_size, round(0.50 * cfg.feature_bank_size))),
            cfg.feature_bank_min_size
        )
    return cfg

# ---------------------------------------------------------------------
def _youden01(sens: float, spec: float) -> float:
    y = float(np.clip(float(sens) + float(spec) - 1.0, -1.0, 1.0))  # [-1,1]
    return float(0.5 * (y + 1.0))  # [0,1]

def _bacc(sens: float, spec: float) -> float:
    return float(0.5 * (float(sens) + float(spec)))

def _sep01_from_logit_margin(logit_p95_p5: float, scale: float = 0.25) -> float:
    """Map a (non-negative) logit separability margin into [0,1].

    Uses robust spread (p95 - p5) of logits because it captures ranking/separability
    before thresholding.
    """
    m = float(logit_p95_p5)
    if not np.isfinite(m):
        return 0.0
    m = max(0.0, m)
    s = max(float(scale), 1e-6)
    return float(np.tanh(m / s))  # 0..1

def _proxy_score_search(
    auc: float,
    sens: float,
    spec: float,
    logit_margin: float | None,
    phase: str,
    cfg: Any,
) -> tuple[float, dict]:
    """Compute proxy score for RL objective.

    Key idea:
      - SEARCH must reward *separability* (ranking): AUC + logit spread
      - avoid accepting episodes that only 'hold' SPEC via threshold tricks
    """
    auc01 = float(np.clip(float(auc), 0.0, 1.0))
    sens01 = float(np.clip(float(sens), 0.0, 1.0))
    spec01 = float(np.clip(float(spec), 0.0, 1.0))
    youden01 = _youden01(sens01, spec01)

    phase_l = str(phase).lower()
    is_final = phase_l.startswith("final")

    # separability from logits (calibration set)
    # sep_scale = float(getattr(cfg, "final_sep_scale" if is_final else "search_sep_scale", 0.25))
    # sep01 = _sep01_from_logit_margin(float(logit_margin) if logit_margin is not None else float("nan"), scale=sep_scale)
    sep_scale = float(getattr(cfg, "final_sep_scale" if is_final else "search_sep_scale", 0.25))
    sep01_raw = _sep01_from_logit_margin(
        float(logit_margin) if logit_margin is not None else float("nan"),
        scale=sep_scale
    )

    # weights (phase-aware)
    w_auc = float(getattr(cfg, "final_w_auc" if is_final else "search_w_auc", 0.60 if not is_final else 0.55))
    w_y  = float(getattr(cfg, "final_w_youden" if is_final else "search_w_youden", 0.20))
    w_s  = float(getattr(cfg, "final_w_sep" if is_final else "search_w_sep", 0.20 if not is_final else 0.25))
    w_sum = max(1e-9, (w_auc + w_y + w_s))

    # base = (w_auc * auc01 + w_y * youden01 + w_s * sep01) / w_sum  # in [0,1]
    
    # --------------------------------------------------
    # CRITICAL FIX:
    # "spread" sozinho NÃO garante ranking. Então:
    # 1) sep só pode contribuir se AUC estiver acima do piso
    # 2) aplica um gate multiplicativo forte em SEARCH quando AUC≈0.5
    # --------------------------------------------------
    auc_floor = float(getattr(cfg, "final_auc_floor" if is_final else "search_auc_floor", 0.55 if not is_final else 0.60))
    tau_gate  = float(getattr(cfg, "final_auc_gate_tau" if is_final else "search_auc_gate_tau", 0.02))

    # gate ∈ (0,1): ~0 quando auc<<auc_floor, ~0.5 em auc=auc_floor, ~1 quando auc>>auc_floor
    # Isso evita "episódios falsos" com AUC≈0.5 mas logits com range ok.
    if not is_final:
        z = (auc01 - auc_floor) / max(tau_gate, 1e-6)
        auc_gate = float(1.0 / (1.0 + np.exp(-z)))
    else:
        auc_gate = 1.0

    # sep só entra se passar no gate (senão vira incentivo errado)
    sep01 = float(sep01_raw * auc_gate)

    base = (w_auc * auc01 + w_y * youden01 + w_s * sep01) / w_sum  # in [0,1]

    # --- CRITICAL: anti-noise gate in SEARCH ---
    # If AUC is too close to 0.5, treat it as near-random and penalize strongly.
    # Em SEARCH, além de "sep gated", aplicamos um gate multiplicativo final no score.
    # Isso derruba score quando AUC ~0.5, mesmo que você consiga segurar SPEC via thr.
    if not is_final:
        base = float(base * auc_gate)
 

    # optional: also penalize very low separability
    sep_floor = float(getattr(cfg, "final_sep_floor" if is_final else "search_sep_floor", 0.15))
    lam_sep_floor = float(getattr(cfg, "final_sep_floor_lam" if is_final else "search_sep_floor_lam", 0.50))
    pen_sep = 0.0
    if (not is_final) and (sep01 < sep_floor):
        pen_sep = lam_sep_floor * float(sep_floor - sep01)

    score = float(np.clip(base - pen_sep, 0.0, 1.0))
    dbg = {
        "auc01": auc01,
        "youden01": youden01,
        #"sep01": sep01,
        "sep01_raw": float(sep01_raw),
        "sep01": float(sep01),
        "base": float(base),
        "pen_sep_floor": float(pen_sep),
        "w": (w_auc, w_y, w_s),
        "auc_floor": float(auc_floor),
        "auc_gate": float(auc_gate),
        "auc_gate_tau": float(tau_gate),
        "sep_scale": float(sep_scale),
    }
    return score, dbg


def run_arch_search_end2end(X_tr, Y_tr, X_val, Y_val, cfg: Config, logger: Logger, seed: int = 0, device: torch.device | str = "cpu"):
    # SEARCH phase: relax threshold constraints
    try:
        cfg.phase = "search"
    except Exception:
        pass
    set_seeds(seed)
    DEVICE = torch.device(device)
    
    env = QMLEnvEnd2End(X_tr, Y_tr, X_val, Y_val, cfg, logger=logger, seed=seed, device=DEVICE)
    agent = DDQNAgent(cfg, n_actions=env.N_ACTIONS, device=DEVICE)

    run_score_std = RunningStd()
    best_arch = None
    best_score = -1.0
    best_nq = cfg.start_qubits
    best_proxy = None

    collapse_count = 0
    collapse_streak = 0
    collapse_streak_max = int(getattr(cfg, "collapse_streak_max", 5))

    
    ep_rewards = []
    for ep in range(cfg.episodes):

        # --------------------------------------------------
        # NÍVEL 3 — curriculum no shaping do reward
        # --------------------------------------------------
        T1 = int(getattr(cfg, "curriculum_T1", 60))
        T2 = int(getattr(cfg, "curriculum_T2", 160))
        if ep < T1:
            env.set_metric_weight(float(getattr(cfg, "curriculum_w_metric_early", 0.0)))
        elif ep < T2:
            env.set_metric_weight(float(getattr(cfg, "curriculum_w_metric_mid", 0.5)))
        else:
            env.set_metric_weight(float(getattr(cfg, "curriculum_w_metric_late", 1.0)))

    
        # decay feature bank schedule
        if cfg.feature_bank_decay_enabled and len(cfg.feature_bank_schedule) > 0:
            stage = ep // max(int(cfg.feature_bank_decay_every), 1)
            stage = int(np.clip(stage, 0, len(cfg.feature_bank_schedule) - 1))
            sched = getattr(env, "feature_bank_schedule_eff", cfg.feature_bank_schedule)
            k_now = int(sched[stage])
            k_now = int(np.clip(k_now, int(env.feature_bank_min_eff), int(env.feature_bank_size_eff)))
            env.set_current_bank_k(k_now)

        # rescore bank by saliency sometimes
        if cfg.feature_bank_update.lower() == "saliency" and ep > 0 and (ep % max(int(cfg.feature_bank_rescore_every), 1) == 0):
            arch_use = best_arch if best_arch is not None else env.state
            env.maybe_rescore_feature_bank_saliency(arch_use)

        s = env.reset()
        done = False
        last_metric = 0.0
        ep_ret = 0.0



        if not hasattr(env, "last_proxy_score"):
            env.last_proxy_score = 0.5
        
        while not done:
            s_vec = state_to_vec(s, last_metric, env.current_n_qubits, cfg)
            s_in  = torch.tensor(s_vec.reshape(1,-1), dtype=torch.float32, device=DEVICE)

            valid_mask = env.valid_action_mask()
            a = agent.select_action(s_in, valid_mask)

            s1, r, done, info = env.step(a)

            # -----------------------------
            # TERMINAL REWARD INJECTION
            # When episode ends, run terminal_evaluate ONCE here,
            # compute score and add strong terminal reward to the last step reward.
            # This makes ep_ret track score and also pushes signal into replay.
            # -----------------------------
            terminal_bonus = 0.0
            terminal_info = None
            if bool(done):
                auc_t, sens_t, spec_t, thr_star, depth_t, cnot_t, tinfo = env.terminal_evaluate()
                logger.log_to_file("rl", f">>> [ep {ep+1:03d}] TERMINAL EVAL: AUC={float(auc_t):.4f} SENS={float(sens_t):.4f} SPEC={float(spec_t):.4f} THR*={float(thr_star):.3f}")
                score = 0.0

                # Balanced accuracy exists and is used later (do NOT remove)
                bacc = _bacc(float(sens_t), float(spec_t))

                # --------------------------------------------------
                # NÍVEL 1 — proxy score (Youden J) + penalizar thr degenerado
                # --------------------------------------------------
                # auc01 = float(np.clip(float(auc_t), 0.0, 1.0))

                # youden01 = _youden01(float(sens_t), float(spec_t))
                # bacc = _bacc(float(sens_t), float(spec_t))

                # # PROXY principal: AUC + Youden01 (anti-colapso)
                # proxy_score = 0.5 * float(auc01) + 0.5 * float(youden01)  # [0,1]
                # --------------------------------------------------
                # SEARCH proxy-score MUST reward separability (ranking), not only threshold tricks.
                # We use (AUC + Youden + logit separability) and penalize AUC≈0.5 in SEARCH.
                # logit separability comes from terminal_evaluate() calib health: tinfo["collapse_dbg"]["logit_p95_p5"].
                # --------------------------------------------------
                phase_l = str(getattr(cfg, "phase", "search")).lower()
                logit_margin = None
                try:
                    cdbg = (tinfo.get("collapse_dbg", {}) if isinstance(tinfo, dict) else {})
                    if isinstance(cdbg, dict):
                        logit_margin = cdbg.get("logit_p95_p5", None)
                except Exception:
                    logit_margin = None

                proxy_score, proxy_dbg = _proxy_score_search(
                    auc=float(auc_t),
                    sens=float(sens_t),
                    spec=float(spec_t),
                    logit_margin=(float(logit_margin) if logit_margin is not None else None),
                    phase=str(phase_l),
                    cfg=cfg,
                )
                score = float(proxy_score)

                # keep debug packed (so you can inspect in logs)
                if isinstance(tinfo, dict):
                    tinfo["proxy_dbg"] = dict(proxy_dbg)
                # Optional: incorporate a tiny balanced-acc term (helps "acc-like" behavior without bias)
                # Disabled by default (weight=0).
                w_bacc = float(getattr(cfg, "terminal_w_bacc", 0.0))
                #proxy_score = float(np.clip(proxy_score + w_bacc * (float(bacc) - 0.5), 0.0, 1.0))
                proxy_score = float(np.clip(float(proxy_score) + w_bacc * (float(bacc) - 0.5), 0.0, 1.0))
                score = float(proxy_score)  # keep consistent after adjustment

                # degenerate threshold penalty
                degenerate_penalty = 0.0
                lo = float(getattr(cfg, "thr_degenerate_lo", 0.05))
                hi = float(getattr(cfg, "thr_degenerate_hi", 0.95))
                if float(thr_star) < lo or float(thr_star) > hi:
                    degenerate_penalty = -float(getattr(cfg, "lambda_thr", 0.75))
                
                
                # --------------------------------------------------
                # NEW: delta-proxy terminal reward (reduces plateau)
                # Default ON (can disable with cfg.terminal_use_delta_proxy=False)
                # --------------------------------------------------
                use_delta = bool(getattr(cfg, "terminal_use_delta_proxy", True))
                ema = float(getattr(cfg, "terminal_proxy_ema", 0.10))  # 0 disables EMA, 0.1 recommended
                prev_proxy = float(getattr(env, "last_proxy_score", 0.5))
                if ema > 0.0:
                    baseline = prev_proxy
                    new_base = float((1.0 - ema) * prev_proxy + ema * float(proxy_score))
                    env.last_proxy_score = float(new_base)
                else:
                    baseline = prev_proxy
                    env.last_proxy_score = float(proxy_score)
                delta_proxy = float(proxy_score - baseline) if use_delta else float(proxy_score - 0.5)



                try:
                    collapse_pen = float(tinfo.get("collapse_pen", 0.0)) if isinstance(tinfo, dict) else 0.0
                except Exception:
                    collapse_pen = 0.0
                if collapse_pen <= 0.0:
                    spec_floor = float(getattr(cfg, "terminal_spec_floor", 0.10))
                    if float(spec_t) < spec_floor:
                        collapse_pen = float(getattr(cfg, "terminal_collapse_penalty", 0.05)) * float(spec_floor - float(spec_t)) / max(spec_floor, 1e-6)

                if float(spec_t) < float(getattr(cfg, "terminal_spec_floor", 0.10)):
                    collapse_count += 1
                    collapse_streak = min(collapse_streak + 1, collapse_streak_max)
                else:
                    collapse_streak = max(collapse_streak - 1, 0)

                # --------------------------------------------------
                # Mantém o "K" (fixo ou adaptativo), mas baseado em PROXY (não SENS pura)
                # --------------------------------------------------
                base_K = float(getattr(cfg, "terminal_reward_K", 10.0))
                adaptive = bool(getattr(cfg, "terminal_adaptive_K", True))

                if adaptive:
                    # Warmup: do NOT adapt K when std is near-zero (early episodes)
                    d_center = float(proxy_score - 0.5)
                    run_score_std.update(d_center)
                    warm = int(getattr(cfg, "terminal_K_warmup", 20))
                    if int(getattr(run_score_std, "n", 0)) < warm:
                        K = float(base_K)
                    else:
                        target_std = float(getattr(cfg, "terminal_target_std", 0.25))
                        scale = float(target_std / (run_score_std.std + 1e-6))
                        # hard cap on scale to avoid exploding K
                        scale = float(np.clip(scale, 0.25, 2.0))
                        K = float(base_K * scale)
                    K_min = float(getattr(cfg, "terminal_K_min", 0.5))
                    K_max = float(getattr(cfg, "terminal_K_max", 10.0))
                    K = float(np.clip(K, K_min, K_max))
                else:
                    K = float(base_K)


                #terminal_bonus = float(K * (float(proxy_score) - 0.5) + float(degenerate_penalty) - float(collapse_pen))
                terminal_bonus = float(K * float(delta_proxy) + float(degenerate_penalty) - float(collapse_pen))
                
                # --------------------------------------------------
                # NÍVEL 2 — penalizar instabilidade do proxy (2 seeds), se env fornecer
                # Espera-se tinfo["std_proxy"] (ou 0.0 se ausente).
                # --------------------------------------------------
                try:
                    std_proxy = float(tinfo.get("std_proxy", 0.0)) if isinstance(tinfo, dict) else 0.0
                except Exception:
                    std_proxy = 0.0
                terminal_bonus -= float(getattr(cfg, "lambda_var", 0.50)) * float(std_proxy)


                try:
                    floor = float(getattr(cfg, "terminal_spec_floor", 0.10))
                    if float(spec_t) < floor:
                        bump = float(getattr(cfg, "eps_boost_on_collapse", 0.05))
                        agent.eps = float(min(1.0, float(agent.eps) + abs(bump)))
                    # If collapse keeps happening, slow down eps decay by effectively raising eps_end for a while
                    if collapse_streak >= int(getattr(cfg, "collapse_streak_trigger", 3)):
                        agent.eps = float(min(1.0, max(agent.eps, float(getattr(cfg, "eps_min_when_collapsing", 0.30)))))
                except Exception:
                    pass

                # clip to avoid dominance
                clip_val = float(getattr(cfg, "terminal_clip", 4.0))
                terminal_bonus = float(np.clip(terminal_bonus, -clip_val, clip_val))


                # salva para logging de melhor arch aqui também
                effective_score = float(np.clip(float(proxy_score) - float(collapse_pen), 0.0, 1.0))
                env.last_ep_score = float(effective_score)
                terminal_info = {
                    "terminal_auc": float(auc_t),
                    "terminal_sens": float(sens_t),
                    "terminal_score": float(score),
                    "terminal_spec": float(spec_t),
                    "thr_star": float(thr_star),
                    "proxy_score": float(proxy_score),
                    "delta_proxy": float(delta_proxy),
                    "degenerate_penalty": float(degenerate_penalty),
                    "effective_score": float(effective_score),
                    "balanced_acc": float(bacc),
                    "collapse_pen": float(collapse_pen),
                    "std_proxy": float(std_proxy),
                    "terminal_bonus": float(terminal_bonus),
                    "terminal_K": float(K),
                    "terminal_depth": depth_t,
                    "terminal_cnot": cnot_t,
                 }

                # melhor arquitetura por score terminal (não pelo last_auc stale)
                if float(effective_score) > float(best_score):
                    best_score = float(effective_score)
                    best_arch  = env.state.copy()
                    best_nq = env.current_n_qubits
                    best_proxy = float(effective_score)

                # injeta o bônus no último reward do episódio
                r = float(r) + float(terminal_bonus)
                info = dict(info)
                info["terminal"] = terminal_info

                # >>> IMPORTANT: account terminal bonus in per-episode breakdown
                try:
                    env._ep_reward_sums["terminal"] += float(terminal_bonus)
                except Exception:
                    pass

            ep_ret += float(r)

            
            try:
                auc_v = float(info.get("auc_val", 0.5))
                sens_v = float(info.get("sens_val", 0.0))
                # prefer env.last_spec if you track it there; else allow info["spec_val"]
                spec_v = float(getattr(env, "last_spec", info.get("spec_val", 0.0)))
                last_metric = 0.5 * float(auc_v) + 0.5 * float(np.clip(sens_v + spec_v - 1.0, -1.0, 1.0))
            except Exception:
                last_metric = 0.0

            s1_vec = state_to_vec(s1, last_metric, env.current_n_qubits, cfg)

            agent.nbuf.push(s_vec, a, r, s1_vec, done)
            if agent.nbuf.is_ready():
                ns = agent.nbuf.pop_nstep(cfg.gamma())
                agent.replay.push(*ns)
                agent.update(cfg.gamma())

            s = s1
            ep_rewards.append(r)

        # -------------------------------------------
        # Per-episode reward breakdown logging
        # -------------------------------------------
        sums = env.flush_episode_reward_sums()
        # terminal score já está em env.last_ep_score (set quando done=True)
        score_print = float(env.last_ep_score) if (env.last_ep_score is not None) else float("nan")
        logger.log_to_file(
            "rl",
            "[ep %03d] score=%.4f best=%.4f nq=%d ep_ret=%.3f | "
            "metric=%.3f depth=%.3f cnot=%.3f qubit=%.3f rot=%.3f dead=%.3f budget=%.3f repeat=%.3f terminal=%.3f"
            % (
                ep+1, score_print, best_score, int(best_nq), float(ep_ret),
                float(sums.get("metric", 0.0)),
                float(sums.get("depth", 0.0)),
                float(sums.get("cnot", 0.0)),
                float(sums.get("qubit", 0.0)),
                float(sums.get("rot", 0.0)),
                float(sums.get("dead", 0.0)),
                float(sums.get("budget", 0.0)),
                float(sums.get("repeat", 0.0)),
                float(sums.get("terminal", 0.0)),
            )
        )

        # extra health signal: how often we collapsed spec under floor
        if (ep + 1) % int(max(1, getattr(cfg, "collapse_log_every", 25))) == 0:
            logger.log_to_file("rl", f"[health] collapse_count={int(collapse_count)} / {int(ep+1)} (spec < terminal_spec_floor)")

        
        if env.last_ep_score is not None:
            logger.log_to_file("rl_terminal", f"[ep {ep+1:03d}] terminal_score={env.last_ep_score:.4f}")

    # # unfreeze all
    # for p in model.parameters():
    #     p.requires_grad_(True)

    # if model.head.bias is not None:
    #     model.head.bias.requires_grad_(True)


    # arch_mat = torch.tensor(best_arch, dtype=torch.int64)
    if best_arch is None:
        best_arch = env.state.copy()
        best_nq = env.current_n_qubits
    arch_mat = torch.tensor(best_arch, dtype=torch.int64)

    return arch_mat, int(best_nq), (None if best_proxy is None else float(best_proxy))

@torch.no_grad()
def eval_metrics_final(model, X, Y, thr: float, device: torch.device | str):
    DEVICE = torch.device(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=DEVICE)
    logits = model(X_t)
    probs  = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    probs  = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

    yt = (Y_t.detach().cpu().numpy().reshape(-1) > 0.5).astype(int)
    n_pos = int(yt.sum()); n_neg = int(len(yt) - n_pos)
    if (n_pos == 0) or (n_neg == 0):
        auc = 0.5
    else:
        try:
            auc = float(roc_auc_score(yt, probs))
            if not np.isfinite(auc): auc = 0.5
        except Exception:
            auc = 0.5
    thr = float(thr)
    yp = (probs >= thr).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    sens = float(tp) / float(max(n_pos, 1))
    return float(auc), float(sens)


def train_final_model_end2end(arch_mat: torch.Tensor, n_qubits: int, X_trval, Y_trval, X_te, Y_te, 
                              cfg: Config, logger: Logger, device: torch.device | str):
    # FINAL phase: strict/clinical threshold constraints
    try:
        cfg.phase = "final"
    except Exception:
        pass
    
    sens_tgt, spec_min, fpr_max = get_thr_targets(cfg)
    DEVICE = torch.device(device)
    arch_mat = sanitize_architecture(arch_mat, int(n_qubits)).to(DEVICE)
    
    model = BinaryCQV_End2End(
        arch_mat=arch_mat,
        n_qubits=n_qubits,
        enc_lambda=float(cfg.enc_lambda),
        diff_method="adjoint",
        input_dim=int(np.asarray(X_trval).shape[1]),
        enc_affine_mode=str(getattr(cfg, "enc_affine_mode", "per_feature")),
        enc_alpha_init=float(getattr(cfg, "enc_alpha_init", 1.0)),
        enc_beta_init=float(getattr(cfg, "enc_beta_init", 0.0)),
        enc_beta_max=float(getattr(cfg, "enc_beta_max", 1.0)),
        use_batched_qnode=bool(getattr(cfg, "use_batched_qnode", True)),
    ).to(DEVICE)

    model.set_logit_scale_trainable(trainable=True)
    
    X_trval_t = torch.tensor(X_trval, dtype=torch.float32, device=DEVICE)
    Y_trval_t = torch.tensor(Y_trval, dtype=torch.float32, device=DEVICE)
    Y_trval_np = (Y_trval_t.detach().cpu().numpy() > 0.5).astype(np.int32)

    init_head_bias_with_prevalence(model, Y_trval_np)
    pos_weight = compute_pos_weight(Y_trval_t, DEVICE)


    # This function must not reference env/self. It trains from provided arrays only.
    # ------------------------------------------------------------------
    # NO-LEAKAGE threshold calibration:
    # Train on FIT split; calibrate thr* on CAL split; evaluate on X_te.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(getattr(cfg, "thr_calib_seed", 12345)))
    ybin = (np.asarray(Y_trval).reshape(-1) > 0.5).astype(np.int32)
    idx = np.arange(len(ybin))

    # stratified split
    idx_pos = idx[ybin == 1]
    idx_neg = idx[ybin == 0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    calib_frac = float(getattr(cfg, "thr_calib_frac", 0.20))
    n_cal_pos = int(max(1, round(calib_frac * len(idx_pos))))
    n_cal_neg = int(max(1, round(calib_frac * len(idx_neg))))

    cal_idx = np.concatenate([idx_pos[:n_cal_pos], idx_neg[:n_cal_neg]])
    fit_idx = np.setdiff1d(idx, cal_idx)

    X_fit, Y_fit = np.asarray(X_trval)[fit_idx], np.asarray(Y_trval)[fit_idx]
    X_cal, Y_cal = np.asarray(X_trval)[cal_idx], np.asarray(Y_trval)[cal_idx]

    X_fit_t = torch.tensor(X_fit, dtype=torch.float32, device=DEVICE)
    Y_fit_t = torch.tensor(Y_fit, dtype=torch.float32, device=DEVICE)

    dl = DataLoader(
        TensorDataset(X_fit_t, Y_fit_t),
        batch_size=int(cfg.batch_size), shuffle=True, num_workers=0, drop_last=False
    )


    # dl = DataLoader(
    #     TensorDataset(X_trval_t, Y_trval_t),
    #     batch_size=int(cfg.batch_size), shuffle=True, num_workers=0, drop_last=False
    # )

    # warmup head only (few batches, publication-stable)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    
    model.logit_scale.requires_grad_(False)


    # >>> CRITICAL FIX: freeze bias during head-only
    if model.head.bias is not None:
        model.head.bias.requires_grad_(False)


    opt_head = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    crit_warm = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()

    n_head_batches = int(max(1, cfg.inner_train_batches_head))

    for bi, (xb, yb) in enumerate(dl):
        logits = model(xb)
        loss = crit_warm(logits, yb)
        opt_head.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
        opt_head.step()
        if (bi + 1) >= n_head_batches:
            break
 
    # 2) unfreeze all and do limited VQC updates on small loader
    for p in model.parameters():
        p.requires_grad_(True)

    if model.head.bias is not None:
        model.head.bias.requires_grad_(True)
        
    opt = torch.optim.Adam(model.parameters(), lr=cfg.final_lr_vqc)

    # final training
    for ep in range(cfg.final_epochs):
        model.train()
        for xb, yb in dl:
            logits = model(xb)
            if bool(cfg.use_focal):
                loss = focal_loss_with_logits(
                    logits, yb,
                    alpha=float(cfg.focal_alpha),
                    gamma=float(cfg.focal_gamma),
                    reduction="mean",
                    pos_weight=pos_weight
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), yb.view(-1), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if (not cfg.phase.startswith("final")) and (model.head.bias is not None):
                bmax = float(getattr(cfg, "search_head_bias_clamp", 2.0))
                with torch.no_grad():
                    model.head.bias.data.clamp_(-bmax, bmax)

        if ep % 3 == 0:
            # calibrate thr* on train (trval) each check
            model.eval()
            Xtr_t = torch.tensor(X_fit, dtype=torch.float32, device=DEVICE)
            Ytr_t = torch.tensor(Y_fit, dtype=torch.float32, device=DEVICE)

            
            logits_tr = model(Xtr_t).detach().cpu().numpy().reshape(-1)
            probs_tr = (1.0 / (1.0 + np.exp(-logits_tr))).astype(np.float32).reshape(-1)



            ytr = (Ytr_t.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
            thr_star = find_threshold(
                ytr, probs_tr,
                mode=str(cfg.thr_mode),
                sens_target=float(sens_tgt),
                grid_size=cfg.grid_size,
                fpr_max=float(fpr_max),
                spec_min=float(spec_min),
                lam_spec=float(getattr(cfg, "thr_lam_spec", 2.0)),
                lam_fpr=float(getattr(cfg, "thr_lam_fpr", 2.0)),
                lam_sens=float(getattr(cfg, "thr_lam_sens", 1.0)),
                logits=logits_tr,
                logger=logger,
                return_info=False
            )
            sens_r, spec_r, fpr_r = _rates_from_thr(ytr, probs_tr, float(thr_star))
            #auc_m, sens_m = eval_metrics_final(model, X_trval, Y_trval, thr=thr_star, device=DEVICE)
            auc_m, sens_m = eval_metrics_final(model, X_fit, Y_fit, thr=thr_star, device=DEVICE)

            logger.log_to_file("final_train", f"[ep={ep:02d}] thr*={thr_star:.3f} AUC_trval={auc_m:.4f} SENS={sens_r:.4f} SPEC={spec_r:.4f} FPR={fpr_r:.4f}")
 
    # calibrate thr* on train (or fold-tr) BEFORE measuring val/test sens
    # model.eval()
    # Xtr_t = torch.tensor(X_trval, dtype=torch.float32, device=DEVICE)
    # Ytr_t = torch.tensor(Y_trval, dtype=torch.float32, device=DEVICE)
    # logits_tr = model(Xtr_t).detach().cpu().numpy().reshape(-1)
    # probs_tr = torch.sigmoid(torch.tensor(logits_tr)).numpy().reshape(-1)
    # ytr = (Ytr_t.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)
    # thr_star = find_threshold(
    #     ytr, probs_tr,
    #     mode=str(cfg.thr_mode),
    #     sens_target=float(sens_tgt),
    #     grid_size=cfg.grid_size,
    #     fpr_max=float(fpr_max),
    #     spec_min=float(spec_min),
    #     lam_spec=float(getattr(cfg, "thr_lam_spec", 2.0)),
    #     lam_fpr=float(getattr(cfg, "thr_lam_fpr", 2.0)),
    #     lam_sens=float(getattr(cfg, "thr_lam_sens", 1.0)),
    #     logits=logits_tr,
    #     logger=logger,
    #     return_info=False
    # )

    # auc_te, sens_te = eval_metrics_final(model, X_te, Y_te, thr=thr_star, device=DEVICE)

    # --- NEW: split X_trval into fit/calib for thr* (no leakage) ---
    # rng = np.random.default_rng(int(getattr(cfg, "thr_calib_seed", 12345)))
    # ybin = (np.asarray(Y_trval).reshape(-1) > 0.5).astype(np.int32)
    # idx = np.arange(len(ybin))

    # # stratified split
    # idx_pos = idx[ybin == 1]
    # idx_neg = idx[ybin == 0]
    # rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    # calib_frac = float(getattr(cfg, "thr_calib_frac", 0.20))
    # n_cal_pos = int(max(1, round(calib_frac * len(idx_pos))))
    # n_cal_neg = int(max(1, round(calib_frac * len(idx_neg))))

    # cal_idx = np.concatenate([idx_pos[:n_cal_pos], idx_neg[:n_cal_neg]])
    # fit_idx = np.setdiff1d(idx, cal_idx)

    # X_fit, Y_fit = np.asarray(X_trval)[fit_idx], np.asarray(Y_trval)[fit_idx]
    # X_cal, Y_cal = np.asarray(X_trval)[cal_idx], np.asarray(Y_trval)[cal_idx]

    # # train model on X_fit/Y_fit (use your same dl but built from X_fit/Y_fit)
    # # (minimal change: rebuild dl right before training)
    # X_fit_t = torch.tensor(X_fit, dtype=torch.float32, device=DEVICE)
    # Y_fit_t = torch.tensor(Y_fit, dtype=torch.float32, device=DEVICE)

    # dl = DataLoader(
    #     TensorDataset(X_fit_t, Y_fit_t),
    #     batch_size=int(cfg.batch_size), shuffle=True, num_workers=0, drop_last=False
    # )

    # # (keep your training loop the same)

    # --- calibrate thr* on X_cal/Y_cal ---
    model.eval()
    Xcal_t = torch.tensor(X_cal, dtype=torch.float32, device=DEVICE)
    Ycal_t = torch.tensor(Y_cal, dtype=torch.float32, device=DEVICE)
    logits_cal = model(Xcal_t).detach().cpu().numpy().reshape(-1)
    #probs_cal  = torch.sigmoid(torch.tensor(logits_cal)).numpy().reshape(-1)
    probs_cal  = (1.0 / (1.0 + np.exp(-logits_cal))).astype(np.float32).reshape(-1)
    ycal = (Ycal_t.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.int32)

    thr_star = find_threshold(
        ycal, probs_cal,
        mode=str(cfg.thr_mode),
        sens_target=float(sens_tgt),
        grid_size=cfg.grid_size,
        fpr_max=float(fpr_max),
        spec_min=float(spec_min),
        lam_spec=float(getattr(cfg, "thr_lam_spec", 2.0)),
        lam_fpr=float(getattr(cfg, "thr_lam_fpr", 2.0)),
        lam_sens=float(getattr(cfg, "thr_lam_sens", 1.0)),
        logits=logits_cal,
        logger=logger,
        return_info=False
    )

    # --- final evaluation on X_te/Y_te ---
    auc_te, sens_te = eval_metrics_final(model, X_te, Y_te, thr=thr_star, device=DEVICE)
    logger.log_to_file("final_test", f"[TEST] thr*={thr_star:.3f} AUC={auc_te:.4f} SENS@thr*={sens_te:.4f}")
    # >>> no final da função, antes do return (pós-treino)
    try:
        safe_prefix = "final_circuit"
        out_img = (logger.log_dir / f"circuit_{safe_prefix}_nq{n_qubits}_posttrain.png")
        save_circuit_image(
            model,
            x_ref=np.asarray(X_trval)[0],
            out_path=str(out_img),
            title=f"Final circuit (nq={n_qubits}) - posttrain"
        )
        logger.log_to_file("circuit", f"[SAVED] {out_img}")
    except Exception as e:
        logger.log_to_file("circuit", f"[WARN] could not save posttrain circuit: {e}")

    return auc_te, sens_te, float(thr_star)

def kfold_eval_fixed_arch(arch_mat, X, Y, cfg: Config, n_qubits: int, logger: Logger, n_splits=5, seed=0,  device: torch.device | str = "cpu"):
    DEVICE = torch.device(device)
    y = Y.reshape(-1).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    aucs, sens_list = [], []
    thrs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xva, Yva = X[va_idx], Y[va_idx]
        arch_mat = sanitize_architecture(arch_mat, int(n_qubits))
        auc, sens, thr = train_final_model_end2end(arch_mat, n_qubits, Xtr, Ytr, Xva, Yva, cfg, logger, device=DEVICE)
        aucs.append(auc); sens_list.append(sens); thrs.append(thr)
        logger.log_to_file("kfold", f"[fold={fold}] thr*={thr:.3f} AUC={auc:.4f} SENS@thr*={sens:.4f}")

    return {
        "aucs": aucs,
        "sens": sens_list,
        "thr_star": thrs,
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "sens_mean": float(np.mean(sens_list)), "sens_std": float(np.std(sens_list))
    }


def split_holdout(X, Y, frac=0.20, seed=0):
    """
    Publication-grade split:
    Hold out a final test set never used by RL search or threshold calibration.
    """
    rng = np.random.default_rng(int(seed))
    y = Y.reshape(-1).astype(int)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    n = len(y)
    n_te = int(max(2, round(frac * n)))
    n_te_pos = int(min(len(idx_pos), n_te // 2))
    n_te_neg = int(min(len(idx_neg), n_te - n_te_pos))
    te_idx = np.concatenate([idx_pos[:n_te_pos], idx_neg[:n_te_neg]])
    rng.shuffle(te_idx)
    tr_idx = np.setdiff1d(np.arange(n), te_idx)
    return tr_idx, te_idx


def nested_cv_eval_fixed_arch(arch_mat, X, Y, cfg: Config, n_qubits: int, logger: Logger, seed=0, device: torch.device | str = "cpu"):
    # FINAL phase: strict/clinical threshold constraints
    try:
        cfg.phase = "final"
    except Exception:
        pass
    """
    Publication protocol:
    Outer CV evaluates generalization on TRAIN ONLY (holdout untouched).
    Inner training calibrates thr* using fold-train only.
    """
    DEVICE = torch.device(device)
    arch_mat = sanitize_architecture(arch_mat, int(n_qubits))
    y = Y.reshape(-1).astype(int)
    outer = StratifiedKFold(n_splits=int(cfg.nested_cv_splits_outer), shuffle=True, random_state=int(seed))
    aucs, sens, thrs = [], [], []
    for fo, (tr_idx, va_idx) in enumerate(outer.split(X, y), 1):
        Xtr, Ytr = X[tr_idx], Y[tr_idx]
        Xva, Yva = X[va_idx], Y[va_idx]
        # Train on Xtr, evaluate on Xva, thr* calibrated only on Xtr inside train_final_model_end2end
        a, s, t = train_final_model_end2end(arch_mat, n_qubits, Xtr, Ytr, Xva, Yva, cfg, logger, device=DEVICE)
        aucs.append(a); sens.append(s); thrs.append(t)
        logger.log_to_file("nested", f"[outer={fo}] thr*={t:.3f} AUC={a:.4f} SENS@thr*={s:.4f}")
    return {
        "aucs": aucs, "sens": sens, "thr_star": thrs,
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "sens_mean": float(np.mean(sens)), "sens_std": float(np.std(sens))
    }
    