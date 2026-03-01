from dataclasses import dataclass
from typing import Tuple
import numpy as np
import numpy as np

from rl_and_qml_in_clinical_images.features import num_patches

@dataclass
class Config:
    # Search space / state
    n_qubits: int = 4
    L_max: int = 20
    allow_nop: bool = True
    calib_frac: float = 0.20
    val_frac_search: float = 0.40  
    # Protocol (publication)
    # - Use reduced PERCENT for RL architecture search
    # - Use full data (100%) only for final evaluation / reporting
    percent_search: int = 60 #25
    percent_eval: int = 100
    holdout_frac: float = 0.20     # final untouched test holdout
    nested_cv_splits_outer: int = 5
    nested_cv_splits_inner: int = 3
    ci_method: str = "bootstrap"           # "t" | "bootstrap"
    bootstrap_B: int = 2000
    ci_alpha: float = 0.05

    # novos knobs
    use_patch_bank: bool = False
    patch_bank_compact_features: bool = False
    search_grid_size: int = 51         # COST: coarse grid during RL search
    search_calib_cap: int = 256        # COST: small calib cap during search (was 1024)
    final_calib_cap: int = 1024        # full cap for final evaluation

    grid_size: int = 201   # threshold calibration grid size
    # feature bank
    feature_bank_size: int = 9
    feature_bank_update: str = "none"   # "saliency" | "chi2" | "random" | "none"
    feature_bank_rescore_every: int = 25

    std_proxy_threshold = 0.05 
    
    # SEARCH AUC floor collapse gate
    search_auc_floor = 0.50
    reliable_score_min  = 0.75 

    #search_auc_floor_collapse = 0.515
    search_auc_gate_tau = 0.03
    search_auc_floor_streak = 3
    collapse_saturation_abslogit_p95_thr: float = 12.0
    collapse_prob_std_saturation: float = 1e-6

    feature_bank_decay_enabled: bool = True
    feature_bank_schedule: Tuple[int, ...] = (128, 96, 64, 32)
    feature_bank_decay_every: int = 50 #10
    feature_bank_min_size: int = 3

    # patch bank
    use_patch_bank: bool = False
    patch_size: int = 4
    patch_stride: int = 4




    eps_decay_mult: float = 1.0

    # NEW: exploration boost if terminal spec collapses repeatedly
    eps_boost_on_collapse: float = 0.05
    eps_min_when_collapsing: float = 0.30
    collapse_streak_trigger: int = 3
    collapse_streak_max: int = 5

    # NEW: stronger default repetition penalty (env-side)
    repeat_penalty: float = 0.01
    # FIX-4: inter-episode architecture repeat penalty (terminal-level, not per-step).
    # Per-step repeat_penalty fires on repeated ACTIONS within an episode; it cannot
    # detect repeated ARCHITECTURES across episodes because recent_actions is cleared
    # on reset(). This was the root cause of repeat=0.000 in eps 2-5 of the logs.
    repeat_arch_penalty: float = 0.20        # NEW
    repeat_arch_window: int = 50             # NEW — max hashes kept (bounded memory)

    # NEW: delta-proxy terminal reward mode (runner-side)
    terminal_use_delta_proxy: bool = True
    terminal_proxy_ema: float = 0.10


    # RL
    episodes: int = 400 #250

    min_steps_before_nop: int = 5 

    n_steps: int = 15
    target_sync_steps: int = 512
    replay_capacity: int = 16384
    batch_size: int = 32
    eps_start: float = 1.0
    eps_end: float = 0.10
    eps_decay_steps: int = 20000 #8000inner_train_batches_head

    search_lr_vqc: float = 0.015 
    # VQC inner training per env-step
    inner_epochs_classif: int = 8 #3
    lr_vqc: float = 0.01

    enc_affine_mode: str = "per_feature"
    enc_alpha_init: float = 1.0
    enc_beta_init: float = 0.0
    enc_beta_max: float = 1.0
    # -------------------------
    # NEW: head optimizer knobs (Priority #1)
    # -------------------------
    lr_head: float = 3e-2#6e-3 #3e-3
    wd_head: float = 0.0 #1e-4
    wd_vqc: float = 1e-4

    # Gradient clipping (Priority #1: allow head to move)
    clip_head: float = 5.0
    clip_body: float = 2.0
    clip_all: float = 2.0 

    search_lr_head: float = 5e-2 

    normalize_inputs: bool = True
    norm_per_feature: bool = True     # per-feature mean/std (recommended for patch bank)
    norm_eps: float = 1e-6
    norm_clip: float = 0.0            # 0 disables; try 3.0 if needed
    norm_tanh: bool = False           # optional extra bounding


    lr_enc: float = 2e-3              # encoding affine params (alpha/beta)
    lr_theta: float = 5e-3            # VQC rotational params (theta) (default uses lr_vqc value style)
    wd_enc: float = 0.0               # encoding WD (default 0 tends to help search)
    wd_theta: float = 0.0             # theta WD (default 0 tends to help search)

    # Phase-aware defaults (SEARCH: less regularization to avoid collapse)
    search_wd_enc: float = 0.0
    search_wd_theta: float = 0.0
    final_wd_enc: float = 0.0
    final_wd_theta: float = 0.0
    
    search_head_only: bool = False
    search_logit_scale_trainable: bool = True 

    search_logit_scale_init:    float = 3.0   # novo parâmetro
    search_lr_logit_scale:      float = 0.001  # novo parâmetro (15x menor que lr_vqc)
    collapse_logit_margin_min = 0.05
    collapse_prob_std_weak    = 0.01
    collapse_auc_eps          = 0.01
    collapse_fail_streak_k    = 2

    terminal_weight          = 2.0
    terminal_depth_penalty   = 0.3
    terminal_cnot_penalty    = 0.3
    
    depth_budget             = 10.0
    # -------------------------
    # NEW: anti-collapse retry (Priority #1)
    # -------------------------
    collapse_retry: bool = True
    collapse_range_min: float = 0.08
    collapse_std_min: float = 0.02
    collapse_max_retry: int = 3
    collapse_lr_scale: float = 2.0
    collapse_boost_epochs: int = 5
    collapse_boost_batches_mult: int = 3
    collapse_boost_head_epochs: int = 1


    # final training
    final_epochs: int = 16
    final_lr_vqc: float = 0.03

    # encoding
    enc_lambda: float = float(np.pi)

    # threshold calibration
    thr_mode: str = "soft"   # "f2" | "sens"
    sens_target: float = 0.85
    thr_fpr_max: float = 0.10   # constraint for @fpr modes (publication-friendly)
    thr_spec_min: float = 0.80  # constraint for @spec modes
    # focal loss
    use_focal: bool = True
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0
    # head clamp/clip
    head_weight_abs_max = 2.0
    head_bias_abs_max   = 2.0
    head_weight_norm_max = 1.0
    head_bias_norm_max   = 2.0
    log_head_clamp = True

    # logit scale
    logit_scale_mode = "target_std"      # ou "fixed" ou "target_p95abs"
    logit_scale_fixed = 10.0
    logit_scale_target_std = 0.65
    logit_scale_target_p95abs = 10.0
    logit_scale_min = 1.0
    logit_scale_max = 30.0

    # thresholding
    thr_grid_n = 201
    thr_soft_viable_extra_pen = 0.0

    # reward weights
    alpha_auc: float = 1.0 #0.6
    beta_sens: float = 0.8 #0.6
    depth_penalty: float = 0.01
    cnot_penalty: float = 0.012
    rot_penalty: float = 0.002

    # variable qubits
    min_qubits: int = 4
    max_qubits: int = 10
    start_qubits: int = 4
    qubit_change_cooldown: int = 1
    qubit_penalty: float = 0.015
    
    search_logit_scale_max = 8.0
    search_logit_scale_min = 2.0 #3.0 se não tiver bom
    search_logit_scale_target = 2.0   # melhor começar menor (2 ou 3), não 8
    search_logit_scale_method = "p95"

    search_inner_train_batches_vqc_override: int = -1 
    # budgets
    ENC_budget: int = 9
    ROT_budget: int = 17
    CNOT_budget: int = 2
    hard_block_budget: bool = True
    budget_penalty: float = 0.25
 
    calibrate_thr_on_train: bool = True    # thr* computed on train-subset, applied on val
    inner_train_subset_size: int = 2048
    inner_train_batches_head: int = 16#3
    inner_train_batches_vqc: int = 32  #16 #2 #3
    inner_eval_batch_cap: int = 512 #1024#999999 #356 #128 #256 
    # Compute depth/cnot by tape less frequently (expensive)
    depth_check_every: int = 5             # compute tape-based depth every N steps
    depth_use_proxy_when_skip: bool = True # proxy when skipping expensive tape
    proxy_depth_per_op: float = 1.0        # simple proxy factor
    proxy_cnot_per_cnot: int = 1           # proxy factor


    calib_prev_override: float | None = None
    recenter_after_warmup: bool = False
    terminal_target_std: float = 0.20 
    terminal_clip: float = 4.0

    # -------------------------
    # Terminal score (patch)
    # -------------------------
    terminal_w_bacc: float = 0.0           # opcional: +acc-like sem enviesar (default OFF)
    terminal_spec_floor: float = 0.10      # se spec cair abaixo disso, conta como colapso
    terminal_collapse_penalty: float = 0.05# penaliza colapso (específico para evitar "all-positive")
    collapse_log_every: int = 25           # log health a cada N episódios

    terminal_reward_K: float = 5.0        # was 10.0 — base K halved for agg=min
    terminal_K_start: float = 1.0         # K at episode 0
    terminal_K_end: float = 5.0           # K at episode terminal_curriculum_K_T
    terminal_curriculum_K_T: int = 120    # ramp duration in episodes
    terminal_K_max: float = 10.0          # hard cap (unchanged)
    # FIX-A: absolute-level bonus knobs
    terminal_agg_floor: float = 0.30      # proxy_score below this → abs_signal = 0
    terminal_K_abs: float = 0.5           # weight of abs_signal term

    
    # -------------------------------------------------
    # NÍVEL 1 — correções obrigatórias
    # -------------------------------------------------
    # Penalidade por thr* degenerado (evita thr*=0/1)
    lambda_thr: float = 0.75          # ex: 0.5 ~ 1.0
    thr_degenerate_lo: float = 0.05
    thr_degenerate_hi: float = 0.95

    # Proxy score: "youden" recomendado
    proxy_metric: str = "youden"      # "youden" | "f2" (f2 exigiria probs)
    thr_policy: str = "youden"          # {"f2","sens","youden"} #

    
    # Depth normalization (p95 online)
    depth_ref_min: float = 8.0        # fallback inicial
    depth_ref_warmup: int = 10        # n episódios antes de confiar no p95

    # -------------------------------------------------
    # NÍVEL 2 — reduzir gap RL -> train
    # -------------------------------------------------
    proxy_two_seeds: bool = True
    proxy_n_seeds: int = 3 
    proxy_seed_delta: int = 1337
    #lambda_var: float = 0.50          # penaliza instabilidade entre seeds
    proxy_aggregation: str = "mean"       # "mean" | "min" | "quantile20"
    proxy_quantile: float = 0.20         # usado se aggregation="quantile20"
    # -------------------------------------------------
    # NÍVEL 3 — curriculum no reward
    # -------------------------------------------------
    curriculum_T1: int = 40 #60
    curriculum_T2: int = 100 #160
    curriculum_w_metric_early: float = 0.4 #0.1
    curriculum_w_metric_mid: float = 0.5
    curriculum_w_metric_late: float = 1.0

    # FIX-3: calib_seed_delta — large offset ensures calib cap never overlaps proxy seeds.
    # val_bce < calib_bce was systematic in all 5 spike episodes (ep047,243,214,158,170).
    # max proxy seed offset = (n_seeds-1)*proxy_seed_delta = 2*1337 = 2674 << 99991.
    calib_seed_delta: int = 99991            # NEW — was implicit 0
    lambda_var: float = 0.50                 # penaliza instabilidade entre seeds
    # FIX-5: thr* stability penalty.
    # 43.9% of evals in logs had prob_std < 0.02. Unstable thr* across seeds signals
    # fragile separation → penalise it to align RL objective with clinical deployability.
    lambda_thr_std: float = 0.60             # NEW
    thr_std_threshold: float = 0.05          # NEW — std above this is penalised
    log_thr_stability: bool = True           # NEW — write thr_stability log channel

    # logging de correlação RL vs final
    log_proxy_final_pairs: bool = True
    
    patch_bank_compact_features: bool = False
    cost_measure_samples: int = 16

    phase: str = "search"  # "search" | "final"

    # During SEARCH (RL proxy): relaxed constraints (avoid viable=0)
    search_sens_target: float = 0.42
    search_thr_spec_min: float = 0.65
    search_thr_fpr_max: float = 0.30

    # During FINAL (nested CV / holdout): strict/clinical constraints
    final_sens_target: float = 0.85
    final_thr_spec_min: float = 0.80
    final_thr_fpr_max: float = 0.10


    def gamma(self) -> float:
        return 0.99
    
    
    # Explicit phase-aware lambdas (avoid falling back to 2/2/2)
    search_thr_lam_spec: float = 1.5
    search_thr_lam_fpr: float = 2.0
    search_thr_lam_sens: float = 1.5 #2.0
    final_thr_lam_spec: float = 2.0
    final_thr_lam_fpr: float = 2.0
    final_thr_lam_sens: float = 2.0


    # Policy-specific multipliers (safe defaults)
    thr_policy_sens_mult: float = 2.0     # sens policy makes sens penalty heavier
    thr_policy_spec_mult: float = 1.0     # sens policy may soften spec a bit
    thr_policy_fpr_mult: float  = 1.0
    thr_policy_youden_spec_mult: float = 1.5
    thr_policy_youden_sens_mult: float = 1.5


    # -------------------------------------------------
    # Phase-aware "cheap separation" knobs (Priority 2)
    # -------------------------------------------------
    # SEARCH: slightly more proxy training (still cheap on subset) to avoid probs collapse (0.35-0.50)
    search_inner_epochs_classif: int = 6 #12 #8#15
    search_inner_train_batches_vqc: int = 64 #32 #16#32   # ~+50% over 16 by default
    search_inner_train_batches_head: int = 16 #50 #20#12 #6#10  # warmup head helps when collapsed
    search_terminal_diff_method: str = "backprop"
    search_head_epochs: int = 8 #8 #2

    # FINAL: keep more strict / stable settings
    final_inner_epochs_classif: int = 15
    final_inner_train_batches_vqc: int = 32
    final_inner_train_batches_head: int = 5
    final_terminal_diff_method: str = "adjoint"
    final_head_epochs: int = 1


    search_head_lr = 5e-3 

    # Backward-compat (if code still reads terminal_diff_method)
    terminal_diff_method: str = "adjoint"

    
# -------------------------
# Helpers: apply overrides safely
# -------------------------
def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """
    Returns a NEW Config with overrides applied.
    Also enforces minimal consistency between patch-bank and feature bank sizing.
    """
    cfg2 = Config(**cfg.__dict__)
    for k, v in (overrides or {}).items():
        if not hasattr(cfg2, k):
            raise AttributeError(f"Config has no field '{k}' (override error)")
        setattr(cfg2, k, v)

    # Hard consistency: if patch-bank is enabled and compact_features True,
    # then feature bank domain should be P patches.
    if bool(cfg2.use_patch_bank) and bool(cfg2.patch_bank_compact_features):
        P = num_patches(28, 28, int(cfg2.patch_size), int(cfg2.patch_stride))
        cfg2.feature_bank_size = int(P)
        cfg2.feature_bank_min_size = int(min(int(cfg2.feature_bank_min_size), int(P)))
        if len(cfg2.feature_bank_schedule):
            cfg2.feature_bank_schedule = tuple(int(min(int(k), int(P))) for k in cfg2.feature_bank_schedule)
        else:
            cfg2.feature_bank_schedule = (int(P),)

    # If patch-bank is OFF, keep your qubit-based pixel-domain heuristic inside make_cfg_for_qubits()
    # (done per nq later). Here we only ensure no obviously invalid schedule:
    if len(cfg2.feature_bank_schedule) == 0:
        cfg2.feature_bank_schedule = (int(cfg2.feature_bank_size),)

    # Clamp min_size to size
    cfg2.feature_bank_min_size = int(min(int(cfg2.feature_bank_min_size), int(cfg2.feature_bank_size)))
    return cfg2

def scenario_tag(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return safe[:120] if len(safe) > 120 else safe


# -------------------------
# Define Ablations
# -------------------------
def build_ablation_scenarios():
    return [
        # --- Core / reference (FASE 1) ---
        {"name": "ref_patch_compact_saliency_focal_hardbudgets_thrsoft_f2_encAff_perFeature",
         "semantics_flag": "strong",
         "notes": "FASE 1 ref: patch-compact (P dims) + saliency bank + focal + hard budgets + soft thr + encoding affine per_feature.",
         "overrides": dict(
             use_patch_bank=True,
             patch_bank_compact_features=True,
             feature_bank_update="saliency",
             use_focal=True,
             hard_block_budget=True,
             thr_mode="soft",
             thr_policy="f2",
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        # --- Patch-bank ablations ---
        {"name": "no_patch_bank_pixel_saliency",
         "semantics_flag": "strong",
         "notes": "No patch bank -> pixel-domain features (784). Bank update uses pixel-saliency.",
         "overrides": dict(
             use_patch_bank=False,
             patch_bank_compact_features=False,
             feature_bank_update="saliency",
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        {"name": "patch_stride2_more_patches",
         "semantics_flag": "strong",
         "notes": "Compact patch bank with more patches (stride=2). Keep enc affine per_feature to avoid param explosion.",
         "overrides": dict(
             use_patch_bank=True,
             patch_bank_compact_features=True,
             patch_size=4,
             patch_stride=2,
             feature_bank_update="saliency",
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        # --- Diagnostic WEAK semantics ---
        {"name": "diag_patchbank_no_compact_WEAK_pixels_firstP",
         "semantics_flag": "weak",
         "notes": (
             "DIAGNOSTIC ONLY. use_patch_bank=True but compact_features=False: ENC selects first P pixels. "
             "Not a main claim; used as stress test for semantics mismatch."
         ),
         "overrides": dict(
             use_patch_bank=True,
             patch_bank_compact_features=False,
             patch_size=4,
             patch_stride=4,            # fix P = 49
             feature_bank_update="random",
             feature_bank_size=49,
             feature_bank_min_size=16,
             enc_affine_mode="global",
             enc_beta_max=1.0,
         )},

        # --- Feature bank update ablations ---
        {"name": "patch_chi2_bank",
         "semantics_flag": "strong",
         "notes": "Compact patch-bank, chi2 bank update.",
         "overrides": dict(
             use_patch_bank=True,
             patch_bank_compact_features=True,
             feature_bank_update="chi2",
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        {"name": "patch_random_bank",
         "semantics_flag": "strong",
         "notes": "Compact patch-bank, random bank update.",
         "overrides": dict(
             use_patch_bank=True,
             patch_bank_compact_features=True,
             feature_bank_update="random",
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        {"name": "patch_none_bank_no_rescore",
         "semantics_flag": "strong",
         "notes": "Compact patch-bank, no bank update and no curriculum decay.",
         "overrides": dict(
             use_patch_bank=True,
             patch_bank_compact_features=True,
             feature_bank_update="none",
             feature_bank_decay_enabled=False,
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        # --- Curriculum ablation ---
        {"name": "no_curriculum_decay",
         "semantics_flag": "strong",
         "notes": "Disables feature bank curriculum decay.",
         "overrides": dict(
             feature_bank_decay_enabled=False,
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        # --- Loss ablation ---
        {"name": "no_focal_loss",
         "semantics_flag": "strong",
         "notes": "Disable focal loss.",
         "overrides": dict(
             use_focal=False,
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        # --- Threshold policy ablation ---
        {"name": "thr_policy_sens_target_085",
         "semantics_flag": "strong",
         "notes": "Threshold calibrated to prioritize sensitivity target.",
         "overrides": dict(
             thr_mode="soft",
             thr_policy="sens",
             sens_target=0.85,
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        # --- Budget regime ablations ---
        {"name": "soft_budgets_shaping_only",
         "semantics_flag": "strong",
         "notes": "Do not hard-block actions; only penalize budget excess.",
         "overrides": dict(
             hard_block_budget=False,
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        {"name": "tight_budgets",
         "semantics_flag": "strong",
         "notes": "Tighter ENC/ROT/CNOT budgets with hard blocking.",
         "overrides": dict(
             ENC_budget=4,
             ROT_budget=8,
             CNOT_budget=1,
             hard_block_budget=True,
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},

        {"name": "no_cnot_budget_zero",
         "semantics_flag": "strong",
         "notes": "CNOT budget=0 with hard blocking.",
         "overrides": dict(
             CNOT_budget=0,
             hard_block_budget=True,
             enc_affine_mode="per_feature",
             enc_beta_max=1.0,
         )},
    ]



def get_thr_targets(cfg: "Config") -> tuple[float, float, float]:
    """
    Returns (sens_target, spec_min, fpr_max) depending on cfg.phase.
    - search: relaxed targets (avoid viable=0)
    - final: strict/clinical targets
    Falls back to legacy cfg.sens_target / cfg.thr_spec_min / cfg.thr_fpr_max if missing.
    """
    phase = str(getattr(cfg, "phase", "final")).lower()
    if phase == "search":
        sens = float(getattr(cfg, "search_sens_target", getattr(cfg, "sens_target", 0.60)))
        spec = float(getattr(cfg, "search_thr_spec_min", getattr(cfg, "thr_spec_min", 0.55)))
        fpr  = float(getattr(cfg, "search_thr_fpr_max", getattr(cfg, "thr_fpr_max", 0.50)))
    else:
        sens = float(getattr(cfg, "final_sens_target", getattr(cfg, "sens_target", 0.85)))
        spec = float(getattr(cfg, "final_thr_spec_min", getattr(cfg, "thr_spec_min", 0.80)))
        fpr  = float(getattr(cfg, "final_thr_fpr_max", getattr(cfg, "thr_fpr_max", 0.10)))
    return sens, spec, fpr