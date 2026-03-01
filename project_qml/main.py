from main_ablation import main_ablation
from main_debug_2 import main_debug_ablation
#from main_debug import main_debug_ablation
from main_debug_cross_circle import main_debug_cross_circle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Run in DEBUG mode (minimal execution, no scientific guarantees)"
    )
    parser.add_argument(
        "--cross_circle",
        dest="cross_circle",
        action="store_true",
        help="Run in CROSS_CIRCLE mode (minimal execution, no scientific guarantees)"
    )
    return parser.parse_args()

ARGS = parse_args()

DEBUG_MODE = ARGS.debug
CROSS_CIRCLE_MODE = ARGS.cross_circle

"""
main_debug_cross_circle_test.py
================================
Sanity checks progressivos para diagnosticar por que o head não aprende.

Execução:
  python rl_and_qml_in_clinical_images/main.py --cross_circle_test

São 4 testes em ordem crescente de complexidade:

  [TEST 1] Head com features REAIS do circle/cross (sklearn LR como baseline)
           → Esperado: AUC > 0.95  (dataset é separável)
           → Se falhar: problema está nos dados/labels, não no modelo

  [TEST 2] Head PyTorch com features REAIS (sem VQC)
           → Esperado: loss < 0.30, AUC > 0.90 em 30 epochs
           → Se falhar: LR ou inicialização do head estão errados

  [TEST 3] Head PyTorch com features ALEATÓRIAS (baseline ruído)
           → Esperado: loss em ~0.693 (não aprende, como esperado)
           → Confirma que o teste 2 realmente precisava do sinal

  [TEST 4] VQC real (nq=4) + head treinado (end-to-end frozen VQC)
           → Esperado: loss < 0.60, AUC > 0.65
           → Se falhar: VQC não está produzindo features discriminativas
"""

"""
main_debug_cross_circle_test.py
================================
Sanity checks progressivos para diagnosticar por que o head não aprende.

Execução:
  python rl_and_qml_in_clinical_images/main.py --cross_circle_test

São 4 testes em ordem crescente de complexidade:

  [TEST 1] Head com features REAIS do circle/cross (sklearn LR como baseline)
           → Esperado: AUC > 0.95  (dataset é separável)
           → Se falhar: problema está nos dados/labels, não no modelo

  [TEST 2] Head PyTorch com features REAIS (sem VQC)
           → Esperado: loss < 0.30, AUC > 0.90 em 30 epochs
           → Se falhar: LR ou inicialização do head estão errados

  [TEST 3] Head PyTorch com features ALEATÓRIAS (baseline ruído)
           → Esperado: loss em ~0.693 (não aprende, como esperado)
           → Confirma que o teste 2 realmente precisava do sinal

  [TEST 4] VQC real (nq=4) + head treinado (end-to-end frozen VQC)
           → Esperado: loss < 0.60, AUC > 0.65
           → Se falhar: VQC não está produzindo features discriminativas
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def main_debug_cross_circle_test():
    print("\n" + "="*60)
    print("  SANITY CHECK — HEAD LEARNING DIAGNOSTICS")
    print("="*60)

    # ----------------------------------------------------------------
    # Imports internos (dentro da função para compatibilidade com main.py)
    # ----------------------------------------------------------------
    from rl_and_qml_in_clinical_images.dataset import (
        load_circle_cross_pool_flatten,
        create_circle_cross_dataset,
    )
    from rl_and_qml_in_clinical_images.rl.rl_config import Config

    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"

    cfg = Config()

    # ----------------------------------------------------------------
    # Carrega dados reais
    # ----------------------------------------------------------------
    X, Y = load_circle_cross_pool_flatten(cfg, percent_total=30, seed=0)
    Y_flat = Y.reshape(-1).astype(np.float32)
    Y_int  = (Y_flat > 0.5).astype(int)

    print(f"\n[DATA] Shape: X={X.shape}, Y={Y.shape}")
    print(f"[DATA] Classes: pos={Y_int.sum()}, neg={(1-Y_int).sum()}, ratio={Y_int.mean():.2f}")
    print(f"[DATA] X stats: mean={X.mean():.4f}, std={X.std():.4f}, min={X.min():.4f}, max={X.max():.4f}")

    # Split simples treino/val
    n = len(X)
    n_tr = int(0.8 * n)
    idx = np.random.default_rng(42).permutation(n)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
    Xtr, Ytr = X[tr_idx], Y_int[tr_idx]
    Xva, Yva = X[va_idx], Y_int[va_idx]

    # ================================================================
    # TEST 1 — Sklearn LR (baseline de referência)
    # ================================================================
    print("\n" + "-"*50)
    print("[TEST 1] Sklearn LogisticRegression com features REAIS")
    print("-"*50)

    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    pipe.fit(Xtr, Ytr)
    p_va = pipe.predict_proba(Xva)[:, 1]
    auc_sk = roc_auc_score(Yva, p_va) if len(np.unique(Yva)) > 1 else 0.5
    print(f"  AUC sklearn LR = {auc_sk:.4f}")

    if auc_sk > 0.90:
        print("  ✅ PASS — dados são separáveis, problema está no modelo/config")
    elif auc_sk > 0.70:
        print("  ⚠️  WARN — separabilidade parcial, checar noise_std ou n_samples")
    else:
        print("  ❌ FAIL — dataset não separável ou labels incorretos!")

    # ================================================================
    # TEST 2 — Head PyTorch com features REAIS
    # ================================================================
    print("\n" + "-"*50)
    print("[TEST 2] Head PyTorch (Linear) com features REAIS")
    print("-"*50)

    # Normalizar como o pipeline faz
    mu  = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / std
    Xva_n = (Xva - mu) / std

    Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32, device=DEVICE)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva_n, dtype=torch.float32, device=DEVICE)
    Yva_t = torch.tensor(Yva, dtype=torch.float32, device=DEVICE)

    in_dim = Xtr_t.shape[1]
    head2 = torch.nn.Linear(in_dim, 1).to(DEVICE)

    # Testar 3 LRs diferentes
    for lr_test in [1e-2, 5e-2, 1e-1]:
        head2_tmp = torch.nn.Linear(in_dim, 1).to(DEVICE)
        opt = torch.optim.Adam(head2_tmp.parameters(), lr=lr_test)

        losses = []
        for ep in range(30):
            head2_tmp.train()
            perm = torch.randperm(len(Xtr_t), device=DEVICE)
            ep_loss = []
            for i in range(0, len(Xtr_t), 32):
                xb = Xtr_t[perm[i:i+32]]
                yb = Ytr_t[perm[i:i+32]]
                loss = F.binary_cross_entropy_with_logits(head2_tmp(xb).view(-1), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss.append(loss.item())
            losses.append(np.mean(ep_loss))

        head2_tmp.eval()
        with torch.no_grad():
            logits_va = head2_tmp(Xva_t).view(-1).cpu().numpy()
            probs_va  = 1 / (1 + np.exp(-logits_va))
        auc_pt = roc_auc_score(Yva, probs_va) if len(np.unique(Yva)) > 1 else 0.5
        final_loss = losses[-1]

        status = "✅" if (final_loss < 0.45 and auc_pt > 0.85) else ("⚠️ " if auc_pt > 0.65 else "❌")
        print(f"  lr={lr_test:.0e} | loss_final={final_loss:.4f} | AUC={auc_pt:.4f} {status}")

    # ================================================================
    # TEST 3 — Head PyTorch com features ALEATÓRIAS (controle)
    # ================================================================
    print("\n" + "-"*50)
    print("[TEST 3] Head PyTorch com features ALEATÓRIAS (controle)")
    print("-"*50)

    Xrnd_t = torch.randn(len(Xtr_t), in_dim, device=DEVICE)
    head3 = torch.nn.Linear(in_dim, 1).to(DEVICE)
    opt3 = torch.optim.Adam(head3.parameters(), lr=5e-2)

    for ep in range(30):
        head3.train()
        for i in range(0, len(Xrnd_t), 32):
            xb = Xrnd_t[i:i+32]
            yb = Ytr_t[i:i+32]
            loss3 = F.binary_cross_entropy_with_logits(head3(xb).view(-1), yb)
            opt3.zero_grad(); loss3.backward(); opt3.step()

    print(f"  loss_final (ruído) = {loss3.item():.4f}  ← esperado ~0.693 (não aprende)")
    if loss3.item() > 0.68:
        print("  ✅ Confirmado: sem sinal = sem aprendizado (controle válido)")
    else:
        print("  ⚠️  Head está memorizando ruído — checar batch size ou n_epochs")

    # ================================================================
    # TEST 4 — VQC real (nq=4) + head treinado com VQC congelado
    # ================================================================
    print("\n" + "-"*50)
    print("[TEST 4] VQC real (nq=4) + head treinado (VQC congelado)")
    print("-"*50)

    try:
        from rl_and_qml_in_clinical_images.modeling.model import BinaryCQV_End2End
        from rl_and_qml_in_clinical_images.rl.env import sanitize_architecture
        from rl_and_qml_in_clinical_images.modeling.train import make_cfg_for_qubits

        NQ = 4
        N_LAYERS = 12   # número de colunas da arch_mat (slots do circuito)

        cfg4 = make_cfg_for_qubits(cfg, NQ)
        cfg4.use_patch_bank = False
        cfg4.feature_bank_size = in_dim
        cfg4.feature_bank_min_size = in_dim
        cfg4.start_qubits = NQ
        cfg4.min_qubits = NQ
        cfg4.max_qubits = NQ

        # -------------------------------------------------------
        # Constrói arch_mat shape=(5, N_LAYERS) com operações reais
        # Linhas: [control, target, op, axis, feat_idx]
        # OpType: ENC=1, ROT=2, CNOT=3, NOP=0
        # -------------------------------------------------------
        from rl_and_qml_in_clinical_images.rl.actions import OpType

        rng4 = np.random.default_rng(42)
        arch_np = np.zeros((5, N_LAYERS), dtype=np.int64)

        # Preenche com ENC e ROT simples (sem CNOT para evitar erros de qubit)
        for col in range(N_LAYERS):
            q = int(rng4.integers(1, NQ + 1))          # qubit 1-indexed
            if col < in_dim:                            # primeiras colunas: ENC
                fi = col + 1                            # feature index 1-indexed
                arch_np[0, col] = 0                     # control (sem uso no ENC)
                arch_np[1, col] = q                     # target qubit
                arch_np[2, col] = OpType.ENC.value      # op = ENC
                arch_np[3, col] = 1                     # axis (RY=1)
                arch_np[4, col] = fi                    # feature index
            else:                                       # restante: ROT
                arch_np[0, col] = 0
                arch_np[1, col] = q
                arch_np[2, col] = OpType.ROT.value      # op = ROT
                arch_np[3, col] = int(rng4.integers(1, 4))  # axis 1-3
                arch_np[4, col] = 0

        arch_mat4 = torch.tensor(arch_np, dtype=torch.int64)
        arch_mat4 = sanitize_architecture(arch_mat4, NQ)

        # -------------------------------------------------------
        # Testa 2 modos:
        #   A) VQC congelado  → head aprende sobre features fixas
        #   B) End-to-end     → VQC + head treináveis juntos
        # -------------------------------------------------------
        for mode_name, freeze_vqc in [("VQC congelado", True), ("End-to-end", False)]:
            model4 = BinaryCQV_End2End(
                arch_mat=arch_mat4,
                n_qubits=NQ,
                enc_lambda=float(cfg4.enc_lambda),
                diff_method="backprop",      # backprop = mais rápido para teste
                input_dim=in_dim,
                enc_affine_mode=str(getattr(cfg4, "enc_affine_mode", "per_feature")),
                enc_alpha_init=float(getattr(cfg4, "enc_alpha_init", 1.0)),
                enc_beta_init=float(getattr(cfg4, "enc_beta_init", 0.0)),
                enc_beta_max=float(getattr(cfg4, "enc_beta_max", 1.0)),
            ).to(DEVICE)

            if freeze_vqc:
                for name, p in model4.named_parameters():
                    if "head" not in name:
                        p.requires_grad_(False)

            trainable = [p for p in model4.parameters() if p.requires_grad]
            opt4 = torch.optim.Adam(trainable, lr=5e-2)

            losses4 = []
            model4.train()
            for ep in range(20):
                ep_losses = []
                perm = torch.randperm(len(Xtr_t), device=DEVICE)
                for i in range(0, len(Xtr_t), 32):
                    xb = Xtr_t[perm[i:i+32]]
                    yb = Ytr_t[perm[i:i+32]]
                    logits = model4(xb).view(-1)
                    loss4 = F.binary_cross_entropy_with_logits(logits, yb)
                    opt4.zero_grad()
                    loss4.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt4.step()
                    ep_losses.append(loss4.item())
                losses4.append(np.mean(ep_losses))

            model4.eval()
            with torch.no_grad():
                logits_va4 = model4(Xva_t).view(-1).cpu().numpy()
                probs_va4  = 1.0 / (1.0 + np.exp(-logits_va4))
                prob_std4  = float(probs_va4.std())

            auc4 = roc_auc_score(Yva, probs_va4) if len(np.unique(Yva)) > 1 else 0.5
            delta4 = losses4[0] - losses4[-1]

            if delta4 > 0.05 and auc4 > 0.65:
                status4 = "✅ VQC produz sinal — head aprende"
            elif delta4 > 0.01:
                status4 = "⚠️  Sinal fraco — VQC tem baixa variância"
            else:
                status4 = "❌ Head NÃO aprende — VQC colapsado"

            print(f"\n  [{mode_name}]")
            print(f"    loss: ep1={losses4[0]:.4f} → ep20={losses4[-1]:.4f}  (Δ={delta4:.4f})")
            print(f"    AUC val = {auc4:.4f}  |  prob_std = {prob_std4:.4f}")
            print(f"    {status4}")

        # Diagnóstico final do TEST 4
        print(f"\n  Comparação:")
        print(f"    VQC (congelado) AUC acima: indica features têm sinal")
        print(f"    End-to-end AUC acima:      indica VQC aprende com gradiente")
        print(f"    Ruído (TEST 3) loss={loss3.item():.4f}: baseline sem sinal")

    except Exception as e:
        import traceback
        print(f"  ❌ TEST 4 falhou: {e}")
        traceback.print_exc()
        print("  → Checar imports: BinaryCQV_End2End, sanitize_architecture, OpType")

    # ================================================================
    # DIAGNÓSTICO FINAL
    # ================================================================
    print("\n" + "="*60)
    print("  DIAGNÓSTICO FINAL")
    print("="*60)
    print(f"""
  Interprete os resultados assim:

  TEST 1 (sklearn) AUC > 0.95  → dados ok, problema é no VQC/head
  TEST 2 (pytorch) loss < 0.45 → head PyTorch funciona com features reais
  TEST 3 (ruído)   loss ~ 0.69 → controle válido
  TEST 4 (VQC)     AUC > 0.65  → VQC produz features discriminativas

  Se TEST 2 passa mas TEST 4 falha:
    → O VQC não está produzindo features com variância suficiente
    → Solução: aumentar n_epochs VQC, LR VQC, ou adicionar re-uploading

  Se TEST 2 falha:
    → O head não aprende mesmo com dados limpos
    → Solução: aumentar lr_head, head_epochs, checar normalização

  Se TEST 1 falha:
    → Problema nos dados / labels
    → Checar create_circle_cross_dataset (noise_std, templates)
""")
    print("="*60 + "\n")

if __name__ == "__main__":
    if CROSS_CIRCLE_MODE:
        print("[INFO] Running in CROSS_CIRCLE_MODE=True")
        #main_debug_cross_circle_test()
        main_debug_cross_circle()
    elif DEBUG_MODE:
        print("[INFO] Running in DEBUG_MODE=True")
        main_debug_ablation()
        #main_debug_ablation()
    else:
        print("[INFO] Running in FULL / PUBLICATION mode")
        main_ablation()