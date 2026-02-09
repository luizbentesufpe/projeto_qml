import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import pennylane as qml
import torch.nn.functional as F
from rl_and_qml_in_clinical_images.rl.actions import OpType


class BinaryCQV_End2End(nn.Module):
    """
    Binário: head Linear(n_qubits->1).
    QNode executa portas a partir do arch_mat.
    """


    def __init__(
        self,
        arch_mat: torch.Tensor,
        n_qubits: int = 4,
        enc_lambda: float = np.pi,
        diff_method: str = "adjoint",
        input_dim: int = 49,
        enc_affine_mode: str = "per_feature",
        use_batched_qnode: bool = True,
        enc_alpha_init: float = 1.0,
        enc_beta_init: float = 0.0,
        enc_beta_max: float = 1.0,
    ):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.register_buffer("arch_mat", arch_mat.to(torch.int64))
        self.enc_lambda = float(enc_lambda)
        self.input_dim = int(input_dim)
        self.use_batched_qnode = bool(use_batched_qnode)

        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.logit_scale_min = 0.5
        self.logit_scale_max = 80.0


        # --- logit clamp control (default ON for safety) ---
        self.clamp_logits_enabled = True
        self.logit_clamp_value = 30.0  # default matches your current behavior


        # Debug hooks (filled during forward; env can log them)
        self._dbg_raw_logits_std = float("nan")
        self._dbg_scaled_logits_std = float("nan")
        self._dbg_logit_scale_value = float("nan")


        # ---- trainable affine encoding params (alpha/beta) ----
        self.enc_affine_mode = str(enc_affine_mode).lower().strip()
        self.enc_beta_max = float(enc_beta_max)

        # stable init helpers
        a0 = float(max(enc_alpha_init, 1e-6))
        # inverse-softplus approx: softplus(z)=a0  => z = log(exp(a0)-1)
        alpha_raw0 = float(np.log(np.exp(a0) - 1.0))
        b0 = float(enc_beta_init)
        denom = max(self.enc_beta_max, 1e-6)
        b0n = float(np.clip(b0 / denom, -0.999, 0.999))
        beta_raw0 = float(np.arctanh(b0n))

        if self.enc_affine_mode == "per_feature_qubit":
            self.enc_alpha_raw = nn.Parameter(torch.full((self.n_qubits, self.input_dim), alpha_raw0, dtype=torch.float32))
            self.enc_beta_raw  = nn.Parameter(torch.full((self.n_qubits, self.input_dim), beta_raw0,  dtype=torch.float32))
        elif self.enc_affine_mode == "per_feature":
            self.enc_alpha_raw = nn.Parameter(torch.full((self.input_dim,), alpha_raw0, dtype=torch.float32))
            self.enc_beta_raw  = nn.Parameter(torch.full((self.input_dim,), beta_raw0,  dtype=torch.float32))
        elif self.enc_affine_mode == "per_qubit":
            self.enc_alpha_raw = nn.Parameter(torch.full((self.n_qubits,), alpha_raw0, dtype=torch.float32))
            self.enc_beta_raw  = nn.Parameter(torch.full((self.n_qubits,), beta_raw0,  dtype=torch.float32))
        else:  # "global" (fallback)
            self.enc_affine_mode = "global"
            self.enc_alpha_raw = nn.Parameter(torch.tensor(alpha_raw0, dtype=torch.float32))
            self.enc_beta_raw  = nn.Parameter(torch.tensor(beta_raw0,  dtype=torch.float32))


        # slots variacionais (ROT) — CRITICAL FIX:
        # create a stable mapping from layer index -> theta index
        self._rot_param_index = {}
        slots = []
        L = int(self.arch_mat.shape[1])
        for l in range(L):
            op = int(self.arch_mat[2, l].item())
            t  = int(self.arch_mat[1, l].item())
            ax = int(self.arch_mat[3, l].item())
            if op == OpType.ROT.value and (t > 0) and (ax > 0):
                idx = len(slots)
                self._rot_param_index[int(l)] = int(idx)
                slots.append((l, t - 1, ax))
        self.theta = nn.Parameter(0.01 * torch.randn(len(slots)))
        if len(slots) > 0:
            with torch.no_grad():
                self.theta.data = torch.clamp(self.theta.data, -0.1, 0.1)

        # device selection
        if str(diff_method).lower() == "backprop":
            dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)
            self._dev_name = "default.qubit"
        else:
            try:
                dev = qml.device("lightning.gpu", wires=self.n_qubits, shots=None)
                self._dev_name = "lightning.gpu"
            except Exception:
                dev = qml.device("lightning.qubit", wires=self.n_qubits, shots=None)
                self._dev_name = "lightning.qubit"

        def circuit(xi, theta_vec, enc_alpha_raw, enc_beta_raw):
            # xi deve ser 1D (D,)
            try:
                xi = xi.reshape(-1)
            except Exception:
                xi = qml.math.reshape(xi, (-1,))
            Lloc = self.arch_mat.shape[1]
            for l in range(Lloc):
                op = int(self.arch_mat[2, l].item())
                c  = int(self.arch_mat[0, l].item())
                t  = int(self.arch_mat[1, l].item())
                ax = int(self.arch_mat[3, l].item())
                f1 = int(self.arch_mat[4, l].item())

                if op == OpType.ENC.value:
                    if t > 0 and ax > 0 and f1 > 0:
                        tgt = t - 1
                        if tgt < 0 or tgt >= self.n_qubits:
                            continue
                        feat_idx = f1 - 1
                        if feat_idx < 0 or feat_idx >= int(xi.shape[0]):
                            continue
                        # -----------------------------
                        # Encoding parametrizado:
                        # angle = enc_lambda * (alpha[...] * x + beta[...])
                        # alpha>0 (softplus), beta bounded (tanh)
                        # Indexing decided by enc_affine_mode.
                        # -----------------------------
                        xval = xi[feat_idx]

                        if self.enc_affine_mode == "per_feature_qubit":
                            a_raw = enc_alpha_raw[tgt, feat_idx]
                            b_raw = enc_beta_raw[tgt, feat_idx]
                        elif self.enc_affine_mode == "per_feature":
                            a_raw = enc_alpha_raw[feat_idx]
                            b_raw = enc_beta_raw[feat_idx]
                        elif self.enc_affine_mode == "per_qubit":
                            a_raw = enc_alpha_raw[tgt]
                            b_raw = enc_beta_raw[tgt]
                        else:  # "global"
                            a_raw = enc_alpha_raw
                            b_raw = enc_beta_raw

                        alpha = F.softplus(a_raw) + 1e-6
                        beta = torch.tanh(b_raw) * self.enc_beta_max

                        angle = self.enc_lambda * (alpha * xval + beta)

                        if   ax == 1: qml.RX(angle, wires=tgt)
                        elif ax == 2: qml.RY(angle, wires=tgt)
                        elif ax == 3: qml.RZ(angle, wires=tgt)

                elif op == OpType.ROT.value:
                    if t > 0 and ax > 0:
                        tgt = t - 1
                        idx = self._rot_param_index.get(int(l), None)
                        if idx is None:
                            continue
                        if tgt < 0 or tgt >= self.n_qubits:
                            continue
                        ang = theta_vec[int(idx)]
                        if   ax == 1: qml.RX(ang, wires=tgt)
                        elif ax == 2: qml.RY(ang, wires=tgt)
                        elif ax == 3: qml.RZ(ang, wires=tgt)

                elif op == OpType.CNOT.value:
                    if c > 0 and t > 0 and c != t:
                        c0 = c - 1
                        t0 = t - 1
                        if (0 <= c0 < self.n_qubits) and (0 <= t0 < self.n_qubits) and (c0 != t0):
                            qml.CNOT(wires=[c0, t0])

            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]



        self._qnode = qml.QNode(
            circuit,
            dev,
            interface="torch",
            diff_method=diff_method,
            cache=True,
            max_diff=1,
        )

        self._qnode_batched = None

        if self.use_batched_qnode:
            try:
                qfunc_b = qml.batch_input(circuit, argnum=0)
                self._qnode_batched = qml.QNode(
                    qfunc_b,
                    dev,
                    interface="torch",
                    diff_method=None,
                    cache=True,
                )
            except Exception:
                self._qnode_batched = None
                self.use_batched_qnode = False
        

        self.head = nn.Linear(self.n_qubits, 1)

        with torch.no_grad():
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)


    def measure_depth_and_cnot(self, x_sample=None):
        if x_sample is None:
            x_sample = torch.zeros((1, self.input_dim), dtype=torch.float32, device=self.theta.device)
        xi = x_sample[0]
        #tape = self._qnode.construct([xi.detach().cpu(), self.theta.detach().cpu()], {})
        tape = self._qnode.construct([xi.detach().cpu(), self.theta.detach().cpu(),
        self.enc_alpha_raw.detach().cpu(), self.enc_beta_raw.detach().cpu()], {})
        
        qubit_timeline = {}
        depth = 0
        cnot_count = 0
        for op in tape.operations:
            wires = op.wires.tolist()
            if op.name.upper() in ("CNOT", "CX"):
                cnot_count += 1
            layer = max([qubit_timeline.get(w, 0) for w in wires], default=0)
            for w in wires:
                qubit_timeline[w] = layer + 1
            depth = max(depth, layer + 1)
        return depth, cnot_count
    
    @torch.no_grad()
    def measure_depth_cnot_mean(self, X: torch.Tensor, n_samples: int = 16, seed: int = 0):
        """
        Publication-grade cost measurement:
        measure depth/CNOT from actual tape on random inputs and average.
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)
        n = int(min(int(n_samples), int(X.shape[0])))
        if n <= 0:
            n = int(X.shape[0])
        g = torch.Generator(device=X.device)
        g.manual_seed(int(seed))
        idx = torch.randperm(X.shape[0], generator=g, device=X.device)[:n]
        depths = []
        cnots = []
        for i in idx.tolist():
            x1 = X[i:i+1]
            _ = self(x1)  # ensure qnode compiled
            d, c = self.measure_depth_and_cnot(x1)
            depths.append(float(d))
            cnots.append(float(c))
        if len(depths) == 0:
            return 0.0, 0.0
        return float(np.mean(depths)), float(np.mean(cnots))
    
    def _ensure_BD(self, x: torch.Tensor) -> torch.Tensor:
        # força (B,D)
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            if x.numel() == self.input_dim:
                x = x.unsqueeze(0)  # (1,D)
            else:
                x = x.view(-1, 1)   # (B,1)
        elif x.dim() > 2:
            x = x.view(x.shape[0], -1)

        if x.dim() != 2:
            raise RuntimeError(f"[forward] expected 2D (B,D); got {tuple(x.shape)}")

        if x.shape[1] != self.input_dim:
            raise RuntimeError(f"[forward] Bad input_dim: got x.shape={tuple(x.shape)} but input_dim={self.input_dim}")

        return x

    def _ev_to_row(self, ev, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Normaliza a saída do QNode para shape (n_qubits,)
        """
        if isinstance(ev, (list, tuple)):
            ev = [t if torch.is_tensor(t) else torch.as_tensor(t, device=device) for t in ev]
            row = torch.stack([t.to(device=device, dtype=dtype).reshape(()) for t in ev], dim=0)  # (nq,)
        else:
            row = ev if torch.is_tensor(ev) else torch.as_tensor(ev, device=device)
            row = row.to(device=device, dtype=dtype).reshape(-1)

        # garante tamanho = n_qubits
        if row.numel() != self.n_qubits:
            # tenta corrigir casos (nq,1) ou (1,nq)
            if row.dim() == 2 and (row.shape == (self.n_qubits, 1) or row.shape == (1, self.n_qubits)):
                row = row.reshape(-1)
            if row.numel() != self.n_qubits:
                raise RuntimeError(f"Bad expvals row: got shape={tuple(row.shape)} numel={row.numel()} expected n_qubits={self.n_qubits}")

        return row

    def forward(self, x_flat):
        device = self.theta.device
        dtype = self.head.weight.dtype

        # x = torch.as_tensor(x_flat, dtype=torch.float32, device=device)
        
        x = x_flat if torch.is_tensor(x_flat) else torch.as_tensor(x_flat)
        x = x.to(device=device, dtype=torch.float32)
        x = self._ensure_BD(x)

        use_fast_batched = (
             (not torch.is_grad_enabled())
             and self.use_batched_qnode
             and (self._qnode_batched is not None)
        )

        # 🔥 CRITICAL FIX:
        # PennyLane batch_input requires x to be fully detached from autograd graph

        if use_fast_batched:
            try:
                # batch_input exige argnum=0 não-trainable; NumPy costuma resolver
                x_np = x.detach().cpu().numpy()
                ev = self._qnode_batched(x_np, self.theta, self.enc_alpha_raw, self.enc_beta_raw)

                if isinstance(ev, (list, tuple)):
                    cols = [t if torch.is_tensor(t) else torch.as_tensor(t, device=device) for t in ev]
                    expvals = torch.stack(
                        [c.to(device=device, dtype=dtype).reshape(-1) for c in cols],
                        dim=1
                    )
                else:
                    expvals = ev.to(device=device, dtype=dtype)
                    if expvals.dim() == 1:
                        expvals = expvals.view(1, -1)

            except ValueError as e:
                # PennyLane ainda pode acusar "trainable" dependendo da versão / interface.
                # Fallback robusto: loop sample-a-sample (sempre funciona).
                B = int(x.shape[0])
                rows = []
                for i in range(B):
                    ev_i = self._qnode(x[i], self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                    rows.append(self._ev_to_row(ev_i, device=device, dtype=dtype))
                expvals = torch.stack(rows, dim=0)
        else:
            # fallback single-sample loop
            B = int(x.shape[0])
            rows = []
            for i in range(B):
                ev_i = self._qnode(x[i], self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                rows.append(self._ev_to_row(ev_i, device=device, dtype=dtype))
            expvals = torch.stack(rows, dim=0)


        raw_logits = self.head(expvals)

        # -----------------------------
        # Logit-scale: optional eval-only
        # -----------------------------
        scale_eval_only = bool(getattr(self, "logit_scale_eval_only", False))

        if self.training and scale_eval_only:
            # Durante treino: NÃO aplica escala (evita compensação via encolher head)
            scale_t = raw_logits.new_tensor(1.0)
        else:
            # EVAL (ou treino normal): aplica escala
            scale_t = torch.clamp(
                self.logit_scale.to(device=device, dtype=raw_logits.dtype),
                min=self.logit_scale_min,
                max=self.logit_scale_max,
            )

        logits = raw_logits * scale_t

        # DEBUG
        try:
            with torch.no_grad():
                self._dbg_raw_logits_std = float(raw_logits.detach().std().cpu().item())
                self._dbg_scaled_logits_std = float(logits.detach().std().cpu().item())
                self._dbg_logit_scale_value = float(scale_t.detach().cpu().item())
                self._dbg_logit_scale_eval_only = float(scale_eval_only)
                self._dbg_training_flag = float(self.training)
        except Exception:
            pass

        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        # Optional clamp (ON in FINAL, OFF in SEARCH)
        if bool(getattr(self, "clamp_logits_enabled", True)):
            c = float(getattr(self, "logit_clamp_value", 30.0))
            logits = torch.clamp(logits, -c, c)

        return logits
    @torch.no_grad()
    def forward_raw_logits(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Returns logits BEFORE applying logit_scale and clamp.
        Useful for deterministic scale calibration.
        """
        device = self.theta.device
        dtype = self.head.weight.dtype

        x = x_flat if torch.is_tensor(x_flat) else torch.as_tensor(x_flat)
        x = x.to(device=device, dtype=torch.float32)
        x = self._ensure_BD(x)

        use_fast_batched = (
            (not torch.is_grad_enabled())
            and self.use_batched_qnode
            and (self._qnode_batched is not None)
        )

        if use_fast_batched:
            try:
                x_np = x.detach().cpu().numpy()
                ev = self._qnode_batched(x_np, self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                if isinstance(ev, (list, tuple)):
                    cols = [t if torch.is_tensor(t) else torch.as_tensor(t, device=device) for t in ev]
                    expvals = torch.stack([c.to(device=device, dtype=dtype).reshape(-1) for c in cols], dim=1)
                else:
                    expvals = ev.to(device=device, dtype=dtype)
                    if expvals.dim() == 1:
                        expvals = expvals.view(1, -1)
            except Exception:
                B = int(x.shape[0])
                rows = []
                for i in range(B):
                    ev_i = self._qnode(x[i], self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                    rows.append(self._ev_to_row(ev_i, device=device, dtype=dtype))
                expvals = torch.stack(rows, dim=0)
        else:
            B = int(x.shape[0])
            rows = []
            for i in range(B):
                ev_i = self._qnode(x[i], self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                rows.append(self._ev_to_row(ev_i, device=device, dtype=dtype))
            expvals = torch.stack(rows, dim=0)

        raw_logits = self.head(expvals)
        return torch.nan_to_num(raw_logits, nan=0.0, posinf=1e6, neginf=-1e6)
    @torch.no_grad()
    def set_logit_scale_calibrated(
        self,
        raw_logits: torch.Tensor,
        method: str = "p95",     # "p95" or "std"
        target: float = 8.0,     # target p95(|logit|) or target std
        eps: float = 1e-6,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
    ) -> float:
        """
        Deterministic (non-learned) scaling to reduce seed-to-seed temperature drift.
        raw_logits must be BEFORE applying self.logit_scale.
        """
        x = raw_logits.detach().view(-1).float()
        if x.numel() == 0:
            return float(self.logit_scale.detach().cpu().item())

        method = str(method).lower().strip()
        if method == "std":
            denom = float(x.std().cpu().item())
        else:
            denom = float(torch.quantile(torch.abs(x), 0.95).cpu().item())

        s = float(target) / float(max(denom, eps))
        lo = float(clamp_min) if clamp_min is not None else float(self.logit_scale_min)
        hi = float(clamp_max) if clamp_max is not None else float(self.logit_scale_max)
        s = float(max(lo, min(hi, s)))
        self.logit_scale.fill_(float(s))
        return float(s)
    # --------------------------------------------------
    # Logit-scale control (SEARCH vs FINAL)
    # --------------------------------------------------
    def set_logit_scale_trainable(self, trainable: bool, value: float | None = None):
        """
        If trainable=False, freezes logit_scale (used in SEARCH).
        If trainable=True, enables learning (used in FINAL).
        Optionally sets a fixed value.
        """
        if value is not None:
            with torch.no_grad():
                self.logit_scale.fill_(float(value))
        self.logit_scale.requires_grad_(bool(trainable))

    
    def set_clamp_logits(self, enabled: bool, clamp_value: float | None = None) -> None:
        self.clamp_logits_enabled = bool(enabled)
        if clamp_value is not None:
            self.logit_clamp_value = float(clamp_value)


def compute_pos_weight(Y_tr: torch.Tensor, device: str) -> torch.Tensor:
    y = Y_tr.detach().cpu().numpy().reshape(-1)
    pos = float(y.sum())
    neg = float(y.shape[0] - pos)
    w = (neg / max(pos, 1.0)) if pos > 0 else 1.0
    if not np.isfinite(w):
        w = 1.0
    device = torch.device(device)
    return torch.tensor([w], device=device)


def init_head_bias_with_prevalence(model: nn.Module, Y_tr_np, *, force_p: float | None = None):
    if force_p is not None:
        p = float(force_p)
    else:
        p = float(Y_tr_np.mean())

    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    b = torch.tensor([np.log(p / (1 - p))], dtype=torch.float32, device=next(model.parameters()).device)
    with torch.no_grad():
        model.head.bias.copy_(b)


def clamp_and_clip_head_(head: torch.nn.Module, cfg) -> dict:
    """
    Aplica clamp/clip SOMENTE no head (peso e bias separados).
    Retorna métricas úteis para log/depuração.
    """
    if head is None:
        return {}

    w = getattr(head, "weight", None)
    b = getattr(head, "bias", None)

    # Clamps (valores seguros por default; tune via cfg)
    w_abs_max = float(getattr(cfg, "head_weight_abs_max", 2.0))
    b_abs_max = float(getattr(cfg, "head_bias_abs_max", 2.0))

    # Clips por norma (se quiser)
    w_norm_max = float(getattr(cfg, "head_weight_norm_max", 1.0))
    b_norm_max = float(getattr(cfg, "head_bias_norm_max", 2.0))

    dbg = {}
    with torch.no_grad():
        if w is not None:
            dbg["w_norm_before"] = float(w.data.norm().item())
            dbg["w_absmax_before"] = float(w.data.abs().max().item())

            # clamp por valor
            if math.isfinite(w_abs_max) and w_abs_max > 0:
                w.data.clamp_(-w_abs_max, w_abs_max)

            # clip por norma
            if math.isfinite(w_norm_max) and w_norm_max > 0:
                wn = float(w.data.norm().item())
                if wn > w_norm_max:
                    w.data.mul_(w_norm_max / (wn + 1e-12))

            dbg["w_norm_after"] = float(w.data.norm().item())
            dbg["w_absmax_after"] = float(w.data.abs().max().item())

        if b is not None:
            dbg["b_norm_before"] = float(b.data.norm().item())
            dbg["b_absmax_before"] = float(b.data.abs().max().item())

            if math.isfinite(b_abs_max) and b_abs_max > 0:
                b.data.clamp_(-b_abs_max, b_abs_max)

            if math.isfinite(b_norm_max) and b_norm_max > 0:
                bn = float(b.data.norm().item())
                if bn > b_norm_max:
                    b.data.mul_(b_norm_max / (bn + 1e-12))

            dbg["b_norm_after"] = float(b.data.norm().item())
            dbg["b_absmax_after"] = float(b.data.abs().max().item())

    return dbg