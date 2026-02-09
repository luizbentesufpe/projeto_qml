import torch
from rl_and_qml_in_clinical_images.rl.rl_config import Config
import numpy as np

from rl_and_qml_in_clinical_images.modeling.model import BinaryCQV_End2End
from rl_and_qml_in_clinical_images.rl.env import sanitize_architecture


def pareto_front(points):
    # points: list of dict with keys: cost, perf
    pts = sorted(points, key=lambda d: (d["cost"], -d["perf"]))
    front = []
    best_perf = -1e9
    for p in pts:
        if p["perf"] > best_perf:
            front.append(p)
            best_perf = p["perf"]
    return front

def measure_cost_from_arch(arch_mat: torch.Tensor, n_qubits: int, cfg: Config, X_ref: np.ndarray, seed=0, device: torch.device | str = "cpu" ):
    """
    Publication fix: cost proxy grounded in measured depth/CNOT (tape-based).
    Returns a dict with measured stats and a single scalar 'cost'.
    """
    # model = MultiLabelCQV_End2End(arch_mat=arch_mat, n_qubits=int(n_qubits), enc_lambda=float(cfg.enc_lambda), diff_method="adjoint").to(DEVICE)
    DEVICE = torch.device(device)
    arch_mat = sanitize_architecture(arch_mat, int(n_qubits))  # <<< FIX
    model = BinaryCQV_End2End(
        arch_mat=arch_mat, n_qubits=int(n_qubits), enc_lambda=float(cfg.enc_lambda),
        diff_method="adjoint", input_dim=int(np.asarray(X_ref).shape[1])
    ).to(DEVICE)
    X_t = torch.tensor(X_ref, dtype=torch.float32, device=DEVICE)
    d_mean, c_mean = model.measure_depth_cnot_mean(X_t, n_samples=int(cfg.cost_measure_samples), seed=int(seed))
    # Scalar cost: interpretability-friendly, justify as weighted NISQ cost
    # (depth dominates latency; CNOT dominates error; qubits dominate footprint)
    cost = float(d_mean + 2.0 * c_mean + 0.2 * float(n_qubits))
    return {
        "depth_mean": float(d_mean),
        "cnot_mean": float(c_mean),
        "n_qubits": int(n_qubits),
        "cost": float(cost),
        "weights": {"depth": 1.0, "cnot": 2.0, "qubits": 0.2}
    }