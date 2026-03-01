from collections import deque
import json, random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch

import math

def set_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def log_to_file(self, filename: str, text: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = self.log_dir / f"{filename}-{self.run_tag}.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")

    def dump_json(self, name: str, obj: Any) -> str:
        path = self.log_dir / f"{name}-{self.run_tag}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return str(path)
    
    def return_log_dir(self, filename: str) -> str:
        path = self.log_dir
        return str(path)

def dump_run_metadata(logger: Logger, cfg, extra: Optional[Dict[str, Any]] = None, device: str = "cpu") -> str:
    import pennylane as qml
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "pennylane_version": getattr(qml, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "cfg": cfg.__dict__,
    }
    if extra:
        meta.update(extra)
    path = logger.dump_json("run_meta", meta)
    logger.log_to_file("results", f"[META] {path}")
    return path

def save_circuit_image(model,
                       x_ref: np.ndarray,
                       out_path: str,
                       title: str = ""):
    """
    Salva o circuito do QNode do modelo como imagem (PNG/PDF).

    FIX-E: três bugs corrigidos:
      1. _q_single → _qnode  (atributo real do BinaryCQV_End2End)
      2. drawer recebe os 4 argumentos de circuit():
         (xi, theta_vec, enc_alpha_raw, enc_beta_raw)
      3. o model deve usar diff_method='backprop' (default.qubit);
         lightning não suporta draw_mpl — ver train.py para o model
         temporário que garante isso.
    """
    import matplotlib.pyplot as plt
    import pennylane as qml
    import numpy as np
    import torch

    # --- x em CPU ---
    x = np.asarray(x_ref, dtype=np.float32).reshape(-1)
    x_t = torch.as_tensor(x, dtype=torch.float32, device="cpu")

    # --- theta (parâmetros variacionais ROT) ---
    theta = getattr(model, "theta", None)
    if theta is None:
        raise AttributeError("Model has no attribute 'theta'")
    theta_t = theta.detach().cpu()

    # --- enc affine params (alpha_raw, beta_raw) ---
    enc_alpha_raw = getattr(model, "enc_alpha_raw", None)
    enc_beta_raw  = getattr(model, "enc_beta_raw",  None)
    if enc_alpha_raw is None or enc_beta_raw is None:
        raise AttributeError("Model has no enc_alpha_raw / enc_beta_raw")
    alpha_t = enc_alpha_raw.detach().cpu()
    beta_t  = enc_beta_raw.detach().cpu()

    # --- FIX-E bug 1: _q_single não existe; usar _qnode ---
    qnode = getattr(model, "_q_single", None) or getattr(model, "_qnode", None)
    if qnode is None:
        raise AttributeError(
            "Model has no drawable QNode (_q_single or _qnode). "
            "Instancie o model com diff_method='backprop' (default.qubit)."
        )

    # --- FIX-E bug 3: passar os 4 argumentos de circuit() ---
    drawer = qml.draw_mpl(qnode, decimals=2, max_length=200)
    out = drawer(x_t, theta_t, alpha_t, beta_t)

    # --- normaliza retorno ---
    figs = []

    if isinstance(out, tuple):
        # caso clássico: (fig, ax)
        figs = [out[0]]

    elif isinstance(out, list):
        # lista de (fig, ax) OU lista de fig
        for item in out:
            if isinstance(item, tuple):
                figs.append(item[0])
            else:
                figs.append(item)

    else:
        # retorno direto de Figure
        figs = [out]

    # --- salva todas (ou só a primeira, se preferir) ---
    for i, fig in enumerate(figs):
        if title:
            try:
                fig.suptitle(title)
            except Exception:
                pass

        fig.tight_layout()

        # se houver múltiplas figuras, adiciona sufixo
        if len(figs) > 1:
            path = out_path.replace(".png", f"_part{i}.png")
        else:
            path = out_path

        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

class RunningStd:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 1e-6
        return math.sqrt(self.M2 / (self.n - 1))

class RunningPctl:
    """
    Simple running percentile using a fixed-size buffer.
    Good enough for reward normalization (p95 depth).

    Usage:
    rp = RunningPctl(p=95, maxlen=512)
    rp.update(x)
    ref = rp.value(default=1.0)
    """
    def __init__(self, p: float = 95.0, maxlen: int = 512):
        self.p = float(p)
        self.buf = deque(maxlen=int(maxlen))

    def update(self, x: float):
        self.buf.append(float(x))

    def value(self, default: float = 1.0) -> float:
        if len(self.buf) < 8:
            return float(default)
        arr = np.asarray(self.buf, dtype=np.float32)
        v = float(np.percentile(arr, self.p))
        if not np.isfinite(v) or v <= 1e-9:
            return float(default)
        return float(v)
    