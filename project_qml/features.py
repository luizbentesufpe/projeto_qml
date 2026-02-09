
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_selection import chi2


def num_patches(h: int, w: int, patch: int, stride: int) -> int:
    # counts patches in a sliding grid
    nr = ((h - patch) // stride) + 1
    nc = ((w - patch) // stride) + 1
    return int(max(0, nr) * max(0, nc))


def make_patch_groups(h: int = 28, w: int = 28, patch: int = 4, stride: int = 4):
    """
    Ex.: patch=4, stride=4 em 28x28 => 7x7=49 patches.
    groups[p] contém os índices (em [0..783]) do patch p.
    """
    groups = []
    for r in range(0, h - patch + 1, stride):
        for c in range(0, w - patch + 1, stride):
            idx = []
            for rr in range(r, r + patch):
                base = rr * w
                for cc in range(c, c + patch):
                    idx.append(base + cc)
            groups.append(np.array(idx, dtype=np.int64))
    return groups

def patchify_mean(X_flat: np.ndarray, groups) -> np.ndarray:
    """
    X_flat: (N,784) -> Xp: (N,P) onde P=len(groups) (ex.: 49).
    """
    N, _ = X_flat.shape
    P = len(groups)
    Xp = np.zeros((N, P), dtype=np.float32)
    for p, g in enumerate(groups):
        Xp[:, p] = X_flat[:, g].mean(axis=1)
    return Xp

def patchify_mean_flat(
    X_flat: np.ndarray | torch.Tensor,
    patch_groups: list[list[int]],
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert flattened 28x28 pixels (B,784) into compact patch features (B,P)
    by taking mean over indices of each patch in patch_groups.

    patch_groups: list of lists of pixel indices (in [0..783]) per patch.
    """
    device = torch.device(device)
    if not torch.is_tensor(X_flat):
        X = torch.as_tensor(X_flat, dtype=torch.float32, device=device)
    else:
        X = X_flat.to(dtype=torch.float32, device=(device or X_flat.device))

    if X.dim() == 1:
        X = X.unsqueeze(0)
    if X.dim() != 2:
        X = X.view(X.shape[0], -1)

    B, D = int(X.shape[0]), int(X.shape[1])
    pg_np = np.asarray(patch_groups, dtype=np.int64)  # (P,K)

    if pg_np.ndim != 2:
        raise ValueError(f"patch_groups must be 2D after asarray; got shape={pg_np.shape}")

    # --- validate bounds BEFORE GPU indexing ---
    mn = int(pg_np.min()) if pg_np.size else 0
    mx = int(pg_np.max()) if pg_np.size else -1
    if mn < 0:
        raise ValueError(f"[patchify_mean_flat] negative indices found (min={mn}); expected [0..{D-1}]")
    if mx >= D:
        raise ValueError(f"[patchify_mean_flat] index out of bounds: max={mx} but D={D}. "
                            f"X must be flattened 28x28 => D=784. Check X shape upstream.")

    idx = torch.from_numpy(pg_np).to(device=X.device, dtype=torch.long)  # (P,K)
    gathered = X[:, idx]              # (B,P,K)
    return gathered.mean(dim=-1)      # (B,P)
def select_feature_bank_random(n_features: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(n_features), size=min(k, n_features), replace=False)
    return np.sort(idx)

def select_feature_bank_chi2_pixels(X_tr: np.ndarray, y_tr_bin: np.ndarray, k: int) -> np.ndarray:
    y = y_tr_bin.reshape(-1).astype(int)

    X = np.asarray(X_tr, dtype=np.float32)
    # chi2 requires X >= 0
    minv = float(np.min(X)) if X.size else 0.0
    if minv < 0.0:
        X = X - minv  # shift to make non-negative

    scores, _ = chi2(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-scores)
    topk = order[:min(k, X.shape[1])]
    return np.sort(topk)

def select_feature_bank_chi2_patches(X_tr_flat: np.ndarray, y_tr_bin: np.ndarray, k: int, patch_groups) -> np.ndarray:
    y = y_tr_bin.reshape(-1).astype(int)
    Xp = patchify_mean(X_tr_flat, groups=patch_groups).astype(np.float32)

    # chi2 requires X >= 0
    minv = float(np.min(Xp)) if Xp.size else 0.0
    if minv < 0.0:
        Xp = Xp - minv

    scores, _ = chi2(Xp, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-scores)
    topk = order[:min(k, Xp.shape[1])]
    return np.sort(topk)


@torch.no_grad()
def compute_saliency_importance_pixels(model, X: torch.Tensor, Y: torch.Tensor=None, top_k=128):
    model.eval()
    B = min(256, X.size(0))
    idx = torch.randperm(X.size(0), device=X.device)[:B]
    with torch.enable_grad():
        xb = X[idx].detach().clone().requires_grad_(True)

        req = [p.requires_grad for p in model.parameters()]
        # opcional: só precisamos de gradiente na entrada (xb), não nos parâmetros
        for p in model.parameters():
            p.requires_grad_(False)

        logits = model(xb).view(-1)
        if Y is not None:
            yb = Y[idx].detach().float().view_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
        else:
            loss = -logits.mean()

        # garante que não há lixo acumulado em gradientes
        if xb.grad is not None:
            xb.grad.zero_()
        loss.backward()
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)
            
        sal = xb.grad.detach().abs().mean(dim=0)  # (784,)
        sal = (sal - sal.mean()) / (sal.std() + 1e-8)
        order = torch.argsort(-sal)

    return torch.sort(order[:top_k]).values.detach().cpu().numpy()


def compute_saliency_importance_patches(model, X: torch.Tensor, Y: torch.Tensor=None, top_k=49):
    model.eval()
    B = min(256, X.size(0))
    idx = torch.randperm(X.size(0), device=X.device)[:B]
    with torch.enable_grad():
        xb = X[idx].detach().clone().requires_grad_(True)

        # opcional: só gradiente na entrada
        for p in model.parameters():
            p.requires_grad_(False)

        logits = model(xb).view(-1)
        if Y is not None:
            yb = Y[idx].detach().float().view_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
        else:
            loss = -logits.mean()

        if xb.grad is not None:
            xb.grad.zero_()
        loss.backward()
        sal = xb.grad.detach().abs().mean(dim=0)  # (P,)
        sal = (sal - sal.mean()) / (sal.std() + 1e-8)
        order = torch.argsort(-sal)
    return torch.sort(order[:top_k]).values.detach().cpu().numpy()


def init_feature_bank(
        X_np: np.ndarray,
        y_np: np.ndarray,
        k_max: int,
        mode: str,
        seed: int = 0,
        use_patch_bank: bool = False,
        patch_groups=None
    ) -> np.ndarray:

    mode = (mode or "none").lower()
    if use_patch_bank:
        P = int(X_np.shape[1])
        k_max = int(min(k_max, P))
        if mode == "random":
            return select_feature_bank_random(P, k_max, seed=seed)
        if mode == "chi2":
            return select_feature_bank_chi2_pixels(X_np, y_np, k_max)
        # mix
        k_chi = int(0.8 * k_max)
        fb_chi = select_feature_bank_chi2_pixels(X_np, y_np, k_chi)
        pool = np.setdiff1d(np.arange(P), fb_chi)
        rng = np.random.default_rng(seed)
        rng.shuffle(pool)
        fb_mix = np.concatenate([fb_chi, pool[:max(0, k_max - k_chi)]])
        return np.sort(fb_mix)

    # pixels
    D = X_np.shape[1]
    k_max = int(min(k_max, D))
    if mode == "random":
        return select_feature_bank_random(D, k_max, seed=seed)
    if mode == "chi2":
        if (D == 784) and (patch_groups is not None):
             return select_feature_bank_chi2_patches(X_np, y_np, k_max, patch_groups)
        return select_feature_bank_chi2_pixels(X_np, y_np, k_max)

    # mix default
    k_chi = int(0.8 * k_max)
    fb_chi = select_feature_bank_chi2_pixels(X_np, y_np, k_chi)

    if (D == 784) and (patch_groups is not None):
        fb_chi = select_feature_bank_chi2_patches(X_np, y_np, k_chi, patch_groups)
    else:
        fb_chi = select_feature_bank_chi2_pixels(X_np, y_np, k_chi)

    pool = np.setdiff1d(np.arange(D), fb_chi)
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    fb_mix = np.concatenate([fb_chi, pool[:max(0, k_max - k_chi)]])
    return np.sort(fb_mix)
