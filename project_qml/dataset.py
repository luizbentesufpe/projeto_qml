import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from medmnist import ChestMNIST
from rl.rl_config import Config
from features import make_patch_groups, patchify_mean

def get_balanced_subset(dataset, percentage=3, seed=0):
    rng = np.random.default_rng(seed)
    labels = dataset.labels
    y_bin = (labels.sum(axis=1) > 0).astype(np.int32)
    N = len(dataset)
    subset_size = max(int(N * (percentage / 100.0)), 2)

    idx_pos = np.where(y_bin == 1)[0]
    idx_neg = np.where(y_bin == 0)[0]

    k_pos = min(len(idx_pos), subset_size // 2)
    k_neg = min(len(idx_neg), subset_size - k_pos)

    sel_pos = rng.choice(idx_pos, k_pos, replace=False) if k_pos > 0 else np.array([], dtype=int)
    sel_neg = rng.choice(idx_neg, k_neg, replace=False) if k_neg > 0 else np.array([], dtype=int)

    sel = np.concatenate([sel_pos, sel_neg])
    if len(sel) == 0:
        sel = rng.choice(np.arange(N), subset_size, replace=False)

    rng.shuffle(sel)
    return Subset(dataset, sel)


def _dataset_to_arrays(ds, batch_size: int, use_patch_bank: bool, compact: bool, patch_size: int, patch_stride: int):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    X, Y = [], []
    for xb, yb in dl:
        B = xb.size(0)
        xflat = xb.view(B, -1).float()  # (B,784)
        if yb.ndim == 3:
            yb = yb.view(yb.size(0), -1)  # (B,14)
        y_bin = (yb.sum(dim=1) > 0).float().unsqueeze(1)  # (B,1)
        X.append(xflat)
        Y.append(y_bin)

    X = torch.cat(X, 0).cpu().numpy().astype(np.float32)
    Y = torch.cat(Y, 0).cpu().numpy().astype(np.float32)

    return X, Y


def load_chestmnist_pool_flatten(cfg: Config, percent_total: int = 100, seed: int = 0):
    """
    CORRETO (para seu protocolo de holdout próprio):
    - Carrega train/val/test completos
    - Concatena tudo em um pool único
    - (Opcional) subamostra UMA vez no pool inteiro (estratificado)
    - Retorna X_all, Y_all
    """
    tf = T.Compose([T.ToTensor()])

    tr = ChestMNIST(split="train", download=True, transform=tf, size=28)
    va = ChestMNIST(split="val",   download=True, transform=tf, size=28)
    te = ChestMNIST(split="test",  download=True, transform=tf, size=28)

    Xtr, Ytr = _dataset_to_arrays(
        tr, batch_size=int(cfg.batch_size),
        use_patch_bank=bool(getattr(cfg, "use_patch_bank", False)),
        compact=bool(getattr(cfg, "patch_bank_compact_features", False)),
        patch_size=int(getattr(cfg, "patch_size", 4)),
        patch_stride=int(getattr(cfg, "patch_stride", 4)),
    )
    Xva, Yva = _dataset_to_arrays(
        va, batch_size=int(cfg.batch_size),
        use_patch_bank=bool(getattr(cfg, "use_patch_bank", False)),
        compact=bool(getattr(cfg, "patch_bank_compact_features", False)),
        patch_size=int(getattr(cfg, "patch_size", 4)),
        patch_stride=int(getattr(cfg, "patch_stride", 4)),
    )
    Xte, Yte = _dataset_to_arrays(
        te, batch_size=int(cfg.batch_size),
        use_patch_bank=bool(getattr(cfg, "use_patch_bank", False)),
        compact=bool(getattr(cfg, "patch_bank_compact_features", False)),
        patch_size=int(getattr(cfg, "patch_size", 4)),
        patch_stride=int(getattr(cfg, "patch_stride", 4)),
    )

    X_all = np.concatenate([Xtr, Xva, Xte], axis=0)
    Y_all = np.concatenate([Ytr, Yva, Yte], axis=0)

    # Subamostra UMA vez no pool inteiro (opcional)
    percent_total = int(percent_total)
    if percent_total >= 100:
        return X_all, Y_all

    rng = np.random.default_rng(int(seed))
    y = (Y_all.reshape(-1) > 0.5).astype(np.int32)
    N = int(len(y))
    n_keep = int(max(10, round(N * (percent_total / 100.0))))

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    k_pos = int(min(len(idx_pos), n_keep // 2))
    k_neg = int(min(len(idx_neg), n_keep - k_pos))
    sel = np.concatenate([idx_pos[:k_pos], idx_neg[:k_neg]])
    if sel.size < 10:
        sel = rng.choice(np.arange(N), size=n_keep, replace=False)
    rng.shuffle(sel)

    return X_all[sel], Y_all[sel]



def load_chestmnist_flatten(cfg: Config, percentage_each_split=3, seed=0):
    tf = T.Compose([T.ToTensor()])
    tr = ChestMNIST(split="train", download=True, transform=tf, size=28)
    va = ChestMNIST(split="val",   download=True, transform=tf, size=28)
    te = ChestMNIST(split="test",  download=True, transform=tf, size=28)

    tr = get_balanced_subset(tr, percentage_each_split, seed=seed)
    va = get_balanced_subset(va, percentage_each_split, seed=seed+1)
    te = get_balanced_subset(te, percentage_each_split, seed=seed+2)

    groups = None
    if bool(getattr(cfg, "use_patch_bank", False)) and bool(getattr(cfg, "patch_bank_compact_features", False)):
        groups = make_patch_groups(28, 28, int(cfg.patch_size), int(cfg.patch_stride))


    def to_arrays(ds):
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        X, Y = [], []
        for xb, yb in dl:
            B = xb.size(0)
            xflat = xb.view(B, -1).float()   # (B,784)
            if yb.ndim == 3:
                yb = yb.view(yb.size(0), -1)  # (B,14)
            y_bin = (yb.sum(dim=1) > 0).float().unsqueeze(1)  # (B,1)
            X.append(xflat)
            Y.append(y_bin)
        X = torch.cat(X, 0).cpu().numpy()
        Y = torch.cat(Y, 0).cpu().numpy()

        # ==========================
        # Patch-bank (publication fix):
        # If enabled + compact_features, convert pixels->patch means
        # so encoder indices match feature_bank domain (P patches).
        # ==========================
        # if bool(cfg.use_patch_bank) and bool(cfg.patch_bank_compact_features):
        #     # groups = make_patch_groups(28, 28, int(cfg.patch_size), int(cfg.patch_stride))
        #     # groups already built once above (consistent across splits)
        #     X = patchify_mean(X.astype(np.float32), groups=groups).astype(np.float32)
        # else:
        #     X = X.astype(np.float32)
        X = X.astype(np.float32)
        return X, Y

    return to_arrays(tr), to_arrays(va), to_arrays(te)

def _balanced_subset_from_arrays(X: np.ndarray, Y: np.ndarray, percentage: int, seed: int = 0):
    """
    Build a balanced subset (pos/neg) directly from numpy arrays.
    This avoids any dataset-index mismatch and guarantees no holdout leakage
    when you pass X_train_all/Y_train_all.
    """
    rng = np.random.default_rng(int(seed))
    y = (Y.reshape(-1) > 0.5).astype(np.int32)
    N = int(len(y))
    subset_size = int(max(2, round(N * (float(percentage) / 100.0))))

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)

    k_pos = int(min(len(idx_pos), subset_size // 2))
    k_neg = int(min(len(idx_neg), subset_size - k_pos))
    sel = np.concatenate([idx_pos[:k_pos], idx_neg[:k_neg]])
    if sel.size < 2:
        sel = rng.choice(np.arange(N), subset_size, replace=False)
    rng.shuffle(sel)
    return X[sel], Y[sel]


def _make_search_splits_from_train_all(
    X_train_all: np.ndarray,
    Y_train_all: np.ndarray,
    percent_search: int,
    seed: int = 0,
    val_frac: float = 0.20,
):
    """
    Build RL-search splits from TRAIN_ALL only (holdout-safe).
    Returns:
    (XtrS, YtrS), (XvaS, YvaS)
    - Stratified by binary label.
    - Uses a small, balanced-ish subset size determined by percent_search.
    - Deterministic by seed.
    """
     # 1) take a balanced subset from TRAIN_ALL
    Xs, Ys = _balanced_subset_from_arrays(X_train_all, Y_train_all, percentage=int(percent_search), seed=int(seed))
    y = (Ys.reshape(-1) > 0.5).astype(np.int32)

    # 2) stratified split subset -> (tr, val)
    # guard: if subset has only one class, fallback to simple split
    if len(np.unique(y)) < 2 or len(y) < 4:
        n_val = int(max(1, round(float(val_frac) * len(y))))
        XvaS, YvaS = Xs[:n_val], Ys[:n_val]
        XtrS, YtrS = Xs[n_val:], Ys[n_val:]
        if len(XtrS) < 2:
            XtrS, YtrS = Xs, Ys
            XvaS, YvaS = Xs[:1], Ys[:1]
        return (XtrS, YtrS), (XvaS, YvaS)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(val_frac), random_state=int(seed))
    tr_idx, va_idx = next(sss.split(Xs, y))
    XtrS, YtrS = Xs[tr_idx], Ys[tr_idx]
    XvaS, YvaS = Xs[va_idx], Ys[va_idx]
    return (XtrS, YtrS), (XvaS, YvaS)