import numpy as np
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from rl_and_qml_in_clinical_images.rl.env import find_threshold
from rl_and_qml_in_clinical_images.rl.rl_config import Config

def _eval_auc_sens(model, X, y):
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        # decision_function -> sigmoid-ish ranking
        s = model.decision_function(X)
        prob = (s - s.min()) / (s.max() - s.min() + 1e-9)

    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.5
    pred = (prob >= 0.5).astype(int)
    sens = recall_score(y, pred, zero_division=0)
    return float(auc), float(sens)




def kfold_baselines_calibrated(X, Y, cfg: Config, seed=0, n_splits=5):
    """
    Publication fix:
    Baselines must be evaluated with the same threshold policy:
    - either report SENS@0.5 AND SENS@thr* (thr* calibrated on fold-train),
    - or only SENS@thr* (recommended).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, recall_score

    y = Y.reshape(-1).astype(int)
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))

    models = {
        "LR": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "SVM": Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True, class_weight="balanced"))]),
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed),
    }

    def _prob(m, X):
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X)[:, 1]
        s = m.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    out = {}
    for name, m in models.items():
        aucs, sens05, sens_star, thrs = [], [], [], []
        for tr, va in skf.split(X, y):
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]
            m.fit(Xtr, ytr)
            p_tr = _prob(m, Xtr)
            p_va = _prob(m, Xva)
            thr_star = find_threshold(ytr, p_tr, mode=str(cfg.thr_mode), sens_target=float(cfg.sens_target), grid_size=cfg.grid_size)
            auc = roc_auc_score(yva, p_va) if len(np.unique(yva)) > 1 else 0.5
            s05 = recall_score(yva, (p_va >= 0.5).astype(int), zero_division=0)
            ss  = recall_score(yva, (p_va >= thr_star).astype(int), zero_division=0)
            aucs.append(float(auc)); sens05.append(float(s05)); sens_star.append(float(ss)); thrs.append(float(thr_star))
        out[name] = {
            "aucs": aucs,
            "sens@0.5": sens05,
            "sens@thr*": sens_star,
            "thr_star": thrs,
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "sens05_mean": float(np.mean(sens05)), "sens05_std": float(np.std(sens05)),
            "sens_star_mean": float(np.mean(sens_star)), "sens_star_std": float(np.std(sens_star)),
        }
    return out