import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

def clasificar_congestion(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5
) -> dict:
    """
    Pipeline de clasificación con clases desbalanceadas.
    
    Args:
        df (pd.DataFrame): DataFrame con features numéricas y columna objetivo binaria.
        target_col (str): Nombre de la columna objetivo (0=normal, 1=congestión crítica).
        n_splits (int): Número de folds para StratifiedKFold. Por defecto 5.
        
    Returns:
        dict: {
            'precision_media': float,
            'recall_medio': float,
            'f1_medio': float,
            'roc_auc_medio': float,
            'pesos_clase': {0: float, 1: float}
        }
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Calcular pesos de clase con compute_class_weight (antes del loop)
    clases = np.unique(y)
    pesos = compute_class_weight(class_weight='balanced', classes=clases, y=y)
    peso_map = dict(zip(clases.astype(int), pesos))

    skf = StratifiedKFold(n_splits=n_splits)
    precisiones, recalls, f1s, aucs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Imputación (fit solo en train)
        imputer = SimpleImputer(strategy='mean')
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        # Escalado (fit solo en train)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Pesos por muestra
        pesos_muestra = np.where(y_tr == 1, peso_map[1], peso_map[0])

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        clf.fit(X_tr, y_tr, sample_weight=pesos_muestra)

        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]

        precisiones.append(precision_score(y_te, y_pred, zero_division=0))
        recalls.append(recall_score(y_te, y_pred, zero_division=0))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_te, y_proba))

    return {
        "precision_media": float(np.mean(precisiones)),
        "recall_medio":    float(np.mean(recalls)),
        "f1_medio":        float(np.mean(f1s)),
        "roc_auc_medio":   float(np.mean(aucs)),
        "pesos_clase":     {int(k): float(v) for k, v in peso_map.items()}
    }
