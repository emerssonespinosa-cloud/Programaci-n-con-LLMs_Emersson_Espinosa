import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_classification

def clasificar_congestion(
    df: pd.DataFrame = None,
    target_col: str = None,
    n_splits: int = 5
) -> dict:
    """
    Pipeline de clasificación con clases desbalanceadas.

    Args:
        df (pd.DataFrame): DataFrame con features numéricas y columna objetivo binaria.
        target_col (str): Nombre de la columna objetivo.
        n_splits (int): Número de folds para StratifiedKFold. Por defecto 5.

    Returns:
        dict: precision_media, recall_medio, f1_medio, roc_auc_medio, pesos_clase
    """
    if df is None:
        np.random.seed(42)
        X_raw, y_raw = make_classification(
            n_samples=500, n_features=5, n_informative=3,
            weights=[0.9, 0.1], random_state=42
        )
        df = pd.DataFrame(X_raw, columns=[f'feature_{i}' for i in range(5)])
        df['congestion'] = y_raw
        target_col = 'congestion'

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Calcular pesos de clase antes del loop
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


def generar_caso_de_uso_clasificar_congestion():
    rng = np.random.default_rng(seed=np.random.randint(0, 99999))

    n_samples    = int(rng.integers(400, 1001))
    n_features   = int(rng.integers(4, 9))
    n_informative = int(rng.integers(2, n_features))
    imbalance    = float(rng.uniform(0.06, 0.15))
    noise        = float(rng.uniform(0.0, 0.05))
    n_splits_val = int(rng.choice([3, 4, 5]))
    random_state_val = int(rng.integers(0, 999))

    feature_names = [f'feature_{i}' for i in range(n_features)]
    target_col = 'congestion'

    X_raw, y_raw = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=max(0, n_features - n_informative - 1),
        n_clusters_per_class=1,
        weights=[1 - imbalance, imbalance],
        flip_y=noise,
        random_state=random_state_val
    )

    df = pd.DataFrame(X_raw, columns=feature_names)

    # Inyectar nulos
    n_null_cols = int(rng.integers(1, min(4, n_features)))
    null_cols = rng.choice(feature_names, size=n_null_cols, replace=False)
    for col in null_cols:
        n_nulls = int(rng.integers(5, max(6, n_samples // 10)))
        null_idx = rng.choice(n_samples, size=n_nulls, replace=False)
        df.loc[null_idx, col] = np.nan

    df[target_col] = y_raw

    input_args = {
        'df': df.copy(),
        'target_col': target_col,
        'n_splits': n_splits_val
    }

    expected_output = clasificar_congestion(df.copy(), target_col, n_splits=n_splits_val)

    return input_args, expected_output


if __n
