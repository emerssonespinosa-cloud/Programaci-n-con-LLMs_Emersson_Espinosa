import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_classification

# --- 1. SOLUCIÓN: Función clasificar_congestion ---
def clasificar_congestion(df: pd.DataFrame, target_col: str, n_splits: int = 5) -> dict:
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Calcular pesos de clase sobre todo el conjunto
    clases = np.unique(y)
    pesos = compute_class_weight(class_weight='balanced', classes=clases, y=y)
    peso_map = dict(zip(clases.astype(int), pesos))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    precisiones, recalls, f1s, aucs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 1. Imputar NaNs (ajuste local al fold)
        imputer = SimpleImputer(strategy='mean')
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        # 2. Escalar (ajuste local al fold)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # 3. Mapeo de pesos para el ajuste del modelo
        pesos_muestra = np.where(y_tr == 1, peso_map[1], peso_map[0])

        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(X_tr, y_tr, sample_weight=pesos_muestra)

        # 4. Predicciones y métricas
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

# --- 2. GENERADOR DEL CASO DE USO ---
def generar_caso_de_uso_clasificar_congestion():
    # Simulamos un dataset de tráfico con desbalance
    X, y = make_classification(n_samples=500, n_features=5, weights=[0.9, 0.1], random_state=42)
    df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    df["congestion"] = y
    # Inyectar algunos nulos
    df.iloc[0:10, 0] = np.nan 
    return {"df": df, "target_col": "congestion", "n_splits": 5}

# --- 3. BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":
    params = generar_caso_de_uso_clasificar_congestion()
    resultado = clasificar_congestion(**params)
    
    print("--- Resultado del Pipeline ---")
    for k, v in resultado.items():
        print(f"{k}: {v}")
