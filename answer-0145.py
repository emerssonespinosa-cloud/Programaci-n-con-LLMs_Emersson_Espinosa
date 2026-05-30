import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 1. Definición del Caso de Uso ---
def generar_caso_de_uso_entrenar_clasificador():
    iris = load_iris()
    X = iris.data
    y = iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["target"] = y
    df = df.sample(frac=1).reset_index(drop=True)
    ruido = np.random.normal(0, 0.1, df.iloc[:, :-1].shape)
    df.iloc[:, :-1] = df.iloc[:, :-1] + ruido
    n_drop = np.random.randint(0, 10)
    if n_drop > 0:
        df = df.iloc[:-n_drop]
    input_data = {"df": df.copy(), "target_col": "target"}
    return input_data

# --- 2. Definición de tu Solución (entrenar_clasificador) ---
def entrenar_clasificador(df, target_col):
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Dividir datos (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_scaled, y_train)
    
    # Predecir y calcular accuracy
    y_pred = modelo.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)

# --- 3. Ejecución (Integración) ---
if __name__ == "__main__":
    # Obtener datos del caso de uso
    datos = generar_caso_de_uso_entrenar_clasificador()
    
    # Probar la solución
    precision = entrenar_clasificador(datos["df"], datos["target_col"])
    
    print(f"--- Resultado del Reto ---")
    print(f"Precisión (Accuracy) obtenida: {precision:.4f}")
