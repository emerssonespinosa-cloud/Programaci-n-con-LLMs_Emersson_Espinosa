import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random

# -------------------------------------------------------------
# Función para predecir dificultad
# -------------------------------------------------------------
def predecir_dificultad(df, target_col):
    """
    Entrena un DecisionTreeClassifier para predecir la dificultad de las recetas.
    Imputa NaNs con la media y reporta accuracy.
    
    Args:
        df (pd.DataFrame): DataFrame con características y target.
        target_col (str): Nombre de la columna target (dificultad).
        
    Returns:
        dict: {'accuracy': float, 'correctas': int, 'incorrectas': int}
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    # 2. Imputar NaNs con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 3. Dividir en train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    # 4. Entrenar DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Predecir y evaluar
    y_pred = clf.predict(X_test)
    
    correctas = np.sum(y_pred == y_test)
    incorrectas = np.sum(y_pred != y_test)
    accuracy = correctas / len(y_test)
    
    return {'accuracy': accuracy, 'correctas': int(correctas), 'incorrectas': int(incorrectas)}

# -------------------------------------------------------------
# Generador de casos de uso aleatorios para preparar_datos
# -------------------------------------------------------------
def generar_caso_de_uso_predecir_dificultad():
 
    # 1. Configuración aleatoria
    n_rows = random.randint(5, 15)       # Entre 5 y 15 filas
    n_features = random.randint(2, 5)    # Entre 2 y 5 columnas de características
    
    # 2. Generar datos aleatorios
    data = np.random.randn(n_rows, n_features)
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_cols)
    
    # Introducir NaNs aleatorios (~10%)
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df[mask] = np.nan
    
    # Columna target
    target_col = 'target_variable'
    df[target_col] = np.random.randint(0, 2, size=n_rows)
    
    # Construir input
    input_data = {'df': df.copy(), 'target_col': target_col}
    
    # Calcular output esperado (imputar y separar X/y)
    X_expected = df.drop(columns=[target_col])
    y_expected = df[target_col].to_numpy()
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_expected)
    
    output_data = (X_imputed, y_expected)
    
    return input_data, output_data

# -------------------------------------------------------------
# Ejemplo de ejecución
# -------------------------------------------------------------
if __name__ == "__main__":
    # Generamos un caso de prueba
    entrada, salida_esperada = generar_caso_de_uso_predecir_dificultad()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Target Column: {entrada['target_col']}")
    print("DataFrame (primeras 5 filas con posibles NaNs):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (Tupla de arrays) ===")
    X_res, y_res = salida_esperada
    print(f"Shape de X procesada: {X_res.shape}")
    print(f"Shape de y: {y_res.shape}")
    print("Ejemplo de primera fila escalada/imputada:", X_res[0])
    
    # Ejemplo rápido de predecir dificultad
    resultado = predecir_dificultad(entrada['df'], entrada['target_col'])
    print("\n=== Resultado predecir_dificultad ===")
    print(resultado)
