import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- 1. Generador del Caso de Uso ---
def generar_caso_de_uso_predecir_dificultad():
    n_rows = random.randint(20, 50)  # Aumentamos un poco el tamaño para mejor entrenamiento
    n_features = 3
    data = np.random.randn(n_rows, n_features)
    df = pd.DataFrame(data, columns=['calorias', 'tiempo_prep_min', 'num_ingredientes'])
    
    # Introducir NaNs aleatorios
    mask = np.random.choice([True, False], size=df.shape, p=[0.2, 0.8])
    df[mask] = np.nan
    
    # Crear columna target (dificultad: 0 o 1)
    df['dificultad'] = np.random.randint(0, 2, size=n_rows)
    
    return {'df': df, 'target_col': 'dificultad'}

# --- 2. Solución: Función predecir_dificultad ---
def predecir_dificultad(df, target_col):
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Imputar NaNs con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Dividir datos (80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    # Entrenar DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predecir
    y_pred = clf.predict(X_test)
    
    # Calcular correctas e incorrectas usando numpy
    correctas = np.sum(y_pred == y_test)
    incorrectas = np.sum(y_pred != y_test)
    
    # Calcular accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'accuracy': float(accuracy), 
        'correctas': int(correctas), 
        'incorrectas': int(incorrectas)
    }

# --- 3. Ejecución ---
if __name__ == "__main__":
    # Generar caso
    caso = generar_caso_de_uso_predecir_dificultad()
    
    # Ejecutar solución
    resultado = predecir_dificultad(caso['df'], caso['target_col'])
    
    print("--- Resultado del Análisis de Recetas ---")
    print(f"Diccionario de resultados: {resultado}")
    print(f"\nResumen: Se obtuvieron {resultado['correctas']} predicciones correctas "
          f"y {resultado['incorrectas']} incorrectas.")
