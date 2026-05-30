import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --- 1. SOLUCIÓN ---
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
    
    # Calcular correctas e incorrectas
    correctas = np.sum(y_pred == y_test)
    incorrectas = np.sum(y_pred != y_test)
    
    # Calcular accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'accuracy': float(accuracy), 
        'correctas': int(correctas), 
        'incorrectas': int(incorrectas)
    }

# --- 2. GENERADOR (Estructura necesaria para el corrector) ---
def generar_caso_de_uso_predecir_dificultad():
    n_rows = 30
    n_features = 3
    data = np.random.randn(n_rows, n_features)
    df = pd.DataFrame(data, columns=['calorias', 'tiempo_prep_min', 'num_ingredientes'])
    
    # Introducir NaNs
    mask = np.random.choice([True, False], size=df.shape, p=[0.2, 0.8])
    df[mask] = np.nan
    
    # Columna target
    df['dificultad'] = np.random.randint(0, 2, size=n_rows)
    
    input_data = {'df': df, 'target_col': 'dificultad'}
    # El corrector espera el input y el resultado esperado
    expected_output = predecir_dificultad(df, 'dificultad')
    
    return input_data, expected_output

# --- 3. Ejecución ---
if __name__ == "__main__":
    input_data, _ = generar_caso_de_uso_predecir_dificultad()
    resultado = predecir_dificultad(input_data['df'], input_data['target_col'])
    print(resultado)
