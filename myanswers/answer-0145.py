import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def entrenar_clasificador(df, target_col):
    """
    Entrena un clasificador de Regresión Logística para predecir la variable objetivo.
    
    Args:
        df (pd.DataFrame): DataFrame con características numéricas y target.
        target_col (str): Nombre de la columna objetivo.
        
    Returns:
        float: Accuracy del modelo en el conjunto de prueba.
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Dividir en train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Escalar características con StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Entrenar Regresión Logística
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train_scaled, y_train)
    
    # 5. Predecir y calcular accuracy
    y_pred = modelo.predict(X_test_scaled)
    accuracy = float(accuracy_score(y_test, y_pred))
    
    return accuracy
