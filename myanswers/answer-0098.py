import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
    
    # Calcular métricas con numpy
    correctas = int(np.sum(y_pred == y_test))
    incorrectas = int(np.sum(y_pred != y_test))
    accuracy = float(accuracy_score(y_test, y_pred))
    
    return {
        'accuracy': accuracy, 
        'correctas': correctas, 
        'incorrectas': incorrectas
    }
