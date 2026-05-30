import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# SOLO define la función que te piden
def predecir_dificultad(df, target_col):
    # 1. Separar
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Imputar
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 3. Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    
    # 4. Entrenar
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Predecir
    y_pred = clf.predict(X_test)
    
    # 6. Calcular métricas
    correctas = int(np.sum(y_pred == y_test))
    incorrectas = int(np.sum(y_pred != y_test))
    accuracy = float(accuracy_score(y_test, y_pred))
    
    return {
        'accuracy': accuracy, 
        'correctas': correctas, 
        'incorrectas': incorrectas
    }

# IMPORTANTE: NO incluyas el generador aquí, ni bloques if __name__ == "__main__"
# si el corrector te dice que falla al ejecutarlo.
