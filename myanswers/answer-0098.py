import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def predecir_dificultad(df, target_col):
    """
    Entrena un DecisionTreeClassifier para predecir la dificultad de las recetas.
    Imputa NaNs con la media y reporta accuracy.
    
    Args:
        df (pd.DataFrame): DataFrame con características y target.
        target_col (str): Nombre de la columna target (dificultad).
        
    Returns:
        tuple: (X_imputed, y) arrays procesados
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    # 2. Imputar NaNs con la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    return (X_imputed, y)
