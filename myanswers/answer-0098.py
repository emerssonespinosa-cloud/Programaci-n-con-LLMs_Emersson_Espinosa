import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random

def predecir_dificultad(df=None, target_col=None):
    """
    Entrena un DecisionTreeClassifier para predecir la dificultad de las recetas.
    Imputa NaNs con la media y reporta accuracy.
    
    Args:
        df (pd.DataFrame): DataFrame con características y target.
        target_col (str): Nombre de la columna target (dificultad).
        
    Returns:
        tuple: (X_imputed, y) arrays procesados
    """
    if df is None:
        np.random.seed(42)
        n_rows = 10
        df = pd.DataFrame(
            np.random.randn(n_rows, 3),
            columns=['feature_0', 'feature_1', 'feature_2']
        )
