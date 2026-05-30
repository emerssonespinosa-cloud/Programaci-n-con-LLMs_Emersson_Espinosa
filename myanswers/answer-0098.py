import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def predecir_dificultad(df=None, target_col=None):
    if df is None:
        np.random.seed(42)
        n_rows = 10
        df = pd.DataFrame(
            np.random.randn(n_rows, 3),
            columns=['feature_0', 'feature_1', 'feature_2']
        )
        df['target_variable'] = np.random.randint(0, 2, size=n_rows)
        target_col = 'target_variable'

    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    return (X_imputed, y)
