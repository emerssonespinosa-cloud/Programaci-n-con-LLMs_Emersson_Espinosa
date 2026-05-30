import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# --- 1. SOLUCIÓN: Función analiar_churn_gradient_boosting ---
def analizar_churn_gradient_boosting(X, y):
    """
    Entrena un GradientBoostingClassifier y extrae importancia de características.
    """
    # Dividir datos 80% train / 20% test con random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Instanciar y entrenar modelo (semilla fija para reproducibilidad)
    modelo = GradientBoostingClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    predicciones = modelo.predict(X_test)
    
    # Extraer importancia de las características
    importancias = modelo.feature_importances_
    
    # Retornar tupla con los dos arrays
    return predicciones, importancias

# --- 2. GENERADOR DEL CASO DE USO ---
def generar_caso_de_uso_analizar_churn_gradient_boosting(n_muestras=200, n_caracteristicas=5):
    X, y = make_classification(
        n_samples=n_muestras, 
        n_features=n_caracteristicas, 
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.7, 0.3]
    )
    return {"X": X, "y": y}, (X, y)

# --- 3. BLOQUE DE EJECUCIÓN (Prueba) ---
if __name__ == "__main__":
    # Generar datos
    input_dict, _ = generar_caso_de_uso_analizar_churn_gradient_boosting(150, 4)
    
    # Llamar a tu función de solución
    preds, importancias = analizar_churn_gradient_boosting(input_dict["X"], input_dict["y"])
    
    print("--- Resultados del Análisis ---")
    print(f"Predicciones (primeras 5): {preds[:5]}")
    print(f"Importancia de las características: {importancias}")
    print(f"Forma del vector de importancia: {importancias.shape}")
