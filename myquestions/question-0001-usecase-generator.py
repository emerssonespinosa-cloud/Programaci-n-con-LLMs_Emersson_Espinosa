import pandas as pd
import numpy as np

def generar_caso_de_uso_preparar_datos():
    """
    Genera casos de prueba aleatorios para una función de preparación de datos
    enfocada en la cosecha de gulupa.
    """
    # 1. Componente Aleatorio: Decidir el tamaño del dataset y valores base
    n_registros = np.random.randint(10, 50)
    factor_productividad = np.random.uniform(0.5, 1.5)
    
    # 2. Generar datos sintéticos realistas
    data = {
        'temp_promedio': np.random.uniform(15, 25, n_registros),
        'humedad_rel': np.random.uniform(60, 90, n_registros),
        'precipitacion_mm': np.random.uniform(0, 100, n_registros),
        'edad_cultivo_sem': np.random.randint(30, 60, n_registros)
    }
    
    # Creamos el target (y) con una relación lineal simple + ruido para el output esperado
    # Producción = (Temp * 10) + (Humedad * 2) - (Lluvia * 0.5) * factor
    data['produccion_kg'] = (
        (data['temp_promedio'] * 10) + 
        (data['humedad_rel'] * 2) - 
        (data['precipitacion_mm'] * 0.5)
    ) * factor_productividad + np.random.normal(0, 5, n_registros)

    df_input = pd.DataFrame(data)
    
    # 3. Definir los argumentos de entrada (Input)
    columnas_x = ['temp_promedio', 'humedad_rel', 'precipitacion_mm', 'edad_cultivo_sem']
    columna_y = 'produccion_kg'
    
    input_dict = {
        'dataset': df_input,
        'features': columnas_x,
        'target': columna_y
    }
    
    # 4. Definir el resultado esperado (Output)
    # Lo que la función preparar_datos debería devolver internamente
    expected_X = df_input[columnas_x]
    expected_y = df_input[columna_y]
    
    output = (expected_X, expected_y)
    
    return input_dict, output

# --- Ejemplo de ejecución ---
inputs, outputs = generar_caso_de_uso_preparar_datos()

print("--- INPUT (Diccionario) ---")
print(f"Número de registros generados: {len(inputs['dataset'])}")
print(f"Columnas detectadas: {inputs['features']}")

print("\n--- OUTPUT ESPERADO (X) ---")
print(outputs[0].head(3)) # Muestra las primeras 3 filas de X
