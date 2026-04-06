import pandas as pd
import numpy as np

def generar_caso_de_uso_entrenar_modelo():
    """
    Genera casos de prueba aleatorios para la segmentación de clientes de acero.
    Crea dos grupos: B2B (grandes toneladas, baja frecuencia) y 
    B2C (pocas toneladas, alta frecuencia).
    """
    # 1. Definir tamaño de la muestra (aleatorio entre 60 y 100 clientes)
    n_b2b = np.random.randint(20, 40)
    n_b2c = np.random.randint(40, 60)
    total_clientes = n_b2b + n_b2c
    
    # 2. Generar datos para B2B (Empresas/Constructoras)
    data_b2b = {
        'id_cliente': [f'CORP_{i}' for i in range(n_b2b)],
        'tipo_metal': np.random.choice(['Acero Carbono', 'Inoxidable', 'Galvanizado'], n_b2b),
        'toneladas_pedidas': np.random.uniform(50, 200, n_b2b), # Mucho volumen
        'frecuencia_compra_mes': np.random.randint(1, 4, n_b2b), # Baja frecuencia
        'valor_contrato_usd': np.random.uniform(50000, 200000, n_b2b)
    }
    
    # 3. Generar datos para B2C (Contratistas/Detal)
    data_b2c = {
        'id_cliente': [f'CONT_{i}' for i in range(n_b2c)],
        'tipo_metal': np.random.choice(['Acero Carbono', 'Galvanizado'], n_b2c),
        'toneladas_pedidas': np.random.uniform(0.5, 5, n_b2c), # Poco volumen
        'frecuencia_compra_mes': np.random.randint(8, 20, n_b2c), # Alta frecuencia
        'valor_contrato_usd': np.random.uniform(1000, 15000, n_b2c)
    }
    
    # Unificar en un solo DataFrame y mezclar filas
    df_b2b = pd.DataFrame(data_b2b)
    df_b2c = pd.DataFrame(data_b2c)
    df_input = pd.concat([df_b2b, df_b2c]).sample(frac=1).reset_index(drop=True)
    
    # 4. Definir Input (Diccionario para la función preparar_datos/segmentar)
    input_dict = {
        'dataset': df_input,
        'features': ['toneladas_pedidas', 'frecuencia_compra_mes'],
        'target_score_parts': ['valor_contrato_usd', 'toneladas_pedidas']
    }
    
    # 5. Definir Output esperado (Lo que la lógica de negocio debería calcular)
    # Calculamos el score de valor: valor_contrato * toneladas
    df_input['score_valor'] = df_input['valor_contrato_usd'] * df_input['toneladas_pedidas']
    
    # Identificamos el top de cada grupo basado en el origen de los IDs
    b2b_mask = df_input['id_cliente'].str.contains('CORP')
    b2c_mask = df_input['id_cliente'].str.contains('CONT')
    
    top_b2b = df_input[b2b_mask].loc[df_input[b2b_mask]['score_valor'].idxmax(), 'id_cliente']
    top_b2c = df_input[b2c_mask].loc[df_input[b2c_mask]['score_valor'].idxmax(), 'id_cliente']
    
    output = {
        'B2B_top_cliente': top_b2b,
        'B2C_top_cliente': top_b2c
    }
    
    return input_dict, output

# --- Ejemplo de validación del generador ---
if __name__ == "__main__":
    params, resultados = generar_caso_de_uso_entrenar_modelo()
    print(f"Dataset generado con {len(params['dataset'])} clientes.")
    print(f"Mejor cliente B2B esperado: {resultados['B2B_top_cliente']}")
    print(f"Mejor cliente B2C esperado: {resultados['B2C_top_cliente']}")
