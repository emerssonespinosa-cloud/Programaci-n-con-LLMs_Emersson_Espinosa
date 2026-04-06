import pandas as pd
import numpy as np

def generar_caso_de_uso_entrenar_modelo():
    """
    Genera casos de prueba aleatorios para la segmentación de clientes de acero.
    Crea dos grupos: B2B (grandes toneladas, baja frecuencia) y 
    B2C (pocas toneladas, alta frecuencia).
    """
    # 1. Definir tamaño de la muestra
    n_b2b = np.random.randint(20, 40)
    n_b2c = np.random.randint(40, 60)
    
    # 2. Generar datos para B2B (Empresas/Constructoras)
    data_b2b = {
        'id_cliente': [f'CORP_{i}' for i in range(n_b2b)],
        'tipo_metal': np.random.choice(['Acero Carbono', 'Inoxidable', 'Galvanizado'], n_b2b),
        'toneladas_pedidas': np.random.uniform(50, 200, n_b2b),
        'frecuencia_compra_mes': np.random.randint(1, 4, n_b2b),
        'valor_contrato_usd': np.random.uniform(50000, 200000, n_b2b)
    }
    
    # 3. Generar datos para B2C (Contratistas/Detal)
    data_b2c = {
        'id_cliente': [f'CONT_{i}' for i in range(n_b2c)],
        'tipo_metal': np.random.choice(['Acero Carbono', 'Galvanizado'], n_b2c),
        'toneladas_pedidas': np.random.uniform(0.5, 5, n_b2c),
        'frecuencia_compra_mes': np.random.randint(8, 20, n_b2c),
        'valor_contrato_usd': np.random.uniform(1000, 15000, n_b2c)
    }
    
    df_b2b = pd.DataFrame(data_b2b)
    df_b2c = pd.DataFrame(data_b2c)
    df_input = pd.concat([df_b2b, df_b2c]).sample(frac=1).reset_index(drop=True)
    
    # 4. Definir Input
    input_dict = {
        'dataset': df_input,
        'features': ['toneladas_pedidas', 'frecuencia_compra_mes'],
        'target_score_parts': ['valor_contrato_usd', 'toneladas_pedidas']
    }
    
    # 5. Definir Output esperado
    df_input['score_valor'] = df_input['valor_contrato_usd'] * df_input['toneladas_pedidas']
    
    b2b_mask = df_input['id_cliente'].str.contains('CORP')
    b2c_mask = df_input['id_cliente'].str.contains('CONT')
    
    top_b2b = df_input[b2b_mask].loc[df_input[b2b_mask]['score_valor'].idxmax(), 'id_cliente']
    top_b2c = df_input[b2c_mask].loc[df_input[b2c_mask]['score_valor'].idxmax(), 'id_cliente']
    
    output = {
        'B2B_top_cliente': top_b2b,
        'B2C_top_cliente': top_b2c
    }
    
    return input_dict, output

# --- VALIDACIÓN (Asegúrate de llamar a la función correcta) ---
if __name__ == "__main__":
    params, resultados = generar_caso_de_uso_entrenar_modelo() # <--- CORREGIDO AQUÍ
    print(f"Dataset generado con {len(params['dataset'])} clientes.")
    print(f"Mejor cliente B2B esperado: {resultados['B2B_top_cliente']}")
