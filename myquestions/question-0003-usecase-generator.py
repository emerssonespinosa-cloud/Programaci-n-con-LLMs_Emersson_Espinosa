import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def generar_caso_de_uso_evaluar_modelo():
    """
    Genera datos sintéticos de biomoléculas basados en tensores de inercia 
    y geometría espacial para clasificación estructural.
    """
    # 1. Definir tamaño de la muestra aleatoria
    n_muestras = np.random.randint(100, 200)
    
    # 2. Generar datos por categorías físicas
    # Globulares: Momentos de inercia casi iguales
    n_glob = n_muestras // 3
    globulares = {
        'id_molecula': [f'GLOB_{i}' for i in range(n_glob)],
        'masa_daltons': np.random.uniform(10000, 50000, n_glob),
        'radio_giro_nm': np.random.uniform(2, 5, n_glob),
        'momento_x': (m := np.random.uniform(100, 150, n_glob)),
        'momento_y': m + np.random.normal(0, 5, n_glob),
        'momento_z': m + np.random.normal(0, 5, n_glob),
        'longitud_secuencia': np.random.randint(100, 500, n_glob),
        'clase_real': ['Globular'] * n_glob
    }

    # Fibrosas: Un momento mucho mayor que otros
    n_fib = n_muestras // 3
    fibrosas = {
        'id_molecula': [f'FIBR_{i}' for i in range(n_fib)],
        'masa_daltons': np.random.uniform(30000, 80000, n_fib),
        'radio_giro_nm': np.random.uniform(10, 30, n_fib),
        'momento_x': np.random.uniform(500, 1000, n_fib),
        'momento_y': np.random.uniform(10, 50, n_fib),
        'momento_z': np.random.uniform(10, 50, n_fib),
        'longitud_secuencia': np.random.randint(600, 1500, n_fib),
        'clase_real': ['Fibrosa'] * n_fib
    }

    # Membrana: Dos momentos grandes y uno pequeño
    n_memb = n_muestras - n_glob - n_fib
    membrana = {
        'id_molecula': [f'MEMB_{i}' for i in range(n_memb)],
        'masa_daltons': np.random.uniform(50000, 120000, n_memb),
        'radio_giro_nm': np.random.uniform(5, 15, n_memb),
        'momento_x': (m2 := np.random.uniform(300, 600, n_memb)),
        'momento_y': m2 + np.random.normal(0, 20, n_memb),
        'momento_z': np.random.uniform(50, 100, n_memb),
        'longitud_secuencia': np.random.randint(300, 800, n_memb),
        'clase_real': ['Membrana'] * n_memb
    }

    df_input = pd.concat([pd.DataFrame(globulares), 
                          pd.DataFrame(fibrosas), 
                          pd.DataFrame(membrana)]).sample(frac=1).reset_index(drop=True)

    input_dict = {
        'dataset': df_input,
        'features': ['masa_daltons', 'radio_giro_nm', 'momento_x', 'momento_y', 'momento_z', 'longitud_secuencia'],
        'target': 'clase_real'
    }

    # Cálculo del índice de asimetría
    m_x, m_y, m_z = df_input['momento_x'], df_input['momento_y'], df_input['momento_z']
    numerador = (m_x - m_y)**2 + (m_y - m_z)**2 + (m_z - m_x)**2
    denominador = (m_x + m_y + m_z)**2
    df_input['indice_asimetria'] = numerador / denominador
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df_input['clase_real'])
    
    output = {
        'df_procesado': df_input,
        'target_vector': y_encoded,
        'clases_mapeadas': dict(zip(le.classes_, le.transform(le.classes_)))
    }

    return input_dict, output

if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_evaluar_modelo()
    print(f"Mapeo de clases: {salida['clases_mapeadas']}")
