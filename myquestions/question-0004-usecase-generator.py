import pandas as pd
import numpy as np

def generar_caso_de_uso_optimizar_proveedores():
    """
    Genera casos de prueba aleatorios para la optimización de proveedores 
    de materia prima en la síntesis de polímeros.
    """
    # 1. Configuración aleatoria de la muestra
    n_proveedores = np.random.randint(50, 100)
    paises = ['Colombia', 'Brasil', 'EE.UU.', 'China', 'Alemania']
    
    # 2. Generación de datos sintéticos
    ids = [f'PROV_{i:03d}' for i in range(n_proveedores)]
    origenes = np.random.choice(paises, n_proveedores, p = [0.3, 0.2, 0.2, 0.2, 0.1])
    
    precios = np.random.uniform(500, 2000, n_proveedores)
    purezas = np.random.uniform(90, 99.9, n_proveedores) 
    
    # Lógica de distancia y envío basada en el origen
    distancias = []
    envios = []
    for pais in origenes:
        if pais == 'Colombia':
            distancias.append(np.random.uniform(10, 500))
            envios.append(np.random.uniform(50, 200))
        elif pais == 'Brasil':
            distancias.append(np.random.uniform(3000, 5000))
            envios.append(np.random.uniform(1000, 3000))
        else: # Internacional lejano
            distancias.append(np.random.uniform(8000, 15000))
            envios.append(np.random.uniform(4000, 8000))

    df_input = pd.DataFrame({
        'id_proveedor': ids,
        'pais_origen': origenes,
        'precio_unitario_usd': precios,
        'pureza_reactivo': purezas,
        'costo_envio_usd': envios,
        'distancia_km': distancias,
        'tiempo_entrega_dias': np.random.randint(2, 60, n_proveedores)
    })

    # 3. Definición del Input (para la función de respuesta)
    input_dict = {
        'dataset': df_input,
        'min_pureza': 95.0,
        'descuento_local': 0.15
    }

    # 4. Cálculo del Output Esperado (Lógica de negocio)
    # Filtrado por pureza técnica
    df_valid = df_input[df_input['pureza_reactivo'] >= 95.0].copy()
    
    # Aplicar descuento local si es Colombia
    df_valid['envio_final'] = np.where(
        df_valid['pais_origen'] == 'Colombia',
        df_valid['costo_envio_usd'] * (1 - 0.15),
        df_valid['costo_envio_usd']
    )
    
    # Cálculo del factor de cercanía (FC) y costo ajustado
    df_valid['FC'] = 1 + np.log10(df_valid['distancia_km'] + 1)
    df_valid['costo_total_ajustado'] = (df_valid['precio_unitario_usd'] + df_valid['envio_final']) * df_valid['FC']
    
    # Índice de Viabilidad (IV): Relación pureza/costo
    df_valid['IV'] = (df_valid['pureza_reactivo'] * 10) / df_valid['costo_total_ajustado']
    mejor_id = df_valid.loc[df_valid['IV'].idxmax(), 'id_proveedor']
    
    output = {
        'mejor_proveedor_id': mejor_id,
        'n_proveedores_aptos': int(len(df_valid)),
        'promedio_pureza_apta': float(df_valid['pureza_reactivo'].mean())
    }

    return input_dict, output

# --- Ejemplo de ejecución (Corregido para este archivo específico) ---
if __name__ == "__main__":
    # Importante: Aquí se llama a la función de este archivo
    params, resultados = generar_caso_de_uso_optimizar_proveedores()
    print(f"Total proveedores analizados: {len(params['dataset'])}")
    print(f"Proveedores que cumplen calidad (>95%): {resultados['n_proveedores_aptos']}")
    print(f"Ganador de la licitación: {resultados['mejor_proveedor_id']}")
    print(f"Pureza promedio de aptos: {resultados['promedio_pureza_apta']:.2f}%")
