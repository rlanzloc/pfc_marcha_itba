import os
import pandas as pd

# Carpeta de trabajo
folder_path = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/pasadas/pasadas_sin_proteccion"

# 1. Identificar archivos del pie izquierdo
archivos_izq = [f for f in os.listdir(folder_path) 
               if f.endswith(".csv") and "izquierda" in f.lower()]


def detect_delimiter(file_path, sample_size=5):
    delimiters = [',', ';', '\t']
    with open(file_path, 'r') as f:
        sample = [f.readline() for _ in range(sample_size)]
    
    for delimiter in delimiters:
        counts = [line.count(delimiter) for line in sample]
        if all(count == counts[0] for count in counts) and counts[0] > 0:
            return delimiter
    return ','  # Por defecto si no se detecta

# 2. Función para invertir sensores
def invertir_sensores(df):
    sensores = [f"Izquierda_S{i}" for i in range(1, 9)]
    
    if all(col in df.columns for col in sensores):
        # Invertir valores manteniendo nombres originales
        df[sensores] = df[sensores].iloc[:, ::-1].values
        return df
    else:
        print(f"Advertencia: Columnas de sensores no encontradas en {archivo}")
        return df

# 3. Procesar y sobrescribir archivos
for archivo in archivos_izq:
    filepath = os.path.join(folder_path, archivo)
    
    try:
        delimiter = detect_delimiter(filepath)
        df = pd.read_csv(filepath, delimiter=delimiter) 
        # Invertir sensores
        df = invertir_sensores(df)
        
        # Sobrescribir archivo original
        df.to_csv(filepath, index=False, sep=";")
        print(f"Archivo procesado y sobrescrito: {archivo}")
        
    except Exception as e:
        print(f"Error procesando {archivo}: {str(e)}")

print("\nProceso completado. Archivos izquierdos actualizados con sensores invertidos.")