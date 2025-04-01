import os
import pandas as pd

# Carpeta de trabajo
folder_path = "C:/Users/Rashel Lanz Lo Curto/pfc_marcha_itba/analisis_cinetico/pasadas"

# 1. Identificar archivos del pie izquierdo
archivos_izq = [f for f in os.listdir(folder_path) 
               if f.endswith(".csv") and "izquierda" in f.lower()]

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
        # Leer archivo
        df = pd.read_csv(filepath, delimiter=";")
        
        # Eliminar columna Hora si existe
        if "Hora" in df.columns:
            df = df.drop(columns=["Hora"])
        
        # Invertir sensores
        df = invertir_sensores(df)
        
        # Sobrescribir archivo original
        df.to_csv(filepath, index=False, sep=";")
        print(f"Archivo procesado y sobrescrito: {archivo}")
        
    except Exception as e:
        print(f"Error procesando {archivo}: {str(e)}")

print("\nProceso completado. Archivos izquierdos actualizados con sensores invertidos.")