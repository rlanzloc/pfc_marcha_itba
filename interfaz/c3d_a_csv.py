import numpy as np
import pandas as pd
import kineticstoolkit.lab as ktk

def convertir_c3d_a_csv(file_path, output_csv_path):
    import numpy as np
    import kineticstoolkit as ktk

    # Lee el archivo C3D usando KineticsToolkit
    c3d_data = ktk.read_c3d(file_path)

    # Extrae el objeto 'Points' del C3D
    markers = c3d_data["Points"]  # Obtiene el objeto TimeSeries con los puntos
    time = markers.time  # Accede a los tiempos

    print(markers.data)

    # Extrae los datos de los marcadores
    marker_data = markers.data  # Los datos de los marcadores están en markers.data

    # Obtener todas las claves de los datos de los marcadores (por ejemplo, 'X', 'Y', 'Z', etc.)
    keys = marker_data.keys()

    print(keys)

    # Crear una lista de las coordenadas para cada clave en markers.data
    coordinates = [marker_data[key] for key in keys]
    print(coordinates)

    concatenated_data = np.vstack(coordinates)

    # Verificar si hay NaN en el array
    nan_indices = np.isnan(concatenated_data)

    # Si hay NaN, se devuelve un array de booleanos con True en las posiciones donde hay NaN
    print("NaN en las posiciones (True significa NaN):")
    print(nan_indices)

    # Mostrar las posiciones de los NaNs en los datos
    nan_positions = np.where(nan_indices)
    print("Posiciones de los NaNs:")
    print(nan_positions[0])

    for i in range(len(nan_positions[0])):
        
        # Obtener los tiempos correspondientes a los NaNs
        nan_times = time[nan_positions[i]]  # Usamos las posiciones válidas para obtener los tiempos

    print("Tiempos correspondientes a los NaNs:")
    print(nan_times)

    # Combina las coordenadas de todas las claves en un solo array
    marker_data_combined = np.column_stack(coordinates)
    return marker_data_combined
    


# Ejemplo de uso
file_path = "C:/Users/Rashel Lanz Lo Curto/Downloads/Caminata_012_cortado.c3d"  # Cambia por la ruta de tu archivo C3D
output_csv_path = 'output_0.csv'  # Cambia por la ruta de salida de tu archivo CSV
convertir_c3d_a_csv(file_path, output_csv_path)


 # Lee el archivo C3D usando KineticsToolkit
c3d_data = ktk.read_c3d(file_path)
    
    # Extrae el objeto 'Points' del C3D
markers = c3d_data["Points"]  # Obtiene el objeto TimeSeries con los puntos
time = markers.time  # Accede a los tiempos
    
    
    # Extrae los datos de los marcadores
marker_data = markers.data  # Los datos de los marcadores están en markers.data

