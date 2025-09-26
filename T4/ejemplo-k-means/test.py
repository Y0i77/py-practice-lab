"""
Demostración de algoritmo K-Means para clustering de datos de viviendas en California.

Este script implementa el algoritmo K-Means para agrupar viviendas basándose en su
ubicación geográfica y valor medio. Incluye visualización de datos, normalización,
selección óptima de clusters usando el método de silueta, y visualización de resultados.

Cumple con buenas prácticas de programación y está preparado para análisis con SonarQube.
"""

# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os


def cargar_datos(ruta_archivo):
    """
    Carga los datos desde un archivo CSV y selecciona las columnas relevantes.

    Parameters:
    ruta_archivo (str): Ruta completa al archivo CSV

    Returns:
    pandas.DataFrame: DataFrame con los datos cargados y columnas seleccionadas
    """
    # Columnas a utilizar en el análisis
    columnas = ['longitude', 'latitude', 'median_house_value']

    # Leer datos desde el archivo CSV
    datos = pd.read_csv(ruta_archivo, usecols=columnas)

    return datos


def limpiar_datos(datos):
    """
    Elimina filas con valores faltantes del DataFrame.

    Parameters:
    datos (pandas.DataFrame): DataFrame con los datos a limpiar

    Returns:
    pandas.DataFrame: DataFrame limpio sin valores faltantes
    """
    # Crear copia para no modificar el DataFrame original
    datos_limpios = datos.copy()

    # Eliminar filas con valores NaN
    datos_limpios = datos_limpios.dropna()

    return datos_limpios


def visualizar_datos_originales(datos):
    """
    Crea un gráfico de dispersión de los datos originales.

    Parameters:
    datos (pandas.DataFrame): DataFrame con los datos a visualizar
    """
    # Configurar estilo de seaborn para mejores visualizaciones
    sns.set(style="whitegrid")

    # Crear figura con tamaño personalizado
    plt.figure(figsize=(10, 8))

    # Crear gráfico de dispersión
    grafico = sns.scatterplot(
        data=datos,
        x='longitude',
        y='latitude',
        hue='median_house_value',
        palette='viridis',
        alpha=0.6
    )

    # Configurar título y etiquetas
    plt.title('Distribución de Viviendas por Ubicación y Valor Medio', fontsize=16)
    plt.xlabel('Longitud', fontsize=12)
    plt.ylabel('Latitud', fontsize=12)

    # Ajustar leyenda
    plt.legend(title='Valor Medio', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Mostrar gráfico
    plt.tight_layout()
    plt.show()

    # Cerrar figura para liberar memoria
    plt.close()


def normalizar_datos(datos):
    """
    Normaliza los datos utilizando StandardScaler.

    Parameters:
    datos (pandas.DataFrame): DataFrame con los datos a normalizar

    Returns:
    tuple: (datos_escalados, escalador) donde:
        - datos_escalados: array numpy con los datos normalizados
        - escalador: instancia de StandardScaler utilizada
    """
    # Crear instancia de StandardScaler
    escalador = StandardScaler()

    # Ajustar y transformar los datos
    datos_escalados = escalador.fit_transform(datos)

    return datos_escalados, escalador


def calcular_puntajes_silueta(datos_escalados, rango_k):
    """
    Calcula los puntajes de silueta para diferentes valores de k.

    Parameters:
    datos_escalados (numpy.array): Datos normalizados
    rango_k (range): Rango de valores de k a evaluar

    Returns:
    list: Lista de puntajes de silueta para cada valor de k
    """
    puntajes_silueta = []

    # Iterar sobre el rango de valores de k
    for k in rango_k:
        # Crear instancia de KMeans con k clusters y semilla aleatoria
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        # Ajustar el modelo a los datos escalados
        kmeans.fit(datos_escalados)

        # Calcular el puntaje de silueta
        puntaje = silhouette_score(datos_escalados, kmeans.labels_)

        # Almacenar el puntaje
        puntajes_silueta.append(puntaje)

        # Imprimir progreso (opcional para depuración)
        print(f"K={k}, Puntaje de silueta: {puntaje:.4f}")

    return puntajes_silueta


def graficar_puntajes_silueta(rango_k, puntajes_silueta):
    """
    Crea un gráfico de los puntajes de silueta para diferentes valores de k.

    Parameters:
    rango_k (range): Rango de valores de k evaluados
    puntajes_silueta (list): Lista de puntajes de silueta
    """
    # Crear figura
    plt.figure(figsize=(10, 6))

    # Graficar puntajes de silueta
    plt.plot(rango_k, puntajes_silueta, marker='o', linestyle='-', color='b')

    # Configurar título y etiquetas
    plt.title('Método de Silueta para Determinar el Número Óptimo de Clusters', fontsize=16)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Coeficiente de Silueta', fontsize=12)

    # Añadir cuadrícula para mejor lectura
    plt.grid(True, linestyle='--', alpha=0.7)

    # Resaltar el valor óptimo de k
    k_optimo = rango_k[np.argmax(puntajes_silueta)]
    plt.axvline(x=k_optimo, color='r', linestyle='--',
                label=f'K óptimo = {k_optimo}')
    plt.legend()

    # Mostrar gráfico
    plt.tight_layout()
    plt.show()

    # Cerrar figura para liberar memoria
    plt.close()

    return k_optimo


def aplicar_kmeans(datos_escalados, k_optimo):
    """
    Aplica el algoritmo K-Means con el número óptimo de clusters.

    Parameters:
    datos_escalados (numpy.array): Datos normalizados
    k_optimo (int): Número óptimo de clusters determinado por el método de silueta

    Returns:
    tuple: (etiquetas, centroides) donde:
        - etiquetas: array con las etiquetas de cluster para cada punto
        - centroides: array con las coordenadas de los centroides
    """
    # Crear instancia de KMeans con el número óptimo de clusters
    kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)

    # Ajustar el modelo a los datos
    kmeans.fit(datos_escalados)

    # Obtener etiquetas y centroides
    etiquetas = kmeans.labels_
    centroides = kmeans.cluster_centers_

    return etiquetas, centroides


def visualizar_resultados_clustering(datos_escalados, etiquetas, centroides):
    """
    Visualiza los resultados del clustering con los centroides.

    Parameters:
    datos_escalados (numpy.array): Datos normalizados
    etiquetas (numpy.array): Etiquetas de cluster para cada punto
    centroides (numpy.array): Coordenadas de los centroides de cada cluster
    """
    # Crear figura
    plt.figure(figsize=(10, 8))

    # Crear gráfico de dispersión con colores por cluster
    plt.scatter(datos_escalados[:, 0], datos_escalados[:, 1],
                c=etiquetas, s=50, cmap='viridis', alpha=0.6)

    # Marcar los centroides
    plt.scatter(centroides[:, 0], centroides[:, 1],
                marker='X', s=200, c='red', label='Centroides')

    # Configurar título y etiquetas
    plt.title('Resultados de Clustering con K-Means', fontsize=16)
    plt.xlabel('Longitud (escalada)', fontsize=12)
    plt.ylabel('Latitud (escalada)', fontsize=12)

    # Añadir leyenda y cuadrícula
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Mostrar gráfico
    plt.tight_layout()
    plt.show()

    # Cerrar figura para liberar memoria
    plt.close()


def main():
    """
    Función principal que orquesta todo el proceso de análisis.
    """
    try:
        # Definir ruta al archivo de datos (usando raw string para evitar problemas con backslashes)
        ruta_archivo = r'C:\Users\oryee\PycharmProjects\PythonProject2\py-practice-lab\T4\ejemplo-k-means\housing.csv'

        # Verificar que el archivo existe
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encuentra el archivo: {ruta_archivo}")

        print("Cargando datos...")
        # Cargar datos
        datos = cargar_datos(ruta_archivo)

        print("Limpiando datos...")
        # Limpiar datos (eliminar valores faltantes)
        datos_limpios = limpiar_datos(datos)

        print("Visualizando datos originales...")
        # Visualizar datos originales
        visualizar_datos_originales(datos_limpios)

        print("Normalizando datos...")
        # Normalizar datos
        datos_escalados, escalador = normalizar_datos(datos_limpios)

        # Definir rango de valores de k a evaluar
        rango_k = range(2, 11)

        print("Calculando puntajes de silueta...")
        # Calcular puntajes de silueta para diferentes valores de k
        puntajes_silueta = calcular_puntajes_silueta(datos_escalados, rango_k)

        print("Graficando puntajes de silueta...")
        # Graficar puntajes de silueta y determinar k óptimo
        k_optimo = graficar_puntajes_silueta(rango_k, puntajes_silueta)

        print(f"Aplicando K-Means con k={k_optimo}...")
        # Aplicar K-Means con el número óptimo de clusters
        etiquetas, centroides = aplicar_kmeans(datos_escalados, k_optimo)

        print("Visualizando resultados del clustering...")
        # Visualizar resultados del clustering
        visualizar_resultados_clustering(datos_escalados, etiquetas, centroides)

        print("Proceso completado exitosamente.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Por favor, verifica la ruta al archivo de datos.")
    except Exception as e:
        print(f"Error inesperado: {e}")
        print("Por favor, revisa los datos y el código.")


# Punto de entrada del script
if __name__ == "__main__":
    main()