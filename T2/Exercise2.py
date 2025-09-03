"""
2.  Con el uso de la Biblioteca de funciones Matplotlib
    (ver ejemplo en documento de regresión lineal): Entender y Codificar
    la interpretación gráfica de la función OR del ejercicio 2.1.1 del documento
    Apuntes de Redes neuronales
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Configuración del Perceptrón
Aquí vamos a definir algunas constantes para que nuestra función tome forma.
"""
THRESHOLD = 0.5
FIRST_ENTRY_WEIGHT = 0.7
SECOND_ENTRY_WEIGHT = 0.7

"""
Datos de la compuerta OR
Aquí vamos a guardar en otra constante de tipo Array los puntos requeridos
"""
LOCATION_POINTS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
LABELS = ['0', '1', '1', '1']
SELECTED_COLORS = ['blue' if label == '0' else 'green' for label in LABELS]

"""
Funciones auxiliares
Definimos unas funciones de tal manera que desacoplemos todo el trabajo y el código
se vea más limpio, en estas utilizamos los terminos x, w1, w2,etc... Para brindar un
enfoque más ligado a la formula que se nos ha dado en el libro.
"""


def decision_boundary(x_values, threshold, w1, w2):
    """En esta parte vamos a calcular la frontera de decisión del perceptrón."""
    return (threshold - w1 * x_values) / w2


def plot_decision_boundary(threshold, w1, w2, points, labels, colors):
    """Esta función genera el gráfico de la frontera de decisión con los puntos de verdad."""
    # 1. Definimos el rango de valores para la recta
    x_range = np.linspace(-0.5, 1.5, 100)
    y_range = decision_boundary(x_range, threshold, w1, w2)

    # 2. Empezamos a crear gráfico
    plt.figure(figsize=(8, 6))
    plt.plot(
        x_range, y_range, 'r-',
        label=f'{w1}x₁ + {w2}x₂ = {threshold}'
    )

    # 3. Luego vamos a graficar puntos con el loop "for"
    for point, label, color in zip(points, labels, colors):
        plt.scatter(*point, color=color, s=100, zorder=3)
        plt.text(point[0] + 0.02, point[1] + 0.02,
                 f'({point[0]},{point[1]})', fontsize=12)

    """
    Área de Personalización
    Aquí trabajare toda la estructura personalizable de mi frontend
    colores, tamaños, lineas, textos, etc...
    """
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('x₁', fontsize=14)
    plt.ylabel('x₂', fontsize=14)
    plt.title('Clasificador Lineal para Compuerta OR', fontsize=16)
    plt.legend(loc='upper right')
    plt.show()


"""
Función __main__ para ejecutar mi código
Desacoplamos todo el código de tal manera que su arranque se vea más organizado y
escalable.
"""
if __name__ == "__main__":
    plot_decision_boundary(
        THRESHOLD,
        FIRST_ENTRY_WEIGHT,
        SECOND_ENTRY_WEIGHT,
        LOCATION_POINTS,
        LABELS,
        SELECTED_COLORS
    )
