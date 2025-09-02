"""
1. Programa que recibe dos valores (0 o 1)en dos variables x1 y x2 y devuelve la función lógica OR

NOTA: para que funcione correctamente recomiendo descargar las librerias:
                - pyinputplus
                - rich
    puede obtenerlas por medio del comando pip install... desde la terminal
"""

import pyinputplus as pyip
from rich.console import Console
console = Console()

#Función para calcular los resultados
def or_function_result(value1, value2):
    match value1 + value2:
        case 1: #Si la suma entre dos numeros me da 1 significa que los dos no son 0
            console.print(1)
        case 2: # si la suma me da 2 ambos son 1
            console.print(1)
        case 0: # si la suma me da 0 no tengo ningún 1
            console.print(0)
        case _:
            console.print(f"[red]ERROR: {value1} or {value2} is not a supported number[/red]")

x1 = pyip.inputInt(
    prompt="Introduce a \033[31mvalue for x1:\033[0m ",
    min=0, max=1 #Limitamos el rango para el valor de x1
)

x2 = pyip.inputInt(
    prompt="Introduce a \033[31mvalue for x1:\033[0m ",
    min=0, max=1 #Limitamos el rango para el valor de x2
)

or_function_result(x1,x2)

