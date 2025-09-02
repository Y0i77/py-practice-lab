"""
Reescribe el programa de calificaciones del capítulo anterior usando una función llamada
calcula_calificacion, que reciba una puntuación como parámetro y devuelva una calificación como cadena.
"""

import pyinputplus as pyip
from rich.console import Console

console = Console()

MIN_RANGE = 0.0
MAX_RANGE = 1.0

def calculate_cualification(score):

    match score:
        case 0.9:
            console.print("\033[34mOutstanding\033[0m")
        case 0.8:
            console.print("\033[32mNotable\033[0m")
        case 0.7:
            console.print("\033[32mWell Done\033[0m")
        case 0.6:
            console.print("\033[33mEnough\033[0m")
        case 0.5 | 0.4 | 0.3 | 0.2 | 0.1:
            console.print("\033[31mNot Enough\033[0m")
        case _:
            console.print("\033[31mInvalid Score Value\033[0m")


assignedScore = pyip.inputFloat(
    prompt=f"Write a score \033[34mbetween {MIN_RANGE} - {MAX_RANGE}\033[0m: ",
    min = 0.1, max = 0.9
)

calculate_cualification(assignedScore)


