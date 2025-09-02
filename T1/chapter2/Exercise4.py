"""
Escribe un programa que le pida al usuario una temperatura en grados Celsius,
la convierta a grados Fahrenheit e imprima por pantalla la temperatura convertida.
"""

import pyinputplus as pyip
from rich.console import Console
console = Console()

FAHRENHEIT_FREEZING = 32
F_TO_C_SCALE = 9 / 5

celciusTemperarture = pyip.inputFloat(
    prompt="Insert a current temperature in \033[31mCelcius\033[0m:"
)

console.print(f"Fahrenheit Value: [blue]{(celciusTemperarture * F_TO_C_SCALE) + FAHRENHEIT_FREEZING}°[/blue]")