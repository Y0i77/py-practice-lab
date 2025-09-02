"""
Escribe otro programa que pida una lista de números como
la anterior y al final muestre por pantalla el máximo y mínimo de los
números, en vez de la media.
"""

import pyinputplus as pyip
from rich.console import Console

console = Console()

content = ""
numbers = []

while content != 'end':

    content = pyip.inputStr(
        prompt = "Insert any kind of number and write \033[33m'end'\033[0m to finish the program: "
    )

    if content.lower() == 'end':
        break

    try:
        number = int(content)
    except ValueError:
        console.print(f"[red]ERROR: {content} Is Not A Numerical Value[/red]")
        continue

    console.print(f"you have written the number: [blue]{number}[/blue]")
    numbers.append(number)

console.print(f"[yellow]Quantity Numbers: {len(numbers)}[/yellow]\n"
              f"[blue]Total: {sum(numbers)}[/blue]\n"
              f"[red]Highest Number: {max(numbers)}[/red]\n"
              f"[red]Lowest Number: {min(numbers)}[/red]")