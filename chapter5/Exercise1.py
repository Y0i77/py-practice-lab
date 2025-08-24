"""
Escribe un programa que lea repetidamente números hasta
que el usuario introduzca “fin”. Una vez se haya introducido “fin”,
muestra por pantalla el total, la cantidad de números y la media de
esos números. Si el usuario introduce cualquier otra cosa que no sea un
número, detecta su fallo usando try y except, muestra un mensaje de
error y pasa al número siguiente.
"""

import pyinputplus as pyip
from rich.console import Console

console = Console()

content = ""
numbers = []

while content != 'end':

    content = pyip.inputStr(
        prompt = "Insert any kind of number and write \033[34m'end'\033[0m to finish the program: "
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
              f"[red]Average Number: {round(sum(numbers)/len(numbers))}[/red]")