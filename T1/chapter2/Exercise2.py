"""
Escribe un programa para pedirle al usuario el número de
horas y la tarifa por hora para calcular el salario bruto.
"""

import pyinputplus as pyip
from rich.console import Console

console = Console()

workedHours = pyip.inputInt(
    prompt = "Insert the current of \033[33mworked hours\033[0m: ",
    min = 0, max = 10,
)

paymentPerHour = pyip.inputFloat(
    prompt = "Insert the \033[33msalary range\033[0m per hour: ",
    min = 10, max = 30,
)

console.print(f"Total payment: [blue]${round(paymentPerHour * workedHours)}[/blue]")
