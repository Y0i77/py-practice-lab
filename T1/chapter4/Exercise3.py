"""
Reescribe el programa de cálculo del salario, con tarifa-ymedia para las horas extras,
y crea una función llamada calculo_salario que reciba dos parámetros (horas y tarifa).
"""

from rich.console import Console

console = Console()

NO_BENEFIT_RATE_OF_HOURS = 40
BENEFIT_PERCENTAGE = 1.5

workedHours = 0
paymentPerHour = 0


def calculate_salary(payment, hours):
    if hours > NO_BENEFIT_RATE_OF_HOURS:
        base = NO_BENEFIT_RATE_OF_HOURS * payment
        extra = (hours - NO_BENEFIT_RATE_OF_HOURS) * payment

        return round(base + (extra * BENEFIT_PERCENTAGE), 2)
    else:
        return round(payment * hours, 2)


while True:

    try:
        workedHours = int(input("Insert the current of worked hours: "))
        paymentPerHour = float(input("Insert the current of payment per hour: "))
        break
    except ValueError:
        console.print("[red]Error: Both entries must be numbers. Please try again.[/red]")
        continue

console.print(f"[red]Total Payment: {calculate_salary(paymentPerHour, workedHours)} [/red]")
