"""
Reescribe el programa del salario usando try y except, de modo que el
programa sea capaz de gestionar entradas no numéricas con elegancia, mostrando
un mensaje y saliendo del programa. A continuación se muestran dos ejecuciones
del programa:

Introduzca las Horas: 20
Introduzca la Tarifa por hora: nueve
Error, por favor introduzca un número

Introduzca las Horas: cuarenta
Error, por favor introduzca un número

"""

from rich.console import Console

console = Console()

NO_BENEFIT_RATE_OF_HOURS = 40
BENEFIT_PERCENTAGE = 1.5
ERROR_MESSAGE = "Error: Please Introduce a number"

notBenefitedPayment = 0
benefitedHours = 0
workedHours = 0
paymentPerHour = 0

while True:

    try:
        workedHours = int(input("Insert the current of worked hours: "))
    except ValueError:
        console.print(f"[red]{ERROR_MESSAGE}[/red]")

    try:
        paymentPerHour = float(input("Insert the current of payment per hour: "))
    except ValueError:
        console.print(f"[red]{ERROR_MESSAGE}[/red]")

    if workedHours > NO_BENEFIT_RATE_OF_HOURS:
        notBenefitedPayment = NO_BENEFIT_RATE_OF_HOURS * paymentPerHour
        benefitedHours = workedHours - NO_BENEFIT_RATE_OF_HOURS

        console.print(
            f"Total payment: [blue]${round((benefitedHours * paymentPerHour) * BENEFIT_PERCENTAGE) + notBenefitedPayment}[/blue]")

    else:
        console.print(f"Total payment: [blue]${round(paymentPerHour * workedHours)}[/blue]")


