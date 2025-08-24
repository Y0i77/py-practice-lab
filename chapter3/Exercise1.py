"""
Reescribe el programa del cálculo del salario para darle al empleado
1.5 veces la tarifa horaria para todas las horas trabajadas que excedan de 40.

Introduzca las Horas: 45
Introduzca la Tarifa por hora: 10
Salario: 475.0
"""

import pyinputplus as pyip
from rich.console import Console

console = Console()

NO_BENEFIT_RATE_OF_HOURS = 40
BENEFIT_PERCENTAGE = 1.5

notBenefitedPayment = 0
benefitedHours = 0

workedHours = pyip.inputInt(
    prompt = "Insert the current of \033[33mworked hours\033[0m: ",
    min = 0, max = 144,
)

paymentPerHour = pyip.inputFloat(
    prompt = "Insert the \033[33msalary range\033[0m per hour: ",
    min = 10, max = 30,
)

if workedHours > NO_BENEFIT_RATE_OF_HOURS:
    notBenefitedPayment = NO_BENEFIT_RATE_OF_HOURS * paymentPerHour
    benefitedHours = workedHours - NO_BENEFIT_RATE_OF_HOURS

    console.print(
        f"Total payment: [blue]${round((benefitedHours * paymentPerHour) * BENEFIT_PERCENTAGE) + notBenefitedPayment}[/blue]")

else:
    console.print(f"Total payment: [blue]${round(paymentPerHour * workedHours)}[/blue]")