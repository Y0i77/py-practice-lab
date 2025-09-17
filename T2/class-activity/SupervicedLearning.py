import pyinputplus as pyip
from rich.console import Console

console = Console()
TIMES_FOR_LOOP = 3

def values_obtainer(times):

    basevalues = []
    multivalues = []

    for _ in range(times):  # usamos for para evitar bucle infinito
        basevalues.append(pyip.inputInt(prompt="Enter a number (base): ",
                                        min=0, max=100))
        multivalues.append(pyip.inputInt(prompt="Enter a number (multiplier): ",
                                         min=0, max=100))
    return basevalues, multivalues

def multiply_lists(a, b):
    # zip corta al menor; si quieres otro comportamiento usa zip_longest
    return [x * y for x, y in zip(a, b)]

def checking_values(results):
    for i, value in enumerate(results, start=1):
        if value % 2 == 0:
            console.print(f"Result {i}: {value} — The number is even.")
        else:
            console.print(f"Result {i}: {value} — The number is odd.")

# flujo
baseValues, multiValues = values_obtainer(TIMES_FOR_LOOP)
result = multiply_lists(baseValues, multiValues)
checking_values(result)
