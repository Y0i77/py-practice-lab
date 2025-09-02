"""
programar la neurona del Modelo McCulloch & Pitts (MCP).

NOTA: para que funcione correctamente recomiendo descargar las librerias:
                - pyinputplus (facilita el control de inserción de datos)
                - rich (Permite mejorar la calidad de los Print clasicos)
    puede obtenerlas por medio del comando pip install... desde la terminal
"""

#Librerias Integradas para mejorar la experiencia en la terminal
import pyinputplus as pyip
from rich.console import Console

console = Console()

THRESHOLD = 0 #lÍMITE MINIMO

selectedOption = '' #Inicialización de variables para mejores prácticas
xnValues = []
wnValues = []


while selectedOption != '1': #Bucle para añadir libertad de datos al usirio

    selectedOption = pyip.inputStr( #Despliegue de opciones con ACSI para dar vida al menú
        prompt="\033[34mChoose an Option\033[0m\n"
               "\033[32m1. Start The Program\033[0m\n"
               "\033[33m2. Add More Values\033[0m\n> "
    )

    match selectedOption: #Match para verificar la opción tomada
        case '1' | 'start': #Opción 1 para empezar el programa con los datos agregados
            if not xnValues or not wnValues:
                console.print("[red]ERROR: You must add inputs and weights first.[/red]")
                continue

            total = sum(x*w for x, w in zip(xnValues, wnValues)) #Realizamos la formula
            console.print(f"\033[35mTotal weighted sum: {total}\033[0m")

            output = 1 if total >= THRESHOLD else 0 #Comprobamos el límite establecido
            console.print(f"\033[36mNeuron output (MCP): {output}\033[0m")
            break #volvemos a iniciar el bucle

        case '2': # Caso 2 para agregar datos nuevos
            x_val = pyip.inputInt(
                prompt="Introduce a \033[31mvalue for x\033[0m: ")
            xnValues.append(x_val)

            w_val = pyip.inputInt(
                prompt="Introduce a \033[31mvalue for w\033[0m: ")
            wnValues.append(w_val)


            THRESHOLD = pyip.inputInt(
                prompt="Introduce a \033[33mthreshold value\033[0m (default 0): ",
                default=0)

        case _: #En caso de error
            console.print("[red]ERROR: Not a valid option.[/red]")
