"""
1.  Realizar un programa que reciba por el usuario los valores de una matriz 3x4
    y otra matriz 4x3 y las multiplique:

    a- Con Listas

"""

import pyinputplus as pyip
from rich.console import Console

console = Console()

# Constantes que defino para mis dimensiones
ROWS_A = 3
COLS_A = 4
ROWS_B = 4
COLS_B = 3

# Mis constantes de Validación
MIN_VALUE = -1000
MAX_VALUE = 1000
INDEX_OFFSET = 1
INDEX_BASE_POSITION = 0


def input_matrix(rows, cols, name):
    """Prompt para indicar al usuario que debe sinsertar datos para la Matriz."""
    console.print(f"\n[yellow]Insert values for matrix {name} ({rows}x{cols}):[/yellow]")
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            val = pyip.inputInt(
                prompt=f"Value for {name}[{i + INDEX_OFFSET}][{j + INDEX_OFFSET}]: ",
                min=MIN_VALUE,
                max=MAX_VALUE
            )
            row.append(val)
        matrix.append(row)
    return matrix


def multiply_matrices(matrix_a, matrix_b):
    """Multiplica las dos matrices usando una función de Listas."""
    # Validamos la compatibilidad primero
    if len(matrix_a[INDEX_BASE_POSITION]) != len(matrix_b):
        raise ValueError("Cannot multiply matrices: incompatible dimensions")

    # Luego inicializamos los resultados 
    result = [[INDEX_BASE_POSITION for _ in range(len(matrix_b[INDEX_BASE_POSITION]))] for _ in range(len(matrix_a))]

    # Realizamos las respectivas multiplicaciones, usamos los vectores i,j,k
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[INDEX_BASE_POSITION])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result


def print_matrix(matrix, title, style="cyan"):
    """Print a matrix with formatted title using Rich."""
    console.print(f"\n[bold {style}]{title}:[/bold {style}]")
    for row in matrix:
        console.print(row)


def main():
    """Función Main para desacoplar el funciónamiendto del codigo y controlar bloque de errores."""
    try:
        # Input matrices with fixed dimensions
        matrix_a = input_matrix(ROWS_A, COLS_A, "A")
        matrix_b = input_matrix(ROWS_B, COLS_B, "B")

        # Calculate result
        result_matrix = multiply_matrices(matrix_a, matrix_b)

        # Display results
        print_matrix(matrix_a, f"Matrix A ({ROWS_A}x{COLS_A})")
        print_matrix(matrix_b, f"Matrix B ({ROWS_B}x{COLS_B})")
        print_matrix(result_matrix,
                     f"Result Matrix C ({len(result_matrix)}x{len(result_matrix[0])}) = A x B",
                     "green")

    except ValueError as e:
        console.print(f"\n[red]Error: {e}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()
