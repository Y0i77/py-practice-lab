"""
1.  Realizar un programa que reciba por el usuario los valores de una matriz 3x4 y
    otra matriz 4x3 y las multiplique:

    b- Con la función np.array de la biblioteca de funciones NumPy

"""
import pyinputplus as pyip
import numpy as np
from rich.console import Console

console = Console()

# Definimos algunas constantes para nuestras dimensiones
ROWS_A = 3
COLS_A = 4
ROWS_B = 4
COLS_B = 3

# Definimos algunas validaciones
MIN_VALUE = -1000
MAX_VALUE = 1000
INDEX_OFFSET = 1


def input_matrix(rows, cols, name):
    """Mi prompt para recivir los valores del usuario"""
    console.print(f"\n[yellow]Insert some values for matrix {name} ({rows}x{cols}):[/yellow]")
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            val = pyip.inputInt(
                prompt=f"Insert a Value for {name}[{i + INDEX_OFFSET}][{j + INDEX_OFFSET}]: ",
                min=MIN_VALUE,
                max=MAX_VALUE
            )
            row.append(val)
        matrix.append(row)
    return matrix


def print_matrix(matrix, title, style="cyan"):
    """Print a matrix with formatted title using Rich."""
    console.print(f"\n[bold {style}]{title}:[/bold {style}]")

    # Handle both regular lists and numpy arrays
    if isinstance(matrix, np.ndarray):
        console.print(matrix)
    else:
        for row in matrix:
            console.print(row)


def main():
    """Función Main para ejecutar las demás funciones."""
    try:
        # Input matrices with fixed dimensions
        matrix_a = input_matrix(ROWS_A, COLS_A, "A")
        matrix_b = input_matrix(ROWS_B, COLS_B, "B")

        # Convert to NumPy arrays
        np_matrix_a = np.array(matrix_a)
        np_matrix_b = np.array(matrix_b)

        # Calculate result using NumPy's matrix multiplication
        result_matrix = np.matmul(np_matrix_a, np_matrix_b)

        # Display results
        print_matrix(np_matrix_a, f"Matrix A ({ROWS_A}x{COLS_A})")
        print_matrix(np_matrix_b, f"Matrix B ({ROWS_B}x{COLS_B})")
        print_matrix(result_matrix,
                     f"Result Matrix C ({result_matrix.shape[0]}x{result_matrix.shape[1]}) = A x B",
                     "green")

    except ValueError as e:
        console.print(f"\n[red]Error: {e}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    main()