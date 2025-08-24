"""
 ¿Cuál es la utilidad de la palabra clave “def” en Python?

    a) Es una jerga que significa “este código es realmente estupendo”
    b) Indica el comienzo de una función.
    c) Indica que la siguiente sección de código indentado debe ser almacenada para
       usarla más tarde.
    d) b y c son correctas ambas.
    e) Ninguna de las anteriores.

    ANSER: b) Indica el comienzo de una función
"""
from rich.console import Console

console = Console()


def hello_world():
    console.print("\033[34mHello World\033[0m")

hello_world()