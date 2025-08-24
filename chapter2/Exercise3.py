"""
 Asume que ejecutamos las siguientes sentencias de asignación:
"""

from rich.console import Console
console = Console()

OBJECT_WIDTH = 17
OBJECT_HEIGH = 12.0

console.print(f"Object Width divided by 2: [blue]{OBJECT_WIDTH / 2}[/blue]\n"
              f"Object Width divided by 2.0: [blue]{OBJECT_WIDTH / 2.0}[/blue]\n"
              f"Object Heigh divided by 3: [blue]{OBJECT_HEIGH / 3}[/blue]\n"
              f"Basic Operation (1 + 2 * 5): [blue]{1 + 2 * 5}[/blue]\n")