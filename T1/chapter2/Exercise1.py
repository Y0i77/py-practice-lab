"""
Escribe un programa que use input para pedirle al usuario
su nombre y luego darle la bienvenida.
"""

import pyinputplus as pyip
from rich.console import Console

console = Console()

userName = pyip.inputStr(
    prompt="Insert \033[33mUser Name:\033[0m ",
    blockRegexes=[(r'^.{0,2}$|^.{16,}$',"\033[31monly names on a range between 5 to 15 characters\033[0m")]
)

console.print(f"Hello: [yellow]{userName}[/yellow]")
