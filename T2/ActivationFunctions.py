import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    """Clase abstracta base para todas las funciones de activación"""

    @abstractmethod
    def function(self, x: np.ndarray) -> np.ndarray:
        """Calcula el valor de la función de activación"""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la función de activación"""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Permite usar la instancia como función"""
        return self.function(x)

    def plot(self, x: np.ndarray) -> None:
        """Método para graficar la función y su derivada"""
        y = self.function(x)
        dy = self.derivative(x)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title(f'Función {self.__class__.__name__}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('f(x)')

        plt.subplot(1, 2, 2)
        plt.plot(x, dy, 'r-', linewidth=2)
        plt.title(f'Derivada de {self.__class__.__name__}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel("f'(x)")

        plt.tight_layout()
        plt.show()


class Step(ActivationFunction):
    """
    Función de activación escalón (Heaviside)

    Matemáticamente:
        f(x) = { 0 si x < 0, 1 si x >= 0 }
        f'(x) = 0 para x ≠ 0 (indefinida en x=0)
    """

    def function(self, x: np.ndarray) -> np.ndarray:
        """Calcula la función escalón"""
        return np.where(x >= 0, 1.0, 0.0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la función escalón"""
        return np.zeros_like(x)


class Linear(ActivationFunction):
    """
    Función de activación lineal

    Matemáticamente:
        f(x) = x
        f'(x) = 1
    """

    def function(self, x: np.ndarray) -> np.ndarray:
        """Calcula la función lineal"""
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la función lineal"""
        return np.ones_like(x)


class Mixed(ActivationFunction):
    """
    Función de activación mixta (Leaky ReLU)

    Matemáticamente:
        f(x) = { αx si x < 0, x si x >= 0 }
        f'(x) = { α si x < 0, 1 si x >= 0 }
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def function(self, x: np.ndarray) -> np.ndarray:
        """Calcula la función mixta"""
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la función mixta"""
        return np.where(x > 0, 1, self.alpha)


class Sigmoid(ActivationFunction):
    """
    Función de activación sigmoide

    Matemáticamente:
        f(x) = 1 / (1 + e^(-x))
        f'(x) = f(x) * (1 - f(x))
    """

    def function(self, x: np.ndarray) -> np.ndarray:
        """Calcula la función sigmoide"""
        # Para estabilidad numérica
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la función sigmoide"""
        fx = self.function(x)
        return fx * (1 - fx)


class Tanh(ActivationFunction):
    """
    Función de activación tangente hiperbólica

    Matemáticamente:
        f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        f'(x) = 1 - f(x)^2
    """

    def function(self, x: np.ndarray) -> np.ndarray:
        """Calcula la tangente hiperbólica"""
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la tangente hiperbólica"""
        return 1 - np.tanh(x) ** 2


class ReLU(ActivationFunction):
    """
    Función de activación ReLU (Rectified Linear Unit)

    Matemáticamente:
        f(x) = max(0, x)
        f'(x) = { 0 si x < 0, 1 si x >= 0 }
    """

    def function(self, x: np.ndarray) -> np.ndarray:
        """Calcula la función ReLU"""
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calcula la derivada de la función ReLU"""
        return np.where(x > 0, 1, 0)


def plot_all_functions():
    """Función para graficar todas las funciones de activación"""
    x = np.linspace(-5, 5, 1000)

    # Crear instancias de todas las funciones
    functions = [
        Step(),
        Linear(),
        Mixed(alpha=0.1),
        Sigmoid(),
        Tanh(),
        ReLU()
    ]

    # Crear figura con subplots
    fig, axes = plt.subplots(len(functions), 2, figsize=(12, 4 * len(functions)))

    for i, func in enumerate(functions):
        y = func.function(x)
        dy = func.derivative(x)

        # Graficar función
        axes[i, 0].plot(x, y, 'b-', linewidth=2)
        axes[i, 0].set_title(f'Función {func.__class__.__name__}')
        axes[i, 0].grid(True, linestyle='--', alpha=0.7)
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('f(x)')

        # Graficar derivada
        axes[i, 1].plot(x, dy, 'r-', linewidth=2)
        axes[i, 1].set_title(f'Derivada de {func.__class__.__name__}')
        axes[i, 1].grid(True, linestyle='--', alpha=0.7)
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel("f'(x)")

    plt.tight_layout()
    plt.show()


def main():
    """Función principal para demostración"""
    x = np.linspace(-5, 5, 1000)

    # Crear instancias de todas las funciones
    functions = [
        Step(),
        Linear(),
        Mixed(alpha=0.1),
        Sigmoid(),
        Tanh(),
        ReLU()
    ]

    print("=== DEMOSTRACIÓN DE FUNCIONES DE ACTIVACIÓN ===")

    # Graficar todas las funciones juntas
    plot_all_functions()

    # Graficar cada función por separado
    for func in functions:
        func.plot(x)

    # Demostración matemática
    print("\n=== VALORES NUMÉRICOS DE EJEMPLO ===")
    test_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    for func in functions:
        print(f"\n{func.__class__.__name__}:")
        print(f"  f({test_values}) = {func.function(test_values)}")
        print(f"  f'({test_values}) = {func.derivative(test_values)}")


if __name__ == "__main__":
    main()