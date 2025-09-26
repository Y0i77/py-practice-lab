import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Datos de ejemplo (serie de tiempo)
# Suponemos 10 secuencias de 5 días de temperaturas
X = np.array([
    [20, 21, 19, 22, 23],
    [21, 19, 22, 23, 24],
    [19, 22, 23, 24, 25],
    [22, 23, 24, 25, 26],
    [23, 24, 25, 26, 27]
])
y = np.array([24, 25, 26, 27, 28])  # valor esperado del día siguiente

# Redimensionamos para [muestras, pasos_tiempo, características]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Modelo RNR
model = Sequential()
model.add(SimpleRNN(10, activation='tanh', input_shape=(5,1)))
model.add(Dense(1))  # salida: predicción de un valor
model.compile(optimizer='adam', loss='mse')

# Entrenamos
model.fit(X, y, epochs=200, verbose=0)

# Predicción: próximo día tras [24,25,26,27,28]
test_input = np.array([24,25,26,27,28]).reshape((1,5,1))
pred = model.predict(test_input, verbose=0)
print(f"Predicción del próximo día: {pred[0][0]:.2f}")
