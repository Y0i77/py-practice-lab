
# ——— Importar librerías ———
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import joblib

# ——— Cargar el archivo Excel ———
filename = "dataexport_20251005T141846.xlsx"
possible_paths = [
    filename,
    "/content/" + filename,
    "/mnt/data/" + filename
]

excel_path = None
for p in possible_paths:
    if os.path.exists(p):
        excel_path = p
        break

if excel_path is None:
    raise FileNotFoundError(f"No encontré el archivo Excel '{filename}' en rutas comunes: {possible_paths}")

print("Archivo encontrado en:", excel_path)

# Leer la hoja “Hoja1”
df = pd.read_excel(excel_path, sheet_name="Hoja1", engine="openpyxl")
print("Columnas originales:", df.columns.tolist())
print("Primeras filas:\n", df.head())

# ——— Preprocesamiento de columnas ———
df = df.rename(columns={
    "timestamp": "timestamp",
    "Basilea Temperature [2 m elevation corrected]": "temperature"
})

# Convertir coma decimal a punto decimal
df["temperature"] = df["temperature"].astype(str).str.replace(",", ".").astype(float)

# Convertir timestamp a tipo datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Ordenar
df = df.sort_values("timestamp").reset_index(drop=True)
print("Después de conversión:\n", df.head())

# ——— Crear serie temporal con índice datetime ———
df = df.set_index("timestamp")
ts = df[["temperature"]].rename(columns={"temperature": "temp"})

# Resample horario e interpolar
ts_hourly = ts.resample("H").mean()
ts_hourly["temp"] = ts_hourly["temp"].interpolate(method="time").bfill().ffill()
print("Datos horarios (primeras filas):\n", ts_hourly.head(24))

# ——— Crear secuencias ———
def create_sequences(values, window_size):
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size])
    return np.array(X), np.array(y)

window_size = 24
values = ts_hourly["temp"].values.reshape(-1, 1)

scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

X, y = create_sequences(values_scaled, window_size)
print("Shape de X:", X.shape, "Shape de y:", y.shape)

# División entrenamiento / prueba
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

# ——— Modelo LSTM ———
tf.keras.backend.clear_session()
model = Sequential([
    LSTM(64, input_shape=(window_size, 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ——— Entrenamiento ———
epochs = 20
batch_size = 32

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=batch_size,
    verbose=2
)

# ——— Guardar modelo y scaler ———
model_save_path = "/mnt/data/lstm_temp_model.h5"
scaler_save_path = "/mnt/data/lstm_scaler.gz"
model.save(model_save_path)
joblib.dump(scaler, scaler_save_path)
print("Modelo guardado en:", model_save_path)
print("Scaler guardado en:", scaler_save_path)

# ——— Evaluar y graficar ———
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_inv = scaler.inverse_transform(y_test)

# Preparar DataFrame de comparación
df_compare = pd.DataFrame({
    "actual": y_test_inv.flatten(),
    "pred": y_pred.flatten()
})
print("Comparación (primeras filas):\n", df_compare.head(20))

plt.figure(figsize=(10,5))
plt.plot(df_compare["actual"].values, label="actual")
plt.plot(df_compare["pred"].values, label="predicción")
plt.title("Temperatura real vs predicha (conjunto de prueba)")
plt.xlabel("índice de muestra")
plt.ylabel("Temperatura")
plt.legend()
plt.show()

# ——— Predecir la siguiente hora ———
last_window = values_scaled[-window_size:].reshape(1, window_size, 1)
next_scaled = model.predict(last_window)
next_temp = scaler.inverse_transform(next_scaled)[0][0]
print(f"Predicción para la siguiente hora: {next_temp:.4f}")
