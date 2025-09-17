import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Cargar y preparar los datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline']

print("Cargando dataset de vino...")
data = pd.read_csv(url, names=column_names)

# Dividir características y objetivo
X = data.drop('Class', axis=1)
y = data['Class']

# Codificar etiquetas (aunque ya son 1,2,3)
le = LabelEncoder()
y = le.fit_transform(y)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Dimensiones de los datos: {X.shape}")
print(f"Clases: {np.unique(y)}")

# 2. Implementación con TensorFlow
print("\n" + "=" * 50)
print("ENTRENAMIENTO CON TENSORFLOW")
print("=" * 50)

import tensorflow as tf

# Resetear cualquier gráfico previo
tf.reset_default_graph()

# Parámetros
learning_rate = 0.01
n_epochs = 1000
n_features = X_train.shape[1]
n_classes = len(np.unique(y))

# Placeholders
X_tf = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
y_tf = tf.placeholder(tf.int32, shape=[None], name='y')

# Arquitectura de la red
n_hidden1 = 16
n_hidden2 = 8

# Pesos y sesgos
W1 = tf.Variable(tf.random_normal([n_features, n_hidden1]), name='W1')
b1 = tf.Variable(tf.zeros([n_hidden1]), name='b1')
W2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]), name='W2')
b2 = tf.Variable(tf.zeros([n_hidden2]), name='b2')
W3 = tf.Variable(tf.random_normal([n_hidden2, n_classes]), name='W3')
b3 = tf.Variable(tf.zeros([n_classes]), name='b3')

# Capas
hidden1 = tf.nn.relu(tf.matmul(X_tf, W1) + b1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
logits = tf.matmul(hidden2, W3) + b3

# Pérdida y optimizador
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_tf, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Predicciones y precisión
preds = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, y_tf), tf.float32))

# Inicialización y sesión
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Entrenamiento
    for epoch in range(n_epochs):
        sess.run(optimizer, feed_dict={X_tf: X_train, y_tf: y_train})

        if epoch % 100 == 0:
            current_loss = sess.run(loss, feed_dict={X_tf: X_train, y_tf: y_train})
            train_acc = sess.run(accuracy, feed_dict={X_tf: X_train, y_tf: y_train})
            print(f'Epoch {epoch}, Loss: {current_loss:.4f}, Accuracy: {train_acc:.4f}')

    # Evaluación
    test_acc = sess.run(accuracy, feed_dict={X_tf: X_test, y_tf: y_test})
    print(f'Precisión final en prueba (TensorFlow): {test_acc:.4f}')

# 3. Implementación con Keras
print("\n" + "=" * 50)
print("ENTRENAMIENTO CON KERAS")
print("=" * 50)

from tensorflow import keras

# Crear modelo
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(n_classes, activation='softmax')
])

# Compilar modelo
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar modelo
history = model.fit(X_train, y_train,
                    epochs=100,
                    validation_split=0.2,
                    verbose=1)

# Evaluar modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Precisión en prueba (Keras): {test_acc:.4f}')

# 4. Implementación con Scikit-learn
print("\n" + "=" * 50)
print("ENTRENAMIENTO CON SCIKIT-LEARN")
print("=" * 50)

from sklearn.neural_network import MLPClassifier

# Crear y entrenar modelo
mlp = MLPClassifier(hidden_layer_sizes=(16, 8),
                    activation='relu',
                    solver='sgd',
                    learning_rate_init=0.01,
                    max_iter=1000,
                    random_state=42,
                    verbose=True)

mlp.fit(X_train, y_train)

# Evaluar modelo
train_acc = mlp.score(X_train, y_train)
test_acc = mlp.score(X_test, y_test)
print(f'Precisión en entrenamiento (Scikit-learn): {train_acc:.4f}')
print(f'Precisión en prueba (Scikit-learn): {test_acc:.4f}')

# 5. Comparativa de resultados
print("\n" + "=" * 50)
print("COMPARATIVA DE RESULTADOS")
print("=" * 50)

# Hacer predicciones con todos los modelos
with tf.Session() as sess:
    sess.run(init)
    tf_preds = sess.run(preds, feed_dict={X_tf: X_test, y_tf: y_test})

keras_preds = np.argmax(model.predict(X_test), axis=1)
sklearn_preds = mlp.predict(X_test)

# Calcular precisiones
tf_acc = accuracy_score(y_test, tf_preds)
keras_acc = accuracy_score(y_test, keras_preds)
sklearn_acc = accuracy_score(y_test, sklearn_preds)

print(f"TensorFlow Accuracy: {tf_acc:.4f}")
print(f"Keras Accuracy: {keras_acc:.4f}")
print(f"Scikit-learn Accuracy: {sklearn_acc:.4f}")

# Visualizar comparativa
models = ['TensorFlow', 'Keras', 'Scikit-learn']
accuracies = [tf_acc, keras_acc, sklearn_acc]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.ylabel('Precisión')
plt.title('Comparación de Precisiones entre Diferentes Implementaciones')
plt.ylim(0.8, 1.0)

# Añadir valores en las barras
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{acc:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()