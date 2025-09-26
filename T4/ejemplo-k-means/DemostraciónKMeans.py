# Importar librerías
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Leemos los datos de vivienda desde un archivo CSV llamado 'housing.csv'
data = pd.read_csv(r'C:\Users\oryee\PycharmProjects\PythonProject2\py-practice-lab\T4\ejemplo-k-means\housing.csv',
                   usecols=['longitude', 'latitude', 'median_house_value'])

# Eliminamos cualquier fila que tenga valores faltantes (NaN) del conjunto de datos
data = data.dropna()

# Gráfico original de seaborn (sin cambios)
sns.scatterplot(data=data, x='longitude', y='latitude', hue='median_house_value')
plt.show()

# Creamos una instancia de StandardScaler para la normalización
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Calculamos los puntajes de silueta
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    score = silhouette_score(data_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Graficar los puntajes de silueta
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel("Coeficiente de Silueta")
plt.title('Coeficiente de Silueta para Diversos Valores de K')
plt.show()

# Establecemos k=2 basado en los resultados de silueta
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data_scaled)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Graficar resultados con datos escalados
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')
plt.xlabel('Longitude (escalada)')
plt.ylabel('Latitude (escalada)')
plt.title('Clusters de Viviendas con K-Means (Datos Escalados)')
plt.show()