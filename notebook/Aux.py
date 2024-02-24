import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Carga del conjunto de datos
datos = pd.read_csv("ruta/al/archivo.csv")

# Preprocesamiento de las features
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos)

**Definición de los modelos**

# KMeans con búsqueda de hiperparámetros
kmeans_params = {
  'n_clusters': np.arange(2, 10),
  'init': ['k-means++', 'random'],
  'max_iter': [100, 300]
}
kmeans_cv = GridSearchCV(KMeans(), kmeans_params, scoring='silhouette_score', n_jobs=-1)

# Agglomerative Clustering con búsqueda de hiperparámetros
agglo_params = {
  'n_clusters': np.arange(2, 10),
  'linkage': ['average', 'ward', 'complete']
}
agglo_cv = GridSearchCV(AgglomerativeClustering(), agglo_params, scoring='silhouette_score', n_jobs=-1)

**Ajuste de los modelos a los datos**

kmeans_cv.fit(datos_escalados)
agglo_cv.fit(datos_escalados)

**Evaluación de los resultados**

# Mejores parámetros
print(f"Mejores parámetros KMeans: {kmeans_cv.best_params_}")
print(f"Mejores parámetros Agglomerative Clustering: {agglo_cv.best_params_}")

# Métricas
metricas = {
  "KMeans": {
    "Silhouette": silhouette_score(datos_escalados, kmeans_cv.best_estimator_.labels_),
    "Calinski-Harabasz": calinski_harabasz_score(datos_escalados, kmeans_cv.best_estimator_.labels_)
  },
  "Agglomerative Clustering": {
    "Silhouette": silhouette_score(datos_escalados, agglo_cv.best_estimator_.labels_),
    "Calinski-Harabasz": calinski_harabasz_score(datos_escalados, agglo_cv.best_estimator_.labels_)
  }
}

# Visualización de los resultados
plt.figure(figsize=(10, 6))
for i, modelo in enumerate([kmeans_cv.best_estimator_, agglo_cv.best_estimator_]):
  plt.subplot(1, 2, i + 1)
  plt.scatter(datos_escalados[:, 0], datos_escalados[:, 1], c=modelo.labels_, cmap='viridis')
  plt.title(modelo.__class__.__name__)
plt.show()

# Impresión de las métricas
for modelo, resultados in metricas.items():
  print(f"Modelo: {modelo}")
  for metrica, valor in resultados.items():
    print(f"  {metrica}: {valor}")

# **Selección del mejor modelo**

# En base a las métricas de evaluación y la visualización de los resultados, se selecciona el modelo que mejor se ajusta a los datos y a los objetivos del análisis.

# **Análisis e interpretación de los resultados**

# * Se realiza un análisis e interpretación de los clusters encontrados por el mejor modelo.
# * Se visualizan los clusters con diferentes técnicas, como gráficos de dispersión, diagramas de caja y heatmaps.
# * Se comparan los clusters entre sí y se identifican las características que los diferencian.
# * Se interpretan los clusters en el contexto del problema de negocio.

# **Recomendaciones:**

# * Ajustar los parámetros de los modelos para optimizar los resultados.
# * Probar diferentes modelos de clustering para comparar resultados.
# * Evaluar la interpretabilidad de los clusters.

# **Este código implementa las sugerencias para mejorar el clustering, como la búsqueda de hiperparámetros y el análisis e interpretación de los resultados.**

# **Recursos adicionales:**

# * [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)
# * [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

# **Espero que este código te haya sido útil.
