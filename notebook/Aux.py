


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

...# 

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering

...#

from sklearn.metrics import silhouette_samples, silhouette_score

 
...#

import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid",
        color_codes=True,
        context="notebook",
        rc={"grid.linewidth":0.25,"grid.color":"grey","grid.linestyle":"-"},
        font_scale=1)

sns.set_palette("deep")
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (9,9)

############################################



# Cargamos el DataFrame
data_df = pd.read_csv('../data/processed/dataset.csv')
subdata_df = pd.read_csv('../data/processed/subdataset.csv')

# Asumiendo que cat_cols son tus columnas categóricas y num_cols son las numéricas
cat_cols = ['gender','etnicity','edu_lvl','month']
num_cols = ['years', 'main_source','total'] # age

# Crear los transformadores para las columnas categóricas y numéricas
cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)  # Añadir sparse=False
num_transformer = StandardScaler()

# Crear el preprocesador que aplicará las transformaciones a las columnas correspondientes
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])

# Crear el pipeline que primero preprocesará los datos y luego aplicará el clustering
clust_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('cluster', AgglomerativeClustering())])

# Ajustar y aplicar el modelo
clust_pipeline.fit(subdata_df)   

# Obtener las etiquetas de los clusters
labels = clust_pipeline.named_steps['cluster'].labels_
subdata_df['cluster_label'] = labels

score = silhouette_score(subdata_df, labels, metric='euclidean')
print(f'Coeficiente de silueta: {score:.3f}')

# Obtener los valores del coeficiente de silueta para cada objeto
silhouette_values = silhouette_samples(subdata_df, labels)

# Crear un diagrama de barras
plt.figure(figsize=(10, 7))
plt.bar(range(len(subdata_df)), silhouette_values)
plt.xlabel('Objetos')
plt.ylabel('Coeficiente de silueta')
plt.title('Diagrama de silueta')

# Colorear las barras según el cluster al que pertenecen
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.fill_between(range(len(subdata_df)), 0, silhouette_values, where=(labels == i), color=colors[i], alpha=0.5)

# Mostrar el gráfico
plt.show()