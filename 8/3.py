import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas
import numpy as np
import random
from sklearn.cluster import KMeans


data_flats_raw = pandas.read_csv('flats_for_clustering.tsv', sep='\t')
data_flats_raw['Piętro'] = data_flats_raw['Piętro'].apply(
    lambda x: 0 if x in ['parter', 'niski parter'] else x
)
data_flats_raw['Piętro'] = data_flats_raw['Piętro'].apply(
    lambda x: 2 if x in ['poddasze'] else x
)
data_flats_raw = data_flats_raw.dropna()

array = data_flats_raw.astype(int).values

scaler = StandardScaler()
scaled_flats = scaler.fit_transform(array)

kmeans = KMeans(init="random", n_clusters=5, n_init=20, max_iter=300, random_state=42)
kmeans.fit(scaled_flats)

print(kmeans.inertia_)
print(kmeans.n_iter_)