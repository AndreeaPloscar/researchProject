import glob

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('TkAgg')
path_days = "cluj_preprocessed.xlsx"

file_days = glob.glob(path_days)
data_days = pd.read_excel(file_days[0])


gender = {'F': 1, 'M': 0}
bool = {'da': 1, 'nu': 0}

columns = ['Max. T. (ºC)', 'Min. T. (ºC)',
           'Prec. (mm)',
                          # 'S.L.Press./ Gheopot.',
                          # 'Wind sp. (Km/h)', 'Insolat. (hours)',
                          # 'Cloud c.'
           ]

X = pd.DataFrame(data_days,
                 columns=['Date', *columns])

X['Max. T. (ºC)'] = pd.to_numeric(X['Max. T. (ºC)'], errors='coerce')


X = X.dropna()

scaled = StandardScaler().fit_transform(X[columns])
kmeans = KMeans(n_clusters=4)
model = kmeans.fit(scaled)

centroids = model.cluster_centers_
print(centroids)

colNames = list(columns)
colNames.append('prediction')

dist = kmeans.transform(X[columns])
print(dist)
days = list(X['Date'])
days_to_keep = []

for i in range(4):
    points = []
    index = 0
    for p in dist:
        points.append((index, p[i]))
        index += 1
    print(points)
    points.sort(key=lambda a: a[1])
    points = points[0:40]

    for p in points:
        days_to_keep.append(days[p[0]])
days_to_keep.sort()
print(days_to_keep)
days_to_keep = list(set(days_to_keep))
print(days_to_keep)

X = pd.DataFrame(data_days)
Y = X[X['Date'].isin(days_to_keep)]
Y.to_excel('DownsampledMeteoCluj.xlsx', index=False)









