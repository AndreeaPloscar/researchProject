import numpy
from sklearn.cluster import KMeans
import glob
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.express as px

path_days = "cluj_preprocessed.xlsx"

# ---------- ORADEA -----------

# path_days = "oradea_preprocessed.xlsx"

# =============================

file_days = glob.glob(path_days)
data_days = pd.read_excel(file_days[0])
dataframe_days = pd.DataFrame(data_days)


# dataframe_days['Date'] = pd.to_datetime(data_days['Date'], dayfirst=False, format="%Y/%m/%d")

x=[]
y=[]
z=[]


for day in dataframe_days['Date']:
    for i in range(0, 24):
        hour_end = str((i + 3) % 24).zfill(2) + 'Z'
        hour_start = str(i).zfill(2) + 'Z'
        hours = hour_end + "-" + hour_start
        temp = dataframe_days.loc[(dataframe_days['Date'] == day)].iloc[0]['Temp. (ºC) ' + hours]
        pressure = dataframe_days.loc[(dataframe_days['Date'] == day)].iloc[0]['Pressure/ Geopot. ' + hours]
        if temp != numpy.nan and pressure != numpy.nan:
            x.append(day + " " + hours)
            y.append(temp)
            z.append(pressure)


data = {'interval': x, 'temp diff': y, 'pressure diff': z}
df = pd.DataFrame(data, columns=['interval', 'temp diff', 'pressure diff'])

# kmeans = KMeans(n_clusters=3).fit(df)
# centroids = kmeans.cluster_centers_
# print(centroids)

fig = px.scatter(df, x="interval", color="temp diff", y="pressure diff")
fig.update_layout(yaxis_range=[-10,10])
fig.show()

# plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
# # plt.show()
# plt.xlabel('Temp. (ºC) 3hrs diff')
# plt.ylabel('Pressure/ Geopot. 3hrs diff')
# plt.savefig('cluster.png')
