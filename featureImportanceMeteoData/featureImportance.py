import glob

import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('TkAgg')
path_days = "cluj_preprocessed.xlsx"

# ---------- ORADEA -----------

# path = '/Users/andreeaploscar/Desktop/THESIS/DATE-METEO/ORADEA/HOURS'
# start_date = "2014-07-01"
# path_days = "oradea_preprocessed.xlsx"

# =============================

file_days = glob.glob(path_days)
data_days = pd.read_excel(file_days[0])


gender = {'F': 1, 'M': 0}
bool = {'da': 1, 'nu': 0}
X = pd.DataFrame(data_days,
                 columns=['Date','Ave. T. (ºC)', 'Max. T. (ºC)', 'Min. T. (ºC)', 'Prec. (mm)',
                          'S.L.Press./ Gheopot.',
                          'Wind sp. (Km/h)', 'Insolat. (hours)',
                          'Cloud c.'])

X['Ave. T. (ºC)'] = pd.to_numeric(X['Ave. T. (ºC)'], errors='coerce')
X['Max. T. (ºC)'] = pd.to_numeric(X['Max. T. (ºC)'], errors='coerce')

X = X.dropna()
print(X.describe())
days = []
critical_days = []
y = []
file_med = open('medical.txt', 'r')
lines = file_med.readlines()
for line in lines:
    critical_days.append(pd.to_datetime(line.strip(), dayfirst=True, format="%d.%m.%Y").strftime("%Y/%m/%d"))
print(critical_days)
for index, row in X.iterrows():
    values = row.values[1:]
    days.append(values.astype(float))
    if row.values[0] in critical_days:
        y.append(1)
    else:
        y.append(0)
print(y)
X.pop('Date')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# clf = DecisionTreeClassifier(criterion='gini')
clf = RandomForestClassifier()

clf = clf.fit(X_train, y_train)

feature_importances = clf.feature_importances_

sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_names = X.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

plt.figure(figsize=(17, 8.27))
plt.bar(sorted_feature_names, sorted_importances)
# plt.savefig("gini.png")
plt.savefig("randomForest.png")

