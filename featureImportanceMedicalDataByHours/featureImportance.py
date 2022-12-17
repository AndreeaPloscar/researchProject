import glob
import math
from datetime import timedelta

import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('TkAgg')
path_days = "cluj_preprocessed.xlsx"

file_days = glob.glob(path_days)
data_days = pd.read_excel(file_days[0])


gender = {'F': 1, 'M': 0}
bool = {'da': 1, 'nu': 0}
X = pd.DataFrame(data_days)
X.pop('index')
X.pop('Statie')
X['Ave. T. (ºC)'] = pd.to_numeric(X['Ave. T. (ºC)'], errors='coerce')
X['Max. T. (ºC)'] = pd.to_numeric(X['Max. T. (ºC)'], errors='coerce')


X = X.dropna()
print(X.describe())

critical_days = []
y = []
file_med = open('medical.txt', 'r')
lines = file_med.readlines()
for line in lines:
    critical_days.append(pd.to_datetime(line.strip(), dayfirst=True, format="%d.%m.%Y").strftime("%Y/%m/%d"))
print(critical_days)
for index, row in X.iterrows():
    date = row['Date']
    print(date)
    if date in critical_days:
        y.append(1)
    else:
        y.append(0)
X['Date'] = pd.to_datetime(X['Date'], dayfirst=True, format="%Y/%m/%d")
print(y)
maxT = [[] for i in range(4)]
maxP = [[] for i in range(4)]
temp = [0 for i in range(4)]
pressure = [0 for i in range(4)]
for datetime in X['Date']:
    # datetime = pd.to_datetime(date, dayfirst=True, format="%Y/%m/%d")
    T = [0, 0, 0, 0]
    P = [0, 0, 0, 0]
    if datetime.year >= 2013:
        for index in range(0, 24):
            hour_end = str((index + 3) % 24).zfill(2) + 'Z'
            hour_start = str(index).zfill(2) + 'Z'
            hours = hour_end + "-" + hour_start
            for i in range(4):
                try:
                    day = (datetime - timedelta(days=i)).strftime("%Y/%m/%d")
                    # print(day)
                    temp[i] = X.loc[(X['Date'] == day)].iloc[0]['Temp. (ºC) ' + hours]
                    pressure[i] = X.loc[(X['Date'] == day)].iloc[0]['Pressure/ Geopot. ' + hours]
                except:
                    continue
            for i in range(4):
                if temp[i]:
                    T[i] = max(T[i], math.fabs(temp[i]))
                if pressure[i]:
                    P[i] = max(P[i], math.fabs(pressure[i]))

        for i in range(4):
            maxT[i].append(T[i])
            maxP[i].append(P[i])

X['Max T today'] = maxT[0]
X['Max P today'] = maxP[0]

X = X[['Max T today', 'Max P today', 'Ave. T. (ºC)', 'Max. T. (ºC)', 'Min. T. (ºC)', 'S.L.Press./ Gheopot.']]

X['Max T 1 day ago'] = maxT[1]
X['Max P 1 day ago'] = maxP[1]
X['Max T 2 days ago'] = maxT[2]
X['Max P 2 days ago'] = maxP[2]
X['Max T 3 days ago'] = maxT[3]
X['Max P 3 days ago'] = maxP[3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# clf = DecisionTreeClassifier(criterion='gini')
# clf = RandomForestClassifier()
clf = SVC(kernel='linear', class_weight='balanced', C=1.0)

clf = clf.fit(X_train, y_train)

feature_importances = clf.coef_

sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_names = X.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

plt.figure(figsize=(17, 8.27))
plt.bar(sorted_feature_names, sorted_importances)
plt.savefig("svc.png")
# plt.savefig("gini.png")
# plt.savefig("randomForest.png")