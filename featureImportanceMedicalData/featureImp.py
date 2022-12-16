import numpy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('TkAgg')
medical_file = "Anevrisme_Meteo_Final_aranjat_pe_date_AI.xlsx"
medical_data = pd.read_excel(medical_file)
gender = {'F': 1, 'M': 0}
bool = {'da': 1, 'nu': 0}
X = pd.DataFrame(medical_data,
                 columns=['Sex', 'Varsta', 'Hunt and Hess', 'Inundare ventriculara',
                          'Hematom intraparenchimatos',
                          'GCS internare', 'Latime colet (mm)',
                          'Inaltime dom (mm)',
                          'Numar clipuri','Deces']).dropna(subset='Deces')
X.pop('Deces')
X['Sex'] = X['Sex'].map(gender)
X['Hematom intraparenchimatos'] = X['Hematom intraparenchimatos'].map(bool)
X['GCS internare'] = X['GCS internare'].map(bool)
X['Inundare ventriculara'] = X['Inundare ventriculara'].map(bool)
X['Latime colet (mm)'] = pd.to_numeric(X['Latime colet (mm)'], errors='coerce')
X['Inaltime dom (mm)'] = pd.to_numeric(X['Inaltime dom (mm)'], errors='coerce')

X = X.replace(numpy.nan, 0)
result = pd.DataFrame(medical_data,columns=['Deces']).dropna()
result['Deces'] = result['Deces'].map(bool)
y = result['Deces'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

clf = DecisionTreeClassifier(criterion='gini')
# clf = RandomForestClassifier()

clf = clf.fit(X_train, y_train)

feature_importances = clf.feature_importances_

sorted_indices = feature_importances.argsort()[::-1]
sorted_feature_names = X.columns[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

plt.figure(figsize=(17, 8.27))
plt.bar(sorted_feature_names, sorted_importances)
plt.savefig("gini.png")
# plt.savefig("randomForest.png")

