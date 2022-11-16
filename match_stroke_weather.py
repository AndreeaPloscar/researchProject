import numpy
import pandas as pd
import glob
import matplotlib
from bokeh.plotting import figure
from pandas_bokeh import show

matplotlib.use('TkAgg')

path_days = "cluj_preprocessed.xlsx"

file_days = glob.glob(path_days)
data_days = pd.read_excel(file_days[0])
dataframe_days = pd.DataFrame(data_days)


dataframe_days['Date'] = pd.to_datetime(data_days['Date'], dayfirst=False, format="%Y/%m/%d")

# dataframe_days['Ave. T. (ºC)'] = pd.to_numeric(dataframe_days['Ave. T. (ºC)'], errors='coerce')
print(dataframe_days)
# dataframe_days.plot_bokeh.scatter(x='Date', y='Ave. T. (ºC)')

weather_special_days = []

for day in dataframe_days['Date']:
    special = False
    for i in range(0, 24):
        hour_end = str((i + 3) % 24).zfill(2) + 'Z'
        hour_start = str(i).zfill(2) + 'Z'
        hours = hour_end + "-" + hour_start
        temp = dataframe_days.loc[(dataframe_days['Date'] == day)].iloc[0]['Temp. (ºC) ' + hours]
        pressure = dataframe_days.loc[(dataframe_days['Date'] == day)].iloc[0]['Pressure/ Geopot. ' + hours]
        if temp != numpy.nan and pressure != numpy.nan:
            special = True
    if special:
        weather_special_days.append(day)

print(weather_special_days)
