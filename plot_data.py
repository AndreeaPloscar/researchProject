import pandas as pd
import glob
import matplotlib
from bokeh.plotting import figure

matplotlib.use('TkAgg')

# -------- CLUJ-NAPOCA --------

path_days = "cluj_preprocessed.xlsx"

# ---------- ORADEA -----------

# path = '/Users/andreeaploscar/Desktop/THESIS/DATE-METEO/ORADEA/HOURS'
# start_date = "2014-07-01"
# path_days = "oradea_preprocessed.xlsx"

# =============================

file_days = glob.glob(path_days)
data_days = pd.read_excel(file_days[0])
dataframe_days = pd.DataFrame(data_days)

dataframe_days['Date'] = pd.to_datetime(data_days['Date'], dayfirst=False, format="%Y/%m/%d")

# dataframe_days['Ave. T. (ºC)'] = pd.to_numeric(dataframe_days['Ave. T. (ºC)'], errors='coerce')
print(dataframe_days)
# dataframe_days.plot_bokeh.scatter(x='Date', y='Ave. T. (ºC)')
p = figure(plot_width=1500, plot_height=800, x_axis_type='datetime')
p.xaxis.axis_label = 'Date'
p.yaxis.axis_label = 'Temp. (ºC) 3hrs diff'

for i in range(0, 24):
    hour_end = str((i + 3) % 24).zfill(2) + 'Z'
    hour_start = str(i).zfill(2) + 'Z'
    hours = hour_end + "-" + hour_start
    p.circle(x=dataframe_days['Date'], y=dataframe_days['Temp. (ºC) ' + hours], line_color='navy', fill_color='orange', fill_alpha=0.5)

show(p)


# matplotlib.interactive(True)
# plt.savefig('figure.png')
# plt.show()
