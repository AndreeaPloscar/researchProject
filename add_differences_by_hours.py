from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as plot
import glob
import os
import xlrd
import openpyxl


def get_difference(frame, col, date, hour_start, hour_end):
    date_time = pd.to_datetime(date)
    date_time += pd.Timedelta(days=1)
    date_time = date_time.date().strftime("%Y/%m/%d")
    if hour_end in ['00Z', '01Z', '02Z']:
        val1 = frame.loc[(frame['UTC time'] == hour_end) & (frame['Date'] == date_time)].iloc[0][col]
    else:
        val1 = frame.loc[(frame['UTC time'] == hour_end) & (frame['Date'] == date)].iloc[0][col]
    val2 = frame.loc[(frame['UTC time'] == hour_start) & (frame['Date'] == date)].iloc[0][col]
    if val1 == 'NaN' or val2 == 'NaN':
        return 'NaN'
    try:
        return val1 - val2
    except ValueError:
        return 'NaN'


# ======== CHOOSE CITY ========

# -------- CLUJ-NAPOCA --------

# path = '/Users/andreeaploscar/Desktop/THESIS/DATE-METEO/CLUJ-NAPOCA/HOURS'
# start_date = "2013-01-01"
# path_days = "cluj_preprocessed.xlsx"

# ---------- ORADEA -----------

path = '/Users/andreeaploscar/Desktop/THESIS/DATE-METEO/ORADEA/HOURS'
start_date = "2014-07-01"
path_days = "oradea_preprocessed.xlsx"


# =============================
#
file_days = glob.glob(path_days)
data_days = pd.read_excel(file_days[0])
dataframe_days = pd.DataFrame(data_days)

file_list = glob.glob(path + "/*.xls")
main_dataframe = pd.DataFrame()
for i in range(0, len(file_list)):
    data = pd.read_excel(file_list[i])
    df = pd.DataFrame(data)
    df = df[['Date', 'UTC time', 'Temp. (ºC)', 'Rel. Hum. (%)', 'Pressure/ Geopot.', 'Wind dir', 'Wins speed (Km/h)']]
    df['Pressure/ Geopot.'] = pd.to_numeric(df['Pressure/ Geopot.'].str.replace('Hpa', ''))
    df['Rel. Hum. (%)'] = df['Rel. Hum. (%)'].str.replace('%', '')
    df['Temp. (ºC)'] = pd.to_numeric(df['Temp. (ºC)'], errors='coerce')
    df['Rel. Hum. (%)'] = pd.to_numeric(df['Rel. Hum. (%)'], errors='coerce')
    df['Wind dir'] = pd.to_numeric(df['Wind dir'].str.replace(r"º \([NESW]{1,2}[ ]{0,1}\)", '', regex=True),
                                   errors='coerce')
    df['Wins speed (Km/h)'] = pd.to_numeric(df['Wins speed (Km/h)'], errors='coerce')
    main_dataframe = pd.concat([main_dataframe, df], axis=0)

print(main_dataframe.describe(include='all'))
main_dataframe.rename({'Wins speed (Km/h)': 'Wind speed (Km/h)'}, axis='columns', inplace=True)
main_dataframe.replace(np.nan, 'NaN', inplace=True)
main_dataframe['Date'] = pd.to_datetime(main_dataframe['Date'], dayfirst=True, format="%d/%m/%Y")
main_dataframe = main_dataframe.sort_values(by='Date')

print(pd.date_range(start=start_date, end="2021-12-31").difference(main_dataframe['Date']))
main_dataframe['Date'] = main_dataframe['Date'].dt.strftime('%Y/%m/%d')
main_dataframe.to_excel("main.xlsx", index=False)
dataframe_days = data_days.reset_index()

for i in range(0, 24):
    hour_end = str((i + 3) % 24).zfill(2) + 'Z'
    hour_start = str(i).zfill(2) + 'Z'
    hours = hour_end + "-" + hour_start
    dataframe_days = pd.concat(
        [
            dataframe_days,
            pd.DataFrame(
                [['NaN', 'NaN', 'NaN', 'NaN', 'NaN']],
                index=dataframe_days.index,
                columns=['Temp. (ºC) ' + hours, 'Rel. Hum. (%) ' + hours, 'Wind dir ' + hours,
                         'Wind speed (Km/h) ' + hours, 'Pressure/ Geopot. ' + hours]
            )
        ], axis=1
    )


for i in range(0, 24):
    hour_end = str((i + 3) % 24).zfill(2) + 'Z'
    hour_start = str(i).zfill(2) + 'Z'
    hours = hour_end + "-" + hour_start
    partial_df = main_dataframe.loc[(main_dataframe['UTC time'] == hour_end) | (main_dataframe['UTC time'] == hour_start)]
    for index, row in dataframe_days.iterrows():
        date = row['Date']
        try:
            dataframe_days.at[index, 'Temp. (ºC) ' + hours] = get_difference(partial_df, 'Temp. (ºC)', date,
                                                                             hour_start, hour_end)
            dataframe_days.at[index, 'Rel. Hum. (%) ' + hours] = get_difference(partial_df, 'Rel. Hum. (%)', date,
                                                                                hour_start, hour_end)
            dataframe_days.at[index, 'Wind dir ' + hours] = get_difference(partial_df, 'Wind dir', date, hour_start,
                                                                           hour_end)
            dataframe_days.at[index, 'Wind speed (Km/h) ' + hours] = get_difference(partial_df, 'Wind speed (Km/h)',
                                                                                    date, hour_start, hour_end)
            dataframe_days.at[index, 'Pressure/ Geopot. ' + hours] = get_difference(partial_df, 'Pressure/ Geopot.',
                                                                                    date, hour_start, hour_end)
        except IndexError:
            print(date, hours)

dataframe_days.to_excel(path_days, index=False)
