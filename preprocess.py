import numpy as np
import pandas as pd
import matplotlib as plot
import glob
import os
import xlrd
import openpyxl

# ======== CHOOSE CITY ========

# -------- CLUJ-NAPOCA --------

# path = '/Users/andreeaploscar/Desktop/THESIS/DATE-METEO/CLUJ-NAPOCA/DAYS'
# start_date = "2013-01-01"
# output_file = "cluj_preprocessed.xlsx"
# city = 'Cluj-Napoca'

# ---------- ORADEA -----------

path = '/Users/andreeaploscar/Desktop/THESIS/DATE-METEO/ORADEA/DAYS'
start_date = "2014-07-01"
output_file = "oradea_preprocessed.xlsx"
city = 'Oradea'

# =============================

file_list = glob.glob(path + "/*.xls")
main_dataframe = pd.DataFrame()
for i in range(0, len(file_list)):
    data = pd.read_excel(file_list[i])
    df = pd.DataFrame(data)
    df.drop("Snow depth (cm)", axis=1, inplace=True, errors='ignore')
    df['S.L.Press./ Gheopot.'] = df['S.L.Press./ Gheopot.'].str.replace('Hpa', '')
    df['Cloud c.'] = pd.to_numeric(df['Cloud c.'].str.replace(r'/8', ''), errors='coerce')
    df['Wind dir'] = df['Wind dir'].str.replace(r"º\([NESW]{1,2}\)", '', regex=True)
    df['Insolat. (hours)'] = pd.to_numeric(df['Insolat. (hours)'], errors='coerce')
    main_dataframe = pd.concat([main_dataframe, df], axis=0)

print(main_dataframe.describe(include='all'))

main_dataframe['Date'] = pd.to_datetime(main_dataframe['Date'], dayfirst=True, format="%d/%m/%Y")
main_dataframe = main_dataframe.sort_values(by='Date')

print(pd.date_range(start=start_date, end="2021-12-31").difference(main_dataframe['Date']))
main_dataframe['Date'] = main_dataframe['Date'].dt.strftime('%Y/%m/%d')

for index, row in main_dataframe.iterrows():
    if row['Prec. (mm)'] == 'Tr':
        main_dataframe.at[index, 'Prec. (mm)'] = 0.01

main_dataframe.insert(0, "Statie", city)

main_dataframe.to_excel(output_file, index=False)
