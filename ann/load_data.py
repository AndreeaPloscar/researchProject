import math
from datetime import timedelta

import pandas as pd


def load_data(weather_records, medical_records):
    days = []
    critical_days = []
    classes = []
    file_med = open(medical_records, 'r')
    lines = file_med.readlines()
    for line in lines:
        critical_days.append(pd.to_datetime(line.strip(), dayfirst=True, format="%d.%m.%Y").strftime("%Y/%m/%d"))
    excel_data = pd.read_excel(weather_records)
    data = pd.DataFrame(excel_data)
    maxT = [[] for i in range(4)]
    maxP = [[] for i in range(4)]
    temp = [0 for i in range(4)]
    pressure = [0 for i in range(4)]
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, format="%Y/%m/%d")
    for datetime in data['Date']:
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
                        temp[i] = data.loc[(data['Date'] == day)].iloc[0]['Temp. (ºC) ' + hours]
                        pressure[i] = data.loc[(data['Date'] == day)].iloc[0]['Pressure/ Geopot. ' + hours]
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

    data = data[['Date','Ave. T. (ºC)', 'Max. T. (ºC)', 'Min. T. (ºC)', 'S.L.Press./ Gheopot.']]

    data['Max T today'] = maxT[0]
    data['Max P today'] = maxP[0]
    data['Max T 1 day ago'] = maxT[1]
    data['Max P 1 day ago'] = maxP[1]
    data['Max T 2 days ago'] = maxT[2]
    data['Max P 2 days ago'] = maxP[2]
    data['Max T 3 days ago'] = maxT[3]
    data['Max P 3 days ago'] = maxP[3]

    for index, row in data.iterrows():
        values = row.values[1:]
        days.append(values.astype(float))
        if row.values[0].strftime("%Y/%m/%d") in critical_days:
            classes.append(1)
        else:
            classes.append(0)
    return days, classes
