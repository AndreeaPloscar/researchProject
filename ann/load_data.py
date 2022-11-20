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
    for index, row in data.iterrows():
        values = row.values[1:]
        days.append(values.astype(float))
        if row.values[0] in critical_days:
            classes.append(1)
        else:
            classes.append(0)
    return days, classes
