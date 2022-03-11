# Mümtaz Cem Eriş
# 504191531
import csv
import codecs
import urllib.request

# Retrieve weather data using Visual Crossing API

BaseURL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/weatherdata/'

QueryLocation = "&location=Thuringia"
QueryType = "HISTORY"
QueryKey = "&key=API_KEY"  # API KEY is needed
FromDateParam = "2020-05-14"
ToDateParam = "2020-12-06"

QueryDate = '&startDateTime=' + FromDateParam + 'T00:00:00&endDateTime=' + ToDateParam + 'T00:00:00'
QueryTypeParams = 'history?&aggregateHours=24&unitGroup=us&dayStartTime=0:0:00&dayEndTime=0:0:00' + QueryDate

URL = BaseURL + QueryTypeParams + QueryLocation + QueryKey
print(' - Running query URL: ', URL)
CSVBytes = urllib.request.urlopen(URL)
CSVText = csv.reader(codecs.iterdecode(CSVBytes, 'utf-8'))
with open("weather/Thuringen.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for Row in CSVText:
        wr.writerow(Row)
RowIndex = 0

for Row in CSVText:
    if RowIndex == 0:
        FirstRow = Row
    else:
        print('Weather in ', Row[0], ' on ', Row[1])
        ColIndex = 0
        for Col in Row:
            if ColIndex >= 4:
                print('   ', FirstRow[ColIndex], ' = ', Row[ColIndex])
            ColIndex += 1
    RowIndex += 1
