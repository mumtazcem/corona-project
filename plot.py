import datetime
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def search_trend_korea():
    search_trend = pd.read_csv('archive/SearchTrend.csv')
    search_trend_date = search_trend[(search_trend['date'] > '2020-01-19')]
    date = search_trend_date['date'].tolist()
    date_time_obj = []
    for i in date:
        obj = datetime.datetime.strptime(i, '%Y-%m-%d')
        date_time_obj.append(obj.date())

    cold = search_trend_date['cold'].tolist()
    flu = search_trend_date['flu'].tolist()
    pneumonia = search_trend_date['pneumonia'].tolist()
    coronavirus = search_trend_date['coronavirus'].tolist()
    plt.plot(date_time_obj, cold, c='red', label="Cold")
    plt.plot(date_time_obj, flu, c='blue', alpha=0.5, label="Flu")
    plt.plot(date_time_obj, pneumonia, c='green', alpha=0.5, label="Pneumonia")
    plt.plot(date_time_obj, coronavirus, c='purple', label="Coronavirus")
    plt.xlabel("Date")
    plt.ylabel("The Search Volume")
    plt.title("The Search Volume of Keywords in Trends")
    plt.legend()
    plt.savefig("coronavirus_search_trend_korea.png")
    plt.show()


def search_trend_ger():
    search_trend = pd.read_csv('trend/trend_all.csv')
    search_trend = search_trend.drop(search_trend.columns[0], axis=1)
    search_trend = search_trend[(search_trend['date'] > '2020-09-10')]
    date = search_trend['date'].tolist()
    date_time_obj = []
    for i in date:
        obj = datetime.datetime.strptime(i, '%Y-%m-%d')
        date_time_obj.append(obj.date())
    search_trend = search_trend.drop(search_trend.columns[0], axis=1)
    for column in list(search_trend):
        data = search_trend[column].tolist()
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        plt.plot(date_time_obj, data, c=color, label=column)
        if column == 'Brandenburg':
            break

    plt.xlabel("Date")
    plt.ylabel("Trend Value")
    plt.title("Trends of some Germany Provinces When Second Wave Happened")
    plt.legend(loc=2, prop={'size': 6})
    plt.legend()

    plt.savefig("coronavirus_search_trend_ger.png")
    plt.show()


def daily_all_ger():
    time_prov = pd.read_csv('corona_data/covid_19_data.csv')
    time_prov_ = time_prov.filter(['ObservationDate', 'Province/State', 'Country/Region', 'Confirmed'])
    time_ger = time_prov_[time_prov_['Country/Region'].str.contains("Germany")]
    time_ger = time_ger.dropna()

    time_ger = time_ger[time_ger['Province/State'] != 'Bavaria']
    time_ger = time_ger[time_ger['Province/State'] != 'Unknown']

    germany_states = ['Baden-Wurttemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg',
                      'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen',
                      'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt',
                      'Schleswig-Holstein', 'Thuringen']
    for index in range(len(germany_states)):
        if index > 4:
            time_ger = time_ger[time_ger['Province/State'] != germany_states[index]]

    time_ger.reset_index(drop=True, inplace=True)
    time_ger.sort_values(by=['Province/State', 'ObservationDate'], inplace=True, ascending=True)
    time_ger['daily'] = time_ger['Confirmed'] - time_ger['Confirmed'].shift(1)
    time_ger.fillna(0, inplace=True)
    time_ger['daily'] = time_ger['daily'].apply(lambda row: 0 if row < 0 or row == 4292 else row)
    time_ger = time_ger.drop(columns=['Country/Region', 'Confirmed'])

    date = time_ger['ObservationDate'].tolist()
    date_time_obj = []
    for i in date:
        obj = datetime.datetime.strptime(i, '%m/%d/%Y')
        date_time_obj.append(obj.date())
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.scatterplot(x=date_time_obj, y='daily', hue="Province/State", data=time_ger, ax=ax)
    sns.lineplot(x=date_time_obj, y='daily', hue="Province/State", data=time_ger)
    plt.xlabel("Date")
    plt.ylabel("Daily Cases")
    plt.title('Daily Coronavirus Cases In Germany')
    plt.yticks(np.arange(time_ger['daily'].min(), time_ger['daily'].max(), 1000))
    plt.savefig('figures/daily/some_provinces.png')
    plt.show()
    plt.clf()


def daily_all_korea():
    date_after = '2020-01-19'
    time_prov = pd.read_csv('archive/TimeProvince.csv')
    time_prov_ = time_prov[['province',
                            'confirmed',
                            'date']]
    # Last recorded day is 2020-06-29
    time_prov_ = time_prov_[(time_prov_['date'] < '2020-06-30') & (time_prov_['date'] > date_after)]
    # Remove Sejong, as it is not presented in weather
    time_prov_ = time_prov_[(time_prov_['province'] != 'Sejong')]
    time_prov_.reset_index(drop=True, inplace=True)
    time_prov_.sort_values(by=['province', 'date'], inplace=True, ascending=True)
    time_prov_['daily'] = time_prov_['confirmed'] - time_prov_['confirmed'].shift(1)
    time_prov_.fillna(0, inplace=True)
    time_prov_['daily'] = time_prov_['daily'].apply(lambda row: 0 if row < 0 else row)
    print(time_prov_['daily'].describe())
    # cumulative status
    time_prov_cumulative = time_prov_.groupby(by='date').sum().reset_index()
    date = time_prov_cumulative['date'].tolist()
    date_time_obj = []
    for i in date:
        obj = datetime.datetime.strptime(i, '%Y-%m-%d')
        date_time_obj.append(obj.date())
    sns.lineplot(x=date_time_obj, y="daily", data=time_prov_cumulative)
    plt.title('Daily Confirmed Cases in Korea')
    plt.savefig('daily_korea.png')
    plt.show()
    plt.clf()


search_trend_ger()
