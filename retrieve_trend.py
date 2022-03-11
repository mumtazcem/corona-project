# Mümtaz Cem Eriş
# 504191531
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from pytrends.request import TrendReq

# Retrieve Google Trends using pytrends

pytrend = TrendReq()

KEYWORDS = ['Coronavirus']
KEYWORDS_CODES = [pytrend.suggestions(keyword=i)[0] for i in KEYWORDS]
df_CODES = pd.DataFrame(KEYWORDS_CODES)

EXACT_KEYWORDS = df_CODES['mid'].to_list()
# Date 2020-05-14 to 2020-12-06
DATE_INTERVAL = '2020-05-14 2020-12-06'

germany_states = ['Baden-Wurttemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg',
                  'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen',
                  'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt',
                  'Schleswig-Holstein', 'Thuringen']

STATE = ['DE-BW', 'DE-BY', 'DE-BE', 'DE-BB', 'DE-HB', 'DE-HH',
         'DE-HE', 'DE-MV', 'DE-NI', 'DE-NW',
         'DE-RP', 'DE-SL', 'DE-SN', 'DE-ST',
         'DE-SH', 'DE-TH']
CATEGORY = 0  # Use this link to select categories
SEARCH_TYPE = ''  # default is 'web searches',others include 'images','news','youtube','froogle' (google shopping)

Individual_EXACT_KEYWORD = list(zip(*[iter(EXACT_KEYWORDS)] * 1))
Individual_EXACT_KEYWORD = [list(x) for x in Individual_EXACT_KEYWORD]
dicti = {}
i = 1
for state in STATE:
    for keyword in Individual_EXACT_KEYWORD:
        pytrend.build_payload(kw_list=keyword,
                              timeframe=DATE_INTERVAL,
                              geo=state,
                              cat=CATEGORY,
                              gprop=SEARCH_TYPE)
        dicti[i] = pytrend.interest_over_time()
        i += 1
df_trends = pd.concat(dicti, axis=1)

df_trends.columns = df_trends.columns.droplevel(0)  # drop outside header
df_trends = df_trends.drop('isPartial', axis=1)  # drop "isPartial"
df_trends.reset_index(level=0, inplace=True)  # reset_index
df_trends.columns = ['date', 'Baden-Wurttemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg',
                     'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen',
                     'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt',
                     'Schleswig-Holstein', 'Thuringen']  # change column names
with open("trend/trend_all.csv", 'w', newline='') as myfile:
    df_trends.to_csv("trend/trend_all.csv")
print(df_trends)

