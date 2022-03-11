# Mümtaz Cem Eriş
# 504191531
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Coronavirus daily case prediction for Germany States.
# Figures and plots are commented out. Make sure to delete
# comments if plots are desired to be printed.

seed = 1075
np.random.seed(seed)
# daily_confirmed_threshold = 500
# daily_confirmed_threshold1 = 1000
# daily_confirmed_threshold2 = 2500
# daily_confirmed_threshold3 = 5000
daily_confirmed_threshold = 100
daily_confirmed_threshold1 = 200
daily_confirmed_threshold2 = 300
daily_confirmed_threshold3 = 400
k_fold = 5
# Date 2020-05-14 to 2020-12-06
# Germany States:
germany_states = ['Baden-Wurttemberg', 'Bayern', 'Berlin', 'Brandenburg', 'Bremen', 'Hamburg',
                  'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen', 'Nordrhein-Westfalen',
                  'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt',
                  'Schleswig-Holstein', 'Thuringen']

# Classifiers
et = ExtraTreesClassifier()
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()


def correlationGraphs(merged_df):
    f, ax = plt.subplots(figsize=(10, 8))
    merged_df = merged_df.rename(columns={"daily": "Daily COVID-19 Cases", "Bayern": "Trend(Bayern)", "Berlin": "Trend(Berlin)", "Baden-Wurttemberg": "Trend(BadenW.)"})
    corr = merged_df.corr()
    # Heatmap
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax,  annot=True)
    plt.title('Correlations Heatmap')
    plt.savefig('correlation_heatmap_ger.png')
    plt.show()
    plt.clf()

    # Scatter plots
    sns.scatterplot(x="Daily COVID-19 Cases", y="Trend(Bayern)", data=merged_df)
    plt.title('Daily & GT results of Bayern' + ' p = ' + str(corr.iat[1, 3]))
    plt.savefig('correlation_scatter_d_bayern.png')
    plt.show()
    plt.clf()

    sns.scatterplot(x="Daily COVID-19 Cases", y="Relative Humidity", data=merged_df)
    # pd.plotting.scatter_matrix(merged_df, figsize=(20, 20))
    plt.title('Daily & Relative Humidity' + ' p = ' + str(corr.iat[0, 1]))
    plt.savefig('correlation_scatter_d_humid.png')
    plt.show()
    plt.clf()


def labelDailyCases(row):
    if row['daily'] > daily_confirmed_threshold3:
        return 4
    elif row['daily'] > daily_confirmed_threshold2:
        return 3
    elif row['daily'] > daily_confirmed_threshold1:
        return 2
    elif row['daily'] > daily_confirmed_threshold:
        return 1
    else:
        return 0


def addLabels(combined):
    combined['label'] = combined.apply(labelDailyCases, axis=1)
    return combined


def trainModel(Xtra_r, Ytra, model):
    # train model using given model
    model.fit(Xtra_r, Ytra)
    return model


def predict(model, Xtst_r):
    prediction = model.predict(Xtst_r)
    return prediction


def writeOutput(prediction, ytst, label):
    # compare prediction and ytst
    accuracy = accuracy_score(ytst, prediction)
    print("Accuracy: {0:.3f},  [{1} is used.]".format(accuracy, label))


time_prov = pd.read_csv('corona_data/covid_19_data.csv')
weather_frames = []
for state in germany_states:
    csv_str = "weather/" + state + ".csv"
    weather = pd.read_csv(csv_str)
    weather_frames.append(weather)

search_trend = pd.read_csv('trend/trend_all.csv')

search_trend_data = search_trend.drop(search_trend.columns[0], axis=1)
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=search_trend_data, ax=ax)
plt.title('Coronavirus Search Trend')
plt.savefig('figures/coronavirus_search_trend.png')
plt.show()
plt.clf()

weather_all = pd.concat(weather_frames, ignore_index=True)
weather_all = weather_all.filter(['Address', 'Date time', 'Relative Humidity'])

# Germany
# Date 2020-05-14 to 2020-12-06
time_prov_ = time_prov.filter(['ObservationDate', 'Province/State', 'Country/Region', 'Confirmed'])
time_ger = time_prov_[time_prov_['Country/Region'].str.contains("Germany")]
time_ger = time_ger.dropna()
time_ger = time_ger[time_ger['Province/State'] != 'Bavaria']
time_ger = time_ger[time_ger['Province/State'] != 'Unknown']
sns.scatterplot(x="Confirmed", y="Province/State", data=time_ger)
plt.title('Provinces')
plt.savefig('figures/germany_provinces.png')
plt.show()
plt.clf()
print(time_ger['Province/State'].describe())
print(time_ger['Province/State'].unique())

time_ger.reset_index(drop=True, inplace=True)
time_ger.sort_values(by=['Province/State', 'ObservationDate'], inplace=True, ascending=True)
time_ger['daily'] = time_ger['Confirmed'] - time_ger['Confirmed'].shift(1)
time_ger.fillna(0, inplace=True)
time_ger['daily'] = time_ger['daily'].apply(lambda row: 0 if row < 0 or row == 4292 else row)
time_ger = time_ger.drop(columns=['Country/Region', 'Confirmed'])

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='ObservationDate', y='daily', hue="Province/State", data=time_ger, ax=ax)
plt.title('Daily All')
plt.yticks(np.arange(time_ger['daily'].min(), time_ger['daily'].max(), 200))
plt.savefig('figures/daily/all.png')
plt.show()
plt.clf()
print(time_ger)
print(time_ger['daily'].describe())


# Refactor column names
weather_all = weather_all.rename(columns={'Date time': 'date', 'Address': 'province'})
time_ger = time_ger.rename(columns={'ObservationDate': 'date', 'Province/State': 'province'})

# Date conversion
weather_all['date'] = pd.to_datetime(weather_all['date']).dt.date
time_ger['date'] = pd.to_datetime(time_ger['date']).dt.date
search_trend_data['date'] = pd.to_datetime(search_trend_data['date']).dt.date

# Thuringia - Thuringen
weather_all['province'] = weather_all['province'].apply(lambda row: "Thuringen" if row == "Thuringia" else row)
weather_for_scatter = weather_all[weather_all['province'].str.contains("Ba")]
sns.scatterplot(x='date', y="Relative Humidity", hue="province", data=weather_for_scatter)
plt.title('Relative Humility')
plt.savefig('figures/humidity.png')
plt.show()
plt.clf()

# Merge weather, time province and search trend
combined = pd.merge(weather_all, time_ger, on=['province', 'date'])
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='date', y='province', data=combined, ax=ax)
plt.title('Combined')
plt.savefig('figures/combined.png')
plt.show()
plt.clf()
combined_all = pd.merge(combined, search_trend_data, on=['date'])

# Print correlation graphs
data_for_correlation = combined_all[combined_all['province'] == 'Berlin']
correlationGraphs(data_for_correlation.drop(columns=['Brandenburg', 'Bremen', 'Hamburg',
                                                     'Hessen', 'Mecklenburg-Vorpommern', 'Niedersachsen',
                                                     'Nordrhein-Westfalen',
                                                     'Rheinland-Pfalz', 'Saarland', 'Sachsen', 'Sachsen-Anhalt',
                                                     'Schleswig-Holstein', 'Thuringen']))

# Add labels using daily_confirmed_threshold
combined_labeled = addLabels(combined_all)
print(combined_labeled['label'].describe())
combined_labeled['index'] = combined_labeled.index
plt.clf()
sns.scatterplot(x='index', y="daily", hue="label", data=combined_labeled)
plt.title('Coronavirus Daily Labeled')
plt.savefig('figures/coronavirus_daily_labeled.png')
plt.show()
plt.clf()

# Encoding provinces with 0s and 1s
encoder = OneHotEncoder(categories='auto', sparse=False)
encoder.fit(np.array(combined_labeled['province']).reshape(-1, 1))
inf_dummies = encoder.transform(np.array(combined_labeled['province']).reshape(-1, 1))
dummies = pd.DataFrame(inf_dummies.astype(int), columns=['province_' + x for x in encoder.categories_[0]])
combined_labeled = combined_labeled.join(dummies)
# Drop province, index and daily column as it is not needed anymore
combined_ready = combined_labeled.drop(columns=['province', 'daily', 'index', 'date'])
print(combined_ready.head(10))

# K-fold cross validation
kf = KFold(n_splits=k_fold, shuffle=True, random_state=2)
X = combined_ready.drop(columns=['label']).to_numpy()
Y = combined_ready['label'].to_numpy()
combined_arr = combined_ready.to_numpy()
labels_clf = ['ExtraTrees', 'KNeighbors', 'SVC', 'Ridge']
i = 1

for train_index, test_index in kf.split(combined_arr):
    print("***", i, "-Fold Results***")
    xtra, xtst = X[train_index], X[test_index]
    ytra, ytst = Y[train_index], Y[test_index]

    # Classifiers cross-validation
    # Each of models would be trained and their
    # cross validation score would be printed.
    for model, label in zip([et, knn, svc, rg], labels_clf):
        trained_model = trainModel(xtra, ytra, model)
        prediction = predict(trained_model, xtst)
        writeOutput(prediction, ytst, label)

    i += 1
