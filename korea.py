# Mümtaz Cem Eriş
# 504191531
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Coronavirus daily case prediction for Korea States.

seed = 1075
np.random.seed(seed)
daily_confirmed_threshold = 5
daily_confirmed_threshold1 = 25
daily_confirmed_threshold2 = 125
daily_confirmed_threshold3 = 625
k_fold = 5
# originally cases start at '2020-01-19'
date_after = '2020-01-19'

# Classifiers
et = ExtraTreesClassifier()
knn = KNeighborsClassifier()
svc = SVC()
rg = RidgeClassifier()
gnb = GaussianNB()


def preprocessing(weather, time_prov):
    # starting from 2020-01-20 to 2020-06-30, 163 days
    ### Weather
    weather_ = weather[['province',
                        'max_wind_speed',
                        'avg_relative_humidity',
                        'date']]
    # First recorded corona date is 2020-01-20
    weather_ = weather_[(weather_['date'] > date_after)]
    weather_.reset_index(drop=True, inplace=True)
    # Correct misspelled province
    weather_['province'] = weather_.apply(lambda row:
                                          row['province'] if row[
                                                                 'province'] != 'Chunghceongbuk-do' else 'Chungcheongbuk-do'
                                          , axis=1)

    ### TimeProvinces
    time_prov_ = time_prov[['province',
                            'confirmed',
                            'date']]
    # Last recorded day is 2020-06-29
    time_prov_ = time_prov_[(time_prov_['date'] < '2020-06-30') & (time_prov_['date'] > date_after)]
    plt.figure(figsize=(12, 6))
    sns.displot(time_prov_, y="province")
    plt.savefig("weather_data_provinces.png")
    plt.title('Weather data provinces')
    plt.show()
    plt.clf()
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
    sns.scatterplot(x="date", y="daily", data=time_prov_cumulative)
    plt.title('Daily confirmed cases')
    plt.show()
    plt.savefig('daily_confirmed_cases.png')
    plt.clf()
    return weather_, time_prov_


def correlationGraphs(merged_df):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = merged_df.corr()

    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax, annot=True)
    plt.title('Correlations Heatmap')
    plt.show()
    plt.savefig('correlation_heatmap.png')
    plt.clf()

    pd.plotting.scatter_matrix(merged_df, figsize=(20, 20))
    plt.title('Correlations Scatter')
    plt.savefig('correlation_scatter.png')
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
    # This code below only labels 0 and 1
    # combined['label'] = combined.apply(lambda row:
    #                                    1 if row['daily'] > daily_confirmed_threshold else 0
    #                                    , axis=1)
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


def pcaAnalysis(df):
    X = df[list(df.columns.values.tolist())]
    print(X.shape)
    X.head()
    X = X.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    plt.clf()
    principalDF = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    ax = sns.scatterplot(x='PC1', y='PC2', data=principalDF)
    sns.regplot(x=principalDF['PC1'], y=principalDF['PC2'], fit_reg=False)
    plt.savefig("pca_analysis.png")
    plt.show()
    plt.clf()


time_prov = pd.read_csv('archive/TimeProvince.csv')
weather = pd.read_csv('archive/Weather.csv')
search_trend = pd.read_csv('archive/SearchTrend.csv')

search_trend_date = search_trend[(search_trend['date'] > '2020-01-19')]
columnss = ['cold', 'flu', 'pneumonia', 'coronavirus']
sns.scatterplot(data=search_trend_date, x='date', y='coronavirus')
plt.title('Coronavirus Search Trend')
plt.savefig('coronavirus_search_trend.png')
# plt.show()
plt.clf()


weather_, time_prov_ = preprocessing(weather, time_prov)
# Merge weather, time province and search trend
combined = pd.merge(weather_, time_prov_, on=['province', 'date'])
combined_all = pd.merge(combined, search_trend_date, on=['date'])
combined_all = combined_all.drop(columns=['date', 'confirmed'])

# Print correlation graphs
correlationGraphs(combined_all.drop(columns=['province']))

# Print PCA Analysis
pcaAnalysis(combined_all.drop(columns=['province']))

# Add labels using daily_confirmed_threshold
combined_labeled = addLabels(combined_all)
print(combined_labeled['label'].describe())
combined_labeled['index'] = combined_labeled.index
plt.clf()
sns.scatterplot(x='index', y="daily", hue="label", data=combined_labeled)
plt.title('Coronavirus Daily Labeled')
plt.savefig('coronavirus_daily_labeled.png')
plt.show()
plt.clf()

# Both avg_relative_humidity and coronavirus have positive correlation.
# Therefore, drop other columns.
combined_reduced = combined_labeled.drop(columns=['cold', 'flu', 'pneumonia', 'max_wind_speed'])

# Encoding provinces with 0s and 1s
encoder = OneHotEncoder(categories='auto', sparse=False)
encoder.fit(np.array(combined_reduced['province']).reshape(-1, 1))
inf_dummies = encoder.transform(np.array(combined_reduced['province']).reshape(-1, 1))
dummies = pd.DataFrame(inf_dummies.astype(int), columns=['province_' + x for x in encoder.categories_[0]])
combined_reduced = combined_reduced.join(dummies)
# Drop province, index and daily column as it is not needed anymore
combined_ready = combined_reduced.drop(columns=['province', 'daily', 'index'])

# K-fold cross validation
kf = KFold(n_splits=k_fold, shuffle=True, random_state=2)
X = combined_ready.drop(columns=['label']).to_numpy()
Y = combined_ready['label'].to_numpy()
combined_arr = combined_ready.to_numpy()
labels_clf = ['ExtraTrees', 'KNeighbors', 'SVC', 'Ridge', 'GaussianNB']
i = 1
for train_index, test_index in kf.split(combined_arr):
    print("***", i, "-Fold Results***")
    xtra, xtst = X[train_index], X[test_index]
    ytra, ytst = Y[train_index], Y[test_index]

    # Classifiers cross-validation
    # Each of models would be trained and their
    # cross validation score would be printed.
    for model, label in zip([et, knn, svc, rg, gnb], labels_clf):
        trained_model = trainModel(xtra, ytra, model)
        prediction = predict(trained_model, xtst)
        writeOutput(prediction, ytst, label)

    i += 1
