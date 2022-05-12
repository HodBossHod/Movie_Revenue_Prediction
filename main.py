from matplotlib import style
from sklearn import preprocessing
import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn import linear_model
from sklearn import metrics
from dateutil import parser
from sklearn import svm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# from googlesearch import search
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import metrics
from sklearn import tree
from dateutil.parser import parse
from sklearn.metrics import r2_score
import time
import requests
from bs4 import BeautifulSoup
from yaml import ScalarEvent
import pickle
revenue_df = pd.read_csv('movies-revenue.csv')
actor_df = pd.read_csv('movie-voice-actors.csv')
director_df = pd.read_csv('new_directors.csv')
movie_success_df = pd.read_csv('movies-revenue-classification.csv')

# one hot encoding for movies genre, MPAA_rating & directors names
def one_hot_encoder(d, columnName):
    y = pd.get_dummies(d[columnName])
    final_df = pd.DataFrame(d.join(y))
    final_df.drop(columnName, axis=1, inplace=True)
    return final_df

def ordinalEncoder(df, column_name, ordinal_list):
    encoded_column = df[column_name].replace(ordinal_list)
    return encoded_column

# modify the date format
def handleDate(dr):
    editDate = []
    now = datetime.now().year
    for i in dr:
        editDate.append(now -i)
    return pd.DataFrame(editDate)


# get directors for na values
# def get_director(movie_name):
#     query = movie_name + ' director'  # the movie name +'director'
#     link_list = []
#     for j in search(query, tld="co.in", num=1, stop=1, pause=2):
#         link_list.append(j)
#
#     if len(link_list) != 0:
#         link_comp = link_list[0].split('/')
#         director = link_comp[-1]  # the director name
#         director = str(director)
#         if len(director) <= 43:
#             while '-' in director:
#                 director = director.replace('-', ' ')
#             while '_' in director:
#                 director = director.replace('_', ' ')
#             if str(movie_name).lower() in director.lower():
#                 return ''
#             director = re.sub("\(.*\)", '', director)
#             if 'film director'.lower() in director.lower():
#                 director = director.replace('film director', '')
#             if re.compile(r"\d+").search(director):
#                 return ''
#
#             print(movie_name + ' : ' + str(director))
#             return director
#         else:
#             return ''
#     else:
#         return ''

def get_director(moive):
    moive1 = moive.replace(" ", "+") + '+movie'
    moive2 = moive.replace(" ", "_")

    URL1 = f'https://www.google.com/search?q={moive1}&hl=en'
    URL2 = URL1
    URL3 = f'https://www.rottentomatoes.com/m/{moive2}'
    URL4 = f'https://disney.fandom.com/wiki/{moive2}'
    URL_list = [URL1, URL2, URL3, URL4]
    URL_info = ['Director', 'Directors', 'Director:', 'Directed by']
    new_moive = ""
    for index in range(len(URL_list)):
        try:
            response = requests.get(URL_list[index])
            soup = BeautifulSoup(response.text, 'html.parser')
            new_moive = str((soup.find(text=URL_info[index]).find_next('a').text))
            break
        except:
            True
    return new_moive


# create a dictionary for directors (keys:movietitle, values:directors names)
def fill_new_director():
    # getting the movie directors
    directors = {'movie_title': [], 'director': []}
    move_title = list(revenue_df['movie_title'])
    for move in move_title:
        directors['movie_title'].append(move)
        directors['director'].append(get_director(move))
    df_directors = pd.DataFrame.from_dict(directors, orient='index').T
    print(df_directors.head())

    df_directors.to_csv('new_directors.csv', index=False)
    return df_directors


# filter features according to correlation
def correlation(df, col_name, value):
    # Feature Selection
    # Get the correlation between the features
    corr = df.corr()
    # Top 0% Correlation training features with the Value
    top_feature = corr.index[abs(corr[col_name]) >= value]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = df[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    return top_feature


# apply polynomial regression model
def poly_reg(degree, X_train, y_train, X_test, y_test, random_state):
    model_1_poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly_model_1 = model_1_poly_features.fit_transform(X_train)
    # fit the transformed features to Linear Regression
    poly_model1 = linear_model.LinearRegression()

    # calculate training time
    start_time = time.time()  # start time before training
    poly_model1.fit(X_train_poly_model_1, y_train)
    end_time = time.time()  # end time

    prediction = poly_model1.predict(model_1_poly_features.fit_transform(X_test))
    mse = metrics.mean_squared_error(y_test, prediction)
    acc = r2_score(y_test, prediction) * 100
    print(
        f'Mean Square Error of polynomial Regression with degree of ({degree}) and random state ({random_state}) : {mse}')
    print(f'Accuracy of polynomial Regression : {acc} %')
    print(f'Training time of polynomial Regression model : {end_time - start_time}')
    return mse, acc


# apply Multiple linear regression model
def multi_reg(X_train, y_train, X_test, y_test, random_state):
    # fit the transformed features to Linear Regression
    multi_model1 = linear_model.LinearRegression()

    # calculate training time
    start_time = time.time()  # start time before training
    multi_model1.fit(X_train, y_train)
    end_time = time.time()  # end time

    prediction = multi_model1.predict(X_test)
    mse = metrics.mean_squared_error(y_test, prediction)
    acc = r2_score(y_test, prediction) * 100
    print(f'Mean Square Error of Multiple Linear Regression with random state ({random_state}) : {mse}')
    print(f'Accuracy of Multiple Linear Regression : {acc} %')
    print(f'Training time of Multiple Linear Regression model : {end_time - start_time}')
    return mse, acc


# merging tables
# director_df = fill_new_director()
print(revenue_df.shape)
print(director_df.shape)
rev_dir_df = pd.merge(revenue_df, director_df, how='inner', on='movie_title')
rev_dir_df.drop_duplicates(inplace=True)
print(rev_dir_df.shape)
rev_dir_df.to_csv('dir.csv', index=False)
#print(rev_dir_df.shape)

# is_animation_d = {'movie_title': [], 'is_animation': []}
# movies_list = list(rev_dir_df['movie_title'])
# actor_movies_list = list(actor_df['movie_title'])
#
# for movie in movies_list:
#     if movie in actor_movies_list:
#         is_animation_d['movie_title'].append(str(movie))
#         is_animation_d['is_animation'].append(1)
#     else:
#         is_animation_d['movie_title'].append(str(movie))
#         is_animation_d['is_animation'].append(0)

# is_animation_df = pd.DataFrame.from_dict(is_animation_d, orient='index').T
# print(is_animation_df.shape)
# is_animation_df = pd.read_csv('movie-voice-actors.csv')
movies_df = pd.merge(rev_dir_df, actor_df, how='outer', on='movie_title')
movies_df.drop_duplicates(inplace=True)
print(movies_df.shape)
movies_df.fillna(0, inplace=True)
movies_df.drop(movies_df[movies_df.revenue == 0].index, inplace=True)
movies_df.to_csv('clean_data.csv', index=False)
# merging ended
# removing the dollar sign from movies_df columns
movies_df = movies_df.apply(lambda x: x.str.strip('$') if x.name == "revenue" else x)
# removing the comma from the numeric columns
movies_df = movies_df.apply(lambda x: x.replace(',', "", regex=True) if x.name == "revenue" else x)
# converting to numerize our columns (where it is possible)
movies_df = movies_df.apply(lambda x: pd.to_numeric(x, errors="ignore") if x.name == "revenue" else x)
# cleaning the data of release date
movies_df = movies_df.apply(
    lambda x: x.replace('((\d\d-...-)|(\d-...-))', '', regex=True) if x.name == "release_date" else x)
movies_df['release_date'] = movies_df['release_date'].astype('int32')
movies_df["release_date"] = np.where(movies_df["release_date"] >= 37, movies_df['release_date'] + 1900,
                                     movies_df['release_date'] + 2000)
# movies_df.to_csv('clean_data.csv', index=False)

# Using One_Hot_Encoding
encodlist = ['genre', 'director', 'MPAA_rating', 'voice-actor']
movies_df = one_hot_encoder(movies_df, encodlist)
for colm in encodlist:
    movies_df.drop(f'{colm}_0', axis=1, inplace=True)
# movies_df.to_csv('clean_data.csv', index=False)
X = movies_df[correlation(movies_df, 'revenue', 0.17)]
Y = movies_df['revenue']  # Label
X = X.drop('revenue', axis=1, inplace=False)
print(Y.shape)
MSE = []
Acc = []
dgree = []
# for i in range(2,15):
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.23, shuffle=True, random_state=20)
mse, acc = poly_reg(2, X_train, y_train, X_test, y_test, 20)
# MSE.append(mse)
# Acc.append(acc)
# dgree.append(i)
#
# plt.xlabel('Random State', fontsize=20)
# plt.ylabel('MSE', fontsize=20)
# plt.plot(dgree, MSE, color='red', linewidth=3)
# plt.show()

multi_reg(X_train, y_train, X_test, y_test, 28)
movies_df.to_csv('clean_data.csv', index=False)



# ------------------------------------------------------------------ Classification ------------------------------------------------------------------
success_encode_list = {"S":4, "A":3, "B":2, "C":1, "D":0}

print(f"\t\t\t\t\t\t\t{Fore.CYAN}Classification Starts Here{Style.RESET_ALL}")
# preparing the dataframe for classification
movie_success_df.drop(columns=["release_date", "genre", "MPAA_rating"], inplace=True)
movie_success_df["MovieSuccessLevel"] = ordinalEncoder(movie_success_df, "MovieSuccessLevel", success_encode_list)
movie_success_df = pd.merge(movie_success_df, movies_df, how="outer", on="movie_title")
movie_success_df.drop(columns="revenue", inplace=True)
movie_success_df.to_csv("classification_with_only_ratings.csv", index=False)

X = movie_success_df[correlation(movie_success_df, 'MovieSuccessLevel', 0.12)]
Y = movie_success_df['MovieSuccessLevel']
X = X.drop('MovieSuccessLevel', axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=21)

# using boosting with decision trees
dt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),algorithm="SAMME.R",n_estimators=100)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

dt.fit(X_train,y_train)
y_prediction = dt.predict(X_test)
accuracy = np.mean(y_prediction == y_test) * 100
print(f"Adaboost decision tree accuracy: {accuracy} %")

# using one vs one classifier
svm_kernel_ovo = OneVsOneClassifier(SVC(kernel='linear', C=0.5)).fit(X_train, y_train)
accuracy = svm_kernel_ovo.score(X_test, y_test) * 100
print(f'Linear Kernel OneVsOne SVM accuracy: {accuracy} %')

# using rbf classifier
rbf_svc = svm.SVC(kernel='rbf', gamma=1, C=1).fit(X_train, y_train)
predictions = rbf_svc.predict(X_test)
accuracy = np.mean(predictions == y_test) * 100
print(f"RBF SVM accuracy: {accuracy} %")

# # using linear classifier
# svc = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# predictions = svc.predict(X_test)
# accuracy = np.mean(predictions == y_test) * 100
# print(f"linear SVM accuracy: {accuracy} %")

#svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
#for i, clf in enumerate((svc, rbf_svc)):
#    predictions = clf.predict(X_test)
#    accuracy = np.mean(predictions == y_test)
#    print(accuracy)