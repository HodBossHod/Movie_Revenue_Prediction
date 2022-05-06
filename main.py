import numpy
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from dateutil import parser
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# from googlesearch import search
from sklearn.preprocessing import OneHotEncoder
import re
from datetime import datetime
from sklearn import metrics
from dateutil.parser import parse
from sklearn.metrics import r2_score
import time
import requests
from bs4 import BeautifulSoup

revenue_df = pd.read_csv('movies-revenue.csv')
actor_df = pd.read_csv('movie-voice-actors.csv')


# one hot encoding for movies genre, MPAA_rating & directors names
def one_hot_encoder(d, columnName):
    y = pd.get_dummies(d[columnName])
    final_df = pd.DataFrame(d.join(y))
    final_df.drop(columnName, axis=1, inplace=True)
    return final_df


# modify the date format
def handleDate(dr):
    editDate = []
    date = dr['release_date']
    now = datetime.now().year
    for i in date:
        editDate.append(datetime.strptime(i, '%dd-%mmm-%yy'))
    print(editDate)

    dr['release_date'] = pd.DataFrame(editDate)

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
    moive = moive.replace(" ", "_")
    URL1 = f'https://disney.fandom.com/wiki/{moive}'
    URL2 = f'https://www.rottentomatoes.com/m/{moive}'
    URL3 = f'https://en.wikipedia.org/wiki/{moive}'

    URL_list = [URL1, URL2, URL3]
    URL_info = [['Directed by', 0], ['Director:', 1], ['Directed by', 0]]
    new_moive = ""

    for index in range(len(URL_list)):
        try:
            response = requests.get(URL_list[index])
            soup = BeautifulSoup(response.text, 'html.parser')
            new_moive = str((soup.find(text=URL_info[index][0]).findNext().contents[URL_info[index][1]]).text)
            index += 1
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
def correlation(df, col_name):
    # Feature Selection
    # Get the correlation between the features
    corr = df.corr()
    # Top 0% Correlation training features with the Value
    top_feature = corr.index[abs(corr[col_name]) >= 0.29]
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
    acc = r2_score(y_test, prediction)
    print(
        f'Mean Square Error of polynomial Regression with degree of ({degree}) and random state ({random_state}) : {mse}')
    print(f'Accuracy of polynomial Regression : {acc}')
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
    acc = r2_score(y_test, prediction)
    print(f'Mean Square Error of Multiple Linear Regression with random state ({random_state}) : {mse}')
    print(f'Accuracy of Multiple Linear Regression : {acc} ')
    print(f'Training time of Multiple Linear Regression model : {end_time - start_time}')
    return mse, acc


# merging tables
# director_df = fill_new_director()
print(revenue_df.shape)
director_df = pd.read_csv('new_directors.csv')
print(director_df.shape)
rev_dir_df = pd.merge(revenue_df, director_df, how='inner', on='movie_title')
rev_dir_df.drop_duplicates(inplace=True)
print(rev_dir_df.shape)
rev_dir_df.to_csv('dir.csv', index=False)
print(rev_dir_df.shape)

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
# movies_df.to_csv('clean_data.csv', index=False)
X = movies_df[correlation(movies_df, 'revenue')]
Y = movies_df['revenue']  # Label
X = X.drop('revenue', axis=1, inplace=False)
print(Y.shape)
# MSE = []
# Acc = []
# dgree = []

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=20)
mse, acc = poly_reg(4, X_train, y_train, X_test, y_test, 20)
# MSE.append(mse)
# Acc.append(acc)
# dgree.append(i)

# plt.xlabel('Random State', fontsize=20)
# plt.ylabel('MSE', fontsize=20)
# plt.plot(dgree, MSE, color='red', linewidth=3)
# plt.show()

multi_reg(X_train, y_train, X_test, y_test, 28)
movies_df.to_csv('clean_data.csv', index=False)
