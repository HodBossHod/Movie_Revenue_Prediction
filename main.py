# import pandas as pd
# import numpy as np
# from sklearn import linear_model
# from sklearn import metrics
# from dateutil import parser
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from googlesearch import search
# from sklearn.preprocessing import OneHotEncoder
# import re
# from datetime import datetime
#
# revenue_df = pd.read_csv('movies-revenue.csv')
# actor_df = pd.read_csv('movie-voice-actors.csv')
#
# #Feature Scaling
# def featureScaling(X,a,b):
#     X = np.array(X)
#     Normalized_X=np.zeros((X.shape[0],X.shape[1]))
#     for i in range(X.shape[1]):
#         Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
#     return Normalized_X
#
# #one hot encoding for movies genre
# def encoder(d, columnName):
#     y = pd.get_dummies(d[columnName])
#     final_df = pd.DataFrame(d.join(y))
#     final_df.drop(columnName, axis=1, inplace=True)
#     return final_df
#
# #modify the date format
# def handleDate(dr):
#     editDate = []
#     date = dr['release_date']
#     now = datetime.now().year
#     for i in date:
#         editDate.append(now - i)
#
#     editDate = pd.DataFrame(editDate, columns=['new_date'])
#     dr['release_date'] = editDate['new_date'].values
#     return dr['release_date']
#
# #get directors for na values
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
#
#
# #create a dictionary for directors (keys:movietitle, values:directors names)
# def fill_new_director():
#     # getting the movie directors
#     directors = {'movie_title': [], 'director': []}
#     move_title = list(revenue_df['movie_title'])
#     for move in move_title:
#         directors['movie_title'].append(move)
#         directors['director'].append(get_director(move))
#     df_directors = pd.DataFrame.from_dict(directors, orient='index').T
#     print(df_directors.head())
#
#     df_directors.to_csv('new_directors.csv', index=False)
#     return df_directors
#
#
# #filter features according to correlation
# def correlation(df, col_name):
#     # Feature Selection
#     # Get the correlation between the features
#     corr = df.corr()
#     # Top 0% Correlation training features with the Value
#     top_feature = corr.index[abs(corr[col_name]) > 0.23]
#     # Correlation plot
#     plt.subplots(figsize=(12, 8))
#     top_corr = df[top_feature].corr()
#     sns.heatmap(top_corr, annot=True)
#     plt.show()
#     return top_feature
#
#
# #apply Multiple linear regression
# def multi_reg(X_train, y_train, X_test, y_test):
#
#     # fit the transformed features to Linear Regression
#     multi_model1 = linear_model.LinearRegression()
#
#     # Using Multiple linear regression
#     multi_model1.fit(X_train, y_train)
#     prediction = multi_model1.predict(X_test)
#
#     print('Mean Square Error of Multiple Linear Regression :',metrics.mean_squared_error(y_test, prediction))
#     print('Accuracy of Multiple Linear Regression : ',r2_score(y_test, prediction))
#     return metrics.mean_squared_error(y_test, prediction)
#
# #apply polynomial regression model
# def poly_reg(degree, X_train, y_train, X_test, y_test):
#     model_1_poly_features = PolynomialFeatures(degree=degree)
#     # transforms the existing features to higher degree features.
#     X_train_poly_model_1 = model_1_poly_features.fit_transform(X_train)
#     # fit the transformed features to Linear Regression
#     poly_model1 = linear_model.LinearRegression()
#
#     poly_model1.fit(X_train_poly_model_1, y_train)
#     prediction = poly_model1.predict(model_1_poly_features.fit_transform(X_test))
#
#     print('Mean Square Error of the at degree {} of polynomial Regression :'.format(degree),metrics.mean_squared_error(y_test, prediction))
#     print('Accuracy of polynomial Regression : ',r2_score(y_test, prediction))
#     return metrics.mean_squared_error(y_test, prediction)
#
#
# #merging tables
# # director_df = fill_new_director()
# print(revenue_df.shape)
# director_df = pd.read_csv('new_directors.csv')
# print(director_df.shape)
# rev_dir_df = pd.merge(revenue_df, director_df, how='inner', on='movie_title')
# rev_dir_df.drop_duplicates(inplace=True)
# print(rev_dir_df.shape)
# rev_dir_df.to_csv('dir.csv', index=False)
# print(rev_dir_df.shape)
# # print(rev_dir_df.isna().sum())
# # print(list(rev_dir_df['director'].value_counts()))
#
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
#
# is_animation_df = pd.DataFrame.from_dict(is_animation_d, orient='index').T
# print(is_animation_df.shape)
# movies_df = pd.merge(rev_dir_df, is_animation_df, how='inner', on='movie_title')
# movies_df.drop_duplicates(inplace=True)
# print(movies_df.shape)
# movies_df.fillna(0, inplace=True)
# #merging ended
#
#
#
# # removing the dollar sign from movies_df columns
# movies_df = movies_df.apply(lambda x: x.str.strip('$') if x.name == "revenue" else x)
# # removing the comma from the numeric columns
# movies_df = movies_df.apply(lambda x: x.replace(',', "", regex=True) if x.name == "revenue" else x)
# # converting to numerize our columns (where it is possible)
# movies_df = movies_df.apply(lambda x: pd.to_numeric(x, errors="ignore") if x.name == "revenue" else x)
# # cleaning the data of release date
# movies_df = movies_df.apply(
#     lambda x: x.replace('((\d\d-...-)|(\d-...-))', '', regex=True) if x.name == "release_date" else x)
# movies_df['release_date'] = movies_df['release_date'].astype('int32')
# # print(movies_df.info())
# movies_df["release_date"] = np.where(movies_df["release_date"] >= 37, movies_df['release_date'] + 1900,
#                                      movies_df['release_date'] + 2000)
#
# movies_df.to_csv('clean_data.csv', index=False)
#
# #Drop ٍSpecial Rows with values 0
# movies_df.drop(movies_df[movies_df['MPAA_rating'] == 0].index,inplace=True)
# movies_df.drop(movies_df[movies_df['genre'] == 0].index,inplace=True)
#
# #using One_hot_encoding
# encodlist = ['genre', 'director','MPAA_rating']
# movies_df = encoder(movies_df, encodlist)
# #handle realsed date
# handleDate(movies_df)
#
# #featureScaling(movies_df['release_date'], 0, 20)
#
#
# movies_df.to_csv('clean_data.csv', index=False)
# X = movies_df[correlation(movies_df, 'revenue')]
# Y = movies_df['revenue']  # Label
# # print(X['revenue'])
# # print(X.head())
# X = X.drop('revenue', axis=1, inplace=False)
# # X.drop('revenue',inplace= True)
# print(Y.shape)
# MSE = []
# dgree = []
#
# #for i in range(1, 15):
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=13)
#     #MSE .append(poly_reg(3, X_train, y_train, X_test, y_test))
#
#
# #print(MSE.index(min(MSE))+1)
# multi_reg(X_train, y_train, X_test, y_test)
# poly_reg(3, X_train, y_train, X_test, y_test)
#
# #dgree.append(i)
#     #print(dgree[i])
#
# plt.xlabel('degree', fontsize=20)
# plt.ylabel('random state', fontsize=20)
# plt.plot(dgree, MSE, color='red', linewidth=3)
# plt.show()
#
# for i in X.columns:
#      plt.xlabel(i, fontsize=20)
#      plt.ylabel('revenue', fontsize=20)
#      plt.plot(X[i], Y, color='red', linewidth=3)
#      plt.show()
#
#
#
#
#
#
#
#
#
#
#
# # print(movies_df.columns)
# # print(movies_df.shape)
# # d = pd.read_csv('clean_data.csv')
# # # Using Ordinal Encoding
# # # Rating_dict = {'G': 4, 'PG': 3, 'PG-13': 2, 'R': 1, 'Not Rated': 0}
# # # d['MPAA_rating'] = d.MPAA_rating.map(Rating_dict)
# # movies_df = encoder(movies_df, 'genre')
# # print(movies_df['MPAA_rating'])
# # movies_df = encoder(movies_df, 'MPAA_rating')
# # movies_df = encoder(movies_df, 'director')
# # d.to_csv('one hot final.csv')
# #
# # print(movies_df.shape)
# # print(movies_df.columns)
# # # print(correlation(movies_df, 'revenue'))
# # movies_df.to_csv('clean_data.csv', index=False)
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
from googlesearch import search
from sklearn.preprocessing import OneHotEncoder
import re
from datetime import datetime
from sklearn import metrics
from dateutil.parser import parse
from sklearn.metrics import r2_score

revenue_df = pd.read_csv('movies-revenue.csv')
actor_df = pd.read_csv('movie-voice-actors.csv')


# one hot encoding for movies genre
def encoder(d, columnName):
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
def get_director(movie_name):
    query = movie_name + ' director'  # the movie name +'director'
    link_list = []
    for j in search(query, tld="co.in", num=1, stop=1, pause=2):
        link_list.append(j)

    if len(link_list) != 0:
        link_comp = link_list[0].split('/')
        director = link_comp[-1]  # the director name
        director = str(director)
        if len(director) <= 43:
            while '-' in director:
                director = director.replace('-', ' ')
            while '_' in director:
                director = director.replace('_', ' ')
            if str(movie_name).lower() in director.lower():
                return ''
            director = re.sub("\(.*\)", '', director)
            if 'film director'.lower() in director.lower():
                director = director.replace('film director', '')
            if re.compile(r"\d+").search(director):
                return ''

            print(movie_name + ' : ' + str(director))
            return director
        else:
            return ''
    else:
        return ''


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
    top_feature = corr.index[abs(corr[col_name]) > 0.1]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = df[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    return top_feature


# apply polynomial regression model
def poly_reg(degree, X_train, y_train, X_test, y_test):
    model_1_poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly_model_1 = model_1_poly_features.fit_transform(X_train)
    # fit the transformed features to Linear Regression
    poly_model1 = linear_model.LinearRegression()

    poly_model1.fit(X_train_poly_model_1, y_train)
    prediction = poly_model1.predict(model_1_poly_features.fit_transform(X_test))
    mse = metrics.mean_squared_error(y_test, prediction)
    acc = r2_score(y_test, prediction)
    print(
        f'Mean Square Error of the at degree of   and random state polynomial Regression    {mse}')
    print('Accuracy of polynomial Regression : ', acc)
    return mse, acc


def multi_reg(X_train, y_train, X_test, y_test):
    # fit the transformed features to Linear Regression
    multi_model1 = linear_model.LinearRegression()

    # Using Multiple linear regression
    multi_model1.fit(X_train, y_train)
    prediction = multi_model1.predict(X_test)
    mse = metrics.mean_squared_error(y_test, prediction)
    acc = r2_score(y_test, prediction)
    print('Mean Square Error of Multiple Linear Regression :', mse)
    print('Accuracy of Multiple Linear Regression : ', abs(acc))
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
# print(rev_dir_df.isna().sum())
# print(list(rev_dir_df['director'].value_counts()))

is_animation_d = {'movie_title': [], 'is_animation': []}
movies_list = list(rev_dir_df['movie_title'])
actor_movies_list = list(actor_df['movie_title'])

for movie in movies_list:
    if movie in actor_movies_list:
        is_animation_d['movie_title'].append(str(movie))
        is_animation_d['is_animation'].append(1)
    else:
        is_animation_d['movie_title'].append(str(movie))
        is_animation_d['is_animation'].append(0)

is_animation_df = pd.DataFrame.from_dict(is_animation_d, orient='index').T
print(is_animation_df.shape)
movies_df = pd.merge(rev_dir_df, is_animation_df, how='inner', on='movie_title')
movies_df.drop_duplicates(inplace=True)
print(movies_df.shape)
movies_df.fillna(0, inplace=True)
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
# print(movies_df.info())
movies_df["release_date"] = np.where(movies_df["release_date"] >= 37, movies_df['release_date'] + 1900,
                                     movies_df['release_date'] + 2000)

movies_df.to_csv('clean_data.csv', index=False)
# using ordinal encoding
Rating_dict = {'G': 4, 'PG': 3, 'PG-13': 2, 'R': 1, 'Not Rated': 0}
# movies_df['MPAA_rating'] = movies_df.MPAA_rating.map(Rating_dict)

encodlist = ['genre', 'director', 'MPAA_rating']
movies_df = encoder(movies_df, encodlist)
# movies_df = handleDate(movies_df)

movies_df.to_csv('clean_data.csv', index=False)
X = movies_df[correlation(movies_df, 'revenue')]
Y = movies_df['revenue']  # Label
# print(X['revenue'])
# print(X.head())
X = X.drop('revenue', axis=1, inplace=False)

print(Y.shape)
MSE = []
Acc = []
dgree = []
# for j in range(1, 15):
#     for i in range(1, 30):
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=28)
mse, acc = poly_reg(6, X_train, y_train, X_test, y_test)
MSE.append(mse)
Acc.append(acc)
# dgree.append(i)

        # print(dgree[i])
multi_reg(X_train, y_train, X_test, y_test)

# print(max(Acc))
# print(min(MSE))
# plt.xlabel('random_state', fontsize=20)
# plt.ylabel('MSE', fontsize=20)
# plt.plot(dgree, MSE, color='red', linewidth=3)
# plt.show()
#
# plt.xlabel('random_state', fontsize=20)
# plt.ylabel('MSE', fontsize=20)
# plt.plot(dgree, Acc, color='red', linewidth=3)
# plt.show()
# print(min(MSE))
# print(movies_df.columns)
# print(movies_df.shape)
# d = pd.read_csv('clean_data.csv')
# # Using Ordinal Encoding
# # Rating_dict = {'G': 4, 'PG': 3, 'PG-13': 2, 'R': 1, 'Not Rated': 0}
# # d['MPAA_rating'] = d.MPAA_rating.map(Rating_dict)
# movies_df = encoder(movies_df, 'genre')
# print(movies_df['MPAA_rating'])
# movies_df = encoder(movies_df, 'MPAA_rating')
# movies_df = encoder(movies_df, 'director')
# d.to_csv('one hot final.csv')
#
# print(movies_df.shape)
# print(movies_df.columns)
# # print(correlation(movies_df, 'revenue'))
# movies_df.to_csv('clean_data.csv', index=False)
