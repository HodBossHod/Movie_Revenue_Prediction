from matplotlib import style
from sklearn import preprocessing
import pandas as pd
import numpy as np
#from colorama import Fore, Style
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
#import requests
#from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
# from yaml import ScalarEvent
# import pickle

# TODO: Feature Scaling on Release Date--->Remove Handle Date Function and make the range from 0 to 1 Hadi Ehab
# TODO: Rating Column --> Ordinal Encoding Hadi Ehab
# TODO: Handling missing values for important features (Release date-MPAA_Rating) --> Hadi Atef
# TODO: movies before 1980 --> G Hadi Ehab


# Reading the csv files
revenue_df = pd.read_csv('movies-revenue.csv')
actor_df = pd.read_csv('movie-voice-actors.csv')
director_df = pd.read_csv('new_directors.csv')
movie_success_df = pd.read_csv('movies-revenue-classification.csv')


# one hot encoding for movies genre, MPAA_rating & directors names
def one_hot_encoder_unit(d, columnName):
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
        editDate.append(now - i)
    return pd.DataFrame(editDate)


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
def fill_new_director(df):
    # getting the movie directors
    directors = {'movie_title': [], 'director': []}
    move_title = list(df['movie_title'])
    for move in move_title:
        directors['movie_title'].append(move)
        directors['director'].append(get_director(move))
    df_directors = pd.DataFrame.from_dict(directors, orient='index').T
    print(df_directors.head())

    # df_directors.to_csv('new_directors.csv', index=False)
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
    top_feature = top_feature.drop(col_name)
    return top_feature


def data_formatting(df):
    if 'revenue' in df.columns:
        # removing the dollar sign from movies_df columns
        df = df.apply(lambda x: x.str.strip('$') if x.name == "revenue" else x)
        # removing the comma from the numeric columns
        df = df.apply(lambda x: x.str.replace(',', '') if x.name == "revenue" else x)
        # converting to numerize our columns (where it is possible)
        df = df.apply(lambda x: pd.to_numeric(x, errors="ignore") if x.name == "revenue" else x)
    # cleaning the data of release date
    df = df.apply(
        lambda x: x.replace('((\d\d-...-)|(\d-...-))', '', regex=True) if x.name == "release_date" else x)
    df['release_date'] = df['release_date'].astype('int32')
    df["release_date"] = np.where(df["release_date"] >= 37, df['release_date'] + 1900,
                                  df['release_date'] + 2000)
    return df


def merging_tables(revenue, directors, v_actors, get_directors=False):
    if get_directors:
        directors = fill_new_director(revenue)
    revenue_directors_df = pd.merge(revenue, directors, how='outer', on='movie_title')
    revenue_directors_df.drop_duplicates(inplace=True)  # removing any duplicate cose Merging
    # revenue_directors_df.to_csv('csvFiles/dir.csv', index=False)     # generate the csv file with the Merged dateBase

    # Merging the result table from the last marge with the Actors Table
    res = pd.merge(revenue_directors_df, v_actors, how='outer', on='movie_title')
    res.drop_duplicates(inplace=True)  # removing any duplicate cose Merging
    res.drop('character', axis=1, inplace=True)  # drop the character colo no need will not matter
    # handling the null vals by filling it by 0
    res.fillna(0, inplace=True)
    if 'revenue' in res.columns:
        res.drop(res[res.revenue == 0].index, inplace=True)  # drop any row with 0 (null) revenue

    return res


def one_hot_encoding(df):
    # Using One_Hot_Encoding
    encode_list = ['genre', 'director', 'voice-actor']
    df = one_hot_encoder_unit(df, encode_list)
    for colm in encode_list:
        if f'{colm}_0' in df.columns:
            # df.drop(f'{colm}_0', axis=1, inplace=True)
            df[f'{colm}_0'] = 1
    return df


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
    mse, acc = poly_evaluation(X_test, y_test, poly_model1, model_1_poly_features)
    print(
        f'Mean Square Error of polynomial Regression with degree of ({degree}) and random state ({random_state}) : {mse}')
    print(f'Accuracy of polynomial Regression : {acc} %')
    print(f'Training time of polynomial Regression model : {end_time - start_time}')
    return poly_model1, model_1_poly_features


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
    return multi_model1, mse, acc


def poly_evaluation(x, y, model, poly):
    prediction = model.predict(poly.fit_transform(x))
    mse = metrics.mean_squared_error(y, prediction)
    acc = r2_score(y, prediction) * 100
    return mse, acc


def classification_preprocessing(df, prev_df, is_drop):
    success_encode_list = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
    if is_drop:
        print('dropint')
        df.drop(columns=["release_date", "genre", "MPAA_rating"], inplace=True)
    df["MovieSuccessLevel"] = ordinalEncoder(df, "MovieSuccessLevel", success_encode_list)
    if is_drop:
        df = pd.merge(df, prev_df, how="outer", on="movie_title")
    if 'revenue' in df.columns:
        df.drop(columns="revenue", inplace=True)
    df.to_csv("classification_with_only_ratings.csv", index=False)
    return df

def getDistinct(dataframe,column):
    distinct_ratings=dataframe[column].unique()
    print(distinct_ratings)
    return distinct_ratings

def getOrdinalRating(keys):
    differentRatings={}
    i=keys.size
    for key in keys:
        # if(key!=0):
        differentRatings[key]=i
        i=i-1
    return differentRatings

def removeZeroRatings(moviesData,ratingsColumn,moviesYear):
    index=0
    for i in moviesData[ratingsColumn]:
        if(i==0&moviesData[moviesYear][index]<=1980):
            moviesData[ratingsColumn][index]="G"
        index=index+1

# def getZeroCount(dataframe,column):
#     zeros=0
#     for i in dataframe[column]:
#         if(i==0):
#            zeros=zeros+1
#     return zeros

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X
# ----------------------------- merging tables -----------------------
movies_df = merging_tables(revenue_df, director_df, actor_df)
# ------------------------------------------ formatting the date ------------------------------
movies_df = data_formatting(movies_df)
#--------------------------------- remove 0 ratings ------------
#removeZeroRatings(movies_df,"MPAA_rating","release_date")
#---------------------------------- ordinal encoding ------------
# zeros=getZeroCount(movies_df,"MPAA_rating")
# print(zeros)
distinctRatings=getDistinct(movies_df,"MPAA_rating")
ordinalDictionary=getOrdinalRating(distinctRatings)
#Rating_dict = {"G": 6, "PG": 5, "PG-13": 4, "Not Rated":0,"R": 2 ,"TV-MA":2,"X": 1,"NC-17":1}

# movies_df['MPAA_rating'] = movies_df.MPAA_rating.map(Rating_dict)
movies_df["MPAA_rating"]=ordinalEncoder(movies_df,"MPAA_rating",ordinalDictionary)
#---------------------------------- feature scaling ------------
#movies_df["release_date"]=featureScaling(movies_df["release_date"],0,1)
# --------------------------------- one hot encoding -----------
movies_df = one_hot_encoding(movies_df)
movies_df.to_csv('clean_data.csv')
# ------------------------------------------ preprocessing end -------------------------------------
milestone1_features = correlation(movies_df, 'revenue', 0.17)
milestone1_label = 'revenue'
X = movies_df[milestone1_features]
Y = movies_df[milestone1_label]  # Label

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=20)
p_model, p_features = poly_reg(2, X_train, y_train, X_test, y_test, 20)
multi_reg(X_train, y_train, X_test, y_test, 28)

# ------------------------ Classification -------------------------------------------


print(f"\t\t\t\t\t\t\tClassification Starts Here")
# preparing the dataframe for classification
movie_success_df = classification_preprocessing(movie_success_df, movies_df, True)

milestone2_features = correlation(movie_success_df, 'MovieSuccessLevel', 0.12)
milestone2_label = 'MovieSuccessLevel'
X = movie_success_df[milestone2_features]
Y = movie_success_df[milestone2_label]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=21)

# using boosting with decision trees
dt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), algorithm="SAMME.R", n_estimators=100)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

start_time=time.time()
dt.fit(X_train, y_train)
end_time=time.time()
y_prediction = dt.predict(X_test)
accuracy = np.mean(y_prediction == y_test) * 100
print(f"Adaboost decision tree accuracy: {accuracy} %")
print(f'Training time of Adaboost : {end_time - start_time}')

# using one vs one classifier
start_time=time.time()
svm_kernel_ovo = OneVsOneClassifier(SVC(kernel='linear', C=0.5)).fit(X_train, y_train)
end_time=time.time()
accuracy = svm_kernel_ovo.score(X_test, y_test) * 100
print(f'Linear Kernel OneVsOne SVM accuracy: {accuracy} %')
print(f'Training time of Linear 1v1 SVM : {end_time - start_time}')

# using rbf classifier
start_time=time.time()
rbf_svc = svm.SVC(kernel='rbf', gamma=1, C=1).fit(X_train, y_train)
end_time=time.time()
predictions = rbf_svc.predict(X_test)
accuracy = np.mean(predictions == y_test) * 100
print(f"RBF SVM accuracy: {accuracy} %")

print(f'Training time of RBF classifier : {end_time - start_time}')


# =================================== The testing script =============================
# TODO: Test Script Hadi Ahmed -DONE
def milestone1_test():
    revenue = pd.read_csv('TestCases/Milestone 1/movies-test-samples.csv')
    director = pd.read_csv('TestCases/Milestone 1/movie-director.csv')
    actor = pd.read_csv('TestCases/Milestone 1/movie-voice-actors.csv')
    director.rename(columns={'name': 'movie_title'}, inplace=True)
    actor.rename(columns={'movie': 'movie_title'}, inplace=True)
    res = merging_tables(revenue, director, actor , True)
    # ------------------------------------------ formatting the date ------------------------------
    res = data_formatting(res)
    #---------------------------------- remove zero ratings --------------
    #removeZeroRatings(res, "MPAA_rating", "release_date")
    # --------------------------------- ordinal encoding -----------
    differentRatings=getDistinct(res,"MPAA_rating")
    ratingsDictionary=getOrdinalRating(differentRatings)
    res["MPAA_rating"]=ordinalEncoder(res,"MPAA_rating",ratingsDictionary)
    # --------------------------------- one hot encoding -----------
    res = one_hot_encoding(res)

    for c in milestone1_features:
        if not (c in res.columns):
            res[c] = 0
    res.to_csv('tmp.csv')
    x = res[milestone1_features]
    y = res[milestone1_label]

    mse, acc = poly_evaluation(x, y, p_model, p_features)
    print('====================  testing  =========')
    print(f'the Mean square error of the model = {mse} and the accuracy = {acc}')


def milestone2_test():
    success = pd.read_csv('TestCases/Milestone 2/movies-revenue-test-samples.csv')
    director = pd.read_csv('TestCases/Milestone 2/movie-director-test-samples.csv')
    actor = pd.read_csv('TestCases/Milestone 2/movie-voice-actors-test-samples.csv')
    director.rename(columns={'name': 'movie_title'}, inplace=True)
    actor.rename(columns={'movie': 'movie_title'}, inplace=True)
    res = merging_tables(success, director, actor, True)
    res = data_formatting(res)
    #removeZeroRatings(res, "MPAA_rating", "release_date")
    differentRatings = getDistinct(res, "MPAA_rating")
    ratingsDictionary = getOrdinalRating(differentRatings)
    res["MPAA_rating"] = ordinalEncoder(res, "MPAA_rating", ratingsDictionary)
    res = one_hot_encoding(res)
    success_encode_list = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
    res["MovieSuccessLevel"] = ordinalEncoder(res, "MovieSuccessLevel", success_encode_list)
    for c in milestone2_features:
        if not (c in res.columns):
            res[c] = 0
    res.to_csv('tmp2.csv', index=False)
    # print(res.columns)
    x = res[milestone2_features]
    y = res[milestone2_label]
    predictions1 = rbf_svc.predict(x)
    accuracy1 = np.mean(predictions1 == (y)) * 100
    print(f"========== testing ===========\nRBF SVM accuracy: {accuracy1} %")
    accuracy = svm_kernel_ovo.score(x, y) * 100
    print(f'Linear Kernel OneVsOne SVM accuracy: {accuracy} %')


milestone1_test()
milestone2_test()
