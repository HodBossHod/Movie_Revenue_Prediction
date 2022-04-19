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

revenue_df = pd.read_csv('movies-revenue.csv')
actor_df = pd.read_csv('movie-voice-actors.csv')



def encoder(d, columnName):
    y = pd.get_dummies(d[columnName])
    final_df = pd.DataFrame(d.join(y))
    final_df.drop(columnName, axis=1, inplace=True)
    return final_df

def handleDate(dr):
    editDate = []
    date = dr['release_date']
    for i in date:
        editDate.append(datetime.strptime(i, '%d-%b-%y').year)
        dr['release_date']=pd.DataFrame(editDate)
    return pd.DataFrame(editDate)






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


def fill_new_director():
    # getting the movie directors
    directors = {'movie_title': [], 'director': []}
    move_title = list(revenue_df['movie_title'])
    for move in move_title:
        directors['movie_title'].append(move)
        directors['director'].append(get_director(move))
    df_directors = pd.DataFrame.from_dict(directors, orient='index').T
    print(df_directors.head())
    df_directors.to_csv('new_directors.csv')
    return df_directors


# director_df = fill_new_director()
print(revenue_df.shape)
director_df = pd.read_csv('new_directors.csv')
print(director_df.shape)
rev_dir_df = pd.merge(revenue_df, director_df, how='outer')
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
movies_df = pd.merge(rev_dir_df, is_animation_df, how='outer')
print(movies_df.shape)
# movies_df.to_csv('final.csv')
# print(movies_df.isna().sum())
movies_df = movies_df.fillna(0)
# removing the dollar sign from movies_df columns
movies_df = movies_df.apply(lambda x: x.str.strip('$') if x.dtype == "object" else x)
# removing the comma from the numeric columns
movies_df = movies_df.apply(lambda x: x.replace(',', "", regex=True) if x.dtype == "object" else x)
movies_df.to_csv('final.csv')

d = pd.read_csv('final.csv')

d = encoder(d, 'genre')
d = encoder(d, 'MPAA_rating')
d = encoder(d, 'director')
d.to_csv('one hot final.csv')

handleDate(d)
d.to_csv('one hot final.csv')

rightData=d.iloc[:,:]

# correlation
featuresCorrelation=rightData.corr()
topFeature=featuresCorrelation.index[abs(featuresCorrelation['revenue']) > 0.1]
plt.subplots(figsize=(11,9))
topCorrelation=d[topFeature].corr()
print(topCorrelation)
sns.heatmap(topCorrelation, annot=True)
plt.show()








