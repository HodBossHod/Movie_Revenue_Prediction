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
#from googlesearch import search
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
    now = datetime.now().year
    for i in date:
        editDate.append(now - datetime.strptime(i, '%d-%b-%y').year)
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

#Using Ordinal Encoding
#Rating_dict = {'G': 4, 'PG': 3, 'PG-13': 2, 'R': 1, 'Not Rated': 0}
#d['MPAA_rating'] = d.MPAA_rating.map(Rating_dict)

d = encoder(d, 'genre')
d = encoder(d, 'MPAA_rating')
d = encoder(d, 'director')
d.to_csv('one hot final.csv')

handleDate(d)
d.to_csv('one hot final.csv')
correct_Data=d.iloc[:,:]
print(correct_Data['release_date'])

# Remove Not Features
correct_Data.drop(correct_Data.columns[[0, 1]], axis = 1, inplace = True)

#load X and Y
index_data = correct_Data.columns.values.tolist()
index_data.remove('revenue')
X = correct_Data[index_data]
Y = correct_Data['revenue']

# correlation
featuresCorrelation = correct_Data.corr()
topFeature = featuresCorrelation.index[abs(featuresCorrelation['revenue']) > 0.1]
plt.subplots(figsize=(11,9))
topCorrelation = d[topFeature].corr()
print(topCorrelation)
sns.heatmap(topCorrelation, annot=True)
plt.show()

#Using Multiple linear regression
list5=[]
list_Multi =[]
for i in range(1,50):
    #Split the data to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True,random_state=i)

    # fit the transformed features to Linear Regression
    cl = linear_model.LinearRegression()

    #Using Multiple linear regression
    cl.fit(X_train, y_train)
    prediction = cl.predict(X_test)


    cl.fit(X_train, y_train)
    prediction = cl.predict(X_test)
    list_Multi.append( metrics.mean_squared_error(np.asarray(y_test), prediction))


#plot relation between MSE and random_state
list_y = [i for i in range(1,50)]
plt.title('Relation between MSE and random_state ')
plt.xlabel('random_state',fontsize=15)
plt.ylabel('MSE',fontsize=15)
plt.plot(list_y, list_Multi, color='red', linewidth=3)
plt.show()

list5.append(min(list_Multi))
index_random_state = list_Multi.index(min(list_Multi))+1
print('Mean Square Error of Multiple linear regression when i = {} is :'.format(index_random_state), min(list_Multi))


#Using polynomial regression

# Split the data to training and testing sets

# for i in range(1, 2):

poly_features = PolynomialFeatures(degree=2)

X_train_poly = poly_features.fit_transform(X_train)

cl.fit(X_train_poly, y_train)

Y_train_predicted = cl.predict(X_train_poly)

Y_test_predicted = cl.predict(poly_features.fit_transform(X_test))


#list_poly.append(metrics.mean_squared_error(np.asarray(y_test), Y_test_predicted))


#plot relation between MSE and random_state
# plt.title('Relation between MSE and degree ')
# plt.xlabel('degree',fontsize=15)
# plt.ylabel('MSE',fontsize=15)
# plt.plot(list_y, list_poly, color='red', linewidth=3)
# plt.show()

print('Mean Square Error of polynomial linear regression when i  is :',metrics.mean_squared_error(np.asarray(y_test), Y_test_predicted))







