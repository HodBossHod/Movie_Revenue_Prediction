from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

directors = pd.read_csv("movie-director.csv")
voice_actors = pd.read_csv("movie-voice-actors.csv")
movie_revenue = pd.read_csv("movies-revenue.csv")

# master = pd.DataFrame(
#     columns=['movie_title', 'release_date', 'genre', 'MPAA_rating', 'revenue', 'director', 'voice-actor', 'movie'])

# master = pd.merge(directors, movie_revenue, how='outer')
# # master = pd.marge(master,movie_revenue)
# # master.shape
# master.to_csv('master.csv', index=False)
# print(master.head())

master = pd.read_csv('master.csv')

master1 = pd.merge(master, voice_actors, how='outer')

master1.to_csv('final.csv', index=False)

final = pd.read_csv('final.csv')
print(final.movie_title.nunique())

# master.insert(-2, 'director', director)
# print(master.head())
