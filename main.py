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

directors=pd.read_csv("movie-director.csv")
voice_actors=pd.read_csv("movie-voice-actors.csv")
movie_revenue=pd.read_csv("movies-revenue.csv")

print("Hello world")