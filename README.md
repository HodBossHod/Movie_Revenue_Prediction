# Movie_Revenue_Prediction
Machine Learning Project  
# Dataset Summary  
At first we had 3 different datasets (csv files); `movie-director.csv`, `movies-revenue.csv` and `movie-voice-actors.csv` which we had to merge in a single dataset.  
# Movie-Director csv file 
Contains about 45 movies where each movie has its title and director.  
# Movies-Revenue csv file
Contains about 463 movies where each movie has its title, release date, genre, MPAA rating and revenue.  
# Movie-Voice-Actors csv file
Contains about 811 records of characters with their voice actors and featured movies.  
# Preprocessing
First, we have merged the three dataframes that read the three csv files into a new dataframe "movies_df". Then, we have applied changes on the new dataframe by calling 
data_formatting function whose functionality will be discussed later. Then, the following preprocessing techniques; feature scaling, ordinal encoding & one hot encoding, have been applied on the dataframe using functions we have defined.
# Data Formatting function
We pass "movie_df" dataframe to it as a parameter to apply some changes on its columns.   
1.	"revenue" column:  
               -"$" and "," symbols are removed from all revenues  
               -the records are numerized (where it is possible)  
2.	"release_date" column:  
-all null values are replaced by zeros  
-its format is changed from the following for example:12-JAN-88 to the year only 1988  
-its type is converted to integer (int32)  

At the end after applying the changes the data frame is returned   
# Feature Scaling function
It's called by passing "movie_df" data frame to it as a paramter to change the scale of numbers for each column to be between 0 and 1 all inclusive.
  
