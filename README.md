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
data_formatting function whose functionality will be discussed later. Then, the following preprocessing techniques; feature scaling, ordinal encoding & one hot encoding, have been applied
on the dataframe using functions we have defined.
