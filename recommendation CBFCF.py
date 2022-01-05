# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import time
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

# Setting plot style
plt.style.use('ggplot')

# Loading the cleaned datasets
ratings_df = pd.read_csv("C:/datasets/rangering.csv")
movies_df = pd.read_csv("C:/datasets/film.csv")
users_df = pd.read_csv("C:/datasets/brucker.csv")
##1.1) Exploring the Users Dataset
# Printing the users dataset
users_df.head()
users_df.describe()
# Grouping user data into male and female age groups
user_age_gender_data = users_df[['Kjonn', 'Alder']]
female_user_ages = user_age_gender_data.loc[user_age_gender_data['Kjonn'] == 'F'].sort_values('Alder')
male_user_ages = user_age_gender_data.loc[user_age_gender_data['Kjonn'] == 'M'].sort_values('Alder')


def get_group_count(min_age, max_age, dataset):
    age_group = dataset.apply(lambda x: True if max_age > x['Alder'] > min_age else False, axis=1)
    count = len(age_group[age_group == True].index)
    return count


G1_male = get_group_count(0, 18, male_user_ages)
G2_male = get_group_count(17, 25, male_user_ages)
G3_male = get_group_count(24, 35, male_user_ages)
G4_male = get_group_count(34, 45, male_user_ages)
G5_male = get_group_count(44, 50, male_user_ages)
G6_male = get_group_count(49, 56, male_user_ages)
G7_male = get_group_count(55, 200, male_user_ages)

G1_female = get_group_count(0, 18, female_user_ages)
G2_female = get_group_count(17, 25, female_user_ages)
G3_female = get_group_count(24, 35, female_user_ages)
G4_female = get_group_count(34, 45, female_user_ages)
G5_female = get_group_count(44, 50, female_user_ages)
G6_female = get_group_count(49, 56, female_user_ages)
G7_female = get_group_count(55, 200, female_user_ages)

# Figure 1: Visualizing the userbase by plotting age and gender to a bar chart
labels = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
men_grouped = [G1_male, G2_male, G3_male, G4_male, G5_male, G6_male, G7_male]
women_grouped = [G1_female, G2_male, G3_female, G4_female, G5_female, G6_female, G7_female]
x = np.arange(len(labels))  # the label locations
width = 0.40  # the width of the bars

# Setting the labels for the bar chart
fig1, ax1 = plt.subplots()
rects1 = ax1.bar(x - width/2, men_grouped, width, label='Male users', color='royalblue')
rects2 = ax1.bar(x + width/2, women_grouped, width, label='Female users', color='darkorange')
ax1.set_ylabel('Number of users', size=12)
ax1.set_xlabel('Age', size=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title('Fig. 1: Users grouped by age and gender', size=15)
ax1.legend()
plt.show()


# Printing the movies dataset
movies_df.head()
movies_df.describe()
# Printing the ratings dataset
ratings_df.head()
ratings_df.describe()
# Generating a new dataframe with rating score count (cols) vs genres (rows)
genres = list(movies_df)[2:]
merged = pd.merge(ratings_df, movies_df, on='FilmID')

cleaned = merged[['FilmID', 'BrukerID', 'Rangering', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'Tittel']]

movie_ratings = cleaned.sort_values('FilmID')
rating_scale = [1, 2, 3, 4, 5]

genres_rating_count = []
for genre in genres:
    for rating in rating_scale:
        genre_list = movie_ratings.loc[movie_ratings[genre] == 1]
        genre_rated_count = len(genre_list.loc[genre_list['Rangering'] == rating])
        genres_rating_count.append([genre, rating, genre_rated_count])


# Function for extracting the count of a specific rating of all genres
def get_rating_count(score):
    rating_count = []
    for count in genres_rating_count:
        if count[1] == score:
            rating_count.append(count[2])

    return rating_count


# Function for extracting the total nr of ratings for all genres
def get_total_rating_count():
    all_ratings = []
    totals = []
    for count in genres_rating_count:
        all_ratings.append(count[2])

    lower = 0
    for i in range(lower, len(all_ratings), 5):
        lower = i
        totals.append(sum(all_ratings[lower:lower+5]))

    return totals


genre_df = pd.DataFrame({'Genre': [genre for genre in genres], 'Rated 1': get_rating_count(1), 'Rated 2': get_rating_count(2),
              'Rated 3': get_rating_count(3), 'Rated 4': get_rating_count(4), 'Rated 5': get_rating_count(5),
              'Total nr of ratings': get_total_rating_count()})
genre_df["Average rating"] = (genre_df["Rated 1"] + (genre_df["Rated 2"] * 2) + (genre_df["Rated 3"] * 3) + 
                             (genre_df["Rated 4"] * 4) + (genre_df["Rated 5"] * 5)) / genre_df['Total nr of ratings']
genre_df

# Figure 2: Visualizing the number of ratings and rating scores for each genre
total_ratings = genre_df.iloc[:, 6].values.tolist()
fig2, ax2 = plt.subplots()
fig2_labels = [genre for genre in genres]
x = np.arange(len(fig2_labels))  # the label locations
width = 0.7  # the width of the bars
ax2.barh(x, total_ratings, width, label='Ratings', color='chocolate')
# Setting labels for bar chart
ax2.set_xlabel('Count of user ratings', size=16)
ax2.set_yticks(x)
ax2.set_yticklabels(fig2_labels, fontdict={'fontsize': 14})
ax2.set_title('Fig. 2: Count of ratings per genre', size=18)
ax2.invert_yaxis()  # Invert y axis so that the genres appear alphabetically
ax2.legend(shadow=0.4, prop={"size": 13})
plt.xticks(fontsize=13)
# Adjusting and displaying the graph
fig2.set_figheight(8)
fig2.set_figwidth(13)
plt.show()

# Figure 3: Visualizing the ratings for each genre as a proportion of the total number of movies in that genre
# function for extracting the count of a specific rating of all genres
def get_rating_percentages(score):
    rated_count = genre_df.iloc[:, score].values.tolist()
    rating_percentages = []
    i = 0

    for total in total_ratings:
        percentage = (100 / total) * rated_count[i]
        i += 1
        rating_percentages.append(round(percentage, 1))

    return rating_percentages


genre_rating_percentage_data = {'Genre': [genre for genre in genres], 'Rated 1': get_rating_percentages(1),
                                'Rated 2': get_rating_percentages(2), 'Rated 3': get_rating_percentages(3),
                                'Rated 4': get_rating_percentages(4), 'Rated 5': get_rating_percentages(5),
                                'Total nr of ratings': get_total_rating_count()}
genre_rating_percent_df = pd.DataFrame(genre_rating_percentage_data)

rated_1_percentage = genre_rating_percent_df.iloc[:, 1].values.tolist()
rated_2_percentage = genre_rating_percent_df.iloc[:, 2].values.tolist()
rated_3_percentage = genre_rating_percent_df.iloc[:, 3].values.tolist()
rated_4_percentage = genre_rating_percent_df.iloc[:, 4].values.tolist()
rated_5_percentage = genre_rating_percent_df.iloc[:, 5].values.tolist()


fig3, ax3 = plt.subplots()
fig3_ylabels = [genre for genre in genres]
fig3_xlabels = ['0', '10%', '20%', '30%', '40%', '50%']
x = np.arange(len(fig3_ylabels))  # the label locations
width = 0.17  # the width of the bars
ax3.barh(x + width*2, rated_1_percentage, width, label='Rated 1', color='darkblue')
ax3.barh(x + width, rated_2_percentage, width, label='Rated 2', color='chocolate')
ax3.barh(x, rated_3_percentage, width, label='Rated 3', color='darkgreen')
ax3.barh(x - width, rated_4_percentage, width, label='Rated 4', color='darkgoldenrod')
ax3.barh(x - width*2, rated_5_percentage, width, label='Rated 5', color='darkred')
# Setting labels for bar chart
ax3.set_xlabel('Percentage of user ratings', size=14)
ax3.set_xticklabels(fig3_xlabels, fontdict={'fontsize': 15})
ax3.set_xlim(right=50)
ax3.set_yticks(x)
ax3.set_yticklabels(fig3_ylabels, fontdict={'fontsize': 15})
ax3.set_title('Fig 3: Genre Popularity', size=16)
ax3.invert_yaxis()  # Invert y axis so that the genres appear alphabetically
ax3.legend(shadow=0.4, prop={"size": 13})
# Adjusting and displaying the graph
fig3.set_figheight(12)
fig3.set_figwidth(10)
plt.show()

# Figure 4: Visualizing the proportional frequency of different rating scores for all genres
def get_rating_percentage_total(score):
    rated_count = genre_df.iloc[:, score].values.tolist()
    total_ratings_count = 0
    sum_ratings = 0

    for count in rated_count:
        sum_ratings += count

    for total in total_ratings:
        total_ratings_count += total

    return round((sum_ratings / total_ratings_count) * 100, 1)


ratings_data = [get_rating_percentage_total(5), get_rating_percentage_total(4), get_rating_percentage_total(3),
                get_rating_percentage_total(2), get_rating_percentage_total(1)]

# Plotting a pie (donut) chart for case distribution visualization
labels = ["Rated 5", "Rated 4", "Rated 3", "Rated 2", "Rated 1"]
fig4, ax4 = plt.subplots(subplot_kw=dict(aspect="equal"))
wedges, texts, autotext = ax4.pie(ratings_data, autopct='%1.1f%%', pctdistance=0.75, shadow=True,
                                  wedgeprops=dict(width=0.5), startangle=100, textprops=dict(color="w"),
                                  explode=(0.02, 0.02, 0.02, 0.02, 0.02), colors=['darkred', 'darkgoldenrod',
                                                                                  'darkgreen', 'chocolate', 'darkblue'])

ax4.legend(wedges, labels, loc="upper right", bbox_to_anchor=(1.0, 0.05, 0.33, 0.9), title='Ratings',
           title_fontsize='12', labelspacing=0.8)
ax4.set_title("Frequency of various ratings for all genres", size=18)

plt.show()

# Figure 5: Visualizing the distribution of average ratings for all users
users_avg_rating = pd.DataFrame(ratings_df.groupby('BrukerID')['Rangering'].mean())
print(users_avg_rating.head())

# Plotting a histogram over the average rating for all users
users_avg_rating = users_avg_rating['Rangering']
fig5, ax5 = plt.subplots()
n, bins, patches = ax5.hist(users_avg_rating, label='Users Average Rating',
                            stacked=True, color='royalblue', bins=50, rwidth=0.8)
# Set labels for histogram
ax5.set_xlabel('Average Rating', size=12)
ax5.set_ylabel('Count', size=12)
ax5.set_title('The Distribution of Average Rating across All Users', size=12)

# Add a line indicating the mean rating for all users
plt.axvline(ratings_df["Rangering"].mean(), color='chocolate', linestyle='dotted', dash_capstyle="round",
            linewidth=3, label="Mean")
ax5.legend(bbox_to_anchor=(1, 0.92))

plt.show()


# Figure 6: Visualizing how the average rating has changed over time
def convert_timestamp_to_year(timestamp):
    date_string = time.ctime(timestamp)
    tokens = date_string.split(sep=" ")
    year = int(tokens[-1])
    return year


ratings_copy = ratings_df.copy()
ratings_copy.isna().sum()
ratings_copy.Rangering.isna().sum()
ratings_copy1=ratings_copy.dropna()
ratings_copy1.isna().sum()


ratings_df.isna().sum()
ratings_df1=ratings_df.dropna()
ratings_df1.isna().sum()

ratings_copy1['Rangerings'] = ratings_df1["Tidstempel"].map(lambda timestamp: convert_timestamp_to_year(timestamp))
yearly_mean_rating = ratings_copy1.groupby('Rangerings')['Rangering'].mean()

# Plotting a bar plot for average rating over time
plt.figure()
plt.rcParams.update({'font.size': 14})
yearly_mean_rating.plot(kind='bar', figsize=(10, 10), title="Fig. 6: How the average rating changed over time",
                        subplots=True, color=['darkred', 'darkgoldenrod', 'darkgreen', 'chocolate'],
                        ylim=(3.05, 3.7), fontsize=15, label='')
plt.xlabel(xlabel='Year of ratings', labelpad=18)
plt.ylabel(ylabel='Avg. rating', labelpad=12)

# Adding an arrow along the y axis to indicate that the axis continues to 0
# and that only an excerpt of the axis has been shown
plt.arrow(x=(-0.4), y=3.2, dx=0, dy=(-0.1), width=0.02, length_includes_head=False, head_length=0.037, head_width=0.06,
          color='black')

plt.show()

#2) First Iteration: Building a Hybrid Recommender System with CBF and CF models

#2.1) Splitting the Dataset
# Splitting the ratings dataset into the feature set (X) and target labels (y)
X = ratings_df1.drop(columns='Rangering')
y = ratings_df1["Rangering"].values  # The movie ratings are the target variables we want to predict

# Preparing train, validation and test datasets.
# I have chosen a split ratio of 70%, 15%, 15%, because I want a somewhat large training set at the cost of a
# smaller validation and test set. I do not think that a smaller validation (or test) dataset will negatively
# impact the generalization ability of the chosen models, because I am only using rather simple ML models
# with few hyperparamaters.
X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5, random_state=101)

# Creating a complete training dataset with X_train and y_train
train_df = X_train.copy()
train_df["Rangering"] = y_train

train_df



#2.2) The Baseline Model


# Creating a new dataframe with the average rating for each movie. 
# The "prediction" for the baseline "model" will actually just be these averages for each movie.
baseline_y_pred = pd.DataFrame(train_df.groupby('FilmID')['Rangering'].mean())

# The baseline model has not yet calculated an average for the movies (if any) with no ratings. 
# We therefore add these movies to the model with a predicted rating equal to the average rating 
# for all movies in the training dataset.
# ADD SOME CODE HERE!!

# Creating a dataframe for the validation data (y_true) in order to calculate RMSE of the baseline model
val_movies_dict = {'FilmID': X_val["FilmID"], 'Actual rating': y_val}  
val_movies_df = pd.DataFrame(val_movies_dict)

# Merging the training and validation datasets on the movie ID
y_pred_and_y_true = pd.merge(baseline_y_pred, val_movies_df, on='FilmID')
baseline_y_pred_vs_y_true = y_pred_and_y_true.rename(columns={"Rangering": "Predicted rating"})

baseline_y_pred_vs_y_true
# calculating RMSE for the baseline model
print("RMSE baseline model: ", sqrt(mean_squared_error(baseline_y_pred_vs_y_true["Predicted rating"], 
                                                       baseline_y_pred_vs_y_true["Actual rating"])))



#2.3) Content Based Filtering
# ----------- PREPARING TRAINING DATA ----------- #
# Adding the movie features (genre, release year) to the training dataset
content_train_df = pd.merge(train_df, movies_df, on='FilmID')
content_train_df.drop(columns=['Tidstempel', 'FilmID', 'Tittel'], inplace=True)  # Remove useless features

content_train_df

# Creating a list of lists with the target attribute (movie ratings), grouped by userID
y_grouped_by_user = content_train_df.groupby(["BrukerID"])
y_train_listed = []

for i, j in y_grouped_by_user:
    y_train_listed.append(j["Rangering"].values)  # Extract target feature (ratings) from sorted data
    
# Target attributes for the first user
y_train_listed[0]

# Creating a list of dataframes with the feature set (movie info about genres and release year), grouped by userID
content_train_df.drop(columns='Rangering', inplace=True)
x_grouped_by_user = content_train_df.groupby(["BrukerID"])
x_train_listed = []

for user_id, group in x_grouped_by_user:
    x_train_listed.append(group.drop(columns='BrukerID'))
    
# The feature set for the first user
x_train_listed[0]

# Creating a new dataframe for the movies data
all_movies = movies_df.drop(columns=['Tittel', 'FilmID'])
all_movies

# ----------- PREPARING VALIDATION DATA ----------- #
# Creating a 2 dimensional matrix for the validation data in order to make it easier to calculate RMSE.

# Listing the user ID's in the same order as in the grouped dataframes
user_ids = []
for user_id, group in x_grouped_by_user:
    user_ids.append(user_id)
    
# Listing the movie IDs in the same order as in the movies dataset
movie_ids = movies_df["FilmID"].values

# Creating the matrix. Axis 1: User IDs, Axis 2: Movie IDs. Elements: True ratings from validation data
df_val = X_val.copy()
df_val["Rangering"] = y_val
validation_matrix = pd.DataFrame(index=user_ids, columns=movie_ids)  # Starting with an empty matrix
for array in df_val.to_records():  # Filling in the true ratings as elements
    user = array['BrukerID']
    movie = array['FilmID']
    true_rating = array['Rangering']
    validation_matrix.loc[user][movie] = true_rating
    
validation_matrix

# ----------- CREATING THE DIFFERENT CONTENT-BASED FILTERING MODELS ----------- #
# Assigning the different machine learning algorithms to be implemented in the models (incl. hyperparameters) to a dictionary
ml_algorithms = {"Linear regression": LinearRegression(), "Lasso": Lasso(alpha=1.0, max_iter=10000), 
                 "KNN_7": KNeighborsRegressor(n_neighbors=7),
                 "RFR": RandomForestRegressor(n_estimators=1000, n_jobs=3, max_features="auto", random_state=0),
                 "SVR": SVR(C=1.0)}

# Saving lists that I later use to construct a dataframe containing the performances of the models
CBF_models_listed = []
RMSE_CBF_listed = []

# For every machine learning algorithm in the dictionary:
for name, ml_alg in ml_algorithms.items():
    # Create an empty list for predictions
    CBF_predictions = []

    # For each user in the training dataset:
    for i, x in enumerate(x_train_listed):
        # Fit a machine learning model
        ml_alg.fit(x_train_listed[i], y_train_listed[i])
        # Predict all the ratings for this user for all movies
        prediction = ml_alg.predict(all_movies)
        prediction = np.clip(prediction, 1, 5)  # Predictions must be minimum 1, maximum 5
        # Append all the predictions to the predictions list
        CBF_predictions.append(prediction)

    # Create a dataframe with the predictions
    df_predict = pd.DataFrame(CBF_predictions, index=user_ids, columns=movie_ids)

    # Create a dataframe with only the predictions for the movies-user combinations that appear in the validation set
    num_actual = validation_matrix.to_numpy().flatten()[validation_matrix.notna().to_numpy().flatten()]
    num_predict = df_predict.to_numpy().flatten()[validation_matrix.notna().to_numpy().flatten()]

    # Calculate the RMSE for the content-based filtering model and add the result to the lists
    RMSE_CBF_listed.append(sqrt(mean_squared_error(num_predict, num_actual)))
    CBF_models_listed.append(name)


















# Printing the results
RMSE_CBF_df = pd.DataFrame({"Model": CBF_models_listed, "RMSE": RMSE_CBF_listed})
print("RMSE of different content-based filtering models without the year of release feature:")
RMSE_CBF_df

# Running the best content-based filtering model so far
model = Lasso(alpha=1.0, max_iter=10000)
CBF_predictions = []

# For each user in the training dataset:
for i, j in enumerate(x_train_listed):
    model.fit(x_train_listed[i], y_train_listed[i])
    prediction = model.predict(all_movies)
    prediction = np.clip(prediction, 1, 5)
    CBF_predictions.append(prediction)

# Creating a dataframe for the predictions
CBF_model = pd.DataFrame(CBF_predictions, index=user_ids, columns=movie_ids)
#############################2.4) Collaborative Filtering##################################



# A quick look at the training data
train_df.head()
# DATA PREPROCESSING: Calculating the Pearson Distance between all users in the training data
# Creating a 2D matrix (user ID vs movie ID) with the ratings as elements
user_matrix = train_df.pivot(index='BrukerID', columns='FilmID', values='Rangering')

# I subtract each user's average rating to magnify individual preferences
user_matrix = user_matrix.sub(user_matrix.mean(axis=1), axis=0)

# Replace NaN with 0.0, as this is now the "neutral" value
user_matrix = user_matrix.fillna(0.0)
# I calculate the Pearson Correlation between each user,
# and subtract this from 1 to get the Pearson Distance between users
user_dist_matrix = 1 - user_matrix.T.corr()
user_dist_matrix

# MODELLING: Predicting ratings for every user with K Nearest Neighbours
# Models with a different number of neighbors
ml_algorithms = {'kNN-5': 5, 'kNN-10': 10, 'kNN-20': 20, 'kNN-30': 30, 'kNN-40': 40, "kNN-60": 60}

models_CF = []
RMSE_CF = []

# Training the models and predicting for the users and movies in the validation data
for name, num_neighbours in ml_algorithms.items():
    predictions = []

    # For every rating in the validation data
    for index, row in X_val.iterrows():
        # If the movie is in the training data
        if row["FilmID"] in X_train["FilmID"].unique():
            # Extract all user ID's for users who have rated the movie
            users_rated_movie = X_train.loc[X_train['FilmID'] == row['FilmID'], 'BrukerID']
            # Sort these users by similarity (Pearson distance)
            users_sorted = (user_dist_matrix.loc[row['BrukerID'], users_rated_movie].sort_values())
            # Select the nearest neighbours
            nearest_neighbours = users_sorted[:num_neighbours]
            # Extract the nearest neighbours' ratings data
            nn_data = train_df.loc[train_df['BrukerID'].isin(nearest_neighbours.index.to_list())]
            # Calculate the weighted average of the nearest neighbours' ratings
            nearest_neighbours_avg_rating = np.average(nn_data.loc[train_df['FilmID'] == row['FilmID'], 'Rangering'],
                                                       axis=0, weights=(1/nearest_neighbours))
        else:
            # There is a small chance that a few movies in the validation set might not appear in the training set.
            # I therefore predict that the user will rate these movies with the average rating for all movies
            nearest_neighbours_avg_rating = 4   # Must be changed!

        # Appending the prediction to the list of predictions
        if not np.isnan(nearest_neighbours_avg_rating):
            predictions.append(nearest_neighbours_avg_rating)
        else:
            predictions.append(3)

    models_CF.append(name)
    RMSE_CF.append(sqrt(mean_squared_error(y_val, predictions)))


# Displaying the results
RMSE_CF_dict = {"Model": models_CF, "RMSE": RMSE_CF}
RMSE_CF_df = pd.DataFrame(RMSE_CF_dict)
RMSE_CF_df

# Visualizing how the number of neighbors effect the root mean sqaured error
fig7, ax7 = plt.subplots()
ax7.plot(RMSE_CF_df.Model, RMSE_CF_df.RMSE, label="RMSE", color='darkred', linewidth=2)
plt.xlabel("Number of nearest neighbors", labelpad=18)
plt.ylabel("Root mean squared error", labelpad=15)
plt.title("K-value effect on RMSE for collaborative filtering models")
fig7.set_figheight(10)
fig7.set_figwidth(16)
plt.show()

# Rerunning the best model so far (kNN-40) and storing the prediction results
best_CF_model = []
RMSE_best_CF = []

# Training the models and predicting for the users and movies in the validation data
CF_predictions = []

# For every movie in the validation data
for index, row in X_val.iterrows():
    # If that movie is in the training data
    if row["FilmID"] in X_train["FilmID"].unique():
        # Extract all user ID's for users who have rated the movie
        users_rated_movie = X_train.loc[X_train['FilmID'] == row['FilmID'], 'BrukerID']
        # Sort these users by similarity (Pearson distance)
        users_sorted = (user_dist_matrix.loc[row['BrukerID'], users_rated_movie].sort_values())
        # Select the nearest neighbours
        nearest_neighbours = users_sorted[:40]
        # Extract the nearest neighbours' ratings data
        nn_data = train_df.loc[train_df['BrukerID'].isin(nearest_neighbours.index.to_list())]
        # Calculate the weighted average of the nearest neighbours' ratings
        nearest_neighbours_avg_rating = np.average(nn_data.loc[train_df['FilmID'] == row['FilmID'], 'Rangering'],
                                                   axis=0, weights=(1/nearest_neighbours))
    else:
        # There is a small chance that a few movies in the validation set might not appear in the training set.
        # I therefore predict that the user will rate these movies with the average rating for all movies
        nearest_neighbours_avg_rating = 4   # Must be changed!

    # Appending the prediction to the list of predictions
    if not np.isnan(nearest_neighbours_avg_rating):
        CF_predictions.append(nearest_neighbours_avg_rating)
    else:
        CF_predictions.append(4)


###2.5) Hybrid Recommender
#Combining the Content-based and Collaborative filtering models to improve results

# Extracting the validation prediction from the CBF dataframe containing all predictions
CBF_predictions = []
for index, row in X_val.iterrows():
    user_predictions = CBF_model.loc[row["BrukerID"], row["FilmID"]]
    CBF_predictions.append(user_predictions)
    

# Calculating the predictions for the different hybrid "models": different weighted averages of CF and CBF filtering
print("RMSE combined approach (Lasso and KNN-40):")
weighted_avgs = [(0.5, 0.5), (0.45, 0.55), (0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.20, 0.80)]
  
for weight in weighted_avgs:
    combined_predictions = np.array([y_pred * weight[0] for y_pred in np.array(CBF_predictions)]) + np.array([y_pred * weight[1] for y_pred in np.array(CF_predictions)])
    print(f"RMSE for combined approach with CBF weighted {weight[0]} and CF weighted {weight[1]}: \n",
          sqrt(mean_squared_error(y_val, combined_predictions)), "\n")


#3) Feature Engineering
#Back to top

#The collaborative filtering model I am using for the hybrid recommender does not take any contextual information (such as time of rating) into account. This means that I cannot improve its performance by engineering any new features to the dataset. I could attempt to use a more elaborate method for calculating similarity, or use a completely different collaborative filtering approach such as multidimensional filtering, but that would be outside the scope of this project (at least for now).

#However, the content-based filtering model predicts new ratings based on the user's movie preferences. By supplying the content-based model with more information about each user's preferences, I aim to improve the performance of the hybrid recommender. I will therefore engineer two new features for the movies dataset, and measure the impact these new features may have on the RMSE of the content-based filtering model and the hybrid recommendations.

# Re-loading clean datasets
ratings_df = pd.read_csv("C:/datasets/rangering.csv")
movies_df = pd.read_csv("C:/datasets/film.csv")
users_df = pd.read_csv("C:/datasets/brucker.csv")
#3.1) Adding release year to the movies dataset

# Engineering feature nr. 1: Adding release year to movies dataset
for index, row in movies_df.iterrows():
    title = row[1]
    release_year = int(title[-5:-1])
    movies_df.loc[index, 'Utgivelsesår'] = release_year
    
# Printing the movies dataset again
movies_df.head()

#3.2) Adding the average age of movie fan to the movies dataset
#Movies, like any product on the market, are designed and marketed to a specific type of audience. From the data analysis in section 1.1, I discovered that the userbase was quite varied across different ages, though not as much as one would expect in regards to the general population. There may yet be a significant correlation between the average age of common viewers and the rating for each movie, especially for children's movies or teenage dramas. However, movies may be designed for a rather narrow audience, but viewed (and rated) by a much larger one. I therefore want to find out whether additional data about the average age of the users who rated each movie highly will improve recommendations. I will call this new feature "Fan Average Age", or "FAA" for short.

# Feature engineering: Adding average viewer age to movies_df
ratings_users_merged = users_df.merge(ratings_df, on=['BrukerID'])
movie_fans = {movie_id: [] for movie_id in movies_df['FilmID']}





for index, row in ratings_users_merged.iterrows():
    rating = row['Rangering']
    movie_id = row['FilmID']
    user_age = row['Alder']

    if int(rating) > 3:  # If the rating the user gave was higher than 3, the user probably enjoyed the movie
        movie_fans[movie_id].append(user_age)


fan_avg_ages = []

for movie_id, ages in movie_fans.items():
    if len(ages) > 0:
        fan_avg_ages.append(np.mean(ages))
    else:
        fan_avg_ages.append(np.NaN)

movies_df['FAA'] = fan_avg_ages

print(f'There are {movies_df.isnull().sum().sum()} missing values for "FAA"')

# Mean imputation for missing values due to low ratings
movies_df['FAA'].fillna(29.894868, inplace=True)
movies_df.head()
#There are 364 missing values for "FAA"

# Writing the engineered movies dataframe to a .csv file
movies_df.to_csv('movies_feature_engineered.csv', index=False)
#4) Second Iteration: Retraining the Content-based Filtering Model
#for an Improved Hybrid Recommender
#Back to top

#Engineering two new features may have improved the content-based filtering model. I will therefore retrain the three most promising CBF models and evaluate them on the validations dataset, before I implement the best one of them in the final hybrid recommender.

# Redoing the train-val-test split with the same random state to get the same split
# Splitting the ratings dataset into the feature set (X) and target labels (y)
X = ratings_df.drop(columns='Rangering')
y = ratings_df["Rangering"].values  # The movie ratings are the target variables we want to predict

# Preparing train, validation and test datasets.
X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5, random_state=101)

# Creating a complete training dataset with X_train and y_train
train_df = X_train.copy()
train_df["Rangering"] = y_train

train_df

#4.1) Improved Content-based Filtering model
# For this next iteration of modelling, I will train the two most promising CBF models 
# on the dataset with three newly engineered features.

# ----------- PREPARING TRAINING DATA ----------- #
# Adding the movie features (genre, release year) to the training dataset
content_train_df = pd.merge(train_df, movies_df, on='FilmID')
content_train_df.drop(columns=['Tidstempel', 'FilmID', 'Tittel'], inplace=True)  # Remove useless features

# Creating a list of lists with the target attribute (movie ratings), grouped by userID
y_grouped_by_user = content_train_df.groupby(["BrukerID"])
y_train_listed = []

for i, j in y_grouped_by_user:
    y_train_listed.append(j["Rangering"].values)  # Extract target feature (ratings) from sorted data
    
# Target attributes for the first user
y_train_listed[0]

# Creating a list of dataframes with the feature set (movie info about genres and release year), grouped by userID
content_train_df.drop(columns='Rangering', inplace=True)
x_grouped_by_user = content_train_df.groupby(["BrukerID"])
x_train_listed = []

for user_id, group in x_grouped_by_user:
    x_train_listed.append(group.drop(columns='BrukerID'))
    
    
# Creating a new dataframe for the movies data
all_movies = movies_df.drop(columns=['Tittel', 'FilmID'])
all_movies


# ----------- PREPARING VALIDATION DATA ----------- #
# Creating a 2 dimensional matrix for the validation data in order to make it easier to calculate RMSE.

# Listing the user ID's in the same order as in the grouped dataframes
user_ids = []
for user_id, group in x_grouped_by_user:
    user_ids.append(user_id)
    
# Listing the movie IDs in the same order as in the movies dataset
movie_ids = movies_df["FilmID"].values

# Creating the matrix. Axis 1: User IDs, Axis 2: Movie IDs. Elements: True ratings from validation data
df_val = X_val.copy()
df_val["Rangering"] = y_val
validation_matrix = pd.DataFrame(index=user_ids, columns=movie_ids)  # Starting with an empty matrix
for array in df_val.to_records():  # Filling in the true ratings as elements
    user = array['BrukerID']
    movie = array['FilmID']
    true_rating = array['Rangering']
    validation_matrix.loc[user][movie] = true_rating
    
# ----------- CREATING THE DIFFERENT IMPROVED CONTENT-BASED FILTERING MODELS ----------- #
ml_algorithms = {"Lasso": Lasso(alpha=1.0, max_iter=10000), "KNN_7": KNeighborsRegressor(n_neighbors=7),
                 "SVR": SVR(C=1.0)}

# Saving lists that I later use to construct a dataframe containing the performances of the models
improved_models_listed = []
improved_models_RMSE = []

# For every machine learning algorithm in the dictionary:
for name, ml_alg in ml_algorithms.items():
    # Create an empty list for predictions
    CBF_predictions = []

    # For each user in the training dataset:
    for i, x in enumerate(x_train_listed):
        # Fit a machine learning model with the year of release feature
        ml_alg.fit(x_train_listed[i], y_train_listed[i])
        # Predict all the ratings for this user for all movies
        prediction = ml_alg.predict(all_movies)
        prediction = np.clip(prediction, 1, 5)  # Predictions must be minimum 1, maximum 5
        # Append all the predictions to the predictions list
        CBF_predictions.append(prediction)

    # Create a dataframe with the predictions
    CBF_y_pred_df = pd.DataFrame(CBF_predictions, index=user_ids, columns=movie_ids)

    # Create a dataframe with only the predictions for the movies-user combinations that appear in the validation set
    num_actual = validation_matrix.to_numpy().flatten()[validation_matrix.notna().to_numpy().flatten()]
    num_predict = CBF_y_pred_df.to_numpy().flatten()[validation_matrix.notna().to_numpy().flatten()]

    # Calculate the RMSE for the content-based filtering model and add the result to the lists
    improved_models_RMSE.append(sqrt(mean_squared_error(num_predict, num_actual)))
    improved_models_listed.append(name)


# Printing the results
RMSE_content_df_improved = pd.DataFrame({"Model": improved_models_listed, "RMSE": improved_models_RMSE})
print("RMSE of different content-based filtering models, including the year of release feature")

# Running the best improved content-based filtering model
model = Lasso(alpha=1.0, max_iter=10000)
CBF_improved_predictions = []

# For each user in the training dataset:
for i, j in enumerate(x_train_listed):
    model.fit(x_train_listed[i], y_train_listed[i])
    prediction = model.predict(all_movies)
    prediction = np.clip(prediction, 1, 5)
    CBF_improved_predictions.append(prediction)

# Creating a dataframe for the predictions
CBF_improved_model = pd.DataFrame(CBF_improved_predictions, index=user_ids, columns=movie_ids)


# Creating a dataframe with only the predictions for the movies-user combinations that appear in the validation set
num_actual = validation_matrix.to_numpy().flatten()[validation_matrix.notna().to_numpy().flatten()]
num_predict = CBF_improved_model.to_numpy().flatten()[validation_matrix.notna().to_numpy().flatten()]
print("RMSE of best content-based filtering model:", sqrt(mean_squared_error(num_predict, num_actual)))


num_actual
# Saving the best CBF model's prediction to disk for later use
CBF_improved_model.to_pickle("./CBF_model.pkl")

#4.2) Improved Hybrid Recommender

# Extracting the validation prediction from the improved content dataframe containing all predictions
CBF_y_pred = []
for index, row in X_val.iterrows():
    user_predictions = CBF_improved_model.loc[row["BrukerID"], row["FilmID"]]
    CBF_y_pred.append(user_predictions)

# Calculating the predictions for the different hybrid "models": 
# different weighted averages of CF and improved CBF filtering
print("RMSE combined approach (Lasso and KNN-40):")
weighted_avgs = [(0.5, 0.5), (0.45, 0.55), (0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.20, 0.80)]
for weight in weighted_avgs:
    combined_predictions = ((np.array(CBF_y_pred) * weight[0]) + (np.array(CF_predictions)) * weight[1])
    print(f"RMSE for combined approach with CBF weighted {weight[0]} and CF weighted {weight[1]}: \n",
          sqrt(mean_squared_error(y_val, combined_predictions)), "\n")

#5) Final Test and Evaluation
#Back to top

#Finally, I will test the generalization ability of the improved hybrid recommender by making predictions on the test dataset. I will weight the CBF model 0.35 and the CF model 0.65, as this combination has proved to be the most optimal from the test made on the validation set.

# Building the hybrid recommender: Collaborative Filtering
CF_predictions_test = []
for index, row in X_test.iterrows():
    if row["FilmID"] in X_train["FilmID"].unique():
        users_rated_movie = X_train.loc[X_train['FilmID'] == row['FilmID'], 'BrukerID']
        users_sorted = (user_dist_matrix.loc[row['BrukerID'], users_rated_movie].sort_values())
        n_neighbours = users_sorted[:40]
        nn_data = train_df.loc[train_df['BrukerID'].isin(n_neighbours.index.to_list())]
        nearest_neighbours_avg_rating = np.average(nn_data.loc[train_df['FilmID'] == row['FilmID'], 'Rangering'],
                                                   axis=0, weights=(1/n_neighbours))
    else:
        nearest_neighbours_avg_rating = train_df["Rangering"].mean()

    # appending the prediction to the list
    if not np.isnan(nearest_neighbours_avg_rating):
        CF_predictions_test.append(nearest_neighbours_avg_rating)
    else:
        CF_predictions_test.append(4)

print("RMSE KNN_40:", sqrt(mean_squared_error(y_test, CF_predictions_test)))
# Building the hybrid recommender: Content-Based filtering
# Extracting the predictions for the movies and users in the test data
# from the CBF dataframe (which contains predictions for all movies and all users)
CBF_predictions_test = []
for index, row in X_test.iterrows():
    user_predictions = CBF_improved_model.loc[row["BrukerID"], row["FilmID"]]
    CBF_predictions_test.append(user_predictions)

print("RMSE Lasso:", sqrt(mean_squared_error(y_test, CBF_predictions_test)))
#RMSE Lasso: 1.019278388038111
# Calculating the hybrid recommendations
hybrid_predictions_test = (np.array([y_pred * 0.35 for y_pred in np.array(CBF_predictions_test)]) 
                           + np.array([y_pred * 0.65 for y_pred in np.array(CF_predictions_test)]))

# Displaying the test results from training the Hybrid Recommender on test data
print(f"RMSE hybrid recommendations (test data): {sqrt(mean_squared_error(y_test, hybrid_predictions_test))} ")
