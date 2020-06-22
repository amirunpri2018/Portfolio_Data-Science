# %%
"""
# COLLABORATIVE BASED MOVIE RECOMMENDER SYSTEM :

"""

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# %%
ratings=pd.read_csv('ratings.csv')
movies=pd.read_csv('movies.csv')

# %%
ratings=pd.merge(movies,ratings)

# %%
"""

 <br>
for collaborative filtering , i am droping some columns<br>

"""

# %%
ratings=ratings.drop(['genres','timestamp'],axis=1)
ratings.head()

# %%
user_ratings=ratings.pivot_table(index=['userId'],columns=['title']
                                 ,values='rating')

# %%
print('\n')
#print(user_ratings.head())
user_ratings.head()

# %%
"""

 <br>
Let us drop some movies which has been rated by <br>
less than 10 users and fill remaining NaN with 0<br>
thresh stands fot threshold<br>

"""

# %%
user_ratings=user_ratings.dropna(thresh=10,axis=1).fillna(0)
print('\n')
#print(user_ratings.head())
user_ratings.head()

# %%
"""

 <br>
Let us build our similarity Matrix<br>
for this collaborative filtering , we are using<br>
pearson coorelation <br>

"""

# %%
print('\n')
item_similarity_Df=user_ratings.corr(method='pearson')
item_similarity_Df.head(50)

# %%
"""

 <br>
As you can see it shows a perfect correlation between action movies and romantic movies<br>
here i have subtracted 2.5 because it is the mean of our data<br>

"""

# %%
def get_similar_movies(movie_name, user_rating):
    similar_Score = item_similarity_Df[movie_name] * (user_rating - 2.5)
    similar_Score = similar_Score.sort_values(ascending=False)
    return similar_Score

# %%
"""

 <br>
Let us Create a Fake Action Movie Lover User to find his/her interests :<br>
i have created a empty dataframe which will store info of specific user<br>
and this will get multiple movies and its ratings<br>

"""

# %%
print('\n')
action_lover = [("Amazing Spider-Man, The (2012)",5),("Mission: Impossible III (2006)",4),("Toy Story 3 (2010)",2),("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)",4)]

# %%
similar_movies=pd.DataFrame()

# %%
for movie,rating in action_lover:
    similar_movies=similar_movies.append(get_similar_movies(movie,rating),ignore_index=True)

# %%
print('Recommended Movies for this Action User :')
print('\n')
#print(similar_movies.head(10))
print('\n')
print(similar_movies.sum().sort_values(ascending=False))

# %%
print('\n')

# %%
"""

 <br>
Let us Create a Fake Romantic Movie Lover User to find his/her interests :<br>

"""

# %%
romantic_lover = [("(500) Days of Summer (2009)",5),("Alice in Wonderland (2010)",3),("Aliens (1986)",1),("2001: A Space Odyssey (1968)",2)]
similar_movies = pd.DataFrame()
for movie,rating in romantic_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie,rating),ignore_index = True)

# %%
print('Recommended Movies for this Romantic User :')
print('\n')
#print(similar_movies.head(10))
print('\n')
print(similar_movies.sum().sort_values(ascending=False))

# %%
