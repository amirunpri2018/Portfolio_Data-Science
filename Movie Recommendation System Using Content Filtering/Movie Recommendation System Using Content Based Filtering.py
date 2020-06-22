# %%
"""
# CONTENT BASED MOVIE RECOMMENDER SYSTEM USING MACHINE LEARNING: 

"""

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('movie_dataset.csv')
df.head()


# %%
print(df.columns)
print('\n')

# %%
"""

 <br>
Creating a column which comibes all selected features in df<br>
And also Getting rid of NULL VAlUES in features:<br>

"""

# %%
features = ['keywords', 'cast', 'genres', 'director']

# %%
for feature in features:
    df[feature] = df[feature].fillna('')

# %%
def combine_feature(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    except:
        print("Error : ", row)
        print('\n')

# %%
df['combined features'] = df.apply(combine_feature, axis=1)
print('\n')
print('Combined Features Are: ')
print('\n')
print(df['combined features'].head())
print('\n')

# %%
"""
### What this basically does is , we will try to find similarity between combined features using cosine similarity and on the basis of the cosie similarity score .. we will try to recommend movies to user
"""

# %%
"""

 <br>
Creating a similarity Matrix and cosine score :<br>

"""

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
cv = CountVectorizer()

# %%
count_matrix = cv.fit_transform(df['combined features'])

# %%
print('\n')
cosine_sim = cosine_similarity(count_matrix)

# %%
print(count_matrix)

# %%
df4=pd.DataFrame(cosine_sim)
df4.head()

# %%
df4[0].sort_values(ascending=False)

# %%
df[df.index==94]

# %%
"""

 <br>
Next We have Two Functions : which will help to obtain title of movie from index and<br>
index of movie from title<br>

"""

# %%
def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]

# %%
def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]
print('\n')

# %%
"""
### you can take any random movie name from database..
"""

# %%
movie_user_likes="Avatar"

# %%
"""

 <br>
we will be getting index of that particular movie which user likes<br>
and then , we will calculate its similarity basis on cosine score <br>
and similarity matrix<br>

"""

# %%
movie_index=get_index_from_title(movie_user_likes)

# %%
similar_movies=cosine_sim[movie_index]

# %%
similar_movies=list(enumerate(cosine_sim[movie_index]))

#sorted similar movies will sort cosine scores based on descending order
#lambda function is just taking second parameter , cosine score


# in given list( index, cosine score)

# %%


sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)
print(sorted_similar_movies)


# %%
print('\n')
print("Recommended Movies Based On ", movie_user_likes , " Are :")
print('\n')
i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i=i+1
    if i>50:
        break

# %%
"""
"""
We have created a loop to show only first 50 movies..you can remove it 
if you want

The mentioned code actually takes movie as iterative variable and with the help
of get title from index function , it will take index and will
print its name respectively( movie[ 0 ] , since 0 is index and 1 is cosine score)
"""
"""

# %%
