import numpy as np
import pandas as pd
import ast

"""# New Section"""

import pandas as pd
credits = pd.read_csv('tmdb_5000_credits.csv', on_bad_lines='skip', engine='python')


"""# New Section"""

movies = pd.read_csv('tmdb_5000_movies.csv')

movies.head(1)

credits.head(1)

movies= movies.merge(credits,on='title')

movies.head(1)

movies.info()

movies['original_language'].value_counts()

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.head()

movies.isnull().sum()

movies.dropna(inplace=True)

movies.duplicated().sum()

movies.iloc[0].genres

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

movies.head()

import ast

def convert(obj):
    # If obj is a string, try to evaluate it
    if isinstance(obj, str):
        try:
            # Use literal_eval only if obj is a string
            obj = ast.literal_eval(obj)
        except (ValueError, SyntaxError):
            # Handle cases where obj is not a valid string representation of a Python literal
            return []

    # Assuming obj is now a list, extract 'name' from each dictionary
    L = []
    if isinstance(obj, list):
        for i in obj:
            if isinstance(i, dict) and 'name' in i:
                L.append(i['name'])
    return L

movies['keywords']= movies['keywords'].apply(convert)

def convert3(obj):
    L = []
    counter = 0

    # Check if obj is a list
    if isinstance(obj, list):
        for name in obj:
            if counter < 3:  # Limit to the first 3 names
                L.append(name)
                counter += 1
    else:
        # Handle unexpected input (like None or other types)
        return []

    return L

movies['cast'].apply(convert3)

movies['cast']= movies['cast'].apply(convert3)

movies.head()

movies['crew'][0]

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

import ast

def fetch_director(obj):
    """
    Fetches the director's name from a crew object string.

    Args:
        obj (str): A string containing a crew object.
                     It's expected to be a list of dictionaries in JSON format,
                     but might actually be a string representation of a list.

    Returns:
        list: A list containing the director's name, or an empty list if
              no director is found or if there's a parsing error.
    """
    L = []
    try:
        # Attempt to parse as JSON first
        data = ast.literal_eval(obj)
    except (SyntaxError, ValueError):
        # If it's not a valid dict, try evaluating it as a list literal
        try:
            data = eval(obj)
            # If this also fails, assume the data is corrupted and skip it
        except:
            return L

    for i in data:
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'].apply(fetch_director)

movies['crew']=movies['crew'].apply(fetch_director)

movies.head()

movies['overview'][0]

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags']=movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

new_df = movies[['movie_id','title','tags']]

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

new_df.head()

new_df['tags'][0]

new_df['tags']= new_df['tags'].apply(lambda x:x.lower())

new_df.head()

import nltk

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def stem(text):
    ps = PorterStemmer()  # Initialize the PorterStemmer
    y = []

    for i in text.split():
        y.append(ps.stem(i))  # Use 'stem' instead of 'Stem'

    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

new_df['tags'][0]

new_df['tags'][1]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors

vectors[0]

cv.get_feature_names_out()

['loved','loving','love']
['love','love','love']

ps.stem('loved')

stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction samworthington zoesaldana sigourneyweaver jamescameron')

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

    return

recommend('Batman Begins')

new_df.iloc[1216].title

pip install streamlit

import streamlit as st
st.title('Movie Recommender System')

import pickle
import streamlit as st
import requests
# import os

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key={YOUR_API_KEY}&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')
movies = pickle.load(open('movie_list.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

import pickle
import streamlit as st
import requests
import os

# Get the absolute path to the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full paths to the pickle files
movie_list_path = os.path.join(script_dir, 'movie_list.pkl')
similarity_path = os.path.join(script_dir, 'similarity.pkl')


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key={YOUR_API_KEY}&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')

# Load the pickle files using the full paths
movies = pickle.load(open(movie_list_path,'rb'))
similarity = pickle.load(open(similarity_path,'rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])

import pickle
import streamlit as st
import requests
import os

# Get the current working directory instead of relying on __file__
script_dir = os.getcwd()

# Construct the full paths to the pickle files
movie_list_path = os.path.join(script_dir, 'movie_list.pkl')
similarity_path = os.path.join(script_dir, 'similarity.pkl')


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key={YOUR_API_KEY}&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)