import streamlit as st
import pandas as pd
import ast
import difflib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="REEL RECOMMENDER",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0E1117;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: white;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #AAAAAA;
    margin-bottom: 20px;
}

.movie-card {
    background-color: #1c1c1c;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    height: 180px;
    overflow: hidden;
}

.movie-card h3 {
    font-size: 20px;
    margin-bottom: 8px;
}

.movie-card p {
    font-size: 13px;
    color: #CCCCCC;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    background-color: #E50914;
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.markdown(
    '<div class="main-title">🎬 REEL RECOMMENDER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Find movies similar to your favorites 🍿</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ---------------- #

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on='title')

movies = movies[[
    'movie_id',
    'title',
    'overview',
    'genres',
    'keywords',
    'cast',
    'crew'
]]

movies.dropna(inplace=True)

movies['original_overview'] = movies['overview']

# ---------------- FUNCTIONS ---------------- #

def convert(text):

    L = []

    for i in ast.literal_eval(text):
        L.append(i['name'])

    return L


def fetch_director(text):

    L = []

    for i in ast.literal_eval(text):

        if i['job'] == 'Director':
            L.append(i['name'])

    return L

# ---------------- FEATURE ENGINEERING ---------------- #

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['tags'] = (
    movies['overview'] +
    movies['genres'] +
    movies['keywords'] +
    movies['cast'] +
    movies['crew']
)

new_df = movies[[
    'movie_id',
    'title',
    'tags',
    'original_overview'
]]

new_df['tags'] = new_df['tags'].apply(
    lambda x: " ".join(x)
)

new_df['tags'] = new_df['tags'].apply(
    lambda x: x.lower()
)

# ---------------- ML ---------------- #

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

# ---------------- RECOMMEND FUNCTION ---------------- #

def recommend(movie):

    movie_list = new_df['title'].tolist()

    close_match = difflib.get_close_matches(
        movie,
        movie_list
    )

    if not close_match:
        return []

    movie = close_match[0]

    movie_index = new_df[
        new_df['title'] == movie
    ].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:7]

    recommended_movies = []

    for i in movie_list:

        movie_data = {
            "title": new_df.iloc[i[0]].title,
            "overview": new_df.iloc[i[0]].original_overview
        }

        recommended_movies.append(movie_data)

    return recommended_movies

# ---------------- SEARCH ---------------- #

movie_list = new_df['title'].values

selected_movie = st.selectbox(
    "🎥 Select Movie",
    movie_list
)

if st.button("🚀 Recommend"):

    recommendations = recommend(selected_movie)

    st.markdown("## 🎯 Recommended For You")

    col1, col2, col3 = st.columns(3)

    cols = [col1, col2, col3]

    for index, movie in enumerate(recommendations):

        with cols[index % 3]:

            st.markdown(f"""
            <div class="movie-card">
                <h3>🎬 {movie['title']}</h3>
                <p>{movie['overview'][:120]}...</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------- HOMEPAGE ---------------- #

st.write("---")

st.markdown("## 🔥 Trending Now")

t1, t2, t3 = st.columns(3)

with t1:
    st.markdown("""
    <div class="movie-card">
    <h3>Interstellar</h3>
    <p>Sci-Fi Adventure</p>
    </div>
    """, unsafe_allow_html=True)

with t2:
    st.markdown("""
    <div class="movie-card">
    <h3>Avengers: Endgame</h3>
    <p>Action Superhero</p>
    </div>
    """, unsafe_allow_html=True)

with t3:
    st.markdown("""
    <div class="movie-card">
    <h3>Joker</h3>
    <p>Crime Drama</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## ⭐ Most Viewed")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="movie-card">
    <h3>Avatar</h3>
    <p>Epic Sci-Fi</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="movie-card">
    <h3>Titanic</h3>
    <p>Romance Drama</p>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="movie-card">
    <h3>The Dark Knight</h3>
    <p>Action Crime</p>
    </div>
    """, unsafe_allow_html=True)