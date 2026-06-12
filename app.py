import streamlit as st
import pandas as pd
import ast
import difflib
import numpy as np
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ai_engine import AIEngine
from translations import TRANSLATIONS
from dotenv import load_dotenv

# ---------------- LOGGING & ENVIRONMENT CONFIG ---------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables (such as OPENAI_API_KEY from .env)
load_dotenv()

# ---------------- PAGE CONFIG (Must be first) ---------------- #

st.set_page_config(
    page_title="MOVIE RECOMMENDER | AI Movie Discovery",
    page_icon="🎬",
    layout="wide"
)

# ---------------- NAVIGATION STATE & LANG SELECTOR ---------------- #

# Language select at the top of the sidebar
st.sidebar.markdown('<div style="font-size: 22px; font-weight: 700; color: #ffffff; margin-bottom: 5px; background: linear-gradient(90deg, #a855f7 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🌐 Language / భాష</div>', unsafe_allow_html=True)
lang = st.sidebar.selectbox("Select Language / భాష ఎంచుకోండి", ["English", "తెలుగు"], label_visibility="collapsed")
t = TRANSLATIONS[lang]

# ---------------- PREMIUM CUSTOM CSS & THEMING ---------------- #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Apply font globally */
html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at top center, #1b0e2d 0%, #08050e 100%) !important;
    color: #e5e0fa !important;
}

/* Main title styling */
.main-title {
    text-align: center;
    font-size: 64px;
    font-weight: 800;
    background: linear-gradient(135deg, #ff3366 0%, #a855f7 50%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
    font-family: 'Outfit', sans-serif;
    letter-spacing: -1px;
}

.sub-title {
    text-align: center;
    font-size: 19px;
    color: #b0a4d4;
    margin-bottom: 35px;
    font-weight: 300;
}

/* Sidebar styling overrides */
[data-testid="stSidebar"] {
    background-color: #0c0816 !important;
    border-right: 1px solid rgba(168, 85, 247, 0.15) !important;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 15px;
    background: linear-gradient(90deg, #a855f7 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Glassmorphic Movie Card styling */
.movie-card {
    background: rgba(23, 16, 38, 0.6) !important;
    border: 1px solid rgba(168, 85, 247, 0.2) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), border-color 0.3s ease, box-shadow 0.3s ease !important;
    height: 230px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    overflow: hidden !important;
}

.movie-card:hover {
    transform: translateY(-6px) scale(1.02) !important;
    border-color: rgba(168, 85, 247, 0.7) !important;
    box-shadow: 0 12px 40px 0 rgba(168, 85, 247, 0.3) !important;
}

.movie-card h3 {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 19px !important;
    margin: 0 0 8px 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.movie-card p {
    color: #c3badc !important;
    font-size: 13.5px !important;
    line-height: 1.5 !important;
    overflow: hidden !important;
    display: -webkit-box !important;
    -webkit-line-clamp: 4 !important;
    -webkit-box-orient: vertical !important;
    margin: 0 !important;
}

.movie-badge {
    background: linear-gradient(90deg, #a855f7 0%, #6366f1 100%) !important;
    color: white !important;
    padding: 3px 10px !important;
    border-radius: 20px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    display: inline-block !important;
    margin-top: 10px !important;
    align-self: flex-start !important;
}

/* Glowing primary button styles */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #ff3366 0%, #a855f7 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 18px rgba(255, 51, 102, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(168, 85, 247, 0.5) !important;
}

/* Chat container styling */
.stChatMessage {
    border-radius: 12px !important;
    background-color: rgba(23, 16, 38, 0.3) !important;
    border: 1px solid rgba(168, 85, 247, 0.1) !important;
    margin-bottom: 10px !important;
}

/* Tabs customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: rgba(23, 16, 38, 0.4) !important;
    border: 1px solid rgba(168, 85, 247, 0.1) !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    color: #c3badc !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    border-color: rgba(168, 85, 247, 0.4) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #a855f7 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE & SUBTITLE ---------------- #

st.markdown(
    f'<div class="main-title">{t["title"]}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="sub-title">{t["subtitle"]}</div>',
    unsafe_allow_html=True
)

# ---------------- DATA LOADING (CACHED) ---------------- #

@st.cache_data
def load_and_preprocess_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    
    # Merge on id/movie_id to avoid title collision duplicates
    credits_clean = credits.drop(columns=['title'])
    merged = movies.merge(credits_clean, left_on='id', right_on='movie_id')
    
    # Select columns we need
    df_processed = merged[[
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew',
        'vote_average',
        'release_date'
    ]].copy()
    
    # Clean rows with critical missing features
    df_processed.dropna(subset=['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew'], inplace=True)
    df_processed['original_overview'] = df_processed['overview']
    
    # Extract release year helper
    def get_year(date_str):
        if pd.isna(date_str) or not isinstance(date_str, str):
            return "N/A"
        parts = date_str.split('-')
        return parts[0] if parts else "N/A"
        
    df_processed['release_year'] = df_processed['release_date'].apply(get_year)
    
    # Helpers to extract list structures from JSON text columns
    def convert(text):
        try:
            L = []
            for i in ast.literal_eval(text):
                L.append(i['name'])
            return L
        except Exception:
            return []

    def fetch_director(text):
        try:
            L = []
            for i in ast.literal_eval(text):
                if i['job'] == 'Director':
                    L.append(i['name'])
            return L
        except Exception:
            return []

    df_processed['genres_list'] = df_processed['genres'].apply(convert)
    df_processed['genres'] = df_processed['genres'].apply(convert)
    df_processed['keywords'] = df_processed['keywords'].apply(convert)
    df_processed['cast'] = df_processed['cast'].apply(convert).apply(lambda x: x[0:3])
    df_processed['crew'] = df_processed['crew'].apply(fetch_director)
    df_processed['overview'] = df_processed['overview'].apply(lambda x: str(x).split())
    
    # Remove spaces from keywords/cast/crew for strict bag-of-words exact matching
    df_processed['genres'] = df_processed['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    df_processed['keywords'] = df_processed['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    df_processed['cast'] = df_processed['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    df_processed['crew'] = df_processed['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    df_processed['tags'] = (
        df_processed['overview'] +
        df_processed['genres'] +
        df_processed['keywords'] +
        df_processed['cast'] +
        df_processed['crew']
    )
    
    df_processed['tags'] = df_processed['tags'].apply(lambda x: " ".join(x).lower())
    
    final_df = df_processed[[
        'movie_id',
        'title',
        'tags',
        'original_overview',
        'genres_list',
        'vote_average',
        'release_year'
    ]].rename(columns={'genres_list': 'genres'}).reset_index(drop=True)
    
    return final_df

df = load_and_preprocess_data()

# ---------------- STATIC SIMILARITY (CACHED RESOURCE) ---------------- #

@st.cache_resource
def get_count_similarity(tags_series):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(tags_series).toarray()
    # Memory Optimization: Use float32 to reduce similarity memory usage by 50%
    similarity = cosine_similarity(vectors).astype(np.float32)
    return similarity

content_sim = get_count_similarity(df['tags'])

# ---------------- SIDEBAR CONTROLS ---------------- #

st.sidebar.markdown(f'<div class="sidebar-title">{t["ai_settings"]}</div>', unsafe_allow_html=True)

# Default to the environment's OpenAI API Key (safe, no hardcoding)
env_key = os.environ.get("OPENAI_API_KEY", "")

# Load/check session state for key validation to avoid network calls on every streamlit rerun
if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = {}

api_key = st.sidebar.text_input(
    t["openai_key"],
    type="password",
    value=env_key,
    help=t["openai_help"]
)

is_validated = False
if api_key and api_key.strip():
    if api_key not in st.session_state.api_key_valid:
        with st.sidebar.spinner("Validating API key..."):
            st.session_state.api_key_valid[api_key] = AIEngine.validate_api_key(api_key)
    is_validated = st.session_state.api_key_valid[api_key]

# Initialize AI Engine
ai_engine = AIEngine(api_key=api_key, is_validated=is_validated)

if ai_engine.has_key:
    st.sidebar.markdown(
        f"<div style='color: #4ade80; font-weight: 600; font-size: 14px;'>{t['mode_active']}</div>",
        unsafe_allow_html=True
    )
elif api_key:
    st.sidebar.markdown(
        f"<div style='color: #ef4444; font-weight: 600; font-size: 14px;'>❌ Invalid API Key / Quota Exceeded</div>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        f"<div style='color: #fbbf24; font-weight: 600; font-size: 14px;'>{t['mode_offline']}</div>",
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.markdown(f'<div class="sidebar-title">{t["user_pref"]}</div>', unsafe_allow_html=True)
st.sidebar.caption(t["pref_help"])

# Extract unique genres list
all_genres = sorted(list(set([g for genres in df['genres'] for g in genres if g])))
user_genres = st.sidebar.multiselect(t["fav_genres"], all_genres)
min_rating = st.sidebar.slider(t["min_rating"], 0.0, 10.0, 6.0, 0.5)

# Initialize / Load embeddings
embeddings = ai_engine.cache_movie_embeddings(df)

# ---------------- HELPER CLASSIC RECOMMEND ---------------- #

def get_classic_recommendations(movie_index, df_data, sim_matrix, top_k=6):
    try:
        if movie_index < 0 or movie_index >= len(df_data):
            return []
        distances = sim_matrix[movie_index]
        movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_k+1]
        
        results = []
        for i in movie_indices:
            results.append({
                "title": df_data.iloc[i[0]]['title'],
                "overview": df_data.iloc[i[0]]['original_overview']
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching classic recommendations: {e}")
        return []

# ---------------- NAVIGATION TABS ---------------- #

tab1, tab2, tab3, tab4 = st.tabs([
    t["tab_hybrid"], 
    t["tab_chat"], 
    t["tab_mood"], 
    t["tab_compare"]
])

# ---------------- TAB 1: CLASSIC & HYBRID FINDER ---------------- #

with tab1:
    st.markdown(t["find_title"])
    st.caption(t["find_caption"])

    # Robust index-based selection to avoid duplicate title warnings
    selected_movie_idx = st.selectbox(
        t["search_select"],
        options=range(len(df)),
        format_func=lambda idx: f"{df.iloc[idx]['title']} ({df.iloc[idx]['release_year']})",
        key="hybrid_movie_select"
    )

    if st.button(t["btn_find"], key="btn_hybrid_find"):
        
        # Get recommendations using index
        classic_recs = get_classic_recommendations(selected_movie_idx, df, content_sim, top_k=6)
        hybrid_recs = ai_engine.get_hybrid_recommendations(
            selected_movie_idx, 
            user_genres, 
            min_rating, 
            df, 
            embeddings, 
            content_sim, 
            top_k=6
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(t["classic_header"])
            st.caption(t["classic_caption"])
            
            if not classic_recs:
                st.write("No recommendations found." if lang == "English" else "సిఫార్సులు కనుగొనబడలేదు.")
            else:
                for idx, movie in enumerate(classic_recs):
                    st.markdown(f"""
                    <div class="movie-card" style="height: 180px !important;">
                        <div>
                            <h3>{movie['title']}</h3>
                            <p style="-webkit-line-clamp: 3 !important;">{movie['overview']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_right:
            st.markdown(t["hybrid_header"])
            st.caption(t["hybrid_caption"])

            if not hybrid_recs:
                st.write("No recommendations found. Adjust your user preference settings in the sidebar." if lang == "English" else "సిఫార్సులు కనుగొనబడలేదు. సైడ్‌బార్‌లో మీ సెట్టింగ్‌లను సర్దుబాటు చేయండి.")
            else:
                for idx, movie in enumerate(hybrid_recs):
                    total_pct = int(movie['final_score'] * 100)
                    st.markdown(f"""
                    <div class="movie-card" style="height: 180px !important; margin-bottom: 2px !important;">
                        <div>
                            <h3>{movie['title']}</h3>
                            <p style="-webkit-line-clamp: 3 !important;">{movie['overview']}</p>
                        </div>
                        <div style="font-size: 12px; color: #a855f7; font-weight: bold; margin-top: 4px;">
                            {t['score_hybrid']}: {total_pct}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visual representation of scores
                    st.caption(
                        f"{t['score_pref']}: {int(movie['pref_score']*100)}% | "
                        f"{t['score_content']}: {int(movie['content_score']*100)}% | "
                        f"{t['score_semantic']}: {int(movie['semantic_score']*100)}%"
                    )
                    st.progress(movie['final_score'])
                    st.write("")

# ---------------- TAB 2: CONVERSATIONAL MOVIE ASSISTANT ---------------- #

with tab2:
    st.markdown(t["chat_title"])
    st.caption(t["chat_caption"])

    # Suggested prompt pills (localized)
    if lang == "తెలుగు":
        suggested_prompts = [
            "ఇన్సెప్షన్ లాంటి మైండ్-బెండింగ్ సినిమాలను సూచించండి.",
            "నాకు రొమాన్స్ లేని ఫన్నీ కామెడీ సినిమా కావాలి.",
            "ఇంటర్‌స్టెల్లార్ లాంటిదే కానీ కొంచెం చిన్నగా ఉండే సినిమాను సిఫార్సు చేయండి."
        ]
    else:
        suggested_prompts = [
            "Suggest mind-bending movies like Inception.",
            "I want a funny comedy with no romance.",
            "Recommend something similar to Interstellar but shorter."
        ]

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant", 
                "content": "Hello! 🎬 I'm your MOVIE RECOMMENDER assistant. Tell me what kind of movie you'd like to watch today, or share your favorite film, and I'll find the perfect match!" if lang == "English" else "హలో! 🎬 నేను మీ మూవీ రికమెండర్ అసిస్టెంట్. ఈరోజు మీరు ఎలాంటి సినిమా చూడాలనుకుంటున్నారో చెప్పండి లేదా మీకు నచ్చిన సినిమాను పంచుకోండి, నేను సరిపోయే మ్యాచ్‌ను కనుగొంటాను!"
            }
        ]

    # Click-to-ask suggestions
    st.markdown(f"<div style='font-size: 14px; margin-bottom: 8px;'>{t['try_asking']}</div>", unsafe_allow_html=True)
    sug_cols = st.columns(3)
    for index, prompt in enumerate(suggested_prompts):
        if sug_cols[index].button(prompt, key=f"chat_sug_{index}"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("Analyzing intent and looking up recommendations..." if lang == "English" else "ఉద్దేశాన్ని విశ్లేషిస్తోంది మరియు సిఫార్సులను వెతుకుతోంది..."):
                reply, rec_movies = ai_engine.chat_with_assistant(
                    st.session_state.chat_history[:-1], 
                    prompt, 
                    df,
                    language=lang
                )
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": reply, 
                    "movies": rec_movies
                })
            st.rerun()

    st.write("---")

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "movies" in msg and msg["movies"]:
                movies_list = msg["movies"]
                cols_chat = st.columns(min(3, len(movies_list)))
                for m_idx, movie in enumerate(movies_list):
                    with cols_chat[m_idx % 3]:
                        st.markdown(f"""
                        <div class="movie-card" style="height: 180px !important;">
                            <div>
                                <h3>{movie['title']}</h3>
                                <p style="-webkit-line-clamp: 3 !important;">{movie['overview']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input
    user_input = st.chat_input(t["chat_placeholder"])
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing intent and database..." if lang == "English" else "విశ్లేషిస్తోంది..."):
                reply, rec_movies = ai_engine.chat_with_assistant(
                    st.session_state.chat_history[:-1], 
                    user_input, 
                    df,
                    language=lang
                )
                st.markdown(reply)
                if rec_movies:
                    cols_chat = st.columns(min(3, len(rec_movies)))
                    for m_idx, movie in enumerate(rec_movies):
                        with cols_chat[m_idx % 3]:
                            st.markdown(f"""
                            <div class="movie-card" style="height: 180px !important;">
                                <div>
                                    <h3>{movie['title']}</h3>
                                    <p style="-webkit-line-clamp: 3 !important;">{movie['overview']}</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply,
            "movies": rec_movies
        })
        st.rerun()

# ---------------- TAB 3: MOOD-BASED RECOMMENDER ---------------- #

with tab3:
    st.markdown(t["mood_title"])
    st.caption(t["mood_caption"])

    if lang == "తెలుగు":
        moods = [
            {"name": "ఆనందంగా 😊", "query": "feel-good happy comedy lighthearted movie to make me smile"},
            {"name": "బాధగా 😔", "query": "sad emotional tearjerker drama touching movie"},
            {"name": "ఉత్సాహంగా 🔥", "query": "exciting action adventure fast paced high energy blockbuster"},
            {"name": "ప్రేమగా ❤️", "query": "romantic love story romance drama comedy"},
            {"name": "స్ఫూర్తిదాయకంగా 💪", "query": "motivational inspiring success triumph sports work hard"},
            {"name": "భయానకంగా 😱", "query": "horror scary spooky ghost thriller creepy"}
        ]
    else:
        moods = [
            {"name": "Happy 😊", "query": "feel-good happy comedy lighthearted movie to make me smile"},
            {"name": "Sad 😔", "query": "sad emotional tearjerker drama touching movie"},
            {"name": "Excited 🔥", "query": "exciting action adventure fast paced high energy blockbuster"},
            {"name": "Romantic ❤️", "query": "romantic love story romance drama comedy"},
            {"name": "Motivational 💪", "query": "motivational inspiring success triumph sports work hard"},
            {"name": "Horror 😱", "query": "horror scary spooky ghost thriller creepy"}
        ]

    st.markdown(f"<div style='font-size: 14px; margin-bottom: 8px;'>{t['feel_label']}</div>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)
    m_cols = [m_col1, m_col2, m_col3]
    selected_mood_query = None
    selected_mood_name = None

    for idx, mood in enumerate(moods):
        if m_cols[idx % 3].button(mood["name"], use_container_width=True, key=f"mood_btn_{idx}"):
            selected_mood_query = mood["query"]
            selected_mood_name = mood["name"]

    st.write("")
    custom_mood = st.text_input(t["mood_custom"])
    if st.button(t["btn_custom_mood"], use_container_width=True):
        if custom_mood:
            selected_mood_query = custom_mood
            selected_mood_name = f"Custom: '{custom_mood}'" if lang == "English" else f"కస్టమ్: '{custom_mood}'"

    if selected_mood_query:
        st.write("---")
        st.markdown(f"{t['mood_header_pre']} {selected_mood_name}")
        
        with st.spinner(t["scanning_db"]):
            recs = ai_engine.get_mood_recommendations(selected_mood_query, df, embeddings, top_k=6, language=lang)

        col_a, col_b, col_c = st.columns(3)
        cols_grid = [col_a, col_b, col_c]
        
        for r_idx, rec in enumerate(recs):
            with cols_grid[r_idx % 3]:
                st.markdown(f"""
                <div class="movie-card" style="height: 250px !important; margin-bottom: 2px !important;">
                    <div>
                        <h3>{rec['title']}</h3>
                        <p style="-webkit-line-clamp: 4 !important;">{rec['overview']}</p>
                    </div>
                    <div class="movie-badge">
                        🎭 Mood Match
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.info(f"💡 {rec['explanation']}")
                st.write("")

# ---------------- TAB 4: SIMILAR MOVIE COMPARISON ---------------- #

with tab4:
    st.markdown(t["compare_title"])
    st.caption(t["compare_caption"])

    selected_compare_movie_idx = st.selectbox(
        t["compare_select_label"],
        options=range(len(df)),
        format_func=lambda idx: f"{df.iloc[idx]['title']} ({df.iloc[idx]['release_year']})",
        key="compare_select"
    )

    if st.button(t["btn_compare_label"], key="btn_compare"):
        
        # Get classic recommendations
        classic_recs = get_classic_recommendations(selected_compare_movie_idx, df, content_sim, top_k=5)

        # Get OpenAI Semantic recommendations
        ai_recs = []
        try:
            movie_index = selected_compare_movie_idx
            target_emb = embeddings[movie_index]
            sims = cosine_similarity([target_emb], embeddings).flatten()
            top_indices = sims.argsort()[-6:][::-1]
            
            for idx in top_indices:
                if idx != movie_index:
                    ai_recs.append({
                        "title": df.iloc[idx]['title'],
                        "overview": df.iloc[idx]['original_overview'],
                        "score": sims[idx]
                    })
            ai_recs = ai_recs[:5]
        except Exception as e:
            logger.error(f"Error finding semantic similarities: {e}")
            ai_recs = []

        comp_left, comp_right = st.columns(2)

        with comp_left:
            st.markdown(t["classic_header"])
            st.caption(t["classic_caption"])
            
            if not classic_recs:
                st.write("No matches found." if lang == "English" else "సరిపోలే మ్యాచ్‌లు లేవు.")
            else:
                for idx, rec in enumerate(classic_recs):
                    st.markdown(f"""
                    <div class="movie-card" style="height: 180px !important;">
                        <div>
                            <h3>{rec['title']}</h3>
                            <p style="-webkit-line-clamp: 3 !important;">{rec['overview']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with comp_right:
            st.markdown(t["semantic_header"])
            st.caption(t["semantic_caption"])
            
            if not ai_recs:
                st.write("No matches found." if lang == "English" else "సరిపోలే మ్యాచ్‌లు లేవు.")
            else:
                for idx, rec in enumerate(ai_recs):
                    sim_pct = int(rec['score'] * 100)
                    st.markdown(f"""
                    <div class="movie-card" style="height: 180px !important; margin-bottom: 2px !important;">
                        <div>
                            <h3>{rec['title']}</h3>
                            <p style="-webkit-line-clamp: 3 !important;">{rec['overview']}</p>
                        </div>
                        <div style="font-size: 12px; color: #a855f7; font-weight: bold; margin-top: 4px;">
                            {t['semantic_match_lbl']}: {sim_pct}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(rec['score']))
                    st.write("")