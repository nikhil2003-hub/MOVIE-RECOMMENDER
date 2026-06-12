import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add root folder to sys.path so we can import from project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import load_and_preprocess_data
from ai_engine import AIEngine

def test_data_loading_and_preprocessing():
    """
    Verifies that the dataset merges correctly (ID-based, length 4800) and contains
    all necessary processed columns, including release_year.
    """
    df = load_and_preprocess_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4800
    
    expected_cols = {
        'movie_id', 'title', 'tags', 'original_overview', 
        'genres', 'vote_average', 'release_year'
    }
    assert expected_cols.issubset(df.columns)
    
    # Verify duplicates were cleaned (we should only have unique titles for non-remakes)
    # Batman and The Host have duplicate entries, let's verify release years distinguish them
    batmans = df[df['title'] == 'Batman']
    assert len(batmans) == 2
    assert set(batmans['release_year']) == {'1966', '1989'}

def test_vectorized_preference_scores():
    """
    Verifies that the vectorized user preferences logic works correctly
    and handles custom ratings and genres.
    """
    # Create mock dataframe
    df_mock = pd.DataFrame({
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'genres': [['Action', 'Adventure'], ['Comedy'], ['Drama', 'Action']],
        'vote_average': [8.5, 4.2, 6.0],
        'original_overview': ['Overview A', 'Overview B', 'Overview C']
    })
    
    # Initialize engine in offline mode
    engine = AIEngine(api_key=None, is_validated=False)
    
    # Selected movie: 'Movie A' (index 0)
    # Target genres: ['Action']
    # Min rating: 6.0
    embeddings = np.random.rand(3, 1536)
    content_sim = np.eye(3)
    
    recs = engine.get_hybrid_recommendations(
        selected_movie_idx=0,
        user_genres=['Action'],
        min_rating=6.0,
        df=df_mock,
        embeddings=embeddings,
        content_sim_matrix=content_sim,
        top_k=2
    )
    
    assert len(recs) <= 2
    # Verify recommendations do not include selected movie (Movie A)
    rec_titles = [r['title'] for r in recs]
    assert 'Movie A' not in rec_titles

def test_api_key_validation_failure():
    """
    Checks that the startup validation helper correctly identifies invalid keys.
    """
    # Verify invalid key fails validation
    assert not AIEngine.validate_api_key("sk-invalidkey")
    # Verify empty key fails validation
    assert not AIEngine.validate_api_key("")

def test_dimension_mismatch_fallback():
    """
    Verifies that get_mood_recommendations catches and handles query/embeddings dimension mismatches
    (e.g., TF-IDF fallback queries compared against OpenAI embeddings matrix) without crashing.
    """
    df = load_and_preprocess_data()
    engine = AIEngine(api_key=None, is_validated=False) # offline engine (will produce TF-IDF query)
    
    # Mocking OpenAI embeddings matrix (dimension 1536)
    openai_mock_embeddings = np.random.rand(len(df), 1536)
    
    # This should run and complete successfully by falling back to TF-IDF matrix internally
    recs = engine.get_mood_recommendations(
        mood="excited",
        df=df,
        embeddings=openai_mock_embeddings,
        top_k=3,
        language="English"
    )
    
    assert len(recs) == 3
    assert 'title' in recs[0]
    assert 'overview' in recs[0]
    assert 'explanation' in recs[0]

def test_chat_candidates_retrieval():
    """
    Verifies that get_chat_candidates successfully extracts relevant titles.
    """
    df = load_and_preprocess_data()
    engine = AIEngine(api_key=None, is_validated=False)
    
    # Query mentions Avatar
    candidates = engine.get_chat_candidates("I really loved Avatar and Sci-Fi movies", df, top_k=5)
    assert len(candidates) >= 1
    titles = [t.lower() for t in candidates['title']]
    assert any("avatar" in t for t in titles)
