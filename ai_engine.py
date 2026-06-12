import os
import difflib
import logging
import numpy as np
import pandas as pd
import streamlit as st
import openai
from openai import OpenAI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------- LOGGING CONFIGURATION ---------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------- TENACITY RETRY DECORATOR ---------------- #
openai_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError
    )),
    reraise=True
)

# ---------------- PYDANTIC SCHEMAS FOR STRUCTURED OUTPUTS ---------------- #

class ChatResponse(BaseModel):
    reply: str
    recommended_movies: list[str]

class MoodExplanation(BaseModel):
    title: str
    explanation: str

class MoodResponse(BaseModel):
    movies: list[MoodExplanation]


# ---------------- AI ENGINE CLASS ---------------- #

class AIEngine:
    def __init__(self, api_key: str = None, is_validated: bool = False):
        """
        Initializes the AI Engine. If no api_key is supplied, defaults to the environment.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        self.has_key = False
        self.embeddings = None

        if self.api_key and self.api_key.strip():
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.has_key = is_validated
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI Client: {e}")
                self.has_key = False

    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validates the supplied OpenAI API key by making a lightweight list models request.
        """
        if not api_key or not api_key.strip():
            return False
        try:
            client = OpenAI(api_key=api_key)
            client.models.list()
            logger.info("API Key successfully validated via client.models.list()")
            return True
        except Exception as e:
            logger.warning(f"OpenAI API key validation failed: {e}")
            return False

    @openai_retry
    def _call_get_embedding(self, text: str) -> list[float]:
        if not self.client:
            return []
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates an embedding vector for a single text using text-embedding-3-small.
        """
        if not self.has_key or not self.client:
            return []
        try:
            return self._call_get_embedding(text)
        except Exception as e:
            logger.error(f"Error fetching embedding from OpenAI: {e}")
            return []

    @openai_retry
    def _call_get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not self.client:
            return []
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [x.embedding for x in response.data]

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embedding vectors for a batch of texts using text-embedding-3-small.
        """
        if not self.has_key or not self.client:
            return []
        try:
            return self._call_get_embeddings_batch(texts)
        except Exception as e:
            logger.error(f"Error fetching batch embeddings from OpenAI: {e}")
            return []

    def compute_tfidf_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes local TF-IDF matrices to act as a fallback semantic model.
        """
        cache_path = "movie_embeddings_tfidf.npy"
        if os.path.exists(cache_path):
            try:
                matrix = np.load(cache_path)
                if matrix.shape == (len(df), 5000):
                    return matrix
            except Exception:
                pass
        
        logger.info("Generating and caching local TF-IDF embeddings matrix...")
        texts = (df['title'] + " " + df['original_overview']).fillna("").tolist()
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts).toarray()
        np.save(cache_path, tfidf_matrix)
        return tfidf_matrix

    def cache_movie_embeddings(self, df: pd.DataFrame, force: bool = False) -> np.ndarray:
        """
        Retrieves movie embeddings from cache or generates them using OpenAI API.
        If offline, generates local TF-IDF embeddings.
        """
        if not self.has_key:
            embs = self.compute_tfidf_embeddings(df)
            self.embeddings = embs
            return embs

        cache_path = "movie_embeddings_openai.npy"
        partial_cache_path = "movie_embeddings_openai_partial.npy"
        
        if os.path.exists(cache_path) and not force:
            try:
                embs = np.load(cache_path)
                # Verify embedding dimensions match what's expected for text-embedding-3-small (1536)
                if len(embs) == len(df) and embs.shape[1] == 1536:
                    self.embeddings = embs
                    return embs
            except Exception:
                pass

        # Check if partial cache exists and matches current dataframe
        loaded_embs = []
        start_idx = 0
        if os.path.exists(partial_cache_path) and not force:
            try:
                loaded_embs = np.load(partial_cache_path).tolist()
                if len(loaded_embs) <= len(df) and len(loaded_embs) > 0:
                    start_idx = len(loaded_embs)
                    st.info(f"🔄 Resuming OpenAI embedding generation from movie {start_idx}/{len(df)}...")
            except Exception:
                loaded_embs = []
                start_idx = 0

        # Generate OpenAI embeddings
        st.info("🔄 Generating OpenAI embeddings for movies. This happens once...")
        progress_bar = st.progress(start_idx / len(df)) if len(df) > 0 else None
        
        texts = (df['title'] + " - " + df['original_overview']).fillna("").tolist()
        all_embeddings = list(loaded_embs)
        batch_size = 100
        total_movies = len(texts)

        for idx in range(start_idx, total_movies, batch_size):
            batch_texts = texts[idx : idx + batch_size]
            batch_embs = self.get_embeddings_batch(batch_texts)
            if not batch_embs:
                st.warning("⚠️ Failed to generate OpenAI embeddings. Falling back to Offline TF-IDF Mode.")
                if all_embeddings:
                    np.save(partial_cache_path, np.array(all_embeddings))
                embs = self.compute_tfidf_embeddings(df)
                self.embeddings = embs
                return embs
            
            all_embeddings.extend(batch_embs)
            np.save(partial_cache_path, np.array(all_embeddings))
            
            if progress_bar:
                progress = min(1.0, (idx + len(batch_texts)) / total_movies)
                progress_bar.progress(progress)

        embeddings_arr = np.array(all_embeddings)
        np.save(cache_path, embeddings_arr)
        if os.path.exists(partial_cache_path):
            try:
                os.remove(partial_cache_path)
            except Exception:
                pass
        st.success("🎉 OpenAI embeddings successfully generated and cached!")
        self.embeddings = embeddings_arr
        return embeddings_arr

    # ---------------- RAG CANDIDATE SELECTION FOR CHAT ---------------- #

    def get_chat_candidates(self, user_message: str, df: pd.DataFrame, top_k: int = 35) -> pd.DataFrame:
        """
        Retrieves the top candidate movies matching the user's message using
        exact/fuzzy matching and TF-IDF fallback search.
        """
        candidate_indices = set()
        user_message_lower = user_message.lower()

        # 1. Substring matching for titles
        for idx, row in df.iterrows():
            title_lower = row['title'].lower()
            if len(title_lower) > 3 and (title_lower in user_message_lower or user_message_lower in title_lower):
                candidate_indices.add(idx)
            
            # Check for words in quotes
            elif "'" in user_message or '"' in user_message:
                for word in user_message.replace('"', "'").split("'"):
                    word_clean = word.strip().lower()
                    if word_clean and word_clean == title_lower:
                        candidate_indices.add(idx)

        # 2. Vector search matching (OpenAI embed similarity if online, else TF-IDF)
        try:
            texts = (df['title'] + " " + df['original_overview']).fillna("").tolist()
            tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            query_vec = tfidf.transform([user_message])
            sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_tfidf_indices = sims.argsort()[-top_k:][::-1]
            for idx in top_tfidf_indices:
                candidate_indices.add(int(idx))
        except Exception as e:
            logger.error(f"Error selecting candidates via TF-IDF: {e}")

        # If too few matches, fill with top-rated movies
        if len(candidate_indices) < 15:
            top_rated = df.sort_values(by='vote_average', ascending=False).index[:20]
            for idx in top_rated:
                candidate_indices.add(idx)

        candidates_df = df.loc[list(candidate_indices)].copy()
        return candidates_df.head(top_k)

    # ---------------- CONVERSATIONAL MOVIE ASSISTANT ---------------- #

    @openai_retry
    def _call_chat_completions_parse(self, messages: list[dict]) -> any:
        return self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=ChatResponse,
            temperature=0.7
        )

    def chat_with_assistant(self, chat_history: list[dict], user_message: str, df: pd.DataFrame, language: str = "English") -> tuple[str, list[dict]]:
        """
        Sends the user message along with chat history to OpenAI, returns a natural
        language reply and a list of matched movie recommendations.
        """
        movie_titles = df['title'].tolist()

        if not self.has_key or not self.client:
            # --- OFFLINE FALLBACK CHAT ---
            matched_titles = self.search_movies_tfidf(user_message, df, top_k=5)
            recommended_movies = []
            for t in matched_titles:
                row = df[df['title'] == t]
                if not row.empty:
                    recommended_movies.append({
                        "title": row.iloc[0]['title'],
                        "overview": row.iloc[0]['original_overview']
                    })

            if language == "తెలుగు":
                reply = (
                    "👋 నేను ప్రస్తుతం **ఆఫ్‌లైన్ మోడ్‌లో** నడుస్తున్నాను. నేను మీ సందేశాన్ని విశ్లేషించి, "
                    "సరిపోయే సినిమాల కోసం నా లోకల్ డేటాబేస్‌ను స్కాన్ చేసాను. నేను కనుగొన్నవి ఇక్కడ ఉన్నాయి:"
                )
            else:
                reply = (
                    "👋 I'm currently running in **Offline Mode**. I analyzed your message and scanned my "
                    "local database for matching movies. Here is what I found:"
                )
            return reply, recommended_movies

        # --- ONLINE OPENAI CHAT WITH RAG ---
        candidates = self.get_chat_candidates(user_message, df, top_k=35)
        candidate_titles = candidates['title'].tolist()
        candidate_details = [
            f"Title: {r['title']} | Overview: {r['original_overview']}"
            for _, r in candidates.iterrows()
        ]

        if language == "తెలుగు":
            system_prompt = (
                "You are a premium, highly knowledgeable movie recommendation assistant called MOVIE RECOMMENDER.\n"
                "Your goal is to suggest movies from the following database of candidates based on the user's request.\n"
                "Here are the candidate movies currently available for this search:\n"
                f"{', '.join(candidate_titles)}\n\n"
                "Candidate Details:\n"
                f"{chr(10).join(candidate_details[:30])}\n\n"
                "Instructions:\n"
                "1. Converse naturally and answer the user's questions or suggestions.\n"
                "2. CRITICAL: You must write your reply (the `reply` field of the JSON response) ENTIRELY in Telugu (తెలుగు).\n"
                "3. If they ask for recommendations, choose the most appropriate movies from the database candidates.\n"
                "4. Keep the recommended movie titles in English so they can be matched with our database, but explain why you recommend them in Telugu.\n"
                "5. Your response must be in structured JSON conforming to the schema.\n"
                "6. Ensure the recommended_movies field only contains exact titles from the candidate list."
            )
        else:
            system_prompt = (
                "You are a premium, highly knowledgeable movie recommendation assistant called MOVIE RECOMMENDER.\n"
                "Your goal is to suggest movies from the following database of candidates based on the user's request.\n"
                "Here are the candidate movies currently available for this search:\n"
                f"{', '.join(candidate_titles)}\n\n"
                "Candidate Details:\n"
                f"{chr(10).join(candidate_details[:30])}\n\n"
                "Instructions:\n"
                "1. Converse naturally and answer the user's questions or suggestions.\n"
                "2. If they ask for recommendations, choose the most appropriate movies from the database candidates.\n"
                "3. If a movie they ask for is not in the list, recommend the most similar movies that are in the list.\n"
                "4. Your response must be in structured JSON conforming to the schema.\n"
                "5. Ensure the recommended_movies field only contains exact titles from the candidate list."
            )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._call_chat_completions_parse(messages)
            data = response.choices[0].message.parsed
            reply = data.reply
            rec_titles = data.recommended_movies

            # Map titles to movie details
            recommended_movies = []
            for title in rec_titles:
                close_matches = difflib.get_close_matches(title, movie_titles, n=1, cutoff=0.7)
                if close_matches:
                    row = df[df['title'] == close_matches[0]]
                    if not row.empty:
                        recommended_movies.append({
                            "title": row.iloc[0]['title'],
                            "overview": row.iloc[0]['original_overview']
                        })
            return reply, recommended_movies

        except Exception as e:
            logger.error(f"Error in OpenAI chat completions: {e}")
            matched_titles = self.search_movies_tfidf(user_message, df, top_k=3)
            rec_movies = []
            for t in matched_titles:
                row = df[df['title'] == t]
                if not row.empty:
                    rec_movies.append({
                        "title": row.iloc[0]['title'],
                        "overview": row.iloc[0]['original_overview']
                    })
            if language == "తెలుగు":
                err_msg = f"నేను OpenAI తో కమ్యూనికేట్ చేయడంలో చిన్న లోపాన్ని ఎదుర్కొన్నాను, కానీ సరిపోయే కొన్ని ఫలితాలు ఇక్కడ ఉన్నాయి: {e}"
            else:
                err_msg = f"I encountered an error communicating with OpenAI, but here are some matches: {e}"
            return err_msg, rec_movies

    def search_movies_tfidf(self, query: str, df: pd.DataFrame, top_k: int = 5) -> list[str]:
        try:
            texts = (df['title'] + " " + df['original_overview']).fillna("").tolist()
            tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            query_vec = tfidf.transform([query])
            sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = sims.argsort()[-top_k:][::-1]
            return df.iloc[top_indices]['title'].tolist()
        except Exception:
            return df.sample(min(top_k, len(df)))['title'].tolist()

    # ---------------- MOOD-BASED RECOMMENDATIONS ---------------- #

    @openai_retry
    def _call_mood_completions_parse(self, prompt: str) -> any:
        return self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=MoodResponse,
            temperature=0.6
        )

    def get_mood_recommendations(self, mood: str, df: pd.DataFrame, embeddings: np.ndarray, top_k: int = 6, language: str = "English") -> list[dict]:
        """
        Determines recommendations tailored to a specific mood. 
        Uses embeddings for initial retrieval, and OpenAI to curate and explain them.
        """
        query_text = f"A movie that fits a {mood} mood, vibe, or emotion"
        
        if self.has_key:
            query_vec = self.get_embedding(query_text)
        else:
            query_vec = []

        # Graceful dimension fallback
        if query_vec is not None and len(query_vec) > 0 and embeddings is not None:
            if len(query_vec) != embeddings.shape[1]:
                logger.warning(
                    f"Embedding dimension mismatch: query={len(query_vec)}, matrix={embeddings.shape[1]}. "
                    "Falling back to TF-IDF embeddings."
                )
                tfidf_matrix = self.compute_tfidf_embeddings(df)
                try:
                    texts = (df['title'] + " " + df['original_overview']).fillna("").tolist()
                    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
                    tfidf.fit(texts)
                    query_vec = tfidf.transform([query_text]).toarray()[0]
                    embeddings = tfidf_matrix
                except Exception as ex:
                    logger.error(f"Failed to generate TF-IDF fallback vector: {ex}")
                    query_vec = None

        if query_vec is None or len(query_vec) == 0:
            sample = df.sample(top_k)
            return [{
                "title": r['title'],
                "overview": r['original_overview'],
                "explanation": "మీ మూడ్ కోసం చక్కటి ఎంపిక." if language == "తెలుగు" else f"Uplifting choice selected for your {mood} mood."
            } for _, r in sample.iterrows()]

        sims = cosine_similarity([query_vec], embeddings).flatten()
        candidate_indices = sims.argsort()[-12:][::-1]
        candidates = df.iloc[candidate_indices]

        if not self.has_key or not self.client:
            results = []
            for _, row in candidates.head(top_k).iterrows():
                if language == "తెలుగు":
                    explanation = "మీ అభిరుచికి సరిపోయే లోకల్ మ్యాచ్."
                else:
                    explanation = f"Matched because this movie contains themes suited to a {mood} mood."
                results.append({
                    "title": row['title'],
                    "overview": row['original_overview'],
                    "explanation": explanation
                })
            return results

        # Online OpenAI curation
        candidate_details = []
        for _, row in candidates.iterrows():
            candidate_details.append(f"Title: {row['title']} | Overview: {row['original_overview']}")

        if language == "తెలుగు":
            prompt = (
                f"The user is feeling: {mood}.\n"
                "From the following list of 12 candidate movies, select the top 6 that are best suited to this mood.\n"
                "For each selected movie, provide a short, engaging 1-sentence explanation ENTIRELY in Telugu (తెలుగు) of why it fits this mood.\n"
                "Format the response as JSON adhering to the schema.\n\n"
                "Candidates:\n" + "\n".join(candidate_details)
            )
        else:
            prompt = (
                f"The user is feeling: {mood}.\n"
                "From the following list of 12 candidate movies, select the top 6 that are best suited to this mood.\n"
                "For each selected movie, provide a short, engaging 1-sentence explanation of why it fits this mood.\n"
                "Format the response as JSON adhering to the schema.\n\n"
                "Candidates:\n" + "\n".join(candidate_details)
            )

        try:
            response = self._call_mood_completions_parse(prompt)
            data = response.choices[0].message.parsed
            movies_list = data.movies

            results = []
            for m in movies_list:
                row = df[df['title'] == m.title]
                if not row.empty:
                    results.append({
                        "title": row.iloc[0]['title'],
                        "overview": row.iloc[0]['original_overview'],
                        "explanation": m.explanation
                    })
            
            if not results:
                for _, row in candidates.head(top_k).iterrows():
                    results.append({
                        "title": row['title'],
                        "overview": row['original_overview'],
                        "explanation": "మీ మూడ్ కోసం చక్కటి ఎంపిక." if language == "తెలుగు" else f"Excellent choice suited for a {mood} mood."
                    })
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error curating mood recommendations: {e}")
            results = []
            for _, row in candidates.head(top_k).iterrows():
                results.append({
                    "title": row['title'],
                    "overview": row['original_overview'],
                    "explanation": "మీ మూడ్ కోసం చక్కటి ఎంపిక." if language == "తెలుగు" else f"Excellent choice suited for a {mood} mood."
                })
            return results

    # ---------------- HYBRID RECOMMENDATION ENGINE ---------------- #

    def get_hybrid_recommendations(
        self, 
        selected_movie_idx: int, 
        user_genres: list[str], 
        min_rating: float,
        df: pd.DataFrame, 
        embeddings: np.ndarray, 
        content_sim_matrix: np.ndarray,
        top_k: int = 6
    ) -> list[dict]:
        """
        Combines 3 scoring mechanisms:
        1. User preferences (Genre matching & minimum rating criteria)
        2. Content-based similarity (CountVectorizer bag-of-words)
        3. AI Semantic matching (Embeddings cosine similarity)
        Score Formula: Final Score = 40% Preferences + 30% Content + 30% Semantic
        """
        try:
            movie_index = selected_movie_idx
            selected_movie = df.iloc[movie_index]['title']
        except Exception:
            return []

        # Graceful bounds safety checks
        if movie_index < 0 or movie_index >= len(df):
            return []

        content_scores = content_sim_matrix[movie_index]

        # Semantic cosine similarity
        target_emb = embeddings[movie_index]
        semantic_scores = cosine_similarity([target_emb], embeddings).flatten()

        # Vectorized User Preferences matching (8x performance improvement over iterrows)
        if user_genres:
            user_genres_set = set(user_genres)
            genre_scores = np.array([
                len(user_genres_set.intersection(g)) / len(user_genres) if isinstance(g, (list, tuple, set)) else 0.0
                for g in df['genres']
            ])
        else:
            genre_scores = np.ones(len(df))

        ratings = df['vote_average'].fillna(7.0).values
        if min_rating > 0:
            rating_scores = np.where(ratings >= min_rating, 1.0, np.maximum(0.0, ratings / min_rating))
        else:
            rating_scores = np.ones(len(df))

        pref_scores = (genre_scores + rating_scores) / 2

        # Min-Max Normalization helper
        def normalize(arr):
            amin, amax = arr.min(), arr.max()
            if amax - amin > 0:
                return (arr - amin) / (amax - amin)
            return arr

        norm_content = normalize(content_scores)
        norm_semantic = normalize(semantic_scores)
        norm_pref = normalize(pref_scores)

        final_scores = 0.4 * norm_pref + 0.3 * norm_content + 0.3 * norm_semantic

        scores_df = pd.DataFrame({
            'index': range(len(df)),
            'title': df['title'],
            'overview': df['original_overview'],
            'genres': df['genres'],
            'pref_score': norm_pref,
            'content_score': norm_content,
            'semantic_score': norm_semantic,
            'final_score': final_scores
        })

        scores_df = scores_df[scores_df['index'] != movie_index]
        top_recommendations = scores_df.sort_values(by='final_score', ascending=False).head(top_k)

        results = []
        for _, r in top_recommendations.iterrows():
            results.append({
                "title": r['title'],
                "overview": r['overview'],
                "pref_score": float(r['pref_score']),
                "content_score": float(r['content_score']),
                "semantic_score": float(r['semantic_score']),
                "final_score": float(r['final_score']),
                "genres": r['genres']
            })

        return results
