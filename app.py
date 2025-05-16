import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import json
import datetime
import os
import streamlit as st
import os
import os

st.write("Files in current directory:", os.listdir())
st.write("Files in data folder:", os.listdir("data"))

BASE_DIR = os.getcwd()  # Or hardcode the path to your project root if needed

ratings_path = os.path.join(BASE_DIR, "data", "u_data.csv")
movies_path = os.path.join(BASE_DIR, "data", "movies.csv")

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

st.write("Current Working Directory:", os.getcwd())

# Initialize session state variables
if "usage_log" not in st.session_state:
    st.session_state["usage_log"] = []
if "favorites" not in st.session_state:
    st.session_state["favorites"] = set()

st.set_page_config(page_title="AI Movie Recommendation System", layout="wide")

st.title("🎬 AI Movie Recommendation System")

st.markdown("""
    <style>
    body, .css-1d391kg { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .movie-card {
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s ease-in-out;
        background-color: var(--background-color);
        margin-bottom: 15px;
    }
    .movie-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }
    .genre-badge {
        display: inline-block;
        background-color: #007bff;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        margin-right: 5px;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    ratings = pd.read_csv("data/u_data.csv")
    movies = pd.read_csv("data/movies.csv")
    return ratings, movies


ratings, movies = load_data()

@st.cache_data(show_spinner=False)
def prepare_matrices(ratings, movies):
    user_movie_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
    genre_features = movies.drop(columns=['movie_id', 'title', 'release_date', 'video_release_date', 'year'])
    genre_sim = cosine_similarity(genre_features)
    genre_sim_df = pd.DataFrame(genre_sim, index=movies['title'], columns=movies['title'])
    user_sim = cosine_similarity(user_movie_matrix)
    user_sim_df = pd.DataFrame(user_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    return user_movie_matrix, genre_sim_df, user_sim_df

user_movie_matrix, genre_sim_df, user_sim_df = prepare_matrices(ratings, movies)

def recommend_similar_movies(title, top_n=10):
    if title not in genre_sim_df:
        return []
    sim_scores = genre_sim_df[title].sort_values(ascending=False)[1:top_n+1]
    return list(sim_scores.index)

def recommend_movies_user_based(user_id, top_n=10):
    if user_id not in user_movie_matrix.index:
        return []
    similar_users = user_sim_df[user_id].sort_values(ascending=False).drop(user_id)
    weighted_ratings = pd.Series(0, index=user_movie_matrix.columns, dtype=float)
    for sim_user, sim_score in similar_users.items():
        other_ratings = user_movie_matrix.loc[sim_user]
        weighted_ratings = weighted_ratings.add(other_ratings * sim_score, fill_value=0)
    already_rated = user_movie_matrix.loc[user_id]
    weighted_ratings = weighted_ratings[already_rated == 0]
    top_movie_ids = weighted_ratings.sort_values(ascending=False).head(top_n).index
    return movies[movies['movie_id'].isin(top_movie_ids)]['title'].tolist()

def hybrid_recommendation(user_id, liked_movies, top_n=10, weight_content=0.5, weight_collab=0.5):
    if user_id not in user_movie_matrix.index or not liked_movies:
        return []
    # Content-based score aggregation
    content_scores = pd.Series(0, index=genre_sim_df.columns)
    for movie in liked_movies:
        if movie in genre_sim_df:
            content_scores += genre_sim_df[movie]
    content_scores = content_scores.drop(labels=liked_movies, errors='ignore')

    # Collaborative filtering score aggregation
    similar_users = user_sim_df[user_id].sort_values(ascending=False).drop(user_id)
    weighted_ratings = pd.Series(0, index=user_movie_matrix.columns, dtype=float)
    for sim_user, sim_score in similar_users.items():
        other_ratings = user_movie_matrix.loc[sim_user]
        weighted_ratings = weighted_ratings.add(other_ratings * sim_score, fill_value=0)
    already_rated = user_movie_matrix.loc[user_id]
    liked_movie_ids = [movies[movies['title'] == m]['movie_id'].values[0] for m in liked_movies if not movies[movies['title'] == m].empty]
    weighted_ratings = weighted_ratings[(already_rated == 0) & (~weighted_ratings.index.isin(liked_movie_ids))]

    collab_scores = pd.Series(0, index=genre_sim_df.columns)
    for movie_id, score in weighted_ratings.items():
        title = movies.loc[movies['movie_id'] == movie_id, 'title'].values
        if len(title) > 0:
            collab_scores[title[0]] = score

    # Normalize scores
    if content_scores.max() > 0:
        content_scores /= content_scores.max()
    if collab_scores.max() > 0:
        collab_scores /= collab_scores.max()

    # Combine weighted scores
    final_scores = content_scores * weight_content + collab_scores * weight_collab
    final_scores = final_scores.sort_values(ascending=False)
    return final_scores.head(top_n).index.tolist()

def show_movie_card_simple(title, key_prefix=""):
    movie_row = movies[movies['title'] == title].iloc[0]
    genres = movie_row[['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
    genre_tags = [genre for genre, val in genres.items() if val == 1]
    year = movie_row['year']
    movie_id = movie_row['movie_id']

    is_fav = title in st.session_state["favorites"]
    fav_button_label = "💖 Remove from Favorites" if is_fav else "🤍 Add to Favorites"

    st.markdown(
        f"""
        <div class="movie-card">
            <h4 style="margin-bottom:4px;">{title} ({year})</h4>
            <div>{' '.join([f'<span class="genre-badge">{g}</span>' for g in genre_tags])}</div>
        </div>
        """, unsafe_allow_html=True
    )

    # Unique key includes prefix + movie_id
    if st.button(fav_button_label, key=f"{key_prefix}_fav_{movie_id}"):
        if is_fav:
            st.session_state["favorites"].remove(title)
            st.success(f"Removed '{title}' from favorites")
        else:
            st.session_state["favorites"].add(title)
            st.success(f"Added '{title}' to favorites")
        st.rerun()

def to_csv(data_list):
    buffer = io.StringIO()
    for item in data_list:
        buffer.write(item + "\n")
    return buffer.getvalue()

def to_json(data_list):
    return json.dumps(data_list, indent=2)

def to_excel(data_list):
    df = pd.DataFrame(data_list, columns=['Title'])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer

def log_usage(method, user_id=None, liked_movies=None, recommendations=None):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "method": method,
        "user_id": user_id,
        "liked_movies": liked_movies if liked_movies else [],
        "recommendations": recommendations if recommendations else []
    }
    st.session_state["usage_log"].append(log_entry)

# Sidebar User Profile & Filters
with st.sidebar.expander("👤 User Profile & Filters", expanded=True):
    if "username" not in st.session_state:
        username = st.text_input("Enter your username:", placeholder="Your name or nickname")
        if username:
            st.session_state["username"] = username
            st.rerun()
    else:
        st.success(f"Welcome back, {st.session_state['username']}! 👋")

    st.markdown("### 🎭 Genre Preferences")
    genres = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    genre_weights = {}
    for genre in genres:
        weight = st.slider(f"{genre}", 0.0, 1.0, 0.0, step=0.05, key=f"weight_{genre}")
        genre_weights[genre] = weight

    filtered_movies = movies.copy()
    genre_weight_series = pd.Series(genre_weights)
    if genre_weight_series.sum() > 0:
        genre_score = filtered_movies[genres].mul(genre_weight_series).sum(axis=1)
        filtered_movies = filtered_movies[genre_score > 0].copy()
        filtered_movies['genre_score'] = genre_score
        filtered_movies = filtered_movies.sort_values(by='genre_score', ascending=False)

# Main Interface
st.markdown("## 🔍 Choose Recommendation Mode")

recommendations = []  # Initialize recommendations variable

mode = st.selectbox("Select Recommendation Type", ["Content-Based", "Collaborative-Based", "Hybrid"], index=0)

if mode == "Content-Based":
    selected_title = st.selectbox("Select a movie you liked", sorted(filtered_movies['title'].unique()))
    if selected_title:
        recommendations = recommend_similar_movies(selected_title, top_n=10)
        log_usage("Content-Based", liked_movies=[selected_title], recommendations=recommendations)

elif mode == "Collaborative-Based":
    user_ids = sorted(user_movie_matrix.index)
    selected_user = st.selectbox("Select your User ID", user_ids)
    if selected_user:
        recommendations = recommend_movies_user_based(selected_user, top_n=10)
        log_usage("Collaborative-Based", user_id=selected_user, recommendations=recommendations)

elif mode == "Hybrid":
    user_ids = sorted(user_movie_matrix.index)
    selected_user = st.selectbox("Select your User ID", user_ids, key="hybrid_user")
    liked_movies = st.multiselect(
        "Select movies you liked (for content-based part)",
        options=sorted(filtered_movies['title'].unique())
    )
    weight_content = st.slider("Content-Based Weight", 0.0, 1.0, 0.5, step=0.05)
    weight_collab = 1.0 - weight_content
    if selected_user and liked_movies:
        recommendations = hybrid_recommendation(selected_user, liked_movies, top_n=10,
                                               weight_content=weight_content,
                                               weight_collab=weight_collab)
        log_usage("Hybrid", user_id=selected_user, liked_movies=liked_movies, recommendations=recommendations)

if recommendations:
    st.subheader("Recommended Movies:")
    for i, title in enumerate(recommendations, 1):
        show_movie_card_simple(title, key_prefix=f"rec_{i}")

    # Favorite movies download
    st.markdown("---")
    st.markdown("### 📥 Download Recommendations")
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = to_csv(recommendations)
        st.download_button("Download CSV", csv_data, file_name="recommendations.csv", mime="text/csv")

    with col2:
        json_data = to_json(recommendations)
        st.download_button("Download JSON", json_data, file_name="recommendations.json", mime="application/json")

    with col3:
        excel_data = to_excel(recommendations)
        st.download_button("Download Excel", data=excel_data, file_name="recommendations.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Show favorites if any
if st.session_state["favorites"]:
    st.markdown("---")
    st.subheader("💖 Your Favorites")
    for fav_title in sorted(st.session_state["favorites"]):
        show_movie_card_simple(fav_title, key_prefix="fav")

# Optional: Show usage logs for debugging or user info (can be hidden)
with st.expander("📝 Usage Logs"):
    st.json(st.session_state["usage_log"])
if st.session_state["favorites"]:
    st.markdown("---")
    st.subheader("💖 Your Favorites")
    for i, fav_title in enumerate(sorted(st.session_state["favorites"]), start=1):
        show_movie_card_simple(fav_title, key_prefix=f"fav_{i}")


    st.markdown("### 📥 Download Your Favorites")
    col1, col2, col3 = st.columns(3)

    with col1:
        fav_csv = to_csv(sorted(st.session_state["favorites"]))
        st.download_button(
            label="Download Favorites CSV",
            data=fav_csv,
            file_name="favorites.csv",
            mime="text/csv"
       )

    with col2:
        fav_json = to_json(sorted(st.session_state["favorites"]))
        st.download_button(
            label="Download Favorites JSON",
            data=fav_json,
            file_name="favorites.json",
            mime="application/json"
        )

    with col3:
        fav_excel = to_excel(sorted(st.session_state["favorites"]))
        st.download_button(
            label="Download Favorites Excel",
            data=fav_excel,
            file_name="favorites.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

