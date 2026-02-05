# Hybrid Movie Recommendation System using MovieLens 100K

This project implements a **hybrid movie recommendation system** using the MovieLens 100K dataset. It combines **content-based filtering** and **collaborative filtering** to recommend movies based on user preferences and similarity scores. The system is deployed with a Streamlit interface for interactive recommendations.

---

## 🔧 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (cosine similarity)
- Streamlit

---

## 📂 Dataset Used

- **Dataset:** MovieLens 100K
- Contains 100,000 ratings from 943 users on 1682 movies
- Includes user IDs, movie IDs, ratings, genres, and titles

---

## 🧠 Recommendation Approach

This system uses a **hybrid strategy**:

### 1️⃣ Content-Based Filtering
- Movies are recommended based on similarity of genres and metadata
- Cosine similarity is computed between movie feature vectors

### 2️⃣ Collaborative Filtering
- Recommendations based on user-item rating patterns
- Identifies similar users and suggests movies they liked

### 🔀 Hybrid Logic
- Combines results from both approaches to improve recommendation quality

---

## ⚙️ Features

- Select a movie and get similar movie recommendations
- Combines content similarity and user rating behavior
- Interactive Streamlit interface
- Structured Python modules for similarity computation and recommendation logic

---

## 🗂️ Project Structure

AI-MOVIE-RECOMMENDATION-SYSTEM/ ├── app.py                  # Streamlit UI ├── recommender.py          # Recommendation logic ├── dataset/                # MovieLens dataset files ├── requirements.txt └── README.md

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/ATUL-SHARMA1215/AI-MOVIE-RECOMMENDATION-SYSTEM
cd AI-MOVIE-RECOMMENDATION-SYSTEM
pip install -r requirements.txt
streamlit run app.py

---

🔍 How It Works

1. Dataset is loaded and preprocessed using Pandas
2. Movie features are extracted from genres and metadata
3. Cosine similarity matrix is computed for content-based filtering
4. User rating matrix is used for collaborative filtering
5. Results from both methods are combined to produce final recommendations

---

🧪 Testing & Debugging Performed

1. Verified recommendation quality using known similar movies
2. Debugged indexing issues between movie IDs and titles
3. Handled missing values and cleaned dataset before similarity computation
4. Structured code to separate recommendation logic from UI

---

📌 Example

Input Movie: Toy Story (1995)
Recommended Movies: Animated and family genre movies with high similarity scores

---

🎯 Learning Outcomes

1. Understanding recommendation system architectures
2. Working with real-world datasets (MovieLens)
3. Implementing cosine similarity for content-based filtering
4. Applying collaborative filtering using user-item interactions
5. Structuring Python code for modular recommendation logic