# 🎬 AI Movie Recommendation System

An interactive web-based movie recommender app built using **Streamlit**. It suggests movies based on your preferences using **Content-Based**, **Collaborative-Based**, and **Hybrid** filtering techniques.

---

## 🚀 Features

- 🔍 **Three Recommendation Modes**:
  - **Content-Based**: Suggests similar movies to the one you liked.
  - **Collaborative-Based**: Suggests movies based on other users' preferences.
  - **Hybrid**: Combines both content and collaborative filtering for smarter recommendations.

- 💡 **Genre Preference Filtering** with sliders
- 💖 Add movies to **Favorites**
- 📥 Download recommendations and favorites in **CSV**, **JSON**, and **Excel** formats
- 📊 Usage logging (timestamp, method used, recommendations)
- 🎨 Clean UI with responsive design and movie cards

---

## 🧠 Built With

- [Streamlit](https://streamlit.io)
- [Pandas](https://pandas.pydata.org)
- [NumPy](https://numpy.org)
- [Scikit-learn](https://scikit-learn.org)
- [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/100k/)

---

## 📁 Project Structure

├── app.py # Main Streamlit app
├── ml-100k/
│ ├── u.data # Ratings file
│ └── u.item # Movies metadata
├── requirements.txt # Python dependencies
└── README.md # You're here!


---

## 🧪 How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/ATUL-SHARMA1215/AI movie-recommender-app.git
   cd movie-recommender-app
pip install -r requirements.txt
streamlit run app.py OR python -m streamlit run app.py
