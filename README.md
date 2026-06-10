

# 🎬 Movie Recommendation System

A **Machine Learning based Movie Recommendation System** that suggests movies similar to a user-entered movie using **content-based filtering**.
This project analyzes movie genres and uses **TF-IDF vectorization and cosine similarity** to recommend similar movies.

The system also includes a **Gradio web interface** that allows users to type a movie name and receive recommended movies instantly.

---

# 📌 Project Overview

This project was developed as part of the **Machine Learning Internship Task at Algonive**.

The system works by:

1. Loading a movie dataset
2. Processing movie genres
3. Converting movie data into numerical vectors using **TF-IDF**
4. Calculating similarity between movies using **cosine similarity**
5. Recommending the most similar movies to the user input

---

# 🚀 Features

✔ Movie recommendations based on similarity
✔ Content-based filtering algorithm
✔ Fast search using cosine similarity
✔ User-friendly web interface using **Gradio**
✔ Handles spelling errors using **difflib closest match**
✔ Displays top recommended movies

---

# 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity
* Gradio (for UI)

---

# 📂 Dataset

This project uses the **MovieLens Dataset**.

Dataset includes:

```
movieId
title
genres
```

Source:

[https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

The dataset contains thousands of movies with genre information which is used for similarity calculation.

---

# ⚙️ How the System Works

### Step 1 – Data Loading

The movie dataset is loaded using **Pandas**.

### Step 2 – Data Preprocessing

Movie genres are cleaned and missing values are handled.

### Step 3 – Feature Extraction

Movie genres are converted into numerical vectors using **TF-IDF Vectorization**.

### Step 4 – Similarity Calculation

Cosine similarity is used to measure how similar two movies are.

### Step 5 – Recommendation

When a user enters a movie name:

1. The closest matching movie is found
2. Similarity scores are calculated
3. Top recommended movies are displayed

---

# 🖥️ User Interface

The system uses **Gradio** to provide an interactive interface.

Users can:

1. Enter a movie name
2. Click submit
3. View recommended movies instantly

---

# 📷 Example Output

Input:

```
Toy Story (1995)
```

Output:

```
Movies Suggested for You:

1. Toy Story 2 (1999)
2. Monsters, Inc. (2001)
3. Antz (1998)
4. Emperor's New Groove (2000)
5. Adventures of Rocky and Bullwinkle (2000)
```

---

# 📁 Project Structure

```
Movie-Recommendation-System
│
├── movie_recommendation.ipynb
├── movie.csv
├── rating.csv
├── README.md
```

---

# ▶️ How to Run the Project

### Step 1

Clone the repository

```
git clone https://github.com/yourusername/movie-recommendation-system.git
```

### Step 2

Install dependencies

```
pip install pandas numpy scikit-learn gradio
```

### Step 3

Run the notebook or Python script

```
python movie_recommendation.py
```

or open in **Google Colab / Jupyter Notebook**.

---

# 📊 Recommendation Algorithm

The project uses **Content-Based Filtering**.

Similarity between movies is calculated using:

```
Cosine Similarity
```

This compares movie genre vectors and finds the most similar movies.

---

# 🎓 Internship Details

This project was completed as part of the **Machine Learning Internship Program at Algonive**.

---

# 👨‍💻 Author

**Shiva Sai**

Machine Learning Intern
Algonive

---

# ⭐ Future Improvements

* Add movie posters using **TMDB API**
* Add rating-based recommendations
* Improve UI with better search suggestions
* Deploy the system online

---


