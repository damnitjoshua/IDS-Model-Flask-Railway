from flask import Flask, jsonify
import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.decomposition import TruncatedSVD
import warnings

app = Flask(__name__)

model = load("model/knn_bookRecom_model.sav")
# Rating count more than 50
# csv_url = "https://drive.google.com/file/d/1tIxt00bOAPEKRkc57uuBhPZGDLBwsvJc/view?usp=share_link"
# csv_url = 'https://drive.google.com/uc?id=' + csv_url.split('/')[-2]
RatingCountDF = pd.read_csv("data/RatingCountDF.csv")


# pivot_url = "https://drive.google.com/file/d/1fmmByHYX0xBDgZCZkMCMS-zWWnlK-Cij/view?usp=share_link"
# pivot_url = 'https://drive.google.com/uc?id=' + pivot_url.split('/')[-2]

# RatingCountDFPivot = pd.read_csv('data/RatingCountDFPivotDF1.csv')
RatingCountDFPivot = RatingCountDF.pivot(
    index='ISBN', columns='UserID', values='Rating').fillna(0)

SVDPivot = RatingCountDF.pivot(
    index='UserID', columns='Title', values='Rating').fillna(0)

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(SVDPivot.values.T)

warnings.filterwarnings("ignore", category=RuntimeWarning)
corr = np.corrcoef(matrix)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


@app.route('/random/<count>')
def random(count):
    books = []

    for i in range(int(count)):
        query_index = np.random.choice(RatingCountDF.shape[0])
        book = RatingCountDF.iloc[query_index]
        books.append({
            "title": book['Title'],
            "author": book['Author'],
            "isbn": book['ISBN'],
            "image": book['Image'],
            "category": book["Category"][2:-2],
            "rating": str(book['Rating']),
            "ratingCount": str(book["RatingCount"]),
            "yearOfPublication": str(book["YearOfPublication"])
        })

    return jsonify(books)


@app.route('/knn/<ISBN>')
def knn(ISBN):
    try:
        search = RatingCountDFPivot.loc[ISBN]

        distances, indices = model.kneighbors(
            search.values.reshape(1, -1), n_neighbors=6)

        books = []

        for i in range(0, len(distances.flatten())):
            if i != 0:
                book = RatingCountDF.iloc[indices.flatten()[i]]
                books.append({
                    "title": book['Title'],
                    "isbn": book['ISBN'],
                    "image": book['Image'],
                    "distance": distances.flatten()[i]
                })

        return jsonify(books[::-1])
    except:
        return jsonify({"err"})


@app.route('/svd/<title>')
def svd(title):
    book_titles = SVDPivot.columns

    book_list = list(book_titles)
    index = book_list.index(title.replace("%20", " "))

    test = list(book_titles[(corr[index] < 1.0)
                            & (corr[index] > 0.9)])[1:5]

    return jsonify(test)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
