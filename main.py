from flask import Flask, jsonify
import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.decomposition import TruncatedSVD
import warnings

app = Flask(__name__)

model = load("model/knn_bookRecom_model.sav")

RatingCountDF = pd.read_csv("data/RatingCountDF.csv")

RatingCountDFPivot = RatingCountDF.pivot(
    index='ISBN', columns='UserID', values='Rating').fillna(0)

SVDPivot = RatingCountDF.pivot(
    index='UserID', columns='ISBN', values='Rating').fillna(0)

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(SVDPivot.values.T)

warnings.filterwarnings("ignore", category=RuntimeWarning)
corr = np.corrcoef(matrix)


@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


@app.route('/book/<ISBN>')
def book(ISBN):
    book = RatingCountDF.loc[RatingCountDF['ISBN'] == ISBN][:1].to_dict()

    book = {
        "title": list(book.get("Title").items())[0][1],
        "author": list(book.get("Author").items())[0][1],
        "publisher": list(book.get("Publisher").items())[0][1],
        "isbn": list(book.get("ISBN").items())[0][1],
        "image": list(book.get("Image").items())[0][1],
        "category": list(book.get("Category").items())[0][1][2:-2],
        "rating": list(book.get("Rating").items())[0][1],
        "ratingCount": list(book.get("RatingCount").items())[0][1],
        "yearOfPublication": list(book.get("YearOfPublication").items())[0][1]
    }

    return jsonify(book)


@app.route('/random/<count>')
def random(count):
    books = []

    for i in range(int(count)):
        query_index = np.random.choice(RatingCountDF.shape[0])
        book = RatingCountDF.iloc[query_index].to_dict()
        books.append({
            "title": book['Title'],
            "author": book['Author'],
            "publisher": book['Publisher'],
            "isbn": book['ISBN'],
            "image": book['Image'],
            "category": book["Category"][2:-2],
            "rating": book['Rating'],
            "ratingCount": book['RatingCount'],
            "yearOfPublication": book['YearOfPublication']
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
                book = RatingCountDF.iloc[indices.flatten()[i]].to_dict()

                books.append({
                    "title": book['Title'],
                    "author": book['Author'],
                    "publisher": book['Publisher'],
                    "isbn": book['ISBN'],
                    "image": book['Image'],
                    "category": book["Category"][2:-2],
                    "rating": book['Rating'],
                    "ratingCount": book['RatingCount'],
                    "yearOfPublication": book['YearOfPublication'],
                    "distance": distances.flatten()[i]
                })

        return jsonify(books[::-1])
    except:
        return jsonify({"err"})


@app.route('/svd/<ISBN>')
def svd(ISBN):
    book_ISBNs = SVDPivot.columns

    book_list = list(book_ISBNs)
    index = book_list.index(ISBN)

    recom_ISBNs = list(book_ISBNs[(corr[index] < 1.0)
                                  & (corr[index] > 0.9)])[1:5]

    books = []

    for isbn in recom_ISBNs:
        book = RatingCountDF.loc[RatingCountDF['ISBN'] == isbn][:1].to_dict()

        books.append({
            "title": list(book.get("Title").items())[0][1],
            "author": list(book.get("Author").items())[0][1],
            "publisher": list(book.get("Publisher").items())[0][1],
            "isbn": list(book.get("ISBN").items())[0][1],
            "image": list(book.get("Image").items())[0][1],
            "category": list(book.get("Category").items())[0][1][2:-2],
            "rating": list(book.get("Rating").items())[0][1],
            "ratingCount": list(book.get("RatingCount").items())[0][1],
            "yearOfPublication": list(book.get("YearOfPublication").items())[0][1]
        })

    return jsonify(books)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
