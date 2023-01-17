from flask import Flask, jsonify
import os
import numpy as np
import pandas as pd
from joblib import load

app = Flask(__name__)

# Rating count more than 50
csv_url = "https://drive.google.com/file/d/1tIxt00bOAPEKRkc57uuBhPZGDLBwsvJc/view?usp=share_link"
csv_url = 'https://drive.google.com/uc?id=' + csv_url.split('/')[-2]
RatingCountDF = pd.read_csv(csv_url)


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


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
