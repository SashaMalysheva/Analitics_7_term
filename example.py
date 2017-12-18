import pandas as pd
# noinspection PyUnresolvedReferences
from extractor import FeaturesExtractor
from flask import Flask, render_template, request
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
clf = joblib.load('clf.pkl')


@app.route('/', methods=['GET', 'POST'])
def index_page(text='', label='', color='orange'):
    if request.method == "POST":
        text = request.form["text"]
    if len(text):
        vectorizer = TfidfVectorizer(max_features=10)
        d = [text]
        df = pd.DataFrame(data=d, columns=['text'])
        x_v = vectorizer.fit_transform(df.text)
        label = clf.predict(x_v)[0]
    return render_template('hello.html', text=text, label=label, color='green')


if __name__ == '__main__':
    app.run()
