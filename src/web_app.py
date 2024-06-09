from flask import Flask, request, render_template
import src.sentiment_analysis as sa
import src.topic_modeling as tm

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Perform sentiment analysis
    sentiment = sa.predict_sentiment(text)
    return render_template('result.html', sentiment=sentiment)

@app.route('/topics', methods=['POST'])
def topics():
    text = request.form['text']
    # Perform topic modeling
    topics = tm.get_topics(text)
    return render_template('topics.html', topics=topics)

if __name__ == '__main__':
    app.run(debug=True)



