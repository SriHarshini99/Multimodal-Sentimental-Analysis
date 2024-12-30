from flask import Flask, render_template, request, jsonify
import os
import whisper
from moviepy.editor import VideoFileClip
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import string

# Initialize Flask app
app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words("english")

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]
    return " ".join(tokens)

# Load and prepare Sentiment Analysis Model
tfidf_sentiment = TfidfVectorizer(max_features=20, stop_words=stop_words)
df_sentiment = pd.read_csv("Sentiment Analysis.csv")
df_sentiment.columns = ["Text", "Label"]
x_sentiment = tfidf_sentiment.fit_transform(df_sentiment["Text"]).toarray()
y_sentiment = df_sentiment["Label"]
x_train_sentiment, x_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(
    x_sentiment, y_sentiment, test_size=0.1, random_state=0)
model_sentiment = LogisticRegression()
model_sentiment.fit(x_train_sentiment, y_train_sentiment)

# Load and prepare Sarcasm Detection Model
tfidf_sarcasm = TfidfVectorizer(max_features=20, stop_words=stop_words)
df_sarcasm = pd.read_csv("Sarcasm Detection.csv")
df_sarcasm.columns = ["Text", "Label"]
x_sarcasm = tfidf_sarcasm.fit_transform(df_sarcasm["Text"]).toarray()
y_sarcasm = df_sarcasm["Label"]
x_train_sarcasm, x_test_sarcasm, y_train_sarcasm, y_test_sarcasm = train_test_split(
    x_sarcasm, y_sarcasm, test_size=0.1, random_state=0)
model_sarcasm = LogisticRegression()
model_sarcasm.fit(x_train_sarcasm, y_train_sarcasm)

# Function to extract text from video using Whisper
def extract_text_from_video(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"

    try:
        # Extract audio from video
        video.audio.write_audiofile(audio_path)
    finally:
        video.close()

    # Use Whisper to transcribe audio to text
    model = whisper.load_model("base")  # Load Whisper model
    result = model.transcribe(audio_path)
    
    # Return the transcribed text
    extracted_text = result['text']
    
    # Remove temporary audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return extracted_text.strip()

# Flask endpoint for video upload and text extraction
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected video'}), 400

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    try:
        # Extract text from video using Whisper
        extracted_text = extract_text_from_video(video_path)
        transformed_text = transform_text(extracted_text)

        # Predict Sentiment
        sentiment_vector = tfidf_sentiment.transform([transformed_text])
        sentiment_prediction = model_sentiment.predict(sentiment_vector)[0]
        sentiment = "Positive" if sentiment_prediction == 1 else "Negative"

        # Predict Sarcasm
        sarcasm_vector = tfidf_sarcasm.transform([transformed_text])
        sarcasm_prediction = model_sarcasm.predict(sarcasm_vector)[0]
        sarcasm = "Sarcastic" if sarcasm_prediction == 1 else "Non-Sarcastic"

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

    return jsonify({
        'extracted_text': extracted_text,
        'sentiment': sentiment,
        'sarcasm': sarcasm
    })

# Flask endpoint for manual text input predictions
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    transformed_text = transform_text(text)

    # Predict Sentiment
    sentiment_vector = tfidf_sentiment.transform([transformed_text])
    sentiment_prediction = model_sentiment.predict(sentiment_vector)[0]
    sentiment = "Positive" if sentiment_prediction == 1 else "Negative"

    # Predict Sarcasm
    sarcasm_vector = tfidf_sarcasm.transform([transformed_text])
    sarcasm_prediction = model_sarcasm.predict(sarcasm_vector)[0]
    sarcasm = "Sarcastic" if sarcasm_prediction == 1 else "Non-Sarcastic"

    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'sarcasm': sarcasm
    })
# Flask endpoint for audio upload and text extraction
@app.route('/audio_predict', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio = request.files['audio']
    if audio.filename == '':
        return jsonify({'error': 'No selected audio'}), 400

    audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
    audio.save(audio_path)

    try:
        # Use Whisper to transcribe audio to text
        model = whisper.load_model("base")  # Load Whisper model
        result = model.transcribe(audio_path)

        # Extracted text from the audio
        extracted_text = result['text']
        transformed_text = transform_text(extracted_text)

        # Predict Sentiment
        sentiment_vector = tfidf_sentiment.transform([transformed_text])
        sentiment_prediction = model_sentiment.predict(sentiment_vector)[0]
        sentiment = "Positive" if sentiment_prediction == 1 else "Negative"

        # Predict Sarcasm
        sarcasm_vector = tfidf_sarcasm.transform([transformed_text])
        sarcasm_prediction = model_sarcasm.predict(sarcasm_vector)[0]
        sarcasm = "Sarcastic" if sarcasm_prediction == 1 else "Non-Sarcastic"

    finally:
        # Remove the uploaded audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return jsonify({
        'extracted_text': extracted_text,
        'sentiment': sentiment,
        'sarcasm': sarcasm
    })


@app.route('/predict_text', methods=['POST'])
def predict_text():
    # Parse JSON data
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Preprocess the text
        transformed_text = transform_text(text)

        # Predict Sentiment
        sentiment_vector = tfidf_sentiment.transform([transformed_text])
        sentiment_prediction = model_sentiment.predict(sentiment_vector)[0]
        sentiment = "Positive" if sentiment_prediction == 1 else "Negative"

        # Predict Sarcasm
        sarcasm_vector = tfidf_sarcasm.transform([transformed_text])
        sarcasm_prediction = model_sarcasm.predict(sarcasm_vector)[0]
        sarcasm = "Sarcastic" if sarcasm_prediction == 1 else "Non-Sarcastic"

        return jsonify({
            'sentiment': sentiment,
            'sarcasm': sarcasm
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Root endpoint
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
