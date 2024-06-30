from flask import Flask, request, jsonify
import pickle

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    # Function to preprocess text (this should match the preprocessing steps in your notebook)
    import string
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import nltk

    # Ensure NLTK data is downloaded
    nltk.download('stopwords')
    nltk.download('wordnet')

    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.json['email']
    processed_text = preprocess_text(email_text)
    features = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(features)
    return jsonify({'spam': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
