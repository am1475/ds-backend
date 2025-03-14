from flask import Flask, request, jsonify
from flask_cors import CORS  # optional, for cross-origin requests
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

app = Flask(__name__)
CORS(app)  # enable CORS if needed

# ---------------------------
# 1. Load the Saved Model
# ---------------------------
model = load_model('autocomplete_model.h5')
print("Model loaded successfully.")

# ---------------------------
# 2. Reload the Dataset and Recreate Encoders
# ---------------------------
df = pd.read_csv('unigram_freq.csv')

# Rename column if necessary
if 'frequency' not in df.columns and 'count' in df.columns:
    df.rename(columns={'count': 'frequency'}, inplace=True)

df['word'] = df['word'].str.lower()
df.dropna(subset=['word', 'frequency'], inplace=True)
df['first_letter'] = df['word'].str[0]
df = df.sort_values(by='frequency', ascending=False).head(1000)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(df[['first_letter']])

label_encoder = LabelEncoder()
label_encoder.fit(df['word'])

# ---------------------------
# 3. Define the Suggestion Function
# ---------------------------
def suggest_words(input_text, top_n=5):
    """
    Given a text input, extract the current word (last token) and use the model
    to predict word probabilities based on its first letter. Return suggestions that
    match the current word.
    """
    # Extract the last word (token) from the input text
    tokens = input_text.strip().split()
    if not tokens:
        return []
    current_word = tokens[-1].lower().strip()
    if not current_word:
        return []

    # Use only the first letter of the current word for model prediction
    first_letter = current_word[0]
    letter_input = np.array([[first_letter]])
    X_input = encoder.transform(letter_input)
    preds = model.predict(X_input)

    # Get candidate words from the model prediction
    top_indices = np.argsort(preds[0])[::-1]
    candidate_words = label_encoder.inverse_transform(top_indices)

    # Filter candidates to include only words starting with the current word
    suggestions = [word for word in candidate_words if word.startswith(current_word)]

    # Supplement suggestions using frequency-based lookup if needed
    if len(suggestions) < top_n:
        additional = df[df['word'].str.startswith(current_word)] \
                     .sort_values(by='frequency', ascending=False)['word'].tolist()
        for word in additional:
            if word not in suggestions:
                suggestions.append(word)
            if len(suggestions) >= top_n:
                break

    return suggestions[:top_n]

# ---------------------------
# 4. Create the Suggestion Endpoint
# ---------------------------
@app.route('/suggest', methods=['GET'])
def get_suggestions():
    input_text = request.args.get('q', default='', type=str)
    top_n = request.args.get('top_n', default=5, type=int)
    suggestions = suggest_words(input_text, top_n)
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)
