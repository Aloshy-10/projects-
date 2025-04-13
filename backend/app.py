from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io
from PIL import Image
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)
CORS(app)

# Load the trained emotion detection model
model = load_model('emotion_cnn_model.h5')  # Ensure correct path to your model

# Emotion to genre mapping (for Spotify)
emotion_to_genre = {
    "Happy": "Pop",
    "Sad": "Acoustic",
    "Neutral": "Classical",
    "Angry": "Rock",
    "Fear": "Instrumental",
    "Surprise": "Electronic",
    "Disgust": "Jazz"
}

# Spotify API setup
client_id = "984b9e03139c40d8ace2c51615b8123f"  
client_secret = "97c256d97abf4ec69efcf20512c822da"  
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Route to upload image and get emotion
@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image file in the request"}), 400

    file = request.files['image']
    
    try:
        # Load the image and preprocess it
        img = Image.open(io.BytesIO(file.read()))

        # Ensure the image is RGB or grayscale as needed (check your model)
        img = img.convert('RGB')  # Convert to RGB if the model expects RGB
        img = img.resize((48, 48))  # Resize to the size your model expects
        img_array = img_to_array(img) / 255.0  # Normalize pixel values

        # Ensure the correct shape (e.g., for a model expecting (48, 48, 3))
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict emotion
        prediction = model.predict(img_array)
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Get the genre for the detected emotion
        genre = emotion_to_genre.get(predicted_emotion, "Pop")  # Default to "Pop" if emotion not found
        playlist = sp.search(q=f"{genre} playlist", type="playlist", limit=1)

        # Get the playlist URL
        if playlist['playlists']['items']:
            playlist_url = playlist['playlists']['items'][0]['external_urls']['spotify']
        else:
            playlist_url = None

        return jsonify({
            "emotion": predicted_emotion,
            "playlist_url": playlist_url
        })

    except Exception as e:
        # Log error for debugging
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
