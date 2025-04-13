import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser
from google.colab import files

# Load the FER-2013 dataset
data = pd.read_csv('/content/fer2013.csv')

# Extract features (pixels) and labels (emotions)
pixels = data['pixels'].tolist()
emotions = data['emotion'].values

# Convert pixel strings to numpy arrays
images = []
for pixel_string in pixels:
    pixel_values = pixel_string.strip().split()
    image = np.array([int(value) for value in pixel_values], dtype='float32').reshape(48, 48)
    images.append(image)

images = np.array(images)  # Convert list to numpy array
images = images.reshape(-1, 48, 48, 1)  # Reshape to (num_samples, 48, 48, 1)

# Normalize pixel values to [0, 1]
images /= 255.0

# Convert labels to one-hot encoding
labels = to_categorical(emotions, num_classes=7)

# Split the data into training, validation, and test sets
x_temp, x_test, y_temp, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")
print(f"Test data shape: {x_test.shape}")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(x_train)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=20,
                    validation_data=(x_val, y_val))

# Print accuracy
print("Training Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])

# Plot accuracy and loss
def plot_history(history):
    plt.figure(figsize=(12, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# Evaluate the model
loss, accuracy = model.evaluate(x_val, y_val)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Generate Predictions
y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

# Classification Report
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
report = classification_report(y_true, y_pred_classes, target_names=emotion_labels)
print("Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
plt.figure(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Music Recommendation based on Emotion
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
client_id = "your_client_id"  # Replace with your Client ID
client_secret = "your_client_secret"  # Replace with your Client Secret

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Process uploaded image and recommend music
uploaded = files.upload()

# Load the trained model
model = load_model('emotion_detection_model.h5')

for file_name in uploaded.keys():
    print(f"Uploaded file: {file_name}")

    # Load the image and check for color mode
    img = load_img(file_name, target_size=(48, 48))

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Check if the image is grayscale (1 channel) and convert to RGB if necessary
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)  # Convert grayscale to RGB by repeating the channel

    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict emotion
    prediction = model.predict(img_array)
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    print(f"Predicted Emotion: {predicted_emotion}")

    # Save the detected emotion to a file
    with open('/content/detected_emotion.txt', 'w') as file:
        file.write(predicted_emotion)

# Recommend music based on emotion
def recommend_music():
    try:
        with open('/content/detected_emotion.txt', 'r') as file:
            emotion = file.read().strip()
    except FileNotFoundError:
        print("Error: Emotion file not found.")
        return

    genre = emotion_to_genre.get(emotion, "Pop")
    print(f"Emotion: {emotion}, Genre: {genre}")

    results = sp.search(q=f"{genre} playlist", type="playlist", limit=1)
    if results['playlists']['items']:
        playlist_url = results['playlists']['items'][0]['external_urls']['spotify']
        print(f"Playlist Name: {results['playlists']['items'][0]['name']}")
        print(f"Playlist URL: {playlist_url}")
        webbrowser.open(playlist_url)
    else:
        print("No playlist found for the detected emotion.")

recommend_music()
