from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import io
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# CORS middleware settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model
model = tf.keras.models.load_model("./modal.keras")

# Function to check if the uploaded file is a valid WAV
def is_valid_wav(audio_data):
    try:
        with sf.SoundFile(io.BytesIO(audio_data)) as f:
            return True
    except Exception as e:
        print(f"Invalid WAV file: {e}")
        return False

# Function to convert speech to Mel spectrogram
def load_mel_spectrogram(audio_data, n_mels=40, max_length=128):
    try:
        # Validate the WAV file
        if not is_valid_wav(audio_data):
            raise ValueError("Invalid WAV file")

        # Load waveform from byte data
        waveform, sr = librosa.load(io.BytesIO(audio_data), sr=None)
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Ensure consistent shape
        if mel_spec_db.shape[1] < max_length:
            pad_width = max_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_length]

        return np.expand_dims(mel_spec_db, axis=(0, -1))  # Shape: (1, n_mels, max_length, 1)
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_data = await file.read()

    try:
        # Extract Mel spectrogram features
        mel_spec = load_mel_spectrogram(audio_data)
        if mel_spec is None:
            return {"error": "Invalid audio file"}

        # Make a prediction
        prediction = model.predict(mel_spec)
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class])

        # Label mapping
        labels = {0: "Healthy", 1: "Impaired"}
        result = labels.get(predicted_class, "Unknown")

        return {
            "prediction": result,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}