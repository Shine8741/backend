import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Load dataset
dataset_path = "torgo_data/data.csv"
df = pd.read_csv(dataset_path)

# Extract folder names (dysarthria_female, dysarthria_male, etc.) from 'filename' column
df['folder'] = df['filename'].apply(lambda x: x.split('/')[1])  # Correct folder extraction

# Display class distribution
print("Dataset Class Distribution by Folder:")
print(df['folder'].value_counts())


# Check class distribution in data.csv
print("Dataset Class Distribution:")
print(df['filename'].apply(lambda x: x.split('/')[0]).value_counts())

# Feature Extraction Function
def feature_extraction(df):
    features = []
    for i, record in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            file_path = record['filename']  # Path is already correct in CSV
            x, sr = librosa.load(file_path)
            # mean_mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128), axis=1)
            mean_mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128, n_fft=2048, hop_length=512), axis=1)

            features.append(mean_mfcc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
    dataf = pd.DataFrame(features)
    dataf['class'] = df['is_dysarthria']
    return dataf

# Extract features
dataf = feature_extraction(df)

# Convert labels
dataf.loc[dataf['class'] == 'non_dysarthria', 'class'] = 0.0
dataf.loc[dataf['class'] == 'dysarthria', 'class'] = 1.0
dataf['class'] = dataf['class'].astype(float)

# Prepare data for training
X = dataf.iloc[:, :-1].values
y = dataf.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Add folder information back to data
dataf['folder'] = df['folder']

# Split the data accordingly
train_df, test_df = train_test_split(dataf, test_size=0.2, stratify=dataf['class'], random_state=42)

# Display test data distribution
print("Test Data Distribution by Folder:")
print(test_df['folder'].value_counts())

print("Train Data Distribution:")
print(pd.Series(y_train).value_counts())

print("Test Data Distribution:")
print(pd.Series(y_test).value_counts())


# Reshape data for CNN
X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)

# Build the model
model = Sequential([
    InputLayer(input_shape=(16, 8, 1)),
    Conv2D(32, (3, 3), activation='relu', padding="same"),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding="same"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks
checkpoint = ModelCheckpoint("model.h5", monitor="val_loss", save_best_only=True, verbose=1)
# earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[checkpoint, earlystopping])

# Save final model
model.save('model.keras')

print("Training complete! Model saved as model.h5")

# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the best model
model.load_weights("model.h5")

# Predict on test data
y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Dysarthria', 'Dysarthria'])

# Display plot with sample count
plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix (Total Samples: {len(y_test)})')
plt.show()
print(classification_report(y_test, y_pred))