'''
This file will create JSON files that will have data necessary for the models
Will start off by creating a file that will use Wei's final_data.csv to create WordRep vs NoStutter JSON file

Creating Files:
    - wordRep.json

'''
import pandas as pd
import librosa
import os
import numpy as np
from collections import Counter
audio_dir = r"C:/Users/andrew/Downloads/ml_stuttering/clips/stuttering-clips/clips"

def audio_to_mfcc(filename, n_fft=2048, hop_length=512):
    audio_path = os.path.join(audio_dir, filename + '.wav')
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        S = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        return S
    except Exception as e:
        print(f"Failed to process {filename}: {str(e)}")
        return None

def audio_to_spectrogram(filename, n_fft=2048, hop_length=512, n_mels=128):
    # Load the audio file
    audio_path = os.path.join(audio_dir, filename + '.wav')
    try:
        # Ensure `sr=None` to use the native sampling rate of the file
        audio, sr = librosa.load(audio_path, sr=None)  
        # Generate a Mel-scaled spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        # Convert to log scale (dB)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    except Exception as e:
        print(f"Failed to process {filename}: {str(e)}")
        return None
from tensorflow.keras.utils import to_categorical
def getDataFrame():
    # will create a DF with WordRep and NoStutter
    csv_file_path = r"C:/Users/andrew/Downloads/final_data.csv"
    data = pd.read_csv(csv_file_path)
    # data["mfcc"] = data['Filename'].apply(lambda f: audio_to_mfcc(f))
    
    data["spectrogram"] = data["Filename"].apply(lambda f: audio_to_spectrogram(f))

    # (13, 94)
    # desired_shape = (13, 94)
    # mask = data['mfcc'].apply(lambda x: x.shape == desired_shape)
    # final1_data = data[mask]

    desired_shape = (128, 94)
    mask = data['spectrogram'].apply(lambda x: x.shape == desired_shape)
    final1_data = data[mask]

    mask1 = data['WordRep'].apply(lambda x: x>=2)
    mask2 = data['label'].apply(lambda x: x == 0)
    combined_mask = mask1 | mask2
    final_data = final1_data[combined_mask]

    # (128, 94)
    return final_data

def balanceDF(df):
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]
    
    min_size = min(len(df_0), len(df_1))
    print(min_size)
    df_0_balanced = df_0.sample(min_size, random_state=42)
    df_1_balanced = df_1.sample(min_size, random_state=42)
    
    df_balanced = pd.concat([df_0_balanced, df_1_balanced])
    
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def createModel1():
    data = getDataFrame()
    data = balanceDF(data)
    
    X = np.stack(data['mfcc'].values)
    X = X[..., np.newaxis]
    y = to_categorical(data['label'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)
    X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=46)
    print(X_train1.shape, X_val.shape, y_train1.shape, y_val.shape)
    model = Sequential([
        # Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(13,94,1)),
        # MaxPooling2D(pool_size=(2, 2)),  # Reduced stride
        # Dropout(0.3),
        # Conv2D(64, (3, 3), activation='relu'),
        # MaxPooling2D(pool_size=(2, 2)),  # Reduced stride
        # Dropout(0.5),
        
        # Flatten(),
        # Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        # Dropout(0.5),
        # Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        # Dropout(0.5),
        # Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        # Dropout(0.5),
        # Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        # Dropout(0.5),
        # Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        # Dropout(0.5),
        # Dense(y_train.shape[1], activation='softmax')
        Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
        MaxPooling2D((2, 2), strides=(1, 1)),  # Reduced stride
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),  # Reduced stride
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        # BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        # BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        # BatchNormalization(),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='softmax')
    ])
    # model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train1, y_train1, epochs=20, validation_data=(X_val, y_val), batch_size=32)

def createModel():
    data = getDataFrame()
    data = balanceDF(data)
    
    X = np.stack(data['spectrogram'].values)[:, :, :, np.newaxis]
    y = to_categorical(data['label'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)
    X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=46)
    print(X_train1.shape, X_val.shape, y_train1.shape, y_val.shape)
    model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 94, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),  # Added dropout here
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),  # Added dropout here
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])
    # model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train1, y_train1, epochs=20, validation_data=(X_val, y_val), batch_size=32)


createModel()