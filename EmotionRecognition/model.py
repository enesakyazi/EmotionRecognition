import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import load_model

def load_audio_file(file_path, sr=16000, duration=3):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
    samples_per_chunk = sr * duration
    if len(audio) > samples_per_chunk:
        audio = audio[:samples_per_chunk]
    else:
        padding = samples_per_chunk - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
    return audio

folder_path = '/Users/akyaz/OneDrive/Desktop/enesProject/Combined1'
model_path = '/Users/akyaz/OneDrive/Desktop/enesProject/cnn_model_16000_3-1.h5'

model = load_model(model_path)

filenames = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
labels = [file.split('_')[0] for file in filenames]

X = np.array([load_audio_file(os.path.join(folder_path, file)) for file in filenames])

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
y = to_categorical(integer_encoded)

X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=1, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('/Users/akyaz/OneDrive/Desktop/enesProject/cnn_model4.h5')