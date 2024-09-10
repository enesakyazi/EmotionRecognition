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

folder_path = '/Users/akyaz/OneDrive/Desktop/enesProject/Combined1'

# Function to load a file, apply data augmentation, and convert to raw waveform
def augment_data(audio, noise_factor=0.005, shift_max=2):
    audio_with_noise = audio + noise_factor * np.random.randn(len(audio))

    shift = np.random.randint(shift_max)
    if np.random.choice([True, False]):
        audio_shifted = np.roll(audio, shift)
    else:
        audio_shifted = np.roll(audio, -shift)

    return audio_with_noise, audio_shifted

def load_audio_file(file_path, sr=22050, duration=3):
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration, res_type='kaiser_fast')
        samples_per_track = sr * duration
        if len(audio) > samples_per_track:
            audio = audio[:samples_per_track]
        else:
            padding = samples_per_track - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        return audio
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return np.zeros(samples_per_track)

# extracting labels
filenames = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
labels = [file.split('_')[0] for file in filenames]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)

X_original = np.array([load_audio_file(os.path.join(folder_path, file)) for file in filenames])
X_augmented = np.array([augment_data(audio) for audio in X_original])
X = np.concatenate((X_original, X_augmented[:, 0], X_augmented[:, 1]))

y_original = to_categorical(integer_encoded)
y = np.concatenate((y_original, y_original, y_original)) 

X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(input_shape, num_classes, l2_factor=0.001):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(4))
    model.add(Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(4))
    model.add(Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(4))
    model.add(Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=l2(l2_factor)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(4))

    model.add(tf.keras.layers.Flatten())

    model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_factor)))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_factor)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)

model = create_model(X_train.shape[1:], y.shape[1])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

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

model.save('/Users/akyaz/OneDrive/Desktop/enesProject/cnn_model_22050-3-1.h5')

