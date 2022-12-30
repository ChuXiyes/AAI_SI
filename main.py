import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import vgg


train_path = 'train'
test_path = 'test'

n_mfcc = 40
max_length = 1000
# Initialize lists to store the features and labels
X = []
y = []
# Loop through each speaker folder
for speaker_id, speaker_folder in enumerate(os.listdir(train_path)):
    # Loop through each audio file in the speaker folder e.g. spk001
    id = speaker_folder[3:]
    id = int(id)
    for audio_file in os.listdir(os.path.join(train_path, speaker_folder)): # train\spk001
        # Load the audio file and extract the features
        audio, sample_rate = librosa.load(os.path.join(train_path, speaker_folder, audio_file))
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        # Add the features and label to the list
        X.append(mfcc)
        y.append(id)

# Print the shape of the MFCC features
print(mfcc.shape)
print(mfcc.shape[1])
# Get the maximum length of the sequences in the training set
# max_length = max([x.shape[1] for x in X])


# Pad the sequences in the training and validation sets to the maximum length
X_row = X
for i in range(len(X)):
    X[i] = pad_sequences(X[i], maxlen=max_length, padding='post')

# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)
# Split the data into a training set and a validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the labels to one-hot encoding
num_classes = len(set(y)) + 1
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Define the model
vgg = vgg.VGG((n_mfcc,max_length,1),num_classes)

# Compile the model
vgg.compile_model(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




# Train the model
vgg.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))


def predict(model, test_folder):
    # Initialize a list to store the mfccs
    mfccs = []

    # Loop through each audio file in the test folder
    for audio_file in os.listdir(test_folder):
        # Load the audio file and extract the features
        audio, sample_rate = librosa.load(os.path.join(test_folder, audio_file))
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Pad or truncate the sequence to the same length as the training set, if necessary
        # max_length = 1000 # Length to pad or truncate the sequence to
        mfcc = pad_sequences(mfcc, maxlen=max_length, padding='post')

        # Add the prediction to the list
        mfccs.append(mfcc)
    # numpy array
    mfccs = np.array(mfccs)
    # Make a prediction on the extracted features
    output = model.predict(mfccs)

    # Extract the predicted class from the model's output
    predicted_class = np.argmax(output, axis=1)

    # Open a file for writing
    with open('a.txt', 'w') as f:
        # Loop through the predictions
        for i, prediction in enumerate(predicted_class):
            # Get the filename of the audio file
            filename = os.listdir(test_folder)[i]
            # Increment the prediction value by one
            # prediction += 1
            # Format the prediction value as a three-digit string with leading zeros
            prediction_str = str(prediction).zfill(3)
            # Write the filename and prediction to the file
            f.write(f"{filename} spk{prediction_str}\n")

    return predicted_class


# Make predictions on the audio samples in the test folder
predictions = predict(vgg, test_path)

# Print the predictions
print(predictions)
