import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import vgg
import os


test_path = 'LibriSpeech-SI/test'

n_mfcc = 40
max_length = 1057

X = np.load('LibriSpeech-SI/train_data.npy')
y = np.load('LibriSpeech-SI/train_label.npy')

# Split the data into a training set and a validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the labels to one-hot encoding
num_classes = len(set(y))
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
    mfccs = np.load('LibriSpeech-SI/test_data.npy')

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
            prediction += 1
            # Format the prediction value as a three-digit string with leading zeros
            prediction_str = str(prediction).zfill(3)
            # Write the filename and prediction to the file
            f.write(f"{filename} spk{prediction_str}\n")

    return predicted_class


# Make predictions on the audio samples in the test folder
predictions = predict(vgg, test_path)

# Print the predictions
print(predictions)
