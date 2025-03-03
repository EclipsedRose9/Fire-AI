import os
from tkinter import *
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Set the data directory
data_dir = "C:/Users/I1d0n/Desktop/FireClass"

# Define image dimensions and other parameters
img_width, img_height = 150, 150
batch_size = 32
num_classes = 13

# Preprocess the images and create data generators
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Get the class labels from folder names
class_labels = sorted(os.listdir(data_dir))

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)

# Function to classify a new image
def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

# Example usage:
image_path = "C:/Users/I1d0n/Desktop/test.png"
predicted_class = classify_image(image_path)
Transcript = open("CallTranscript.txt", "r")
CallData = Transcript.readlines()
Adress = CallData[1]
FireLocation = CallData[4]
LiveAtRisk = CallData[7]

# Logic
if int(predicted_class) == 2:
    if str(LiveAtRisk) == "Yes" or str(FireLocation) == "Roof":
        print("Rosy Suggests: (4) Pump")
    else:
        print("Rosy Suggests: (2) Pump")



#Defines the tkinter window
root = Tk()

RosyText = Label(root, height=1, width=1, bg="#FFFFFF", fg="#000000")
RosyText.place(x=0, y=0)
InfoText = Label(root, height=1, width=1, bg="#FFFFFF", fg="#000000")
InfoText.place(x=500, y=500)

DevButton = Button(root, height=1, width=10, bg="#FFFFFF", fg="#000000")
DevButton.place(x=1000, y=600)
    
#Sets the tkinter window 
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title("Rosy")
root.mainloop()

