import os
from tkinter import *
import numpy as np

InfoCounter = False

def Info():
    global InfoCounter
    if InfoCounter == False:
        InfoCounter = True
        RosyCool.place(x=1565, y=60)
        InfoButton.config(text="Close")
        InfoText.config(text='''





























Rosey is an Artificial Intelligence model that
uses predetermined Fire-fighting rules along
with trained decision making based off the
brilliant mind of an experienced firefighter.
''')
    else:
        InfoCounter = False
        RosyCool.place_forget()
        InfoText.config(text="")
        InfoButton.config(text="Info")
    

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Set the data directory
data_dir = "D:\Python Codes\Fire\FireClass"

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
def TestBTN():
    ""
    


def GetPDA(x, y):
    if str(x) == "House Fire" and str(LiveAtRisk) == "No":
        PDA = ""+y+" fire engines, 1 ladder and 1 officer in charge"
    elif str(x) == "House Fire" and str(LiveAtRisk) == "Yes":
        PDA = ""+y+" fire engines, 1 ladder and 1 officer in charge"
    elif str(x) == "High-Rise Fire":
        PDA = ""+y+" fire engines, 1 aerial appliance, 1 command unit and multiple officers"
    elif str(x) == "Large Vehicle Fire":
        PDA = "2 fire engines, 1 ladder and 1 officer in charge"
    elif str(x) == "Bin/Small Rubbish Fire":
        PDA = "1 fire engine"
    elif str(x) == "Cladding Fire (High-Rise)":
        PDA = ""+y+" fire engines, 3 ladders, multiple officers and command units."
    elif str(x) == "Fire with Fatality":
        PDA = ""+y+" fire engines, 1 fire investigation unit and press officer notified"
    elif str(x) == "Person Trapped (Non-Road Traffic Collision)":
        PDA = "2 fire engines, 1 heavy rescue unit and 1 officer in charge"     
    elif str(x) == "Road Traffic Collision (Person Trapped)":
        PDA = "2 fire engines, 1 heavy rescue unit and 1 officer in charge, ambulance notified"    
    elif str(x) == "Person Shut in Lift (Non-Emergency)":
        PDA = "1 fire engine (only if no other help available)"   
    elif str(x) == "Animal Rescue (Large Animal)":
        PDA = "1 fire engine, 1 heavy rescue unit and specialist officers"   
    elif str(x) == "Flooding (Non-Commercial)":
        PDA = "1 fire engine"    
    elif str(x) == "Gas Leak (Domestic/Commercial)":
        PDA = "1 fire engine and gas authority notified"    
    elif str(x) == "Minor Hazardous Spill (under 100L)":
        PDA = "1 fire engine"    
    elif str(x) == "Major Hazardous Spill (over 100L)":
        PDA = "2 fire engines, HazMat unit and command unit"    
    elif str(x) == "Suspicious Substance/White Powder":
        PDA = "1 HazMat officer and police notified"    
    elif str(x) == "Person Threatening to Jump/Rescue from Height":
        PDA = "1 fire engine, ladder appliance, ambulance and police notified"    
    elif str(x) == "Person Collapsed Behind Locked Door":
        PDA = "1 fire engine and ambulance notified"    
    elif str(x) == "Vehicle into Building":
        PDA = "2 fire engines, 1 heavy rescue unit, 1 structural officer and ambulance"
    else:
        pass
    return PDA
    
    
def Logic():
    PDA = GetPDA(IncidentType, predicted_class)
    RoseyText.config(text="Rosey says: 'There is a "+str(IncidentType)+" at "+str(Adress)+". I recommend the attendance of "+PDA+".")

def Update():
    global NB
    global Adress
    global IncidentType
    global LiveAtRisk
    global predicted_class
    print(NB)
    try:
        Transcript = open("CallTranscript"+str(NB)+".txt", "r")
        image_path = r"D:\Python Codes\Fire\Image"+str(NB)+".png"
        NB = NB + 1
        predicted_class = classify_image(image_path)
        CallData = Transcript.readlines()
        Adress = (CallData[1].strip())
        IncidentType = (CallData[4].strip())
        LiveAtRisk = (CallData[7].strip())
        Logic()
    except:
        pass
    root.after(1000, Update)

#Defines the tkinter window
root = Tk()

#Images
LFBMap = PhotoImage(file="LFBMap.png")
RoseyIcon = PhotoImage(file="RoseyIcon.png")
RoseyInfo = PhotoImage(file="RoseyInfo.png")

#Labels
Map = Label(root, image=LFBMap, bd=0)
Map.place(x=0, y=0)

RoseyText = Label(root, text="Rosey", height=8, width=160, bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
RoseyText.place(x=127, y=953)
InfoText = Label(root, text="Info", height=64, width=40,  bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
InfoText.place(x=1565, y=0)

RoseyFace = Label(root, image=RosyIcon, bd=0)
RoseyFace.place(x=0, y=953)
RoseyCool = Label(root, image=RosyInfo, bd=0)

#Buttons
DevButton = Button(root, text="Dev Mode", height=5, width=25, bg="#FFFFFF", fg="#000000", command=TestBTN)
DevButton.place(x=1380, y=867)
InfoButton = Button(root, text="Info", height=5, width=25, bg="#FFFFFF", fg="#000000", command=Info)
InfoButton.place(x=1200, y=867)


Btn1 = Button(root, text="Barking and\nDagenham")
Btn2 = Button(root, text="Barnet")
Btn3 = Button(root, text="Bexley")
Btn4 = Button(root, text="Brent")
Btn5 = Button(root, text="Bromley")
Btn6 = Button(root, text="Camden")
Btn7 = Button(root, text="City")
Btn8 = Button(root, text="Croydon")
Btn9 = Button(root, text="Ealing")
Btn10 = Button(root, text="Enfield")
Btn11 = Button(root, text="Greenwich")
Btn12 = Button(root, text="Hackney")	
Btn13 = Button(root, text="Hammersmith\nand Fulham")
Btn14 = Button(root, text="Haringey")
Btn15 = Button(root, text="Harrow")
Btn16 = Button(root, text="Havering")
Btn17 = Button(root, text="Hillingdon")
Btn18 = Button(root, text="Hounslow")
Btn19 = Button(root, text="Islington")
Btn20 = Button(root, text="Kensington\nand Chelsea")
Btn21 = Button(root, text="Kingston\nupon\nThames")
Btn22 = Button(root, text="Lambeth")
Btn23 = Button(root, text="Lewisham")
Btn24 = Button(root, text="Merton")
Btn25 = Button(root, text="Newham")	
Btn26 = Button(root, text="Redbridge")
Btn27 = Button(root, text="Richmond\nupon Thames")
Btn28 = Button(root, text="Southwark")
Btn29 = Button(root, text="Sutton")	
Btn30 = Button(root, text="Tower\nHamlets")
Btn31 = Button(root, text="Waltham\nForest")	
Btn32 = Button(root, text="Wandsworth")
Btn33 = Button(root, text="Westminster")

Btn1.place(x=1060, y=360)
Btn2.place(x=620, y=200)
Btn3.place(x=1095, y=530)
Btn4.place(x=550, y=325)
Btn5.place(x=970, y=725)
Btn6.place(x=669, y=335)
Btn7.place(x=782, y=415)
Btn8.place(x=800, y=750)
Btn9.place(x=470, y=405)
Btn10.place(x=775, y=125)
Btn11.place(x=957, y=505)
Btn12.place(x=825, y=332)
Btn13.place(x=610, y=470)
Btn14.place(x=750, y=255)
Btn15.place(x=450, y=235)
Btn16.place(x=1210, y=320)
Btn17.place(x=285, y=360)
Btn18.place(x=410, y=500)
Btn19.place(x=755, y=350)
Btn20.place(x=590, y=425)
Btn21.place(x=520, y=650)
Btn22.place(x=730, y=525)
Btn23.place(x=870, y=550)
Btn24.place(x=635, y=635)
Btn25.place(x=940, y=390)	
Btn26.place(x=990, y=270)
Btn27.place(x=470, y=556)
Btn28.place(x=784, y=478)
Btn29.place(x=665, y=745)	
Btn30.place(x=840, y=394)
Btn31.place(x=873, y=235)
Btn32.place(x=625, y=548)
Btn33.place(x=680, y=424)

#Running
NB = 1
Update()
    
#Sets the tkinter window 
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title("Rosey")
root.mainloop()

