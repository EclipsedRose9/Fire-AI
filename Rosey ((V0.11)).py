import os
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    CurrentData = str("'There is a "+str(IncidentType)+" at "+str(Adress)+" "+str(Borough)+" "+str(Postcode)+". I recommend the attendance of "+PDA+".")
    RoseyText.config(text="Rosey says: "+CurrentData)
    AddData(Borough, CurrentData)

def AddData(x, y):
    CurrentFile = open(str(x)+".txt", "a")
    CurrentFile.write(y+"\n\n")
    CurrentFile.close()
    
    AllFile = open("All.txt", "a")
    AllFile.write(y+"\n\n")
    AllFile.close()
    
def DisplayInfo(x):
    InfoBox.configure(state='normal')
    InfoBox.delete('1.0', END)
    InfoBox.configure(state='disabled')
    
    if x == 1:
        F1 = open("Barking and Dagenham.txt", "r")
        Data = F1.readlines()
        F1.close()

    elif x == 2:
        F2 = open("Barnet.txt", "r")
        Data = F2.readlines()
        F2.close()

    elif x == 3:
        F3 = open("Bexley.txt", "r")
        Data = F3.readlines()
        F3.close()

    elif x == 4:
        F4 = open("Brent.txt", "r")
        Data = F4.readlines()
        F4.close()

    elif x == 5:
        F5 = open("Bromley.txt", "r")
        Data = F5.readlines()
        F5.close()

    elif x == 6:
        F6 = open("Camden.txt", "r")
        Data = F6.readlines()
        F6.close()

    elif x == 7:
        F7 = open("City.txt", "r")
        Data = F7.readlines()
        F7.close()

    elif x == 8:
        F8 = open("Croydon.txt", "r")
        Data = F8.readlines()
        F8.close()

    elif x == 9:
        F9 = open("Ealing.txt", "r")
        Data = F9.readlines()
        F9.close()

    elif x == 10:
        F10 = open("Enfield.txt", "r")
        Data = F10.readlines()
        F10.close()

    elif x == 11:
        F11 = open("Greenwich.txt", "r")
        Data = F11.readlines()
        F11.close()

    elif x == 12:
        F12 = open("Hackney.txt", "r")
        Data = F12.readlines()
        F12.close()

    elif x == 13:
        F13 = open("Hammersmith and Fulham.txt", "r")
        Data = F13.readlines()
        F13.close()

    elif x == 14:
        F14 = open("Haringey.txt", "r")
        Data = F14.readlines()
        F14.close()

    elif x == 15:
        F15 = open("Harrow.txt", "r")
        Data = F15.readlines()
        F15.close()

    elif x == 16:
        F16 = open("Havering.txt", "r")
        Data = F16.readlines()
        F16.close()

    elif x == 17:
        F17 = open("Hillingdon.txt", "r")
        Data = F17.readlines()
        F17.close()

    elif x == 18:
        F18 = open("Hounslow.txt", "r")
        Data = F18.readlines()
        F18.close()

    elif x == 19:
        F19 = open("Islington.txt", "r")
        Data = F19.readlines()
        F19.close()

    elif x == 20:
        F20 = open("Kensington and Chelsea.txt", "r")
        Data = F20.readlines()
        F20.close()

    elif x == 21:
        F21 = open("Kingston upon Thames.txt", "r")
        Data = F21.readlines()
        F21.close()

    elif x == 22:
        F22 = open("Lambeth.txt", "r")
        Data = F22.readlines()
        F22.close()

    elif x == 23:
        F23 = open("Lewisham.txt", "r")
        Data = F23.readlines()
        F23.close()

    elif x == 24:
        F24 = open("Merton.txt", "r")
        Data = F24.readlines()
        F24.close()

    elif x == 25:
        F25 = open("Newham.txt", "r")
        Data = F25.readlines()
        F25.close()

    elif x == 26:
        F26 = open("Redbridge.txt", "r")
        Data = F26.readlines()
        F26.close()

    elif x == 27:
        F27 = open("Richmond upon Thames.txt", "r")
        Data = F27.readlines()
        F27.close()

    elif x == 28:
        F28 = open("Southwark.txt", "r")
        Data = F28.readlines()
        F28.close()

    elif x == 29:
        F29 = open("Sutton.txt", "r")
        Data = F29.readlines()
        F29.close()

    elif x == 30:
        F30 = open("Tower Hamlets.txt", "r")
        Data = F30.readlines()
        F30.close()

    elif x == 31:
        F31 = open("Waltham Forest.txt", "r")
        Data = F31.readlines()
        F31.close()

    elif x == 32:
        F32 = open("Wandsworth.txt", "r")
        Data = F32.readlines()
        F32.close()

    elif x == 33:
        F33 = open("Westminster.txt", "r")
        Data = F33.readlines()
        F33.close()

    InfoBox.configure(state='normal')
    InfoBox.insert("end", "".join(Data))
    InfoBox.configure(state='disable')

def Update():
    global NB
    global Adress
    global Borough
    global Postcode
    global IncidentType
    global LiveAtRisk
    global predicted_class
    try:
        Transcript = open("CallTranscript"+str(NB)+".txt", "r")
        RoseyCool.place_forget()
        InfoBox.configure(state='normal')
        InfoBox.delete('1.0', END)
        InfoBox.configure(state='disabled')
        image_path = r"D:\Python Codes\Fire\Image"+str(NB)+".png"
        NB = NB + 1
        predicted_class = classify_image(image_path)
        CallData = Transcript.readlines()
        Adress = (CallData[1].strip())
        Borough = (CallData[2].strip())
        Postcode = (CallData[3].strip())
        IncidentType = (CallData[6].strip())
        LiveAtRisk = (CallData[9].strip())
        Logic()
        AllFile = open("All.txt", "r")
        AllData = AllFile.readlines()
        AllFile.close()
        InfoBox.configure(state='normal')
        InfoBox.insert("end", "".join(AllData))
        InfoBox.configure(state='disable')
    except:
       pass
    root.after(1000, Update)



#Defines the tkinter window
root = Tk()

#Text boxes
InfoBox = Text(root, state='disabled', height=65, width=45,  bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
InfoBox.place(x=1565, y=0)

#Images
LFBMap = PhotoImage(file="LFBMap.png")
RoseyIcon = PhotoImage(file="RoseyIcon.png")
RoseyInfo = PhotoImage(file="RoseyInfo.png")

#Labels
Map = Label(root, image=LFBMap, bd=0)
Map.place(x=0, y=0)

RoseyText = Label(root, text="Rosey", height=8, width=160, bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
RoseyText.place(x=127, y=953)

RoseyFace = Label(root, image=RoseyIcon, bd=0)
RoseyFace.place(x=0, y=953)
RoseyCool = Label(root, image=RoseyInfo, bd=0)


#Buttons
Btn1 = Button(root, text="Barking and\nDagenham", command=lambda: DisplayInfo(1))
Btn2 = Button(root, text="Barnet", command=lambda: DisplayInfo(2))
Btn3 = Button(root, text="Bexley", command=lambda: DisplayInfo(3))
Btn4 = Button(root, text="Brent", command=lambda: DisplayInfo(4))
Btn5 = Button(root, text="Bromley", command=lambda: DisplayInfo(5))
Btn6 = Button(root, text="Camden", command=lambda: DisplayInfo(6))
Btn7 = Button(root, text="City", command=lambda: DisplayInfo(7))
Btn8 = Button(root, text="Croydon", command=lambda: DisplayInfo(8))
Btn9 = Button(root, text="Ealing", command=lambda: DisplayInfo(9))
Btn10 = Button(root, text="Enfield", command=lambda: DisplayInfo(10))
Btn11 = Button(root, text="Greenwich", command=lambda: DisplayInfo(11))
Btn12 = Button(root, text="Hackney", command=lambda: DisplayInfo(12))	
Btn13 = Button(root, text="Hammersmith\nand Fulham", command=lambda: DisplayInfo(13))
Btn14 = Button(root, text="Haringey", command=lambda: DisplayInfo(14))
Btn15 = Button(root, text="Harrow", command=lambda: DisplayInfo(15))
Btn16 = Button(root, text="Havering", command=lambda: DisplayInfo(16))
Btn17 = Button(root, text="Hillingdon", command=lambda: DisplayInfo(17))
Btn18 = Button(root, text="Hounslow", command=lambda: DisplayInfo(18))
Btn19 = Button(root, text="Islington", command=lambda: DisplayInfo(19))
Btn20 = Button(root, text="Kensington\nand Chelsea", command=lambda: DisplayInfo(20))
Btn21 = Button(root, text="Kingston\nupon\nThames", command=lambda: DisplayInfo(21))
Btn22 = Button(root, text="Lambeth", command=lambda: DisplayInfo(22))
Btn23 = Button(root, text="Lewisham", command=lambda: DisplayInfo(23))
Btn24 = Button(root, text="Merton", command=lambda: DisplayInfo(24))
Btn25 = Button(root, text="Newham", command=lambda: DisplayInfo(25))	
Btn26 = Button(root, text="Redbridge", command=lambda: DisplayInfo(26))
Btn27 = Button(root, text="Richmond\nupon Thames", command=lambda: DisplayInfo(27))
Btn28 = Button(root, text="Southwark", command=lambda: DisplayInfo(28))
Btn29 = Button(root, text="Sutton", command=lambda: DisplayInfo(29))	
Btn30 = Button(root, text="Tower\nHamlets", command=lambda: DisplayInfo(30))
Btn31 = Button(root, text="Waltham\nForest", command=lambda: DisplayInfo(31))	
Btn32 = Button(root, text="Wandsworth", command=lambda: DisplayInfo(32))
Btn33 = Button(root, text="Westminster", command=lambda: DisplayInfo(33))
Btn1.place(x=1060, y=360)
Btn2.place(x=620, y=200)
Btn3.place(x=1095, y=530)
Btn4.place(x=550, y=325)
Btn5.place(x=970, y=725)
Btn6.place(x=669, y=335)
Btn7.place(x=782, y=414)
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
RoseyCool.place(x=1565, y=60)
InfoBox.configure(state='normal')
InfoBox.insert("end", '''
 



































         Rosey is an Artificial Intelligence model that
          uses predetermined Fire-fighting rules along
          with trained decision making based off the
         brilliant mind of an experienced firefighter.''')
InfoBox.configure(state='disabled')
NB = 1
Update()
    
#Sets the tkinter window 
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title("Rosey")
root.mainloop()

