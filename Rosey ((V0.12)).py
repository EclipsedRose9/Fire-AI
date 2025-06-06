import os
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import re
import ast
import requests

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


def extract_and_process_stations(file_path, target_borough):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    # Split content into records using blank lines as delimiters
    records = re.split(r'\n\s*\n', content)
    output_sections = []
    
    for record in records:
        lines = record.splitlines()
        if len(lines) < 4:
            continue  # Skip incomplete records
        
        borough = lines[0].strip()
        if borough.lower() != target_borough.lower():
            continue  # Only process records for the target borough
        
        station_name = lines[1].strip()  # Fire station name
        station_address = lines[2].strip()
        # Remove commas from the fire station address for comparison
        cleaned_station_address = station_address.replace(",", "").strip()
        
        # Parse the 2D array line (line 4) containing the code/address pairs
        array_line = lines[3].strip()
        try:
            code_entries = ast.literal_eval(array_line)
        except Exception as e:
            print(f"Error parsing 2D array in record starting with '{station_name}': {e}")
            continue
        
        valid_codes = []
        # Loop through each code entry (expected format: [code, address])
        for entry in code_entries:
            if len(entry) < 2:
                continue
            code = entry[0].strip()
            entry_address = entry[1].strip()
            # Only accept the code if the entry address matches the cleaned station address
            if entry_address == cleaned_station_address:
                valid_codes.append(code)
        
        # Only output the station if there's at least one valid code
        if valid_codes:
            section_output = station_name + "\n"
            # Append each valid code on a new line with ' at Station'
            for code in valid_codes:
                section_output += f"{code} at Station\n"
            output_sections.append(section_output.strip())
    
    # Combine all sections into one output string
    final_output = "\n\n".join(output_sections)
    return final_output

def GetPDA(x, y):
    # Fire Incidents
    if str(x) == "House Fire" and str(LiveAtRisk) == "No":
        PDA = ""+y+" P, 1 PL and 1 officer in charge"
    elif str(x) == "House Fire" and str(LiveAtRisk) == "Yes":
        PDA = ""+y+" P, 1 PL and 1 officer in charge, ambulance notified"
    elif str(x) == "High-Rise Fire":
        PDA = ""+y+" P, 1 ALP, 1 CU and multiple officers"
    elif str(x) == "Large Vehicle Fire":
        PDA = "2 P, 1 PL and 1 officer in charge"
    elif str(x) == "Bin/Small Rubbish Fire":
        PDA = "1 P"
    elif str(x) == "Cladding Fire (High-Rise)":
        PDA = ""+y+" P, 3 PL, multiple officers and CU"
    elif str(x) == "Fire with Fatality":
        PDA = ""+y+" P, 2 FIU and press officer notified"
    elif str(x) == "Fire Survival Guidance (FSG) Calls":
        PDA = "Additional CU and personnel dispatched for complex life-risk situations"
    
    # Special Services (Rescues & Non-Fire Emergencies)
    elif str(x) == "Person Trapped (Non-Road Traffic Collision)":
        PDA = "2 P, 1 HRU and 1 officer in charge"
    elif str(x) == "Road Traffic Collision (Person Trapped)":
        PDA = "2 P, 1 HRU, 1 officer in charge, ambulance notified"
    elif str(x) == "Person Shut in Lift (Non-Emergency)":
        PDA = "1 P (only if no other help available)"
    elif str(x) == "Animal Rescue (Large Animal)":
        PDA = "1 P, 1 HRU and specialist officers"
    elif str(x) == "Flooding (Non-Commercial)":
        PDA = "1 P"
    
    # Hazardous Materials (HazMat) Incidents
    elif str(x) == "Gas Leak (Domestic/Commercial)":
        PDA = "1 P and gas authority notified"
    elif str(x) == "Minor Hazardous Spill (under 100L)":
        PDA = "1 P"
    elif str(x) == "Major Hazardous Spill (over 100L)":
        PDA = "2 P, HazMat unit and CU"
    elif str(x) == "Suspicious Substance/White Powder":
        PDA = "1 HazMat officer and police notified"
    elif str(x) == "Deliberate HazMat Release":
        PDA = "6 P, multiple CU, HazMat and specialist units"
    
    # Other Emergencies
    elif str(x) == "Person Threatening to Jump/Rescue from Height":
        PDA = "1 P, 1 PL, ambulance and police notified"
    elif str(x) == "Person Collapsed Behind Locked Door":
        PDA = "1 P and ambulance notified"
    elif str(x) == "Vehicle into Building":
        PDA = "2 P, 1 HRU, 1 structural officer and ambulance"
    elif str(x) == "Train or Tram Crash":
        PDA = "4 P, 1 CU, 1 HRU, 1 USAR team"
    elif str(x) == "Aircraft Accident":
        PDA = "Multiple P, CU, specialist officers"
    elif str(x) == "Fire on a Vessel":
        PDA = "3 P, fireboat, CU support"
    
    # Additional Response Triggers
    elif str(x) == "Multiple Calls (4 or more)":
        PDA = "Additional CU and station commander dispatched"
    elif str(x) == "High-Rise Cladding Fire":
        PDA = "8 P, multiple PL, CU and senior officers"
    elif str(x) == "Confirmed Life Risk":
        PDA = "Additional P and ambulance dispatched immediately"
    elif str(x) == "Large Fires (6+ P)":
        PDA = "Extra CU, fire safety officers and senior officers dispatched"
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

    if __name__ == '__main__':
        file_path = 'Stations.txt'  # Replace with the path to your test file
        target_borough = x  # Change this to the borough you want to find
    
        result = extract_and_process_stations(file_path, target_borough)
    

    InfoBox.configure(state='normal')
    InfoBox.insert("end", "".join(result))
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
        MapAdress = str(Adress)+", "+str(Borough)+", "+str(Postcode)
        print(MapAdress)
        station_file = "Stations.txt"
        best_pair, travel_time = find_best_fire_engine(MapAdress, station_file)
        if best_pair:
            print("\nNearest fire engine (from 2D array) info:")
            print(f"Station: {best_pair['station_name']} (Borough: {best_pair['borough']})")
            print(f"Engine Code: {best_pair['code']}")
            print(f"Engine Address: {best_pair['engine_address']}")
            print(f"Travel time: {travel_time}")
        else:
            print("No fire engine found or an error occurred.")
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

def normalize_address_for_geocoding(address):
    """
    Attempts to normalize an address for geocoding without changing the underlying data.
    This function uses a regular expression to find a UK postcode pattern and ensures there's
    a comma before it. If no postcode is found, it falls back to the simple heuristic.
    """
    # If the address already contains a comma, assume it's reasonably formatted.
    if ',' in address:
        return address

    # Regex for a UK postcode pattern (roughly)
    postcode_pattern = r'([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})'
    match = re.search(postcode_pattern, address, re.IGNORECASE)
    if match:
        start = match.start()
        # Insert a comma before the postcode if not already there.
        if start > 0 and address[start-1] != ',':
            normalized = address[:start].strip() + ', ' + address[start:].strip()
            return normalized
    # Fallback: insert a comma before the last two tokens.
    tokens = address.split()
    if len(tokens) >= 4:
        normalized = " ".join(tokens[:-2]) + ", " + " ".join(tokens[-2:])
        return normalized
    return address

def geocode_address(address):
    """
    Uses Nominatim (OpenStreetMap) to geocode an address.
    First, it normalizes the address (for the query only) and appends ', United Kingdom'
    if not already present.
    Returns a (lat, lon) tuple if successful, or None.
    """
    normalized_address = normalize_address_for_geocoding(address)
    if "UK" not in normalized_address and "United Kingdom" not in normalized_address:
        normalized_address = normalized_address + ", United Kingdom"
    print("Geocoding:", normalized_address)
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": normalized_address,
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MyScript/1.0; +http://example.com)"}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return (lat, lon)
    except Exception as e:
        print(f"Geocoding error for address '{normalized_address}': {e}")
    return None

def parse_station_array(array_line):
    pattern = r'\[\s*"?([A-Za-z]+)"?\s*,\s*"?([^"\]]+)"?\s*\]'
    matches = re.findall(pattern, array_line)
    return matches

def load_station_pairs(station_file):
    with open(station_file, 'r') as f:
        content = f.read().strip()
    records = re.split(r'\n\s*\n', content)
    pairs = []
    for record in records:
        lines = record.splitlines()
        if len(lines) < 4:
            continue
        borough = lines[0].strip()
        station_name = lines[1].strip()
        station_address = lines[2].strip()
        array_line = lines[3].strip()
        code_entries = parse_station_array(array_line)
        for entry in code_entries:
            code = entry[0].strip()
            engine_address = entry[1].strip()
            pairs.append({
                'borough': borough,
                'station_name': station_name,
                'station_address': station_address,
                'code': code,
                'engine_address': engine_address
            })
    return pairs

def get_route_duration_osrm(origin, destination):
    url = f"http://router.project-osrm.org/route/v1/driving/{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
    params = {"overview": "false"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "routes" in data and data["routes"]:
            return data["routes"][0]["duration"]
    except Exception as e:
        print(f"OSRM error for route from {origin} to {destination}: {e}")
    return None

def find_best_fire_engine(destination_address, station_file):
    fire_engines = load_station_pairs(station_file)
    if not fire_engines:
        print("No fire engine pairs found.")
        return None, None

    dest_coords = geocode_address(destination_address)
    if not dest_coords:
        print("Could not geocode destination address.")
        return None, None

    best_pair = None
    best_duration = None

    for engine in fire_engines:
        coords = geocode_address(engine['engine_address'])
        if not coords:
            print(f"Could not geocode engine address for station: {engine['station_name']} (code: {engine['code']})")
            continue

        duration = get_route_duration_osrm(coords, dest_coords)
        if duration is None:
            print(f"Could not get route duration for station: {engine['station_name']} (code: {engine['code']})")
            continue

        print(f"Station {engine['station_name']} ({engine['code']}): duration {duration} sec")

        if best_duration is None or duration < best_duration:
            best_duration = duration
            best_pair = engine

    if best_pair and best_duration is not None:
        minutes = int(best_duration // 60)
        seconds = int(best_duration % 60)
        travel_time_str = f"{minutes} min {seconds} sec"
        return best_pair, travel_time_str
    else:
        return None, None

#Defines the tkinter window
root = Tk()

#Text boxes
InfoBox = Text(root, state='disabled', height=65, width=45,  bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
InfoBox.place(x=1565, y=0)

#Images
LFBMap = PhotoImage(file="LFBMap.png")
RoseyIcon = PhotoImage(file="RoseyIcon.png")

#Labels
Map = Label(root, image=LFBMap, bd=0)
Map.place(x=0, y=0)

RoseyText = Label(root, text="Rosey", height=8, width=160, bg="#FFFFFF", fg="#000000", bd=0, font=("Arial",11))
RoseyText.place(x=127, y=953)

RoseyFace = Label(root, image=RoseyIcon, bd=0)
RoseyFace.place(x=0, y=953)


#Buttons
Btn1 = Button(root, text="Barking and\nDagenham", command=lambda: DisplayInfo("Barking and Dagenham"))
Btn2 = Button(root, text="Barnet", command=lambda: DisplayInfo("Barnet"))
Btn3 = Button(root, text="Bexley", command=lambda: DisplayInfo("Bexley"))
Btn4 = Button(root, text="Brent", command=lambda: DisplayInfo("Brent"))
Btn5 = Button(root, text="Bromley", command=lambda: DisplayInfo("Bromley"))
Btn6 = Button(root, text="Camden", command=lambda: DisplayInfo("Camden"))
Btn7 = Button(root, text="City", command=lambda: DisplayInfo("City"))
Btn8 = Button(root, text="Croydon", command=lambda: DisplayInfo("Croydon"))
Btn9 = Button(root, text="Ealing", command=lambda: DisplayInfo("Ealing"))
Btn10 = Button(root, text="Enfield", command=lambda: DisplayInfo("Enfield"))
Btn11 = Button(root, text="Greenwich", command=lambda: DisplayInfo("Greenwich"))
Btn12 = Button(root, text="Hackney", command=lambda: DisplayInfo("Hackney"))	
Btn13 = Button(root, text="Hammersmith\nand Fulham", command=lambda: DisplayInfo("Hammersmith and Fulham"))
Btn14 = Button(root, text="Haringey", command=lambda: DisplayInfo("Haringey"))
Btn15 = Button(root, text="Harrow", command=lambda: DisplayInfo("Harrow"))
Btn16 = Button(root, text="Havering", command=lambda: DisplayInfo("Havering"))
Btn17 = Button(root, text="Hillingdon", command=lambda: DisplayInfo("Hillingdon"))
Btn18 = Button(root, text="Hounslow", command=lambda: DisplayInfo("Hounslow"))
Btn19 = Button(root, text="Islington", command=lambda: DisplayInfo("Islington"))
Btn20 = Button(root, text="Kensington\nand Chelsea", command=lambda: DisplayInfo("Kensington and Chelsea"))
Btn21 = Button(root, text="Kingston\nupon\nThames", command=lambda: DisplayInfo("Kingston upon Thames"))
Btn22 = Button(root, text="Lambeth", command=lambda: DisplayInfo("Lambeth"))
Btn23 = Button(root, text="Lewisham", command=lambda: DisplayInfo("Lewisham"))
Btn24 = Button(root, text="Merton", command=lambda: DisplayInfo("Merton"))
Btn25 = Button(root, text="Newham", command=lambda: DisplayInfo("Newham"))	
Btn26 = Button(root, text="Redbridge", command=lambda: DisplayInfo("Redbridge"))
Btn27 = Button(root, text="Richmond\nupon Thames", command=lambda: DisplayInfo("Richmond upon Thames"))
Btn28 = Button(root, text="Southwark", command=lambda: DisplayInfo("Southwark"))
Btn29 = Button(root, text="Sutton", command=lambda: DisplayInfo("Sutton"))	
Btn30 = Button(root, text="Tower\nHamlets", command=lambda: DisplayInfo("Tower Hamlets"))
Btn31 = Button(root, text="Waltham\nForest", command=lambda: DisplayInfo("Waltham Forest"))	
Btn32 = Button(root, text="Wandsworth", command=lambda: DisplayInfo("Wandsworth"))
Btn33 = Button(root, text="Westminster", command=lambda: DisplayInfo("Westminster"))
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

NB = 1
Update()
    
#Sets the tkinter window 
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.title("Rosey")
root.mainloop()

